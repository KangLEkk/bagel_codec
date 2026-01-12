from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from data.data_utils import create_sparse_mask

from .compressors.recon_cb_varbpp_type2 import ReconCBVarBppType2
from .compressors.semantic_tiktok_vq import SemanticTikTokVQ


@dataclass
class CodecOut:
    total: torch.Tensor
    flow_mse: torch.Tensor
    recon_rate_bits: torch.Tensor
    sem_rate_bits: torch.Tensor
    vq_loss: torch.Tensor


class BagelCodecPacked(nn.Module):
    """
    PackedDataset-compatible codec training module.

    - Input keys align with Bagel.forward(...)
    - Insert recon compression on t=0 (packed_timesteps == -inf) latent segments
    - Insert semantic compression on ViT connector embeddings
    """

    def __init__(
        self,
        bagel_model,
        new_token_ids: Dict[str, int],
        # semantic vq
        sem_code_dim: int = 256,
        sem_codebook_size: int = 4096,
        sem_beta: float = 0.25,
        sem_use_l2_norm: bool = False,
        sem_use_prior: bool = True,
        # recon
        recon_quant_dim: int = 8,
        recon_bpp_num: int = 6,
        # loss weights
        lambda_recon_rate: float = 1e-3,
        lambda_sem_rate: float = 1e-3,
        lambda_vq: float = 1.0,
        detach_vit: bool = True,
    ):
        super().__init__()
        self.model = bagel_model
        self.new_token_ids = dict(new_token_ids)

        hidden = getattr(self.model, "hidden_size", None)
        if hidden is None:
            hidden = self.model.language_model.config.hidden_size
        self.hidden_size = int(hidden)

        self.recon = ReconCBVarBppType2(
            feat_dim=int(self.model.latent_channel),
            quant_dim=int(recon_quant_dim),
            bpp_num=int(recon_bpp_num),
        )
        self.semantic = SemanticTikTokVQ(
            hidden_size=self.hidden_size,
            code_dim=int(sem_code_dim),
            codebook_size=int(sem_codebook_size),
            commitment_cost=float(sem_beta),
            use_l2_norm=bool(sem_use_l2_norm),
            use_prior=bool(sem_use_prior),
        )

        self.lambda_recon_rate = float(lambda_recon_rate)
        self.lambda_sem_rate = float(lambda_sem_rate)
        self.lambda_vq = float(lambda_vq)
        self.recon_bpp_num = int(recon_bpp_num)
        self.detach_vit = bool(detach_vit)

    def train(self, mode: bool = True):
        """Keep the Bagel backbone in eval-mode.

        We run the backbone under `torch.no_grad()`; however, dropout / stochastic layers
        still depend on the module's train/eval flag. Keeping the backbone in eval-mode
        avoids training-time randomness leaking into the frozen path.
        """
        super().train(mode)
        self.model.eval()
        return self

    def _pack_latent_one(self, latent: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        latent: (C, H_lat, W_lat)  (H_lat/W_lat are in VAE latent pixels)
        h,w: token grid size (H_img/vae_image_downsample, W_img/vae_image_downsample)
        """
        p = int(self.model.config.latent_patch_size)
        Hp = h * p
        Wp = w * p
        latent = latent[:, :Hp, :Wp].contiguous()
        C = latent.shape[0]
        latent = latent.view(C, h, p, w, p).permute(1, 3, 0, 2, 4).contiguous()
        return latent.view(h * w, C * p * p)

    def forward(
        self,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        sample_lens: Sequence[int] | torch.Tensor,
        split_lens: Optional[Sequence[int]] = None,
        attn_modes: Optional[Sequence[str]] = None,
        nested_attention_masks: Optional[List[torch.Tensor]] = None,

        # VAE packed fields
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[Sequence[Sequence[int]]] = None,
        packed_latent_position_ids: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_timesteps: Optional[torch.Tensor] = None,
        mse_loss_indexes: Optional[torch.Tensor] = None,

        # ViT packed fields
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_position_ids: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.Tensor] = None,
        vit_token_seqlens: Optional[torch.Tensor] = None,

        # codec controls
        q_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        device = packed_position_ids.device
        dtype = self.model.language_model.model.embed_tokens.weight.dtype
        if isinstance(sample_lens, torch.Tensor):
            sample_lens_list: List[int] = [int(x) for x in sample_lens.tolist()]
            total_len = int(sample_lens.sum().item())
        else:
            sample_lens_list = [int(x) for x in sample_lens]
            total_len = int(sum(sample_lens_list))

        packed_sequence = torch.zeros((total_len, self.hidden_size), device=device, dtype=dtype)

        # ---- text embedding ----
        text_emb = self.model.language_model.model.embed_tokens(packed_text_ids.long().to(device))
        packed_sequence[packed_text_indexes.long()] = text_emb.to(dtype)

        # ---- vit embedding + semantic compression ----
        sem_rate_bits = torch.zeros((), device=device, dtype=torch.float32)
        vq_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.model.config.visual_und:
            with torch.no_grad():
                vit_out = self.model.vit_model(
                    packed_pixel_values=packed_vit_tokens.to(device),
                    packed_flattened_position_ids=packed_vit_position_ids.to(device),
                    cu_seqlens=torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32),
                    max_seqlen=int(vit_token_seqlens.max().item()),
                )
                vit_embed = self.model.connector(vit_out)

            if self.detach_vit:
                vit_embed = vit_embed.detach()

            sem_out = self.semantic(vit_embed.float())
            vit_embed_hat = sem_out.embed_hat.to(dtype)

            vit_pos = self.model.vit_pos_embed(packed_vit_position_ids.to(device))
            vit_embed_hat = vit_embed_hat + vit_pos

            packed_sequence[packed_vit_token_indexes.long()] = vit_embed_hat

            sem_rate_bits = sem_out.rate_bits.float()
            vq_loss = sem_out.vq_loss.float()

        # ---- vae latent embedding + recon compression on t=0 ----
        recon_rate_bits = torch.zeros((), device=device, dtype=torch.float32)
        flow_mse = torch.zeros((), device=device, dtype=torch.float32)

        if self.model.config.visual_gen:
            assert padded_latent is not None
            assert patchified_vae_latent_shapes is not None
            assert packed_latent_position_ids is not None
            assert packed_vae_token_indexes is not None
            assert packed_timesteps is not None
            assert mse_loss_indexes is not None

            # pack clean x0
            packed_latents = []
            seg_ranges = []
            cur = 0
            for lat, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                h = int(h); w = int(w)
                tok = self._pack_latent_one(lat, h, w)  # (h*w, dim)
                n = tok.shape[0]
                packed_latents.append(tok)
                seg_ranges.append((cur, cur + n, h, w))
                cur += n
            x0_clean = torch.cat(packed_latents, dim=0).to(device).float()

            # identify cond tokens (t=-inf)
            is_cond = torch.isneginf(packed_timesteps.to(device))

            # build x0 used for input (overwrite cond segments by recon(x0))
            x0_for_input = x0_clean.clone()
            cond_segs = [i for i, (s, e, _, _) in enumerate(seg_ranges) if is_cond[s].item()]

            if len(cond_segs) > 0:
                if q_idx is None:
                    q_idx = int(torch.randint(0, self.recon_bpp_num, (1,), device=device).item())

                # NOTE: cond latents can have different spatial shapes. Group by shape to batch.
                p = int(self.model.config.latent_patch_size)
                latent_downsample = int(getattr(self.model, "latent_downsample", 16))

                # collect per-cond-segment tensors
                cond_tensors: List[torch.Tensor] = []
                cond_hw: List[Tuple[int, int]] = []  # (h,w) token-grid
                for i in cond_segs:
                    _, _, h, w = seg_ranges[i]
                    Hp, Wp = h * p, w * p
                    cond_tensors.append(padded_latent[i][:, :Hp, :Wp])
                    cond_hw.append((h, w))

                # group by (Hp,Wp)
                groups: Dict[Tuple[int, int], List[int]] = {}
                for idx, (h, w) in enumerate(cond_hw):
                    groups.setdefault((h, w), []).append(idx)

                recon_rate_bits = torch.zeros((), device=device, dtype=torch.float32)
                total_bits = torch.zeros((), device=device, dtype=torch.float32)
                total_px: int = 0

                # hold outputs in original cond-seg order
                latent_hat_by_condidx: Dict[int, torch.Tensor] = {}

                for (h, w), idxs in groups.items():
                    batch = torch.stack([cond_tensors[j] for j in idxs], dim=0).to(device).float()
                    H_img = h * latent_downsample
                    W_img = w * latent_downsample
                    out = self.recon(batch, img_HW=(H_img, W_img), q_idx=q_idx)

                    # `out.rate_bits` is averaged over the batch; convert to totals for bpp
                    bsz = len(idxs)
                    total_bits = total_bits + out.rate_bits.float() * bsz
                    total_px += int(H_img) * int(W_img) * bsz

                    # save per-item latent_hat
                    for local_k, j in enumerate(idxs):
                        latent_hat_by_condidx[j] = out.latent_hat[local_k].detach()

                # mean bits per conditioned image
                recon_rate_bits = total_bits / max(1, len(cond_segs))
                recon_bpp = (total_bits / max(1, total_px)).detach()

                # overwrite token regions in the packed latent stream
                for seg_i, cond_i in enumerate(cond_segs):
                    s, e, h, w = seg_ranges[cond_i]
                    tok_hat = self._pack_latent_one(latent_hat_by_condidx[seg_i], h, w)
                    x0_for_input[s:e] = tok_hat.to(device).float()

            # noising (Bagel)
            noise = torch.randn_like(x0_clean)
            t = torch.sigmoid(packed_timesteps.to(device).float())
            ts = float(self.model.config.timestep_shift)
            t = (t * ts) / (1 + (ts - 1) * t)
            # keep the shifted sigmoid timesteps for later masking
            packed_timesteps = t

            x_t = (1.0 - t[:, None]) * x0_for_input + t[:, None] * noise
            target = noise - x0_clean

            # embed latent tokens into LLM space
            time_emb = self.model.time_embedder(t.to(dtype))
            pos_emb = self.model.latent_pos_embed(packed_latent_position_ids.to(device))
            lat_emb = self.model.vae2llm(x_t.to(dtype)) + time_emb + pos_emb
            packed_sequence[packed_vae_token_indexes.long()] = lat_emb

        # ---- MoT indexes (very important to match Bagel) ----
        extra_inputs = {}
        if getattr(self.model, "use_moe", False):
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs = dict(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,  # ✅所有 latent 都是 gen token
            )
        
        if nested_attention_masks is None:
            # use_flex=True path
            if split_lens is None or attn_modes is None:
                raise ValueError("use_flex path requires split_lens and attn_modes")
            sparse_mask = create_sparse_mask(
                sample_lens_list,
                [int(x) for x in split_lens],
                [str(x) for x in attn_modes],
                device,
            )
            seqlen = total_len
            num_heads = getattr(self.model, "num_heads", None)
            if num_heads is None:
                num_heads = int(self.model.language_model.config.num_attention_heads)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        # ---- LM forward (Bagel style) ----
        last_hidden = self.model.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens_list,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids.long(),
            **extra_inputs,
        )

        # ---- flow mse loss (Bagel style) ----
        if self.model.config.visual_gen:
            assert mse_loss_indexes is not None
            pred = self.model.llm2vae(last_hidden[mse_loss_indexes.to(device)])
            # match Bagel: only tokens with packed_timesteps > 0 have MSE targets
            has_mse = packed_timesteps > 0
            flow_mse = ((pred - target[has_mse]) ** 2).mean()

        total = flow_mse + self.lambda_recon_rate * recon_rate_bits + self.lambda_sem_rate * sem_rate_bits + self.lambda_vq * vq_loss
        return dict(
            loss_total=total,
            loss_flow_mse=flow_mse,
            recon_rate_bits=recon_rate_bits,
            sem_rate_bits=sem_rate_bits,
            vq_loss=vq_loss,
        )
