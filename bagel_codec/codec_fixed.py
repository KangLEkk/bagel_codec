from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bagel_adapter import forward_cache_update_vae_from_latent, forward_cache_update_vit_from_embeds, compute_packed_vit_embed
from .compressors.recon_cb_varbpp_type2 import ReconCBVarBppType2
from .compressors.semantic_tiktok_vq import SemanticTikTokVQ

@dataclass
class CodecLoss:
    total: torch.Tensor
    flow_mse: torch.Tensor
    recon_rate_bits: torch.Tensor
    sem_rate_bits: torch.Tensor
    vq_loss: torch.Tensor

class BagelCodec(nn.Module):
    def __init__(
        self,
        bagel_model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids: Dict[str, int],
        sem_code_dim: int = 256,
        sem_codebook_size: int = 4096,
        sem_beta: float = 0.25,
        sem_use_l2_norm: bool = False,
        sem_use_prior: bool = True,
        recon_quant_dim: int = 8,
        recon_bpp_num: int = 6,
        lambda_recon_rate: float = 1e-3,
        lambda_sem_rate: float = 1e-3,
        lambda_vq: float = 1.0,
        timestep_shift: float = 3.0,
    ):
        super().__init__()
        self.model = bagel_model
        self.vae = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        hidden = getattr(self.model, "hidden_size", None)
        if hidden is None:
            hidden = self.model.language_model.config.hidden_size
        self.hidden_size = int(hidden)

        self.recon = ReconCBVarBppType2(feat_dim=self.model.latent_channel, quant_dim=int(recon_quant_dim), bpp_num=int(recon_bpp_num))
        self.semantic = SemanticTikTokVQ(hidden_size=self.hidden_size, code_dim=sem_code_dim,
                                         codebook_size=sem_codebook_size, commitment_cost=float(sem_beta),
                                         use_l2_norm=bool(sem_use_l2_norm), use_prior=bool(sem_use_prior))

        self.lambda_recon_rate = float(lambda_recon_rate)
        self.lambda_sem_rate = float(lambda_sem_rate)
        self.lambda_vq = float(lambda_vq)
        self.timestep_shift = float(timestep_shift)
        self.recon_bpp_num = int(recon_bpp_num)

    def _encode_image_to_latent(self, img_01: torch.Tensor) -> torch.Tensor:
        x = img_01 * 2.0 - 1.0
        # x = self.model.pad_image(x)
        lat = self.vae.encode(x)
        return lat

    def forward_train(
        self,
        img_01: torch.Tensor,
        pil_images: List,
        prompt_text: Optional[str] = None,
        q_idx: Optional[int] = None,
    ) -> Tuple[CodecLoss, Dict[str, float]]:
        """Train-time forward.

        Notes:
            - Follows Bagel's training `forward(...)` pattern: build a packed sequence and call
              `self.model.language_model(...)` (NOT `forward_inference` / KV-cache incremental path).
            - We insert three conditioning blocks per sample:
                (1) recon-latent tokens (t=0) from ReconCompressor output (padded_latent_hat)
                (2) semantic ViT tokens from SemanticCompressor output (packed_vit_embed_hat)
                (3) flow-matching generation tokens x_t to predict v_target = x1 - x0
        """
        device = img_01.device
        B, _, H, W = img_01.shape

        # ---------------------------------------------------------------------
        # 1) Reconstruction branch (VAE-latent compression)
        # ---------------------------------------------------------------------
        padded_latent = self._encode_image_to_latent(img_01)  # [B, C, Hc, Wc], latent is padded to model grid
        if q_idx is None:
            q_idx = int(torch.randint(low=0, high=max(1, self.recon_bpp_num), size=(1,), device=device).item())
        recon_out = self.recon(padded_latent, img_HW=(H, W), q_idx=q_idx)
        padded_latent_hat = recon_out.latent_hat  # [B, C, Hc, Wc]

        # ---------------------------------------------------------------------
        # 2) Semantic branch (ViT token compression)
        # ---------------------------------------------------------------------
        # Build packed vit tokens / positions (same as Bagel, but without calling prepare_vit_images twice)
        from data.data_utils import patchify

        vit_token_seqlens: List[int] = []
        vit_tokens_list: List[torch.Tensor] = []
        vit_pos_list: List[torch.Tensor] = []

        for img in pil_images:
            img_t = self.vit_transform(img)  # CHW
            if not torch.is_tensor(img_t):
                raise TypeError("vit_transform must return a torch.Tensor (CHW).")
            img_t = img_t.to(device)
            vit_pos = self.model.get_flattened_position_ids(
                img_t.size(1), img_t.size(2),
                self.model.vit_patch_size,
                max_num_patches_per_side=self.model.vit_max_num_patch_per_side,
            )
            vit_pos = vit_pos.to(device)
            vit_tokens = patchify(img_t, self.model.vit_patch_size)  # [Npatch, patch_dim]
            vit_tokens_list.append(vit_tokens.to(device))
            vit_pos_list.append(vit_pos)
            vit_token_seqlens.append(int(vit_tokens.shape[0]))

        packed_vit_tokens = torch.cat(vit_tokens_list, dim=0)
        packed_vit_position_ids = torch.cat(vit_pos_list, dim=0)
        vit_token_seqlens_t = torch.tensor(vit_token_seqlens, device=device, dtype=torch.int)

        # Run Bagel ViT + connector to get LLM-space embeddings
        cu_seqlens = F.pad(torch.cumsum(vit_token_seqlens_t, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(vit_token_seqlens_t.max().item()) if vit_token_seqlens_t.numel() else 0
        packed_vit_embed = self.model.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_embed = self.model.connector(packed_vit_embed)  # [Nvit, hidden]
        sem_out = self.semantic(packed_vit_embed)
        packed_vit_embed_hat = sem_out.embed_hat  # [Nvit, hidden]

        # ---------------------------------------------------------------------
        # 3) Prepare latent tokens: x0 (clean), recon-hat (cond), x_t (gen)
        # ---------------------------------------------------------------------
        p = self.model.latent_patch_size
        C = self.model.latent_channel
        _, _, Hc, Wc = padded_latent.shape
        h = Hc // p
        w = Wc // p
        L_lat = h * w
        D_lat = p * p * C

        def _latent_to_tokens(lat: torch.Tensor) -> torch.Tensor:
            # lat: [B, C, Hc, Wc] -> [B, L_lat, D_lat]
            _lat = lat.reshape(B, C, h, p, w, p)
            _lat = torch.einsum("nchpwq->nhwpqc", _lat).contiguous()
            return _lat.reshape(B, L_lat, D_lat)

        x0_tokens = _latent_to_tokens(padded_latent)        # [B, L, D]
        recon_tokens = _latent_to_tokens(padded_latent_hat) # [B, L, D]

        x0_packed = torch.cat([x0_tokens[i] for i in range(B)], dim=0)         # [B*L, D]
        recon_packed = torch.cat([recon_tokens[i] for i in range(B)], dim=0)   # [B*L, D]

        # flow-matching: x_t = (1-t)*x0 + t*x1, v_target = x1 - x0
        x1 = torch.randn_like(x0_packed)
        u = torch.rand((x0_packed.size(0),), device=device, dtype=x0_packed.dtype)
        if self.timestep_shift == 1.0:
            t = u
        else:
            t = self.timestep_shift * u / (1.0 + (self.timestep_shift - 1.0) * u)
        x_t = (1.0 - t).unsqueeze(-1) * x0_packed + t.unsqueeze(-1) * x1
        v_target = x1 - x0_packed

        # latent 2D position ids (for latent_pos_embed), length == B*L_lat
        lat_pos_list: List[torch.Tensor] = []
        for _ in range(B):
            lat_pos = self.model.get_flattened_position_ids(
                H, W,
                self.model.latent_downsample,
                max_num_patches_per_side=self.model.max_latent_size,
            ).to(device)
            lat_pos_list.append(lat_pos)
        packed_latent_position_ids = torch.cat(lat_pos_list, dim=0)  # [B*L]

        # ---------------------------------------------------------------------
        # 4) Build a packed sequence (Bagel forward style) and run language_model(...)
        # ---------------------------------------------------------------------
        # Prompt token ids (shared across batch in this simple codec training)
        prompt_ids: List[int] = []
        if prompt_text:
            prompt_ids = self.tokenizer.encode(prompt_text)
            prompt_ids = [self.new_token_ids["bos_token_id"]] + prompt_ids + [self.new_token_ids["eos_token_id"]]

        soi = int(self.new_token_ids["start_of_image"])
        eoi = int(self.new_token_ids["end_of_image"])

        packed_text_ids: List[int] = []
        packed_text_indexes: List[int] = []
        packed_position_ids: List[int] = []

        recon_token_indexes: List[int] = []
        vit_token_indexes: List[int] = []
        gen_token_indexes: List[int] = []

        sample_lens: List[int] = []

        offset = 0
        vit_cursor = 0  # cursor into packed_vit_* (already packed in sample order)

        for i in range(B):
            rope = 0  # per-sample rope counter (text uses increasing positions, each image block uses one shared pos)

            # --- prompt ---
            for tid in prompt_ids:
                packed_text_ids.append(int(tid))
                packed_text_indexes.append(offset)
                packed_position_ids.append(rope)
                rope += 1
                offset += 1

            # --- recon condition latent block (t=0) ---
            packed_text_ids.append(soi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            recon_token_indexes.extend(range(offset, offset + L_lat))
            packed_position_ids.extend([rope] * L_lat)
            offset += L_lat
            packed_text_ids.append(eoi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            rope += 1

            # --- semantic vit block ---
            L_vit = vit_token_seqlens[i]
            packed_text_ids.append(soi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            vit_token_indexes.extend(range(offset, offset + L_vit))
            packed_position_ids.extend([rope] * L_vit)
            offset += L_vit
            packed_text_ids.append(eoi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            rope += 1

            # --- generation latent block (x_t, predict v) ---
            packed_text_ids.append(soi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            gen_token_indexes.extend(range(offset, offset + L_lat))
            packed_position_ids.extend([rope] * L_lat)
            offset += L_lat
            packed_text_ids.append(eoi); packed_text_indexes.append(offset); packed_position_ids.append(rope); offset += 1
            rope += 1

            # per-sample length
            sample_lens.append(len(prompt_ids) + (L_lat + 2) + (L_vit + 2) + (L_lat + 2))

            vit_cursor += L_vit

        sequence_length = offset
        packed_text_ids_t = torch.tensor(packed_text_ids, device=device, dtype=torch.long)
        packed_text_indexes_t = torch.tensor(packed_text_indexes, device=device, dtype=torch.long)
        packed_position_ids_t = torch.tensor(packed_position_ids, device=device, dtype=torch.long)

        # Embed text ids into packed_sequence
        packed_text_embedding = self.model.language_model.model.embed_tokens(packed_text_ids_t)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, packed_text_embedding.size(-1)))
        packed_sequence[packed_text_indexes_t] = packed_text_embedding

        # Fill recon condition latent tokens (t=0)
        recon_token_indexes_t = torch.tensor(recon_token_indexes, device=device, dtype=torch.long)
        t0 = torch.zeros_like(t)
        recon_embed = self.model.vae2llm(recon_packed) + self.model.time_embedder(t0) + self.model.latent_pos_embed(packed_latent_position_ids)
        packed_sequence[recon_token_indexes_t] = recon_embed

        # Fill semantic vit tokens
        vit_token_indexes_t = torch.tensor(vit_token_indexes, device=device, dtype=torch.long)
        # vit_pos embedding is on flattened position ids
        vit_embed_hat = packed_vit_embed_hat + self.model.vit_pos_embed(packed_vit_position_ids)
        packed_sequence[vit_token_indexes_t] = vit_embed_hat

        # Fill generation latent tokens x_t
        gen_token_indexes_t = torch.tensor(gen_token_indexes, device=device, dtype=torch.long)
        gen_embed = self.model.vae2llm(x_t) + self.model.time_embedder(t) + self.model.latent_pos_embed(packed_latent_position_ids)
        packed_sequence[gen_token_indexes_t] = gen_embed

        # Build nested full-attention masks (one per sample) to match Bagel's forward API.
        # 0.0 means attend; -inf means mask out (we keep all-to-all within each sample).
        nested_attention_masks = [packed_sequence.new_zeros((L, L)) for L in sample_lens]

        extra_inputs = {}
        if getattr(self.model, "use_moe", False):
            # understanding tokens: all discrete text + vit tokens (conditioning)
            und_idx = torch.cat([packed_text_indexes_t, vit_token_indexes_t], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_token_indexes_t,
            )

        last_hidden_state = self.model.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=nested_attention_masks,
            packed_position_ids=packed_position_ids_t,
            **extra_inputs,
        )

        # Predict v on generation tokens only
        v_pred = self.model.llm2vae(last_hidden_state[gen_token_indexes_t])

        flow_mse = F.mse_loss(v_pred, v_target)
        total = flow_mse + self.lambda_recon_rate * recon_out.rate_bits + self.lambda_sem_rate * sem_out.rate_bits + self.lambda_vq * sem_out.vq_loss

        loss = CodecLoss(
            total=total,
            flow_mse=flow_mse.detach(),
            recon_rate_bits=recon_out.rate_bits.detach(),
            sem_rate_bits=sem_out.rate_bits.detach(),
            vq_loss=sem_out.vq_loss.detach(),
        )
        logs = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_flow_mse": float(flow_mse.detach().cpu().item()),
            "recon_rate_bits": float(recon_out.rate_bits.detach().cpu().item()),
            "recon_bpp": float(recon_out.aux.get("bpp_value", 0.0)),
            "recon_q_idx": float(q_idx),
            "sem_rate_bits": float(sem_out.rate_bits.detach().cpu().item()),
            "vq_loss": float(sem_out.vq_loss.detach().cpu().item()),
        }
        return loss, logs

