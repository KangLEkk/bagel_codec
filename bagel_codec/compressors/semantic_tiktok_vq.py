from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiktok_quantizer import VectorQuantizer
from ..huffman import build_huffman_code, encode_symbols, decode_symbols


@dataclass
class SemanticOut:
    embed_hat: torch.Tensor          # (N, hidden)
    indices: torch.LongTensor        # (N,)
    rate_bits: torch.Tensor          # scalar total bits
    vq_loss: torch.Tensor            # scalar
    aux: Dict[str, Any]


class SemanticTikTokVQ(nn.Module):
    """TikTok/Bytedance VectorQuantizer adapted for packed sequence embeddings.

    Input: embed (N, hidden). We quantize over the sequence dimension by reshaping:
        (N, code_dim) -> (1, code_dim, N, 1)  (BCHW)
    """

    def __init__(
        self,
        hidden_size: int,
        code_dim: int = 256,
        codebook_size: int = 4096,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
        use_prior: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)

        self.pre = nn.Linear(self.hidden_size, self.code_dim)
        self.vq = VectorQuantizer(
            codebook_size=self.codebook_size,
            token_size=self.code_dim,
            commitment_cost=float(commitment_cost),
            use_l2_norm=bool(use_l2_norm),
        )
        self.post = nn.Linear(self.code_dim, self.hidden_size)

        self.use_prior = bool(use_prior)
        if self.use_prior:
            self.prior_logits = nn.Parameter(torch.zeros(self.codebook_size))
        else:
            self.register_parameter("prior_logits", None)

        self._cached_code: Optional[Dict[int, str]] = None
        self._cached_hash: Optional[int] = None

    def forward(self, embed: torch.Tensor) -> SemanticOut:
        if embed.dim() != 2:
            raise ValueError(f"Expected embed (N, hidden), got {tuple(embed.shape)}")
        N, _ = embed.shape

        z = self.pre(embed)  # (N, code_dim)

        # (N, code_dim) -> (1, code_dim, N, 1)
        z_bchw = z.transpose(0, 1).unsqueeze(0).unsqueeze(-1).contiguous()
        z_q, qinfo = self.vq(z_bchw)

        # back to (N, code_dim)
        z_q_seq = z_q.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
        embed_hat = self.post(z_q_seq)

        idx_hw = qinfo["min_encoding_indices"]  # (1, N, 1)
        indices = idx_hw.reshape(-1).to(torch.long)

        vq_loss = qinfo.get("quantizer_loss", None)
        if vq_loss is None:
            vq_loss = torch.tensor(0.0, device=embed.device, dtype=embed.dtype)

        if self.use_prior:
            logp = F.log_softmax(self.prior_logits, dim=0)
            rate_bits = (-logp[indices] / math.log(2.0)).sum()
        else:
            rate_bits = torch.tensor(float(indices.numel()) * math.log2(self.codebook_size), device=embed.device)

        aux = {"N": int(N), "K": int(self.codebook_size)}
        return SemanticOut(embed_hat=embed_hat, indices=indices, rate_bits=rate_bits, vq_loss=vq_loss, aux=aux)

    @torch.no_grad()
    def _get_huffman_code(self) -> Dict[int, str]:
        if not self.use_prior:
            probs = [1.0 / self.codebook_size] * self.codebook_size
            return build_huffman_code(probs)

        h = int(torch.sum(self.prior_logits.detach().cpu() * 1000).item())
        if self._cached_code is None or self._cached_hash != h:
            probs = torch.softmax(self.prior_logits.detach().cpu(), dim=0).tolist()
            self._cached_code = build_huffman_code(probs)
            self._cached_hash = h
        return self._cached_code

    @torch.no_grad()
    def compress_indices(self, indices: torch.LongTensor) -> Tuple[bytes, Dict[str, Any]]:
        idx = indices.detach().cpu().tolist()
        code = self._get_huffman_code()
        blob = encode_symbols(idx, code)
        meta = {"N": len(idx), "codec": "huffman", "K": self.codebook_size, "use_prior": self.use_prior}
        return blob, meta

    @torch.no_grad()
    def decompress_indices(self, blob: bytes, meta: Dict[str, Any], device: torch.device) -> torch.LongTensor:
        code = self._get_huffman_code()
        idx = decode_symbols(blob, code, n_symbols=int(meta["N"]))
        return torch.tensor(idx, dtype=torch.long, device=device)
