from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..huffman import build_huffman_code, encode_symbols, decode_symbols

@dataclass
class SemanticOut:
    embed_hat: torch.Tensor
    indices: torch.LongTensor
    rate_bits: torch.Tensor
    vq_loss: torch.Tensor
    aux: Dict

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.K = int(codebook_size)
        self.D = int(dim)
        self.beta = float(beta)
        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.K, 1.0 / self.K)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        x_flat = x.reshape(-1, self.D)
        x2 = (x_flat ** 2).sum(dim=1, keepdim=True)
        e2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)
        xe = x_flat @ self.codebook.weight.t()
        dist = x2 + e2 - 2.0 * xe
        idx = dist.argmin(dim=1)
        x_q = self.codebook(idx).view_as(x)
        x_q_st = x + (x_q - x).detach()
        loss_codebook = F.mse_loss(x_q, x.detach())
        loss_commit = F.mse_loss(x, x_q.detach())
        vq_loss = loss_codebook + self.beta * loss_commit
        return x_q_st, idx, vq_loss

class SemanticVQCompressor(nn.Module):
    def __init__(self, hidden_size: int, code_dim: int = 256, codebook_size: int = 4096, beta: float = 0.25):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.pre = nn.Linear(self.hidden_size, self.code_dim)
        self.vq = VectorQuantizer(self.codebook_size, self.code_dim, beta=beta)
        self.post = nn.Linear(self.code_dim, self.hidden_size)
        self.prior_logits = nn.Parameter(torch.zeros(self.codebook_size))
        self._cached_code: Optional[Dict[int, str]] = None
        self._cached_hash: Optional[int] = None

    def forward(self, embed: torch.Tensor) -> SemanticOut:
        z = self.pre(embed)
        z_q, idx, vq_loss = self.vq(z)
        embed_hat = self.post(z_q)
        logp = F.log_softmax(self.prior_logits, dim=0)
        bits = (-logp[idx] / torch.log(torch.tensor(2.0, device=embed.device))).mean() * idx.numel()
        return SemanticOut(embed_hat=embed_hat, indices=idx, rate_bits=bits, vq_loss=vq_loss, aux={"K": self.codebook_size})

    @torch.no_grad()
    def _get_huffman_code(self) -> Dict[int, str]:
        h = int(torch.sum(self.prior_logits.detach().cpu() * 1000).item())
        if self._cached_code is None or self._cached_hash != h:
            probs = torch.softmax(self.prior_logits.detach().cpu(), dim=0).tolist()
            self._cached_code = build_huffman_code(probs)
            self._cached_hash = h
        return self._cached_code

    @torch.no_grad()
    def compress_indices(self, indices: torch.LongTensor) -> Tuple[bytes, Dict]:
        idx = indices.detach().cpu().tolist()
        code = self._get_huffman_code()
        blob = encode_symbols(idx, code)
        meta = {"N": len(idx), "codec": "huffman", "K": self.codebook_size}
        return blob, meta

    @torch.no_grad()
    def decompress_indices(self, blob: bytes, meta: Dict, device: torch.device) -> torch.LongTensor:
        code = self._get_huffman_code()
        idx = decode_symbols(blob, code, n_symbols=int(meta["N"]))
        return torch.tensor(idx, dtype=torch.long, device=device)
