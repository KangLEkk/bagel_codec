from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import zlib
import torch
import torch.nn as nn

@dataclass
class ReconOut:
    latent_hat: torch.Tensor
    rate_bits: torch.Tensor
    aux: Dict

class ReconQuantCompressor(nn.Module):
    def __init__(self, channels: int, init_log_step: float = -2.0, init_log_b: float = 0.0):
        super().__init__()
        self.log_step = nn.Parameter(torch.full((channels,), float(init_log_step)))
        self.log_b = nn.Parameter(torch.full((channels,), float(init_log_b)))

    def forward(self, latent: torch.Tensor) -> ReconOut:
        B, C, H, W = latent.shape
        step = self.log_step.exp().view(1, C, 1, 1)
        b = self.log_b.exp().view(1, C, 1, 1).clamp_min(1e-6)
        y = latent / step
        y_q = torch.round(y)
        y_hat = y + (y_q - y).detach()
        latent_hat = y_hat * step
        nll = (latent_hat.abs() / b) + torch.log(2.0 * b)
        rate_bits = (nll / torch.log(torch.tensor(2.0, device=latent.device))).sum(dim=(1,2,3)).mean()
        return ReconOut(latent_hat=latent_hat, rate_bits=rate_bits, aux={"step": step.detach(), "b": b.detach()})

    @torch.no_grad()
    def compress(self, latent: torch.Tensor) -> Tuple[bytes, Dict]:
        assert latent.dim() == 4 and latent.size(0) == 1
        _, C, H, W = latent.shape
        step = self.log_step.exp().view(1, C, 1, 1).to(latent.device)
        y_q = torch.round(latent / step).to(torch.int16).cpu().numpy()
        raw = y_q.tobytes()
        blob = zlib.compress(raw, level=9)
        meta = {"C": C, "H": H, "W": W, "dtype": "int16", "codec": "zlib"}
        return blob, meta

    @torch.no_grad()
    def decompress(self, blob: bytes, meta: Dict, device: torch.device) -> torch.Tensor:
        import numpy as np
        raw = zlib.decompress(blob)
        C, H, W = meta["C"], meta["H"], meta["W"]
        arr = np.frombuffer(raw, dtype=np.int16).reshape(1, C, H, W)
        y_q = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        step = self.log_step.exp().view(1, C, 1, 1).to(device)
        return y_q * step
