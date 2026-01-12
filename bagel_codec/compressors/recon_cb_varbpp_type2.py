from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

# Wrap user's implementation.
# NOTE: recon_quant_user.py has extra deps (blocks.py, entropy/*). Ensure those are importable in your env.
from .recon_quant_user import Compressive_bottleneck_varbpp_type2


@dataclass
class ReconOut:
    latent_hat: torch.Tensor
    rate_bits: torch.Tensor  # total bits per image (mean over batch)
    aux: Dict[str, Any]


class ReconCBVarBppType2(nn.Module):
    """Recon compressor wrapper using Compressive_bottleneck_varbpp_type2.

    - forward(): differentiable path with bpp estimation
    - compress()/decompress(): real entropy-coded stream (depends on your entropy coder impl)
    """

    def __init__(self, feat_dim: int, quant_dim: int, bpp_num: int):
        super().__init__()
        self.core = Compressive_bottleneck_varbpp_type2(
            feat_dim=int(feat_dim),
            quant_dim=int(quant_dim),
            bpp_num=int(bpp_num),
        )
        self.bpp_num = int(bpp_num)

    def forward(self, latent: torch.Tensor, img_HW: Tuple[int, int], q_idx: int) -> ReconOut:
        y_hat, info = self.core(latent.float(), img_HW=img_HW, q_idx=int(q_idx))
        if "bpp" not in info:
            raise KeyError("Expected info['bpp'] from Compressive_bottleneck_varbpp_type2")
        bpp = info["bpp"]
        H, W = int(img_HW[0]), int(img_HW[1])
        rate_bits = bpp * (H * W)

        aux = dict(info)
        try:
            aux["bpp_value"] = float(bpp.detach().cpu().item())
        except Exception:
            aux["bpp_value"] = float(bpp)
        aux["q_idx"] = int(q_idx)
        return ReconOut(latent_hat=y_hat, rate_bits=rate_bits, aux=aux)

    @torch.no_grad()
    def compress(self, latent: torch.Tensor, q_idx: int) -> Tuple[bytes, Dict[str, Any]]:
        feat_shape = tuple(latent.shape)
        stream = self.core.compress(latent.float(), q_idx=int(q_idx))
        meta = {"feat_shape": feat_shape, "q_idx": int(q_idx)}
        return stream, meta

    @torch.no_grad()
    def decompress(self, stream: bytes, meta: Dict[str, Any], device: torch.device) -> torch.Tensor:
        feat_shape = meta["feat_shape"]
        q_idx = int(meta["q_idx"])
        y_hat = self.core.decompress(stream, feat_shape=feat_shape, q_idx=q_idx)
        return y_hat.to(device=device)
