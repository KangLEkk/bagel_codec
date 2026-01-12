from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import zlib
import torch, math
import torch.nn as nn

from collections import OrderedDict
from torch import nn
from einops import rearrange
from .blocks import DepthConvBlock4
from .entropy.compression_model import CompressionModel, CompressionModel_type2, CompressionModel_type3

class zero_Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        for p in self.parameters():
            p.detach().zero_()
    
    def set_to_zero(self):
        for p in self.parameters():
            p.detach().zero_()


class zero_Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        for p in self.parameters():
            p.detach().zero_()

    def set_to_zero(self):
        for p in self.parameters():
            p.detach().zero_()

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

class Compressive_bottleneck(CompressionModel):
    def __init__(self, N):
        super().__init__(y_distribution='gaussian', z_channel=N, ec_thread=False, stream_part=1)
        inplace = False
        # factorized hype-prior
        self.factorized_prior_vec = nn.Parameter(torch.ones((1, N, 1, 1)), requires_grad=True)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(N, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 3, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock4(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock4(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock4(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock4(N * 2, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 2, inplace=inplace),
        )

    
    def forward(self, y, img_HW):
        B, C, H, W = y.shape
        dummy_param = self.factorized_prior_vec.repeat(B, 1, H, W)
        params = self.y_prior_fusion(dummy_param)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        # calculate
        if self.training:
            y_for_bit = self.add_noise(y_res)
        else:
            y_for_bit = y_q
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        H_img, W_img = img_HW
        pixel_num = H_img * W_img
        bpp_y = torch.mean(torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num)
        return y_hat, {"y_hat": y_hat, "bpp": bpp_y}


class Compressive_bottleneck_varbpp_type2(CompressionModel):
    def __init__(self, feat_dim, quant_dim, bpp_num):
        super().__init__(y_distribution='gaussian', z_channel=quant_dim, ec_thread=False, stream_part=1)
        inplace = False
        # bpp list
        self.enc_q = nn.Parameter(torch.ones((bpp_num, feat_dim, 1, 1)), requires_grad=True)
        self.dec_q = nn.Parameter(torch.ones((bpp_num, feat_dim, 1, 1)), requires_grad=True)

        # encoder transforms
        self.enc_trans_0 = nn.Sequential(
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
        )
        self.enc_trans_1 = nn.Sequential(
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
            DepthConvBlock4(feat_dim, quant_dim, inplace=False)
        )

        # decoder transforms
        self.dec_trans_0 = nn.Sequential(
            DepthConvBlock4(quant_dim, feat_dim, inplace=False),
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
        )
        self.dec_trans_1 = nn.Sequential(
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
            DepthConvBlock4(feat_dim, feat_dim, inplace=False),
        )

        # factorized hype-prior
        self.factorized_prior_vec = nn.Parameter(torch.ones((bpp_num, quant_dim, 1, 1)), requires_grad=True)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(quant_dim, quant_dim * 2, inplace=inplace),
            DepthConvBlock4(quant_dim * 2, quant_dim * 3, inplace=inplace),
        )

        # 4-step spatial prior
        self.y_spatial_prior_reduction = nn.Conv2d(quant_dim * 3, quant_dim * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace),
            DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace),
            DepthConvBlock4(quant_dim * 2, quant_dim * 2, inplace=inplace),
        )

    def get_qp(self, q_idx, shape):
        B, C, H, W = shape
        q_enc = self.enc_q[q_idx].unsqueeze(0).repeat(B, 1, H, W)
        q_dec = self.dec_q[q_idx].unsqueeze(0).repeat(B, 1, H, W)
        q_prior = self.factorized_prior_vec[q_idx].unsqueeze(0).repeat(B, 1, H, W)
        return q_enc, q_dec, q_prior

    def encode(self, y, qp):
        y = self.enc_trans_0(y)
        y = y * qp
        y = self.enc_trans_1(y)
        return y

    def decode(self, y_hat, qp):
        y_hat = self.dec_trans_0(y_hat)
        y_hat = y_hat * qp
        y_hat = self.dec_trans_1(y_hat)
        return y_hat
    
    def quant_4_part(self, y, qp):
        params = self.y_prior_fusion(qp)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)
        return y_res, y_q, y_hat, scales_hat
    
    def forward(self, y, img_HW, q_idx):
        q_enc, q_dec, q_prior = self.get_qp(q_idx, y.shape)

        y = self.encode(y, q_enc)
        y_res, y_q, y_hat, scales_hat = self.quant_4_part(y, q_prior)
        y_hat = self.decode(y_hat, q_dec)

        # calculate
        H_img, W_img = img_HW
        pixel_num = H_img * W_img

        # bpp without noise, for evaluate bpp
        y_for_bit_noise = self.add_noise(y_res)
        bits_y_noise = self.get_y_gaussian_bits(y_for_bit_noise, scales_hat)
        bpp_y_noise = torch.mean(torch.sum(bits_y_noise, dim=(1, 2, 3)) / pixel_num)

        # bpp with noise, for model training
        y_for_bit_direct = y_q.detach()
        bits_y_direct = self.get_y_gaussian_bits(y_for_bit_direct, scales_hat)
        bpp_y_direct = torch.mean(torch.sum(bits_y_direct, dim=(1, 2, 3)) / pixel_num)

        if self.training:
            bpp_y = bpp_y_noise
        else:
            bpp_y = bpp_y_direct

        return y_hat, {"y_hat": y_hat, "bpp": bpp_y, "bpp_direct": bpp_y_direct, "bpp_noise": bpp_y_noise}
    

    @torch.no_grad()
    def compress(self, y, q_idx):
        q_enc, q_dec, q_prior = self.get_qp(q_idx, y.shape)
        y = self.encode(y, q_enc)
        params = self.y_prior_fusion(q_prior)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
        scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
            y, 
            params, 
            self.y_spatial_prior_adaptor_1, 
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, 
            self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction
        )
        
        self.entropy_coder.reset()
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2, skip_thres=self.force_zero_thres)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3, skip_thres=self.force_zero_thres)
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()
        return bit_stream
    

    @torch.no_grad()
    def decompress(self, bit_stream, feat_shape, q_idx):
        self.entropy_coder.reset()
        self.entropy_coder.set_stream(bit_stream)
        device = next(self.parameters()).device
        q_enc, q_dec, q_prior = self.get_qp(q_idx, feat_shape)
        params = self.y_prior_fusion(q_prior)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior,
                                                self.y_spatial_prior_reduction)
        y_hat = self.decode(y_hat, q_dec)
        return y_hat


    @torch.no_grad()
    def compress_decompress(self, y, img_HW, q_idx):
        feat_shape = y.shape
        bit_stream = self.compress(y, q_idx)
        y_hat = self.decompress(bit_stream, feat_shape, q_idx)
        bpp = len(bit_stream) * 8 / (img_HW[0] * img_HW[1])

        # assert
        y_hat_for_valid, est_result = self.forward(y, img_HW, q_idx)
        assert torch.sum(torch.abs(y_hat - y_hat_for_valid)).item() == 0.0
        bpp_est = est_result['bpp'].item()
        bpp_diff = bpp - est_result['bpp'].item()

        return y_hat, {"y_hat": y_hat, "bpp": bpp, "bit_stream": bit_stream, 
                       "bpp_est": bpp_est, "bpp_diff": bpp_diff}


    @torch.no_grad()
    def get_entropy_map(self, y, q_idx):
        """ used for ablation study only.
        """
        # get bpp map
        y = y.clone().detach()
        q_enc, q_dec, q_prior = self.get_qp(q_idx, y.shape)
        y = self.encode(y, q_enc)
        y_res, y_q, y_hat, scales_hat = self.quant_4_part(y, q_prior)

        y_for_bit_direct = y_q.detach()
        bits_y_direct = self.get_y_gaussian_bits(y_for_bit_direct, scales_hat)
        return bits_y_direct

    
    @torch.no_grad()
    def compress_decompress_entropy_map(self, y, img_HW, q_idx):
        """ used for ablation study only.
        """
        # get entropy map
        entropy_map = self.get_entropy_map(y, q_idx)

        feat_shape = y.shape
        bit_stream = self.compress(y, q_idx)
        y_hat = self.decompress(bit_stream, feat_shape, q_idx)
        bpp = len(bit_stream) * 8 / (img_HW[0] * img_HW[1])

        # assert
        y_hat_for_valid, est_result = self.forward(y, img_HW, q_idx)
        assert torch.sum(torch.abs(y_hat - y_hat_for_valid)).item() == 0.0
        bpp_est = est_result['bpp'].item()
        bpp_diff = bpp - est_result['bpp'].item()

        return y_hat, {"y_hat": y_hat, "bpp": bpp, "bit_stream": bit_stream, 
                       "bpp_est": bpp_est, "bpp_diff": bpp_diff, "entropy_map": entropy_map}