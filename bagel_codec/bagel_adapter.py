from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any

import torch

from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.bagel.qwen2_navit import NaiveCache

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

@dataclass
class BagelBundle:
    model: Any
    vae_model: Any
    tokenizer: Any
    vae_transform: Any
    vit_transform: Any
    new_token_ids: Dict[str, int]

def load_bagel_from_dir(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> BagelBundle:
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)

    if load_safetensors is None:
        raise ImportError("safetensors is required to load ema.safetensors")
    state = load_safetensors(os.path.join(model_path, "ema.safetensors"))
    model.load_state_dict(state, strict=False)

    model = model.to(device=device, dtype=dtype).eval()
    vae_model = vae_model.to(device=device, dtype=dtype).eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    return BagelBundle(model=model, vae_model=vae_model, tokenizer=tokenizer,
                       vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids)

def _device_of(model) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")

def forward_cache_update_vae_from_latent(bagel_model, past_key_values, generation_input: Dict[str, torch.Tensor], padded_latent: torch.Tensor, timestep: float = 0.0):
    device = _device_of(bagel_model)
    def mv(x): return x.to(device) if torch.is_tensor(x) else x

    packed_text_ids = mv(generation_input["packed_text_ids"])
    packed_text_indexes = mv(generation_input["packed_text_indexes"])
    patchified_shapes = generation_input["patchified_vae_latent_shapes"]
    packed_vae_position_ids = mv(generation_input["packed_vae_position_ids"])
    packed_vae_token_indexes = mv(generation_input["packed_vae_token_indexes"])
    packed_seqlens = mv(generation_input["packed_seqlens"])
    packed_position_ids = mv(generation_input["packed_position_ids"])
    key_values_lens = mv(generation_input["key_values_lens"])
    packed_indexes = mv(generation_input["packed_indexes"])
    packed_key_value_indexes = mv(generation_input["packed_key_value_indexes"])

    dtype = bagel_model.language_model.model.embed_tokens.weight.dtype
    packed_sequence = torch.zeros((packed_seqlens.sum().item(), bagel_model.hidden_size), device=device, dtype=dtype)

    packed_text_embedding = bagel_model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence[packed_text_indexes] = packed_text_embedding

    padded_latent = padded_latent.to(device=device, dtype=torch.float32)
    p = bagel_model.latent_patch_size
    B, C, H, W = padded_latent.shape
    h = H // p
    w = W // p
    latent = padded_latent.reshape(B, C, h, p, w, p)
    latent = torch.einsum("nchpwq->nhwpqc", latent).contiguous()
    latent = latent.reshape(B, h * w, p * p * C)

    packed_vae_tokens = torch.cat([latent[i, :patchified_shapes[i][0]] for i in range(B)], dim=0).to(device=device, dtype=dtype)
    packed_vae_token_embed = bagel_model.vae2llm(packed_vae_tokens)
    packed_vae_token_embed = packed_vae_token_embed + bagel_model.get_pos_embed(packed_vae_position_ids, bagel_model.latent_pos_embed)
    t = torch.full((packed_vae_token_embed.size(0),), float(timestep), device=device, dtype=packed_vae_token_embed.dtype)
    packed_vae_token_embed = packed_vae_token_embed + bagel_model.time_embedder(t)
    packed_sequence[packed_vae_token_indexes] = packed_vae_token_embed

    out = bagel_model.language_model.forward_inference(
        inputs_embeds=packed_sequence.unsqueeze(0),
        query_lens=packed_seqlens,
        packed_query_position_ids=packed_position_ids,
        packed_query_indexes=packed_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes,
        update_past_key_values=True,
        is_causal=False,
    )
    return out.past_key_values

def forward_cache_update_vit_from_embeds(bagel_model, past_key_values, generation_input: Dict[str, torch.Tensor], packed_vit_token_embed: torch.Tensor):
    device = _device_of(bagel_model)
    def mv(x): return x.to(device) if torch.is_tensor(x) else x

    packed_text_ids = mv(generation_input["packed_text_ids"])
    packed_text_indexes = mv(generation_input["packed_text_indexes"])
    packed_vit_position_ids = mv(generation_input["packed_vit_position_ids"])
    packed_vit_token_indexes = mv(generation_input["packed_vit_token_indexes"])
    packed_seqlens = mv(generation_input["packed_seqlens"])
    packed_position_ids = mv(generation_input["packed_position_ids"])
    key_values_lens = mv(generation_input["key_values_lens"])
    packed_indexes = mv(generation_input["packed_indexes"])
    packed_key_value_indexes = mv(generation_input["packed_key_value_indexes"])

    dtype = bagel_model.language_model.model.embed_tokens.weight.dtype
    packed_sequence = torch.zeros((packed_seqlens.sum().item(), bagel_model.hidden_size), device=device, dtype=dtype)

    packed_text_embedding = bagel_model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence[packed_text_indexes] = packed_text_embedding

    packed_vit_token_embed = packed_vit_token_embed.to(device=device, dtype=dtype)
    packed_vit_token_embed = packed_vit_token_embed + bagel_model.get_pos_embed(packed_vit_position_ids, bagel_model.vit_pos_embed)
    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

    out = bagel_model.language_model.forward_inference(
        inputs_embeds=packed_sequence.unsqueeze(0),
        query_lens=packed_seqlens,
        packed_query_position_ids=packed_position_ids,
        packed_query_indexes=packed_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes,
        update_past_key_values=True,
        is_causal=False,
    )
    return out.past_key_values

def compute_packed_vit_embed(bagel_model, generation_input: Dict[str, torch.Tensor]) -> torch.Tensor:
    device = _device_of(bagel_model)
    packed_vit_tokens = generation_input["packed_vit_tokens"].to(device)
    packed_vit_position_ids = generation_input["packed_vit_position_ids"].to(device)
    vit_out = bagel_model.vit_model(
        pixel_values=packed_vit_tokens.unsqueeze(0),
        position_ids=packed_vit_position_ids.unsqueeze(0),
    )
    vit_embed = vit_out.last_hidden_state.squeeze(0)
    vit_embed = bagel_model.connector(vit_embed)
    return vit_embed
