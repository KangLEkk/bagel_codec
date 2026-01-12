# train/train_codec_packed_fsdp.py
import os
import yaml
import functools
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import set_seed

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens

from modeling.autoencoder import load_ae
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer

from bagel_codec.codec_packed import BagelCodecPacked

# ---- FSDP imports ----
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


def _get_layer_cls_set():
    """
    Return a set of transformer layer classes to auto-wrap.
    Works even if some imports are unavailable.
    """
    layer_cls = set()

    # LLM decoder layers (Bagel customized)
    try:
        from modeling.bagel.qwen2_navit import Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer
        layer_cls.update([Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer])
    except Exception:
        pass

    # SigLIP encoder layer (Transformers)
    try:
        from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
        layer_cls.add(SiglipEncoderLayer)
    except Exception:
        pass

    return layer_cls


def _maybe_apply_activation_ckpt(model: torch.nn.Module, layer_cls_set: set, enable: bool):
    if not enable:
        return
    check_fn = lambda m: m.__class__ in layer_cls_set
    wrapper_fn = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn)


def main():
    # ---- basic args (use env or edit here) ----
    # 你也可以改成 argparse，我这里保持最短可跑
    dataset_config_file = os.environ.get("DATASET_CONFIG", "data/configs/codec_t2i.yaml")
    llm_path = os.environ.get("LLM_PATH", "")
    vit_path = os.environ.get("VIT_PATH", "")
    vae_path = os.environ.get("VAE_PATH", "")

    lr = float(os.environ.get("LR", "1e-4"))
    total_steps = int(os.environ.get("TOTAL_STEPS", "100000"))
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    use_flex = os.environ.get("USE_FLEX", "1") == "1"
    use_act_ckpt = os.environ.get("ACT_CKPT", "1") == "1"

    # ---- dist init ----
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed(42 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- load dataset config ----
    with open(dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)

    # ---- build backbone on CPU first (safer for FSDP) ----
    llm_config = Qwen2Config.from_pretrained(llm_path)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"  # align Bagel default
    language_model = Qwen2ForCausalLM.from_pretrained(llm_path, config=llm_config, torch_dtype=torch.bfloat16)

    vit_config = SiglipVisionConfig.from_pretrained(vit_path)
    vit_model = SiglipVisionModel.from_pretrained(vit_path, config=vit_config, torch_dtype=torch.bfloat16)

    vae_model, vae_config = load_ae(local_path=vae_path)  # usually fp32/bf16 ok

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=3.0,
    )
    bagel = Bagel(language_model, vit_model, bagel_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(llm_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    bagel.language_model.resize_token_embeddings(len(tokenizer))

    # ---- freeze backbone (only train codec) ----
    for p in bagel.parameters():
        p.requires_grad = False
    bagel.eval()

    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad = False

    # ---- build codec (trainable) ----
    codec = BagelCodecPacked(bagel_model=bagel, new_token_ids=new_token_ids).train()

    # ---- dataset (PackedDataset aligns Bagel training) ----
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    dataset_config.vit_patch_size = 14
    dataset_config.max_num_patch_per_side = 70
    dataset_config.vae_image_downsample = 2 * vae_config.downsample
    dataset_config.max_latent_size = 64

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        expected_num_tokens=int(os.environ.get("EXPECTED_TOKENS", "32768")),
        use_flex=use_flex,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_wrapper(),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # ---- FSDP wrap codec root + auto-wrap transformer layers ----
    layer_cls_set = _get_layer_cls_set()
    auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls_set)

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    codec = FSDP(
        codec,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
    )

    _maybe_apply_activation_ckpt(codec, layer_cls_set, enable=use_act_ckpt)

    # ---- optimizer only on trainable params ----
    opt = torch.optim.AdamW([p for p in codec.parameters() if p.requires_grad], lr=lr)

    # ---- train loop ----
    step = 0
    for batch in train_loader:
        batch = batch.cuda(device)
        data = batch.to_dict()

        # VAE encode outside (as Bagel does)
        with torch.no_grad():
            # padded_images: (N_img, 3, H, W)
            data["padded_latent"] = vae_model.encode(data.pop("padded_images")).to(device)

        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = codec(**data)
            loss = out["loss_total"]

        loss.backward()
        opt.step()

        if rank == 0 and step % 20 == 0:
            print(
                f"[{step}] loss={loss.item():.4f} "
                f"flow={out['loss_flow_mse'].item():.4f} "
                f"recon_bits={out['recon_rate_bits'].item():.2f} "
                f"sem_bits={out['sem_rate_bits'].item():.2f} "
                f"vq={out['vq_loss'].item():.4f}"
            )

        step += 1
        if step >= total_steps:
            break

    dist.barrier()
    if rank == 0:
        print("done")


if __name__ == "__main__":
    main()
