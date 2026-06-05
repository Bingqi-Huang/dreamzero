"""Merge a Realman LoRA checkpoint into the DreamZero-AgiBot base -> full checkpoint.

WHY: the Realman run used `save_lora_only=true`, so checkpoint-N contains ONLY the
trainable params (LoRA adapters on the DiT + the new action encoder/decoder). The
inference loader `GrootSimPolicy -> base_vla.load_lora()` is wrong for this kind of
checkpoint: it rebuilds the DiT base from RAW Wan2.1 (which (a) isn't even fully
downloaded here, and (b) is the wrong base -- the LoRA was trained on DreamZero-AgiBot,
not on raw Wan2.1). The correct base is the full DreamZero-AgiBot checkpoint, which is
self-contained (DiT + T5 + CLIP + VAE + action head).

This script reproduces the TRAINING-TIME assembly (experiment/base.py::create_model):
  1. instantiate(cfg.model)            # skip_component_loading=true -> no Wan2.1 load
  2. load DreamZero-AgiBot full shards # DiT + encoders + action head
  3. inject_lora_after_loading()       # create LoRA layers on the loaded DiT
  4. load the Realman LoRA checkpoint  # 800 LoRA + 14 action enc/dec keys
  5. merge_and_unload()                # fold LoRA into the DiT, drop adapters
then saves a FULL checkpoint with train_architecture="full" so the standard,
proven `from_pretrained` path serves it (exactly like DreamZero-AgiBot is served).

CPU-only (CUDA_VISIBLE_DEVICES="" recommended) so it never touches the GPUs.

Usage:
  CUDA_VISIBLE_DEVICES="" uv run python scripts/inference/merge_realman_lora.py \
      --ckpt   ./checkpoints/dreamzero_realman_lora_5k/checkpoint-1000 \
      --base   ./checkpoints/DreamZero-AgiBot \
      --out    ./checkpoints/dreamzero_realman_lora_5k/checkpoint-1000-merged
"""

import argparse
import json
import os
import shutil

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file


def _load_full_state_into(model, base_dir: str) -> None:
    """Load a full (sharded or single) safetensors checkpoint into model, strict=False."""
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    single_path = os.path.join(base_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        for i, shard in enumerate(shards):
            sd = load_file(os.path.join(base_dir, shard))
            model.load_state_dict(sd, strict=False)
            print(f"  loaded base shard {i+1}/{len(shards)}: {shard}")
            del sd
    elif os.path.exists(single_path):
        model.load_state_dict(load_file(single_path), strict=False)
    else:
        raise FileNotFoundError(f"No base weights in {base_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA-only checkpoint dir (has model.safetensors + experiment_cfg/)")
    ap.add_argument("--base", default=None, help="Full base checkpoint dir (default: cfg.pretrained_model_path)")
    ap.add_argument("--out", required=True, help="Output dir for the merged full checkpoint")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    exp_cfg_dir = os.path.join(args.ckpt, "experiment_cfg")
    conf_path = os.path.join(exp_cfg_dir, "conf.yaml")
    cfg = OmegaConf.load(conf_path)
    base_dir = args.base or cfg.pretrained_model_path
    print(f"[merge] ckpt={args.ckpt}")
    print(f"[merge] base={base_dir}")
    print(f"[merge] out ={args.out}")

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # 1. Instantiate the empty model (skip_component_loading=true -> no Wan2.1).
    print("[merge] 1/5 instantiate(cfg.model) ...")
    model = instantiate(cfg.model)
    model.eval()

    # 2. Load the full DreamZero-AgiBot base (DiT + T5 + CLIP + VAE + action head).
    print("[merge] 2/5 loading full base checkpoint ...")
    _load_full_state_into(model, base_dir)

    # 3. Inject LoRA layers onto the now-loaded base DiT (matches create_model).
    ah = model.action_head
    defer = bool(getattr(ah.config, "defer_lora_injection", False))
    if defer and hasattr(ah, "inject_lora_after_loading"):
        print("[merge] 3/5 inject_lora_after_loading() ...")
        ah.inject_lora_after_loading()
    else:
        print("[merge] 3/5 (no deferred LoRA injection needed)")

    # 4. Load the Realman LoRA + action encoder/decoder weights.
    print("[merge] 4/5 loading Realman LoRA checkpoint ...")
    lora_sd = load_file(os.path.join(args.ckpt, "model.safetensors"))
    n_lora = sum("lora" in k.lower() for k in lora_sd)
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    loaded = [k for k in lora_sd if k not in set(unexpected)]
    print(f"    ckpt keys={len(lora_sd)} (lora={n_lora}) | loaded={len(loaded)} "
          f"| unexpected={len(unexpected)} | model-missing={len(missing)}")
    if unexpected:
        print(f"    !! UNEXPECTED (first 5): {unexpected[:5]}")
        raise SystemExit("Unexpected keys means the LoRA structure did not match; aborting.")

    # 5. Merge LoRA into the DiT and drop the adapters.
    print("[merge] 5/5 merge_and_unload() ...")
    model.action_head.model = model.action_head.model.merge_and_unload()

    # Flip config to a plain FULL checkpoint so from_pretrained serves it directly.
    def _set_full(cfg_dict):
        inner = cfg_dict.get("config", cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        if isinstance(inner, dict):
            inner["train_architecture"] = "full"
            inner["defer_lora_injection"] = False
    _set_full(model.config.action_head_cfg)
    if hasattr(model.action_head, "config"):
        try:
            model.action_head.config.train_architecture = "full"
        except Exception:
            pass

    # Uniform bf16. merge_and_unload upcasts the merged DiT layers to fp32, leaving
    # a MIXED-dtype model (~80GB). The bf16 server hides this by casting on load,
    # but the NVFP4/TensorRT export path does NOT, and dies with
    # "mat1 and mat2 must have the same dtype (BFloat16 vs Half)" during calibration.
    # Casting to clean bf16 here (matching DreamZero-AgiBot) makes the checkpoint
    # ~44GB AND lets the engine build succeed. Training + eval both run bf16 anyway.
    print("[merge] casting merged model to bfloat16 ...")
    model = model.to(torch.bfloat16)

    print("[merge] saving merged full checkpoint ...")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    # Carry over the experiment_cfg, but mark save_lora_only=false so the loader
    # uses from_pretrained (full) instead of load_lora.
    out_exp = os.path.join(args.out, "experiment_cfg")
    os.makedirs(out_exp, exist_ok=True)
    shutil.copy(os.path.join(exp_cfg_dir, "metadata.json"), os.path.join(out_exp, "metadata.json"))
    merged_cfg = OmegaConf.load(conf_path)
    merged_cfg.save_lora_only = False
    OmegaConf.save(merged_cfg, os.path.join(out_exp, "conf.yaml"))

    print(f"[merge] DONE -> {args.out}")
    print("[merge] serve with: --model-path", args.out)


if __name__ == "__main__":
    main()
