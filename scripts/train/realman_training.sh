#!/bin/bash
# DreamZero Realman LoRA post-training script.
#
# REALMAN_DATA_ROOTS is a comma-separated list of GEAR-converted LeRobot roots.
# Each root must contain meta/modality.json, meta/embodiment.json,
# meta/stats.json, and meta/relative_stats_dreamzero.json.

set -euo pipefail

export HYDRA_FULL_ERROR=1

# Safe defaults for multi-GPU fine-tuning.
#
# The first attempted 6-GPU smoke test triggered Torch Inductor compilation in
# every rank. PyTorch defaulted to many compile workers per rank, which can
# exhaust host RAM before the model even reaches the GPUs. Fine-tuning does not
# need these inference-oriented compile optimizations, so keep them off unless
# explicitly re-enabled.
SAFE_TRAINING=${SAFE_TRAINING:-1}

# TORCH_COMPILE_DIT=1 turns on torch.compile (Inductor kernel fusion) on the
# trainable CausalWanModel DiT blocks (consumed in groot/vla/experiment/base.py
# create_model). This targets the same WanRMSNorm / adaLN(modulate) ops RLinf
# compiled for its ~34% DreamZero-14B SFT speedup. When on, we must NOT globally
# disable Dynamo, and we keep TORCHINDUCTOR_COMPILE_THREADS=1 plus a shared
# Inductor cache so the 6 ranks don't each spawn a swarm of compile workers
# (the original host-RAM OOM concern) and so only the first run pays compile time.
# First step is slow while it compiles; default OFF keeps the proven baseline.
TORCH_COMPILE_DIT=${TORCH_COMPILE_DIT:-0}
if [ "$TORCH_COMPILE_DIT" = "1" ] || [ "$TORCH_COMPILE_DIT" = "true" ]; then
  export TORCH_COMPILE_DIT=1
  DISABLE_TORCH_COMPILE=${DISABLE_TORCH_COMPILE:-0}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/dreamzero_inductor}
  export TORCH_COMPILE_DIT_MODE=${TORCH_COMPILE_DIT_MODE:-default}
else
  export TORCH_COMPILE_DIT=0
  DISABLE_TORCH_COMPILE=${DISABLE_TORCH_COMPILE:-$SAFE_TRAINING}
fi
if [ "$DISABLE_TORCH_COMPILE" = "1" ] || [ "$DISABLE_TORCH_COMPILE" = "true" ]; then
  export DISABLE_TORCH_COMPILE=1
  export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
  export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
fi

export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DREAMZERO_MODEL_INIT_DTYPE=${DREAMZERO_MODEL_INIT_DTYPE:-bfloat16}
export STAGGER_MODEL_INIT=${STAGGER_MODEL_INIT:-1}
export STAGGER_MODEL_INIT_GROUP_SIZE=${STAGGER_MODEL_INIT_GROUP_SIZE:-3}
export MOVE_MODEL_TO_CUDA_AFTER_LOAD=${MOVE_MODEL_TO_CUDA_AFTER_LOAD:-1}
export STAGGER_MODEL_INIT_TIMEOUT_SEC=${STAGGER_MODEL_INIT_TIMEOUT_SEC:-1800}
export DREAMZERO_INIT_SESSION_ID=${DREAMZERO_INIT_SESSION_ID:-"realman-$(date +%Y%m%d-%H%M%S)-$$"}

DEFAULT_REALMAN_DATA_ROOTS=(
  "/home/bingqi/data/bingqi/CoRL26/Task1_new"
  "/home/bingqi/data/bingqi/CoRL26/Task1_CoRL"
  "/home/bingqi/data/bingqi/CoRL26/Task2_CoRL/Task2"
  "/home/bingqi/data/bingqi/CoRL26/Task3_CoRL/Task3"
  "/home/bingqi/data/bingqi/CoRL26/Task3_new"
)

if [ -z "${REALMAN_DATA_ROOTS:-}" ]; then
  REALMAN_DATA_ROOTS=$(IFS=,; echo "${DEFAULT_REALMAN_DATA_ROOTS[*]}")
fi

OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_realman_lora"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-AgiBot"}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"google/umt5-xxl"}
REPORT_TO=${REPORT_TO:-"wandb"}
WANDB_PROJECT=${WANDB_PROJECT:-"dreamzero_realman"}
MAX_STEPS=${MAX_STEPS:-8000}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LOGGING_STEPS=${LOGGING_STEPS:-10}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}

# Performance toggles (defaults chosen for the 6x RTX PRO 6000 96GB box).
#
# GRADIENT_CHECKPOINTING MUST stay 1 (on) on this 96GB box. Measured 2026-06-04:
# with GC off, the forward pass alone climbs to ~94.75/96GB and OOMs at the very
# first DiT block (idle model ~42GB, GC-on peak ~69GB, but GC-off keeps all 40
# layers' activations resident -> >94GB forward, backward needs even more).
# batch=1 is already minimal, so there is no headroom to disable it. Setting this
# to 0 will CUDA-OOM. Keep =1 unless the model/resolution/batch shrinks.
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
if [ "$GRADIENT_CHECKPOINTING" = "1" ] || [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
  USE_GRADIENT_CHECKPOINTING=true
else
  USE_GRADIENT_CHECKPOINTING=false
fi

# IMPORTANT: ShardedLeRobot's get_shard() decodes and caches an ENTIRE shard of
# video frames into RAM *per worker*, so host RAM scales ~linearly with
# DATALOADER_NUM_WORKERS x NUM_GPUS. 8 workers x 6 ranks OOM-killed a 500GB box.
# The GPUs were already ~100% utilized with a single worker, so the dataloader is
# not the bottleneck here -- keep workers low. Raise only if you have measured
# both GPU starvation AND plenty of free host RAM (each extra worker ~= one more
# decoded-video shard resident in memory). pin_memory adds pinned host RAM and is
# kept off by default to match the proven-safe baseline.
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-1}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-false}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-false}
PYTHON=${PYTHON:-"uv run python"}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
MAX_USED_GPU_MEM_MB=${MAX_USED_GPU_MEM_MB:-2048}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}

if [ "$SAVE_STRATEGY" != "no" ] && [ "$SAVE_TOTAL_LIMIT" -lt 5 ]; then
  echo "WARNING: BaseExperiment requires save_total_limit >= 5; bumping SAVE_TOTAL_LIMIT from $SAVE_TOTAL_LIMIT to 5."
  SAVE_TOTAL_LIMIT=5
fi

MIN_AVAILABLE_RAM_GB=${MIN_AVAILABLE_RAM_GB:-$((NUM_GPUS * 45))}
AVAILABLE_RAM_GB=$(awk '/MemAvailable/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
if [ "$AVAILABLE_RAM_GB" -lt "$MIN_AVAILABLE_RAM_GB" ]; then
  echo "ERROR: Only ${AVAILABLE_RAM_GB}GiB host RAM available; refusing to start ${NUM_GPUS}-GPU training."
  echo "Set MIN_AVAILABLE_RAM_GB to override after checking other jobs on the server."
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  fi

  for gpu_id in "${GPU_IDS[@]}"; do
    gpu_id="${gpu_id#"${gpu_id%%[![:space:]]*}"}"
    gpu_id="${gpu_id%"${gpu_id##*[![:space:]]}"}"
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
      continue
    fi
    used_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -dc '0-9')
    used_mem=${used_mem:-0}
    if [ "$used_mem" -gt "$MAX_USED_GPU_MEM_MB" ]; then
      echo "ERROR: GPU $gpu_id is already using ${used_mem}MiB; refusing to start training."
      echo "Set MAX_USED_GPU_MEM_MB to override after checking other GPU jobs."
      exit 1
    fi
  done
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
  echo "ERROR: DreamZero-AgiBot checkpoint not found at $PRETRAINED_MODEL_PATH"
  exit 1
fi

if [ -d "$TOKENIZER_DIR" ]; then
  echo "Using local tokenizer: $TOKENIZER_DIR"
else
  echo "Using tokenizer id: $TOKENIZER_DIR"
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
  echo "Wan2.1 component directory not found at $WAN_CKPT_DIR; continuing because skip_component_loading=true and pretrained_model_path is used."
fi

IFS=',' read -r -a ROOTS <<< "$REALMAN_DATA_ROOTS"
HYDRA_ROOTS="["
for root in "${ROOTS[@]}"; do
  root="${root#"${root%%[![:space:]]*}"}"
  root="${root%"${root##*[![:space:]]}"}"
  if [ -z "$root" ]; then
    continue
  fi
  if [ ! -d "$root" ]; then
    echo "ERROR: Realman dataset root not found: $root"
    exit 1
  fi
  for meta_file in modality.json embodiment.json stats.json relative_stats_dreamzero.json tasks.jsonl episodes.jsonl; do
    if [ ! -f "$root/meta/$meta_file" ]; then
      echo "ERROR: Missing $root/meta/$meta_file"
      echo "Run scripts/data/prepare_realman_gear_metadata.sh before training."
      exit 1
    fi
  done
  if ! grep -q '"embodiment_tag"[[:space:]]*:[[:space:]]*"realman"' "$root/meta/embodiment.json"; then
    echo "ERROR: $root/meta/embodiment.json does not contain embodiment_tag realman"
    exit 1
  fi
  HYDRA_ROOTS="${HYDRA_ROOTS}'${root}',"
done
HYDRA_ROOTS="${HYDRA_ROOTS%,}]"

if [ "$HYDRA_ROOTS" = "[]" ]; then
  echo "ERROR: REALMAN_DATA_ROOTS did not contain any dataset roots"
  exit 1
fi

echo "Realman dataset roots: $HYDRA_ROOTS"
echo "Output directory: $OUTPUT_DIR"
echo "NUM_GPUS: $NUM_GPUS"
echo "MAX_STEPS: $MAX_STEPS"
echo "SAVE_STRATEGY: $SAVE_STRATEGY"
echo "SAVE_STEPS: $SAVE_STEPS"
echo "SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT"
echo "LOGGING_STEPS: $LOGGING_STEPS"
echo "SAFE_TRAINING: $SAFE_TRAINING"
echo "DISABLE_TORCH_COMPILE: $DISABLE_TORCH_COMPILE"
echo "TORCHINDUCTOR_COMPILE_THREADS: $TORCHINDUCTOR_COMPILE_THREADS"
echo "DREAMZERO_MODEL_INIT_DTYPE: $DREAMZERO_MODEL_INIT_DTYPE"
echo "STAGGER_MODEL_INIT: $STAGGER_MODEL_INIT"
echo "STAGGER_MODEL_INIT_GROUP_SIZE: $STAGGER_MODEL_INIT_GROUP_SIZE"
echo "MOVE_MODEL_TO_CUDA_AFTER_LOAD: $MOVE_MODEL_TO_CUDA_AFTER_LOAD"
echo "Available host RAM: ${AVAILABLE_RAM_GB}GiB"
echo "GRADIENT_CHECKPOINTING: $GRADIENT_CHECKPOINTING (use_gradient_checkpointing=$USE_GRADIENT_CHECKPOINTING)"
echo "TORCH_COMPILE_DIT: $TORCH_COMPILE_DIT (DISABLE_TORCH_COMPILE=$DISABLE_TORCH_COMPILE, mode=${TORCH_COMPILE_DIT_MODE:-n/a})"
echo "DATALOADER_NUM_WORKERS: $DATALOADER_NUM_WORKERS"
echo "DATALOADER_PIN_MEMORY: $DATALOADER_PIN_MEMORY"
echo "DATALOADER_PERSISTENT_WORKERS: $DATALOADER_PERSISTENT_WORKERS"
echo "PREFLIGHT_ONLY: $PREFLIGHT_ONLY"

read -r -a PYTHON_CMD <<< "$PYTHON"
RUN_CMD=(
  "${PYTHON_CMD[@]}" -m torch.distributed.run
  --nproc_per_node "$NUM_GPUS"
  --standalone
  groot/vla/experiment/experiment.py
)

if [ "$PREFLIGHT_ONLY" = "1" ] || [ "$PREFLIGHT_ONLY" = "true" ]; then
  echo "Preflight checks passed. Not launching distributed training because PREFLIGHT_ONLY=$PREFLIGHT_ONLY."
  exit 0
fi

"${RUN_CMD[@]}" \
  report_to=$REPORT_TO \
  data=dreamzero/realman_relative \
  wandb_project=$WANDB_PROJECT \
  train_architecture=lora \
  model.config.model_dtype=$DREAMZERO_MODEL_INIT_DTYPE \
  num_frames=33 \
  action_horizon=24 \
  num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-5 \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  logging_steps=$LOGGING_STEPS \
  save_steps=$SAVE_STEPS \
  training_args.warmup_ratio=0.05 \
  output_dir=$OUTPUT_DIR \
  per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
  max_steps=$MAX_STEPS \
  weight_decay=1e-5 \
  save_total_limit=$SAVE_TOTAL_LIMIT \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=$DATALOADER_PIN_MEMORY \
  dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  dataloader_persistent_workers=$DATALOADER_PERSISTENT_WORKERS \
  image_resolution_width=320 \
  image_resolution_height=176 \
  save_lora_only=true \
  max_chunk_size=4 \
  frame_seqlen=880 \
  save_strategy=$SAVE_STRATEGY \
  realman_data_roots="$HYDRA_ROOTS" \
  dit_version=$WAN_CKPT_DIR \
  text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
  tokenizer_path=$TOKENIZER_DIR \
  pretrained_model_path=$PRETRAINED_MODEL_PATH \
  action_head_cfg.config.model_dtype=$DREAMZERO_MODEL_INIT_DTYPE \
  ++action_head_cfg.config.skip_component_loading=true \
  ++action_head_cfg.config.defer_lora_injection=true \
  ++action_head_cfg.config.use_gradient_checkpointing=$USE_GRADIENT_CHECKPOINTING
