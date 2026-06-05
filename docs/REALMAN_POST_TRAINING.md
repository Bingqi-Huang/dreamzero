# Realman Post-Training Workflow

This document records the local Realman-specific work needed before training a
DreamZero policy on the synchronized 3-view LeRobot datasets in
`/home/bingqi/data/bingqi/CoRL26`.

The goal is to post-train the released `DreamZero-AgiBot` checkpoint into a
Realman policy, then reuse the Blackwell TensorRT/NVFP4 inference path for
robot rollout.

## Data Assumptions

We first train only on the stable 3-view datasets:

```text
/home/bingqi/data/bingqi/CoRL26/Task1_new
/home/bingqi/data/bingqi/CoRL26/Task1_CoRL
/home/bingqi/data/bingqi/CoRL26/Task2_CoRL/Task2
/home/bingqi/data/bingqi/CoRL26/Task3_CoRL/Task3
/home/bingqi/data/bingqi/CoRL26/Task3_new
```

Each dataset has:

- `fps = 15`
- `observation.state`: 8 dimensions
- `action`: 8 dimensions
- `task_index` in parquet
- real task text in `meta/tasks.jsonl`
- three synchronized camera streams:
  - `observation.images.nominal_image`
  - `observation.images.purturbated_c1_image`
  - `observation.images.purturbated_c2_image`

The 8-dimensional state/action vectors are split as:

```text
joint_pos   = dims [0:7]
gripper_pos = dims [7:8]
```

`joint_pos` uses relative-action normalization. `gripper_pos` stays absolute in
the first training path.

## View Layout

The Realman modality order is:

```yaml
- video.nominal_image
- video.purturbated_c1_image
- video.purturbated_c2_image
```

`DreamTransform._prepare_video()` converts any non-DROID multi-view input into a
single 2x2 grid:

```text
[ view 0 | view 2 ]
[ view 1 | black  ]
```

For Realman this means:

```text
[ nominal_image        | purturbated_c2_image ]
[ purturbated_c1_image | black screen         ]
```

This view order must remain identical for training, testing, and robot rollout.

## Code Added

Realman support is registered in:

- `groot/vla/data/schema/embodiment_tags.py`
- `scripts/data/convert_lerobot_to_gear.py`
- `groot/vla/configs/model/dreamzero/transform/base.yaml`
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `groot/vla/configs/data/dreamzero/realman_relative.yaml`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `scripts/data/prepare_realman_gear_metadata.sh`
- `scripts/train/realman_training.sh`

The converter now preserves existing `meta/tasks.jsonl` task text when
`--task-key task_index` is used, including when `FORCE=1` is set on the helper
script.

## Prepare Metadata

Run the helper from the repo root:

```bash
PYTHON='uv run python' bash scripts/data/prepare_realman_gear_metadata.sh
```

This generates or refreshes:

```text
meta/modality.json
meta/embodiment.json
meta/stats.json
meta/relative_stats_dreamzero.json
```

It keeps the original parquet files, videos, `tasks.jsonl`, and `episodes.jsonl`
intact.

To force metadata regeneration:

```bash
FORCE=1 PYTHON='uv run python' bash scripts/data/prepare_realman_gear_metadata.sh
```

To run on a custom comma-separated dataset list:

```bash
REALMAN_DATA_ROOTS='/path/to/ds1,/path/to/ds2' \
PYTHON='uv run python' \
bash scripts/data/prepare_realman_gear_metadata.sh
```

## Metadata Checklist

Before training, each dataset root must contain:

```text
meta/modality.json
meta/embodiment.json
meta/stats.json
meta/relative_stats_dreamzero.json
meta/tasks.jsonl
meta/episodes.jsonl
```

`meta/embodiment.json` must contain:

```json
{"embodiment_tag": "realman"}
```

`meta/modality.json` should include:

```json
{
  "state": {"joint_pos": "...", "gripper_pos": "..."},
  "action": {"joint_pos": "...", "gripper_pos": "..."},
  "video": {
    "nominal_image": "...",
    "purturbated_c1_image": "...",
    "purturbated_c2_image": "..."
  },
  "annotation": {"task_index": {"original_key": "task_index"}}
}
```

## Training

Run preflight first. This validates data roots, required metadata, available
host RAM, and selected GPU memory without launching distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NUM_GPUS=6 \
PREFLIGHT_ONLY=1 \
REPORT_TO=none \
PYTHON='uv run python' \
bash scripts/train/realman_training.sh
```

First run a short sanity check. The training script defaults to safe
fine-tuning mode:

- `DISABLE_TORCH_COMPILE=1`
- `TORCHINDUCTOR_COMPILE_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `DREAMZERO_MODEL_INIT_DTYPE=bfloat16`
- `STAGGER_MODEL_INIT=1`
- `STAGGER_MODEL_INIT_GROUP_SIZE=3`
- `MOVE_MODEL_TO_CUDA_AFTER_LOAD=1`
- a host-RAM preflight check before distributed launch

This is intentional. There are two separate host-RAM failure modes:

- Torch Inductor can compile in every rank and spawn many compile workers.
- Even with compile disabled, 6 ranks can otherwise instantiate and load the
  14B checkpoint at the same time on CPU.

The Realman script still supports 6-GPU training, but rank initialization is
staggered by default. On this machine the default group size is 3: ranks
`0,1,2` build/load in parallel, move to `cuda:0,1,2`, write markers, then ranks
`3,4,5` proceed. All ranks wait for the final rank before continuing into
dataset/trainer setup.

If the server is busy or host RAM is lower than expected, reduce the group size:

```bash
STAGGER_MODEL_INIT_GROUP_SIZE=1 bash scripts/train/realman_training.sh
```

Metrics are logged through Hugging Face Trainer. `scripts/train/realman_training.sh`
defaults to `REPORT_TO=wandb` and `WANDB_PROJECT=dreamzero_realman`, so formal
training will appear in Weights & Biases as long as the machine is logged in:

```bash
uv run wandb login
```

For smoke tests, use `REPORT_TO=none` if you do not want noisy W&B runs. For
formal training, either omit `REPORT_TO` or set `REPORT_TO=wandb`.

Logged metrics include:

- `loss`
- `dynamics_loss_avg`
- `action_loss_avg`
- `learning_rate`
- timing values emitted through the trainer timer, such as
  `model_forward_time` and `training_step_time`

Rank 0 also writes local loss snapshots to:

```text
<OUTPUT_DIR>/loss_log.jsonl
```

The logging interval is controlled by `LOGGING_STEPS` and defaults to `10`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NUM_GPUS=6 \
MAX_STEPS=1 \
SAVE_STRATEGY=no \
REPORT_TO=none \
OUTPUT_DIR=./checkpoints/dreamzero_realman_lora_smoke_6gpu \
PYTHON='uv run python' \
bash scripts/train/realman_training.sh
```

Then run the first real LoRA training job:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NUM_GPUS=6 \
MAX_STEPS=5000 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=5 \
LOGGING_STEPS=10 \
REPORT_TO=wandb \
WANDB_PROJECT=dreamzero_realman \
WANDB_NAME=dreamzero_realman_lora_5k \
OUTPUT_DIR=./checkpoints/dreamzero_realman_lora_5k \
PYTHON='uv run python' \
bash scripts/train/realman_training.sh
```

The training script defaults to:

- `train_architecture=lora`
- `pretrained_model_path=./checkpoints/DreamZero-AgiBot`
- `skip_component_loading=true`, so the Wan2.1 component `.pth` files are not
  downloaded or loaded before the full AgiBot checkpoint is applied
- `per_device_train_batch_size=1`
- `learning_rate=1e-5`
- `action_horizon=24`
- `num_views=3`
- `image_resolution_width=320`
- `image_resolution_height=176`
- `save_lora_only=true`
- `SAFE_TRAINING=1`
- `DISABLE_TORCH_COMPILE=1`

`TOKENIZER_DIR` defaults to the Hugging Face tokenizer id `google/umt5-xxl`.
Transformers will fetch the tokenizer files as needed. Override it with a local
directory if the tokenizer is already available:

```bash
TOKENIZER_DIR=./checkpoints/umt5-xxl bash scripts/train/realman_training.sh
```

If the server is busy, the script may refuse to start because host RAM is below
the safety threshold. By default the threshold is `45GiB * NUM_GPUS`. Override
only after checking other jobs:

```bash
MIN_AVAILABLE_RAM_GB=200 bash scripts/train/realman_training.sh
```

## LoRA vs Full Fine-Tuning

Use LoRA first.

Reasons:

- The current 3-view Realman set is small enough that full 14B fine-tuning can
  overfit or damage the pretrained video/action prior.
- The released new-embodiment path is LoRA-based.
- In this repo, LoRA training also keeps the state/action encoder and decoder
  trainable, which is important for the new robot action space.
- LoRA is fast enough to iterate with real robot rollout.

Full fine-tuning is possible to test later on the 6x RTX PRO 6000 Blackwell
machine, but it should be a controlled comparison after LoRA has produced a
working rollout baseline. If we try it, use low learning rate, ZeRO/offload,
short runs first, and compare rollout behavior rather than only training loss.

## Rollout Integration

The current optimized inference server is still AgiBot/AR-Droid-shaped:

- AgiBot camera keys
- AgiBot action keys
- AgiBot state padding

Before robot rollout, add a Realman inference wrapper or parameterize the
existing wrapper so it uses:

```text
video.nominal_image
video.purturbated_c1_image
video.purturbated_c2_image
state.joint_pos
state.gripper_pos
action.joint_pos
action.gripper_pos
```

The wrapper should return an `(N, 8)` action chunk:

```text
[7 joint positions, 1 gripper position]
```

## Blackwell Inference After Training

After selecting a Realman checkpoint, rebuild the NVFP4 TensorRT engine using
real Realman calibration data:

```bash
PYTHON='uv run python' bash scripts/inference/build_trt_engine.sh \
  --model-path ./checkpoints/dreamzero_realman_lora_5k/<checkpoint> \
  --embodiment-tag realman \
  --model-type ar_14B \
  --tensorrt nvfp4 \
  --dataset-path /home/bingqi/data/bingqi/CoRL26/Task1_new \
  --num-calibration-trajs 5 \
  --cuda-device 0
```

Then use the same RTX PRO 6000 runtime pattern documented in
`docs/BLACKWELL_INFERENCE_OPTIMIZATION.md`, replacing `--model-path` and
`LOAD_TRT_ENGINE` with the Realman checkpoint and engine paths.

Keep:

```bash
unset DYNAMIC_CACHE_SCHEDULE
export NUM_DIT_STEPS=8
export ATTENTION_BACKEND=FA2
```

unless a later benchmark proves another setting is better for Realman.
