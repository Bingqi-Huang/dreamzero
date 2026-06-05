# Blackwell Inference Optimization Notes

This document records the optimized DreamZero-AgiBot inference path for our
NVIDIA RTX PRO 6000 Blackwell Server Edition setup. It is the source of truth
for how to launch inference before and after post-training on our own robot
data.

## Recommended Runtime

Use this path by default on RTX PRO 6000 Blackwell:

```bash
unset DYNAMIC_CACHE_SCHEDULE
export NUM_DIT_STEPS=8
export LOAD_TRT_ENGINE=checkpoints/DreamZero-AgiBot/tensorrt/wan/WanModel_nvfp4.trt

ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path checkpoints/DreamZero-AgiBot
```

Then benchmark or test with:

```bash
uv run python scripts/inference/benchmark_phase0.py \
  --host localhost \
  --port 5000 \
  --warmup-chunks 3 \
  --measure-chunks 10 \
  --tmux-pane Dreamzero:0.0
```

or:

```bash
uv run python test_client_AR.py --port 5000
```

## Why This Is The Default

The upstream README's GB200 TensorRT path is:

```bash
export LOAD_TRT_ENGINE=<checkpoint>/tensorrt/wan/WanModel_nvfp4.trt
export DYNAMIC_CACHE_SCHEDULE=true
python -m torch.distributed.run ... socket_test_optimized_AR.py --enable-dit-cache ...
```

That path is still useful as the upstream GB200 reference, but on our RTX PRO
6000 Blackwell machine it is not the fastest stable setting. In our AgiBot
benchmark, `DYNAMIC_CACHE_SCHEDULE=true` made the runtime execute 16 DiT compute
steps instead of the fixed 8-step mask, which largely removed the TensorRT speed
gain.

The recommended runtime above keeps the same DiT compute step count as our
original non-TensorRT baseline:

- Scheduler timesteps: 16
- Actual DiT model forward calls: 8
- Action horizon: unchanged
- Chunk size: unchanged

So the speedup is not coming from reducing the model's compute steps below the
baseline. It comes primarily from running the DiT diffusion model through a
Blackwell NVFP4 TensorRT engine.

## Current Benchmark

Hardware:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- TensorRT Python package: 11.0.0.114
- ModelOpt: 0.44.0

Baseline before TensorRT:

```text
Measured chunks mean: 2.696s
Server total mean:    2.564s
Diffusion mean:       2.041s
DIT compute steps:    8
```

Recommended NVFP4 TensorRT runtime:

```text
Measured chunks mean: 1.564s
Server total mean:    1.465s
Diffusion mean:       0.957s
KV cache creation:    0.221s
Image encoder:        0.189s
DIT compute steps:    8
```

Observed speedup:

- End-to-end measured chunk latency: about 42% faster
- Diffusion latency: about 53% faster

The current remaining bottlenecks are:

1. Diffusion through TensorRT, still about 65% of server time.
2. KV cache packing/creation, about 15% of server time.
3. Image encoder cache-reset outliers, about 13% of server time on average.

## What We Optimized

### 1. NVFP4 TensorRT Engine

We built a real NVFP4 TensorRT engine for the AgiBot `ar_14B` DiT:

```text
checkpoints/DreamZero-AgiBot/tensorrt/wan/WanModel_nvfp4.trt
```

Current engine size:

```text
NVFP4 engine: 8.9G
FP16 engine: 31G
```

The engine was validated with a dummy forward smoke test:

```text
video  (1, 16, 2, 44, 80) torch.bfloat16 cuda
action (1, 48, 32)        torch.bfloat16 cuda
```

### 2. DiT Cache

Always launch inference with:

```bash
--enable-dit-cache
```

This enables the cache path in the DiT layers and avoids redoing all historical
frame work for every chunk.

### 3. Fixed 8 DiT Compute Steps

By default, the policy has 16 scheduler timesteps and a fixed mask that runs
the DiT on 8 of those timesteps:

```bash
export NUM_DIT_STEPS=8
unset DYNAMIC_CACHE_SCHEDULE
```

This matches our original baseline behavior and is the current best
speed/quality tradeoff on RTX PRO 6000 Blackwell.

Do not set `DYNAMIC_CACHE_SCHEDULE=true` for the default Pro 6000 runtime. In
our test sequence it caused 16 DiT compute steps and increased measured latency
to about 2.58s per chunk.

### 4. Attention Backend

Use:

```bash
ATTENTION_BACKEND=FA2
```

Transformer Engine is installed in this environment, but the repository's
current cuDNN fused-attention wrapper does not match the installed TE 2.15
`fused_attn_fwd` API. The stable runtime path is therefore FlashAttention 2 for
the PyTorch portions plus TensorRT for the DiT diffusion engine.

## Building Or Rebuilding The Engine

For the current AgiBot checkpoint:

```bash
PYTHON='uv run python' bash scripts/inference/build_trt_engine.sh \
  --model-path checkpoints/DreamZero-AgiBot \
  --embodiment-tag agibot \
  --model-type ar_14B \
  --tensorrt nvfp4 \
  --cuda-device 0
```

For best quality, especially after post-training on our own robot data, rebuild
with real calibration data:

```bash
PYTHON='uv run python' bash scripts/inference/build_trt_engine.sh \
  --model-path <path/to/posttrained/checkpoint> \
  --embodiment-tag <our_robot_tag> \
  --model-type ar_14B \
  --tensorrt nvfp4 \
  --dataset-path <path/to/lerobot/calibration_dataset> \
  --num-calibration-trajs 5 \
  --cuda-device 0
```

Notes:

- The current engine was initially built without a real LeRobot calibration
  dataset. It is good for validating speed and runtime integration.
- For robot deployment quality, use real calibration trajectories from the same
  camera/action distribution as the post-trained model.
- Rebuild the TensorRT engine every time the checkpoint weights change after
  fine-tuning or post-training.

## After Post-Training On Our Robot

Use this sequence:

1. Train or post-train a checkpoint for our robot.
2. Prepare a small LeRobot-format calibration dataset from the same robot.
3. Rebuild `WanModel_nvfp4.trt` from the post-trained checkpoint using the
   real calibration dataset.
4. Launch inference with the recommended runtime command:

```bash
unset DYNAMIC_CACHE_SCHEDULE
export NUM_DIT_STEPS=8
export LOAD_TRT_ENGINE=<path/to/posttrained/checkpoint>/tensorrt/wan/WanModel_nvfp4.trt

ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path <path/to/posttrained/checkpoint>
```

5. Run `benchmark_phase0.py` and record the result before robot deployment.
6. Validate action quality on safe replay/open-loop tests before closing the
   control loop.

## Known Pitfalls

- `DYNAMIC_CACHE_SCHEDULE=true` is the upstream GB200 reference setting, but on
  our Pro 6000 AgiBot benchmark it ran 16 DiT compute steps and was slower.
- `NUM_DIT_STEPS=5/6/7` may further reduce latency, but it changes the number
  of DiT forward calls below the baseline and must be treated as a quality
  experiment, not the default deployment path.
- The TensorRT engine can prune unused ONNX inputs. Runtime wrappers must feed
  tensors by input name, not by positional order.
- NVFP4 ONNX export can leave TensorRT type mismatches around MatMul and
  LayerNorm nodes. The local TensorRT build utilities include a sanitizer for
  these ModelOpt/TensorRT graph typing issues.
- Do not compare results unless `DIT compute steps` is also recorded. A faster
  or slower wall time can simply mean the runtime used a different number of
  DiT forwards.

