# 启动 DreamZero Realman 推理服务器（NVFP4 快速推理）

本文档说明如何在 6× RTX PRO 6000 (Blackwell) 机器上,用 **NVFP4 TensorRT** 引擎启动
Realman 的 DreamZero 推理服务器 `socket_test_realman.py`,供 RoboCOIN 侧的
`robot_client_dreamzero.py` 连接。

> 服务器讲 openpi websocket + msgpack 协议,返回 `{"actions": (N, 8)}`(机器人单位:
> 7 关节角(弧度) + 1 夹爪)。所有归一化 / 相对动作解码由模型 transform 内部按
> embodiment=`realman` 完成;服务器只额外做 **视图映射 + 帧累积**
> （集中在可单测的 `scripts/inference/realman_conversion.py`)。
> 单位:全程**弧度**(模型单位)。RoboCOIN 的 realman 机器人类自己做硬件度↔弧度,server 不转。

---

## 0. 这套东西的组成

| 文件 | 作用 | 状态 |
|---|---|---|
| `socket_test_realman.py` | Realman 推理服务器（改自 `socket_test_optimized_AR.py`） | **需 GPU + checkpoint 验证** |
| `scripts/inference/realman_conversion.py` | obs/action 转换核心（3 视图 / 帧累积，不转单位） | 已单测通过 |
| `scripts/inference/mock_realman_server.py` | 无 GPU mock 服务器（冒烟测试用） | 已通过 |
| `scripts/inference/smoke_test_realman_client.py` | 无 GPU 传输/协议冒烟测试 | 已通过 |

⚠️ **先做开环验证再上真机**：服务器里标了 “VERIFY via open-loop” 的边界 ——
**相对动作解码(delta→绝对)** 和 **每次推理喂几帧（num_frames）**。这两个连同
normstats,必须先用训练数据做开环对比(预测 vs 真值)确认,再驱动机器人。
（单位已确认全程弧度,不在待验证项。）

---

## 1. 先建 NVFP4 引擎（用真实 Realman 数据校准）

选好一个 checkpoint 后:

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

产物：`<checkpoint>/tensorrt/wan/WanModel_nvfp4.trt`

> 注意：你训练用了 `save_lora_only=true`，checkpoint 里只有 LoRA。
> 建引擎/推理时需要 “base(`DreamZero-AgiBot`) + 套上 LoRA”，或先把 LoRA merge 进 base。
> 先用 step-1000 的早期 checkpoint 把这条链路跑通。

---

## 2. 启动服务器（NVFP4 运行时）

```bash
unset DYNAMIC_CACHE_SCHEDULE
export NUM_DIT_STEPS=8
export ATTENTION_BACKEND=FA2
export LOAD_TRT_ENGINE=./checkpoints/dreamzero_realman_lora_5k/<checkpoint>/tensorrt/wan/WanModel_nvfp4.trt

uv run python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  socket_test_realman.py \
  --model-path ./checkpoints/dreamzero_realman_lora_5k/<checkpoint> \
  --port 8000
```

- `--nproc_per_node`：分布式推理的 GPU 数（沿用 AR 的 2，可按引擎/显存调整）。
- `--port`：要和客户端 `--port` 一致。
- rank 0 处理 websocket 连接，其余 rank 跑分布式 worker。
- 这些 Blackwell 运行时设置（`NUM_DIT_STEPS=8` / `ATTENTION_BACKEND=FA2` /
  不设 `DYNAMIC_CACHE_SCHEDULE`）见 `docs/BLACKWELL_INFERENCE_OPTIMIZATION.md`,
  除非后续 benchmark 证明别的更好,否则保持。

---

## 3. 服务器对外的 schema（和客户端一致）

请求（客户端 → 服务器）：

```text
video.nominal_image / purturbated_c1_image / purturbated_c2_image : (H,W,3) uint8 原始相机帧
state.joint_pos   : (7,) 关节角，弧度（模型单位）
state.gripper_pos : (1,) 夹爪，机器人原始值
prompt            : str
```

响应（服务器 → 客户端）：

```text
{"actions": (N, 8)}   # [7 关节角(弧度), 1 夹爪]，模型单位
```

服务器内部：3 视图按训练顺序送进模型 → 模型 transform 做归一化/拼图/相对解码 →
输出 `action.joint_pos`(弧度) / `action.gripper_pos` → 直接组成 `(N,8)`，不转单位。
（机器人类在 send_action 里把弧度转成硬件的度。）

---

## 4. 上真机前的无 GPU 冒烟测试（验证传输 + 协议）

不占 GPU、可在训练时跑。验证客户端↔服务器的传输和数据契约：

```bash
# 终端 A：起 mock 服务器（不加载模型）
uv run python scripts/inference/mock_realman_server.py --port 8123

# 终端 B：跑冒烟客户端（用和真客户端相同的 WebsocketClientPolicy 传输）
uv run python scripts/inference/smoke_test_realman_client.py --port 8123 --steps 5
# 期望输出: [smoke] PASS - 5/5 round-trips, schema + transport OK
```

转换核心的单元自检（帧累积 / (N,8) 形状，不转单位）：见
`scripts/inference/realman_conversion.py` 的逻辑,已随开发单测通过。

---

## 5. 推进顺序

1. **现在（无 checkpoint）**：上面第 4 节的冒烟测试已通过 —— 传输 + 协议 OK。
2. **step-1000 早期 checkpoint**：建引擎 → 起服务器 → **开环验证**(拿训练 episode
   对比预测 vs 真值),把 normstats / 相对解码 / 视图顺序 / num_frames 全部确认。
3. **最终 checkpoint**：换路径，真机闭环（`robot_client_dreamzero.py`，见 RoboCOIN
   仓库 `dev_docs/dreamzero_client.md`）。

---

## 6. 已知待验证项（服务器里已标注）

- **弧度↔度**：已确认全程弧度,`joint_deg_to_rad=False`（机器人类自己转硬件度）。无需在 server 转。
- **num_frames**：服务器用 `num_frames=4`（对齐 AR 流式的 `FRAMES_PER_CHUNK`），非 33 帧训练片长。
- **LoRA 加载**：`save_lora_only` 下 base+LoRA 的加载方式。
- 这三项 + normstats/相对解码,都由**开环验证**一次性确认。
