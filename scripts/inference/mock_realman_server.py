"""Mock DreamZero Realman policy server (NO GPU, NO model).

Speaks the exact same openpi websocket + msgpack protocol as the real server
(``socket_test_optimized_AR.py``), but instead of running the 14B model it
validates the incoming DreamZero-native observation schema and returns a dummy
``(N, 8)`` action chunk. Use it to smoke-test the transport + wire contract
against ``robot_client_dreamzero.py`` without touching the GPU (so it can run
alongside training).

Run:
    uv run python scripts/inference/mock_realman_server.py --port 8123
"""

import argparse
import asyncio
import logging

import numpy as np
import websockets
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_realman_server")

# The DreamZero-native request schema (must match robot_client_dreamzero.py).
EXPECTED_VIEW_KEYS = [
    "video.nominal_image",
    "video.purturbated_c1_image",
    "video.purturbated_c2_image",
]
EXPECTED_STATE_KEYS = {"state.joint_pos": 7, "state.gripper_pos": 1}
ACTION_DIM = 8  # [7 joint, 1 gripper]


def _validate_obs(obs: dict) -> None:
    """Assert the obs matches the DreamZero schema; raise with a clear message."""
    if not isinstance(obs, dict):
        raise ValueError(f"obs must be a dict, got {type(obs)}")
    for key in EXPECTED_VIEW_KEYS:
        if key not in obs:
            raise ValueError(f"missing view key '{key}'. got keys: {sorted(obs.keys())}")
        img = np.asarray(obs[key])
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"view '{key}' must be (H, W, 3), got shape {img.shape}")
    for key, dim in EXPECTED_STATE_KEYS.items():
        if key not in obs:
            raise ValueError(f"missing state key '{key}'. got keys: {sorted(obs.keys())}")
        vec = np.asarray(obs[key]).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"state '{key}' must have dim {dim}, got {vec.shape[0]}")
    if "prompt" not in obs or not isinstance(obs["prompt"], str):
        raise ValueError(f"missing/invalid 'prompt' (str). got: {obs.get('prompt')!r}")


class MockRealmanServer:
    def __init__(self, host: str, port: int, actions_per_chunk: int = 4):
        self._host = host
        self._port = port
        self._actions_per_chunk = actions_per_chunk
        self._metadata = {"model_name": "mock-dreamzero-realman", "action_dim": ACTION_DIM}
        self._msg_count = 0

    async def _handler(self, websocket):
        logger.info("Connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        # Mirror the real server: send metadata on connect.
        await websocket.send(packer.pack(self._metadata))
        try:
            async for data in websocket:
                obs = msgpack_numpy.unpackb(data)
                self._msg_count += 1
                _validate_obs(obs)

                joint = np.asarray(obs["state.joint_pos"]).reshape(-1)
                gripper = np.asarray(obs["state.gripper_pos"]).reshape(-1)
                if self._msg_count == 1:
                    logger.info(
                        "msg#%d OK | views=%s | joint_pos%s | gripper_pos%s | prompt=%r",
                        self._msg_count,
                        [tuple(np.asarray(obs[k]).shape) for k in EXPECTED_VIEW_KEYS],
                        tuple(joint.shape),
                        tuple(gripper.shape),
                        obs["prompt"][:40],
                    )

                # Dummy action chunk: hold current joint state, gripper from state.
                # (N, 8) so we exercise the client's chunk-iteration path.
                row = np.concatenate([joint, gripper]).astype(np.float32)
                actions = np.tile(row, (self._actions_per_chunk, 1))
                response = {"actions": actions, "server_timing": {"infer_ms": 0.0}}
                await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            logger.info("Connection closed (served %d messages)", self._msg_count)

    async def _run(self):
        # max_size=None: three raw camera frames (~2.7 MB) exceed the 1 MB default
        # websocket recv limit. The real server must allow large messages too.
        async with _server.serve(self._handler, self._host, self._port, max_size=None) as server:
            logger.info("Mock Realman server listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    def serve_forever(self):
        asyncio.run(self._run())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--actions-per-chunk", type=int, default=4)
    args = ap.parse_args()
    MockRealmanServer(args.host, args.port, args.actions_per_chunk).serve_forever()


if __name__ == "__main__":
    main()
