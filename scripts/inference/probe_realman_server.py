"""Probe the REAL DreamZero Realman server (raw websocket, keepalive disabled).

The openpi WebsocketClientPolicy sends keepalive pings; the server runs the 14B
diffusion forward synchronously inside its async handler, so during the (slow,
torch.compile-laden) FIRST inference it cannot answer pings and the client drops
the connection with a ping timeout. This probe uses a raw websockets.sync
connection with ping_interval=None and a long recv timeout, so a slow first call
is fine. It checks the response contract (actions present, (N,8), all finite)
and prints the predicted action. No GPU here; the GPU work is in the server.

Run (server already listening on --port):
    uv run python scripts/inference/probe_realman_server.py --port 8000 --steps 4
"""

import argparse
import sys
import time

import numpy as np
from websockets.sync.client import connect
from openpi_client import msgpack_numpy

VIEW_KEYS = [
    "video.nominal_image",
    "video.purturbated_c1_image",
    "video.purturbated_c2_image",
]
ACTION_DIM = 8


def build_obs(joint, gripper, prompt, h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    obs = {
        "state.joint_pos": np.asarray(joint, dtype=np.float32),
        "state.gripper_pos": np.asarray([gripper], dtype=np.float32),
        "prompt": prompt,
    }
    for key in VIEW_KEYS:
        obs[key] = (rng.integers(0, 40, size=(h, w, 3)) + 110).astype(np.uint8)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt", default="pick up the object")
    args = ap.parse_args()

    packer = msgpack_numpy.Packer()
    uri = f"ws://{args.host}:{args.port}"
    # ping_interval=None: never send keepalive pings (server is busy in forward).
    # max_size=None: large multi-frame obs. open/close timeouts generous.
    with connect(uri, ping_interval=None, max_size=None, open_timeout=60, close_timeout=10) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
        print(f"[probe] connected {uri} | server metadata: {meta}")

        base_joint = np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.3, 0.0], dtype=np.float32)
        failures = 0
        for step in range(args.steps):
            joint = base_joint + 0.02 * step
            gripper = float(step % 2)
            obs = build_obs(joint, gripper, args.prompt, seed=step)

            t0 = time.perf_counter()
            ws.send(packer.pack(obs))
            raw = ws.recv(timeout=600)  # first call compiles + runs 14B diffusion
            dt = time.perf_counter() - t0
            response = msgpack_numpy.unpackb(raw)

            if not isinstance(response, dict) or "actions" not in response:
                print(f"[probe] step {step}: FAIL ({dt:.1f}s) - no 'actions': {str(response)[:200]}")
                failures += 1
                continue
            actions = np.asarray(response["actions"])
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                print(f"[probe] step {step}: FAIL ({dt:.1f}s) - shape {actions.shape}, want (N,{ACTION_DIM})")
                failures += 1
                continue
            if not np.all(np.isfinite(actions)):
                print(f"[probe] step {step}: FAIL ({dt:.1f}s) - non-finite values")
                failures += 1
                continue
            print(f"[probe] step {step}: OK ({dt:.1f}s) shape={actions.shape}")
            print(f"          joint[0] = {np.array2string(actions[0,:7], precision=4, suppress_small=True)}")
            print(f"          grip[0]  = {actions[0,7]:.4f} | joint range [{actions[:,:7].min():.3f}, {actions[:,:7].max():.3f}]")

    if failures == 0:
        print(f"[probe] PASS - {args.steps}/{args.steps} real-model round-trips, contract OK")
        sys.exit(0)
    print(f"[probe] FAIL - {failures}/{args.steps}")
    sys.exit(1)


if __name__ == "__main__":
    main()
