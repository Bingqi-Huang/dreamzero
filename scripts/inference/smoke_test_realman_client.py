"""No-GPU smoke test for the DreamZero Realman client<->server wire contract.

Uses the SAME transport the real robot client uses
(``openpi_client.websocket_client_policy.WebsocketClientPolicy``) to send the
exact DreamZero-native observation schema that
``robot_client_dreamzero.py::_prepare_observation`` produces, then checks the
``(N, 8)`` action response. No model, no GPU, no robot hardware -> safe to run
while training is using the GPUs.

Run (with the mock server already listening on the same port):
    uv run python scripts/inference/smoke_test_realman_client.py --port 8123
"""

import argparse
import sys

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Must match robot_client_dreamzero.py / mock_realman_server.py.
VIEW_KEYS = [
    "video.nominal_image",
    "video.purturbated_c1_image",
    "video.purturbated_c2_image",
]
ACTION_DIM = 8


def build_obs(joint_deg, gripper, prompt):
    """Replicate robot_client_dreamzero.py::_prepare_observation output."""
    obs = {
        "state.joint_pos": np.asarray(joint_deg, dtype=np.float32),   # (7,) degrees
        "state.gripper_pos": np.asarray([gripper], dtype=np.float32),  # (1,)
        "prompt": prompt,
    }
    for key in VIEW_KEYS:
        obs[key] = np.zeros((480, 640, 3), dtype=np.uint8)  # raw camera frame
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    policy = WebsocketClientPolicy(args.host, args.port)
    print(f"[smoke] connected to {args.host}:{args.port}")
    print(f"[smoke] server metadata: {policy.get_server_metadata()}")

    failures = 0
    for step in range(args.steps):
        joint = np.linspace(-10, 10, 7) + step  # fake degrees, varies per step
        gripper = float(step % 2)
        obs = build_obs(joint, gripper, prompt="pick up the cube and place it in the box")

        response = policy.infer(obs)

        # Validate the response contract the real client relies on.
        if not isinstance(response, dict) or "actions" not in response:
            print(f"[smoke] step {step}: FAIL - response missing 'actions': {type(response)}")
            failures += 1
            continue
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            print(f"[smoke] step {step}: FAIL - actions shape {actions.shape}, expected (N, {ACTION_DIM})")
            failures += 1
            continue
        # The mock echoes state back; confirm the round-trip carried our values.
        if not np.allclose(actions[0, :7], joint, atol=1e-3):
            print(f"[smoke] step {step}: FAIL - joint round-trip mismatch")
            failures += 1
            continue
        if step == 0:
            print(f"[smoke] step 0: OK - got actions {actions.shape}, first row joints match input")

    if failures == 0:
        print(f"[smoke] PASS - {args.steps}/{args.steps} round-trips, schema + transport OK")
        sys.exit(0)
    else:
        print(f"[smoke] FAIL - {failures}/{args.steps} round-trips failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
