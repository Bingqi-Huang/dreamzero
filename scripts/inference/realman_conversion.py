"""Realman <-> DreamZero observation/action conversion (pure, GPU-free, testable).

This is the shared "core" of the Realman rollout: it owns the conversions that
are the three classic silent-failure risks, so they live in ONE place that can
be unit-tested offline and reused by both the inference server and the open-loop
validation script.

Design note (mirrors socket_test_optimized_AR.py): the DreamZero model's own
transform handles state normalization and relative-action decoding INTERNALLY,
keyed by the embodiment metadata. So this core feeds the model state in the
model's training units and reads back action in those units. The only things
this core must get right at the boundary are:

  1. View key mapping   -> the 3 named views into the model modality keys.
  2. Frame accumulation -> the model expects a short video per view, not 1 frame.

UNITS (verified 2026-06-04): NO unit conversion happens here. The RoboCOIN
Realman robot class already converts hardware DEGREES <-> RADIANS internally
(realman config: joint_units=degree, model_joint_units defaults to radian):
``robot.get_observation()`` returns RADIANS, and ``robot.send_action()`` takes
RADIANS and converts to degrees for the SDK. Training data was recorded via
``get_observation()`` (record.py), so it is RADIANS too -- confirmed by the
joint stats magnitude (~[-1.2, 0.4], i.e. radians, not degrees). Therefore the
client sends radians, the model trains/predicts in radians, and the client's
``send_action`` does the rad->deg for hardware. ``joint_deg_to_rad`` defaults
to False (no-op); leave it off unless an open-loop test proves otherwise.

RELATIVE ACTION (training uses relative_action=true for joint_pos): the joint
action is a DELTA. Whether the delta is turned back into an absolute target by
(a) the DreamZero model transform, or (b) the RoboCOIN robot ``delta_with``
mode, MUST be pinned down by the open-loop validation before driving the robot.
This core does NOT add current state -- it only maps keys and shapes.

What this core does NOT do: normalization with stats.json / relative-action
delta integration. Those happen inside the model transform (embodiment=realman),
exactly as in the AgiBot server. The open-loop validation script checks the end
result against ground-truth actions.
"""

from __future__ import annotations

import numpy as np

# Client-side request schema (matches robot_client_dreamzero.py).
CLIENT_VIEW_KEYS = [
    "video.nominal_image",
    "video.purturbated_c1_image",
    "video.purturbated_c2_image",
]
# Model modality keys (matches meta/modality.json for embodiment realman).
MODEL_VIEW_KEYS = [
    "video.nominal_image",
    "video.purturbated_c1_image",
    "video.purturbated_c2_image",
]
N_JOINTS = 7
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


class RealmanConverter:
    """Stateful obs/action converter for one rollout session.

    Holds a rolling per-view frame buffer (the model consumes a short video, not
    a single frame). Reset between episodes with ``reset()``.
    """

    def __init__(self, num_frames: int = 33, joint_deg_to_rad: bool = False):
        self.num_frames = num_frames
        self.joint_deg_to_rad = joint_deg_to_rad
        self._frame_buffers: dict[str, list[np.ndarray]] = {k: [] for k in MODEL_VIEW_KEYS}
        self._is_first_call = True

    def reset(self) -> None:
        for k in self._frame_buffers:
            self._frame_buffers[k] = []
        self._is_first_call = True

    # ----------------------------- observation -----------------------------
    def obs_to_model(self, obs: dict) -> dict:
        """Map a client obs dict to the model's input dict.

        Client obs (raw robot units)::
            video.nominal_image / purturbated_c1_image / purturbated_c2_image : (H,W,3) uint8
            state.joint_pos   : (7,)  joint angles, model units (RADIANS)
            state.gripper_pos : (1,)  gripper, raw robot value
            prompt            : str

        Returns the model modality dict::
            video.<view>                 : (T,H,W,3) uint8   accumulated video
            state.joint_pos              : (1,1,7)  joint angles in MODEL units (radians)
            state.gripper_pos            : (1,1,1)  gripper raw
            annotation.language.action_text : str
        """
        converted: dict = {}

        # Accumulate one frame per view into the rolling buffer.
        for client_key, model_key in zip(CLIENT_VIEW_KEYS, MODEL_VIEW_KEYS):
            if client_key not in obs:
                raise KeyError(f"missing view '{client_key}'. got {sorted(obs.keys())}")
            frame = np.asarray(obs[client_key])
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            self._frame_buffers[model_key].append(np.ascontiguousarray(frame))

        # First call uses a single frame; later calls use the last `num_frames`.
        n = 1 if self._is_first_call else self.num_frames
        for model_key, buf in self._frame_buffers.items():
            if not buf:
                continue
            frames = buf[-n:] if len(buf) >= n else buf.copy()
            while len(frames) < n:  # pad by repeating the oldest frame
                frames.insert(0, buf[0])
            converted[model_key] = np.stack(frames, axis=0)  # (T,H,W,3)

        # State: pass joints through as model units. The robot class already did
        # hardware deg->rad, so by default (joint_deg_to_rad=False) this is a no-op.
        joint = np.asarray(obs["state.joint_pos"], dtype=np.float64).reshape(-1)
        if joint.shape[0] != N_JOINTS:
            raise ValueError(f"state.joint_pos must have {N_JOINTS} dims, got {joint.shape}")
        if self.joint_deg_to_rad:
            joint = joint * DEG2RAD
        gripper = np.asarray(obs["state.gripper_pos"], dtype=np.float64).reshape(-1)

        converted["state.joint_pos"] = joint.reshape(1, 1, N_JOINTS)
        converted["state.gripper_pos"] = gripper.reshape(1, 1, 1)
        converted["annotation.language.action_text"] = obs.get("prompt", "")
        return converted

    def mark_not_first(self) -> None:
        self._is_first_call = False

    # -------------------------------- action --------------------------------
    def action_to_chunk(self, action_dict: dict) -> np.ndarray:
        """Map model action.* outputs to an (N, 8) chunk in ROBOT units.

        Expects ``action.joint_pos`` (N,7) and ``action.gripper_pos`` (N,1) in
        model units (radians, already un-normalized + relative-decoded by the
        model transform). Returns [7 joint (model units / radians), 1 gripper];
        the robot class converts radians->degrees for the SDK on send_action.
        """
        joint = self._as_2d(action_dict, "action.joint_pos", N_JOINTS)
        gripper = self._as_2d(action_dict, "action.gripper_pos", 1)

        n = max(joint.shape[0], gripper.shape[0])
        joint = self._broadcast_rows(joint, n)
        gripper = self._broadcast_rows(gripper, n)

        if self.joint_deg_to_rad:  # opt-in escape hatch only; default off (no-op)
            joint = joint * RAD2DEG

        return np.concatenate([joint, gripper], axis=-1).astype(np.float32)  # (N,8)

    # ------------------------------- helpers --------------------------------
    @staticmethod
    def _as_2d(action_dict: dict, key: str, dim: int) -> np.ndarray:
        if key not in action_dict:
            raise KeyError(f"model output missing '{key}'. got {sorted(action_dict.keys())}")
        value = action_dict[key]
        if hasattr(value, "detach"):  # torch tensor
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1:
            value = value.reshape(1, -1) if value.shape[0] == dim else value.reshape(-1, 1)
        elif value.ndim > 2:
            value = value.reshape(-1, value.shape[-1])
        if value.shape[-1] != dim:
            raise ValueError(f"{key} expected last dim {dim}, got shape {value.shape}")
        return value

    @staticmethod
    def _broadcast_rows(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.shape[0] == n:
            return arr
        if arr.shape[0] == 1:
            return np.repeat(arr, n, axis=0)
        raise ValueError(f"cannot broadcast {arr.shape[0]} rows to {n}")
