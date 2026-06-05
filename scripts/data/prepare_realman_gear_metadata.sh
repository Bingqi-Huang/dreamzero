#!/bin/bash
# Generate DreamZero/GEAR metadata for the local Realman 3-view LeRobot datasets.
#
# This script does not rewrite parquet files or videos. It writes metadata under
# each dataset's meta/ directory.

set -euo pipefail

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

PYTHON=${PYTHON:-"uv run python"}
STATE_KEYS=${STATE_KEYS:-'{"joint_pos":[0,7],"gripper_pos":[7,8]}'}
ACTION_KEYS=${ACTION_KEYS:-'{"joint_pos":[0,7],"gripper_pos":[7,8]}'}
TASK_KEY=${TASK_KEY:-"task_index"}
ACTION_HORIZON=${ACTION_HORIZON:-24}
FORCE=${FORCE:-0}

read -r -a PYTHON_CMD <<< "$PYTHON"
IFS=',' read -r -a ROOTS <<< "$REALMAN_DATA_ROOTS"

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

  cmd=(
    "${PYTHON_CMD[@]}" scripts/data/convert_lerobot_to_gear.py
    --dataset-path "$root"
    --embodiment-tag realman
    --state-keys "$STATE_KEYS"
    --action-keys "$ACTION_KEYS"
    --relative-action-keys joint_pos
    --task-key "$TASK_KEY"
    --action-horizon "$ACTION_HORIZON"
  )
  if [ "$FORCE" = "1" ]; then
    cmd+=(--force)
  fi

  echo "Preparing Realman metadata: $root"
  "${cmd[@]}"
done
