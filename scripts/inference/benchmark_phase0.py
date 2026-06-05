#!/usr/bin/env python3
"""Benchmark DreamZero websocket inference and parse server-side timing.

This script is intentionally non-invasive: it talks to an already running
socket_test_optimized_AR.py server and, optionally, captures its tmux pane to
extract the model's printed timing breakdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy
from test_client_AR import (
    _make_obs_from_video,
    build_frame_schedule,
    load_camera_frames,
)


SERVER_TIMING_RE = re.compile(
    r"Time taken: Total (?P<total>[0-9.]+) seconds, "
    r"Text Encoder (?P<text_encoder>[0-9.]+) seconds, "
    r"Image Encoder (?P<image_encoder>[0-9.]+) seconds, "
    r"VAE (?P<vae>[0-9.]+) seconds, "
    r"KV Cache Creation (?P<kv_cache_creation>[0-9.]+) seconds, "
    r"Diffusion (?P<diffusion>[0-9.]+) seconds, "
    r"DIT Compute Steps (?P<dit_compute_steps>[0-9]+) steps, "
    r"Scheduler (?P<scheduler>[0-9.\\-]+) seconds"
)


@dataclass
class CallRecord:
    index: int
    phase: str
    frame_indices: list[int]
    client_seconds: float
    action_shape: tuple[int, ...]


def capture_tmux_pane(pane: str, history_lines: int = 20000) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-J", "-S", f"-{history_lines}", "-t", pane],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def new_tmux_text(before: str, after: str) -> str:
    if after.startswith(before):
        return after[len(before) :]

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    max_overlap = min(len(before_lines), len(after_lines), 2000)
    for overlap in range(max_overlap, 0, -1):
        if before_lines[-overlap:] == after_lines[:overlap]:
            return "\n".join(after_lines[overlap:])
    return after


def parse_server_timings(text: str) -> list[dict[str, float]]:
    timings: list[dict[str, float]] = []
    for match in SERVER_TIMING_RE.finditer(text):
        item: dict[str, float] = {}
        for key, value in match.groupdict().items():
            item[key] = int(value) if key == "dit_compute_steps" else float(value)
        timings.append(item)
    return timings


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def format_summary(label: str, values: list[float]) -> str:
    stats = summarize(values)
    if stats["n"] == 0:
        return f"{label}: no samples"
    return (
        f"{label}: n={stats['n']:.0f} mean={stats['mean']:.3f}s "
        f"p50={stats['p50']:.3f}s p90={stats['p90']:.3f}s "
        f"min={stats['min']:.3f}s max={stats['max']:.3f}s"
    )


def print_bottleneck_report(server_measure: list[dict[str, float]]) -> None:
    if not server_measure:
        print("\nServer timing breakdown: no parsed samples")
        return

    keys = [
        "text_encoder",
        "image_encoder",
        "vae",
        "kv_cache_creation",
        "diffusion",
        "scheduler",
    ]
    means = {key: statistics.fmean(float(row[key]) for row in server_measure) for key in keys}
    total_mean = statistics.fmean(float(row["total"]) for row in server_measure)
    ranked = sorted(keys, key=lambda key: means[key], reverse=True)

    print("\nServer timing breakdown on measured chunks:")
    print(format_summary("  Total", [float(row["total"]) for row in server_measure]))
    for key in ranked:
        share = 100.0 * means[key] / total_mean if total_mean > 0 else float("nan")
        print(f"  {key:17s} mean={means[key]:.3f}s share={share:5.1f}%")

    print(f"  DIT compute steps mean={statistics.fmean(float(row['dit_compute_steps']) for row in server_measure):.2f}")
    print(f"\nLikely primary bottleneck: {ranked[0]} ({means[ranked[0]]:.3f}s mean)")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running DreamZero inference server.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--prompt", default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan")
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--measure-chunks", type=int, default=10)
    parser.add_argument("--tmux-pane", default="Dreamzero:0.0", help="tmux pane containing rank-0 server logs; use '' to disable")
    parser.add_argument("--tmux-history-lines", type=int, default=20000)
    parser.add_argument("--output-jsonl", default="runs/phase0_benchmark/latest.jsonl")
    parser.add_argument("--reset-at-end", action="store_true", help="Send reset after benchmark. This may trigger slow video saving.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(message)s")

    tmux_before = ""
    if args.tmux_pane:
        tmux_before = capture_tmux_pane(args.tmux_pane, args.tmux_history_lines)

    logging.info("Connecting to %s:%s", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    server_config = policy_server.PolicyServerConfig(**metadata)
    logging.info("Server config: %s", server_config)

    camera_frames = load_camera_frames()
    total_frames = min(v.shape[0] for v in camera_frames.values())
    needed_chunks = args.warmup_chunks + args.measure_chunks
    chunks = build_frame_schedule(total_frames, needed_chunks)
    if len(chunks) < needed_chunks:
        raise RuntimeError(f"Need {needed_chunks} chunks but only built {len(chunks)} from debug videos.")

    session_id = str(uuid.uuid4())
    logging.info("Session ID: %s", session_id)

    records: list[CallRecord] = []

    def run_call(phase: str, frame_indices: list[int]) -> None:
        obs = _make_obs_from_video(camera_frames, frame_indices, args.prompt, session_id)
        t0 = time.perf_counter()
        actions = client.infer(obs)
        dt = time.perf_counter() - t0
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"Expected numpy action array, got {type(actions)!r}")
        record = CallRecord(
            index=len(records),
            phase=phase,
            frame_indices=list(frame_indices),
            client_seconds=dt,
            action_shape=tuple(actions.shape),
        )
        records.append(record)
        logging.info("%s call %02d frames=%s client=%.3fs action_shape=%s", phase, record.index, frame_indices, dt, actions.shape)

    run_call("initial", [0])
    for frame_indices in chunks[: args.warmup_chunks]:
        run_call("warmup", frame_indices)
    for frame_indices in chunks[args.warmup_chunks : args.warmup_chunks + args.measure_chunks]:
        run_call("measure", frame_indices)

    if args.reset_at_end:
        logging.info("Sending reset")
        client.reset({})

    tmux_new = ""
    server_timings: list[dict[str, float]] = []
    if args.tmux_pane:
        tmux_after = capture_tmux_pane(args.tmux_pane, args.tmux_history_lines)
        tmux_new = new_tmux_text(tmux_before, tmux_after)
        server_timings = parse_server_timings(tmux_new)
        if len(server_timings) < len(records):
            logging.warning("Parsed %d server timing rows for %d client calls. Increase --tmux-history-lines if needed.", len(server_timings), len(records))

    # Align parsed server rows to calls from the end. This tolerates old timing
    # lines in tmux history if prefix diffing could not isolate exactly.
    aligned_timings: list[dict[str, float] | None] = [None] * len(records)
    if server_timings:
        recent = server_timings[-len(records) :]
        offset = len(records) - len(recent)
        for idx, timing in enumerate(recent, start=offset):
            aligned_timings[idx] = timing

    rows: list[dict[str, Any]] = []
    for record, timing in zip(records, aligned_timings):
        row = asdict(record)
        row["action_shape"] = list(record.action_shape)
        if timing is not None:
            row["server"] = timing
        rows.append(row)

    output_path = Path(args.output_jsonl)
    write_jsonl(output_path, rows)

    measure_client = [row["client_seconds"] for row in rows if row["phase"] == "measure"]
    warmup_client = [row["client_seconds"] for row in rows if row["phase"] == "warmup"]
    initial_client = [row["client_seconds"] for row in rows if row["phase"] == "initial"]

    print("\nClient wall-time summary:")
    print(format_summary("  Initial", initial_client))
    print(format_summary("  Warmup chunks", warmup_client))
    print(format_summary("  Measured chunks", measure_client))

    server_measure = [
        row["server"]
        for row in rows
        if row["phase"] == "measure" and "server" in row
    ]
    print_bottleneck_report(server_measure)
    print(f"\nWrote per-call records to {output_path}")

    if args.tmux_pane and tmux_new and not server_timings:
        print("\nNo server timing rows were parsed from tmux. Check that the server pane is correct and rank 0 prints 'Time taken: Total ...'.")


if __name__ == "__main__":
    main()
