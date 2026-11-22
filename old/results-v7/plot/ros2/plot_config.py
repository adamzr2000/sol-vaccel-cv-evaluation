#!/usr/bin/env python3
"""
plot_config.py (ros2)

Shared config for the results/plot/ros2/ plot family only - intentionally
separate from results/plot/vaccel/plot_config.py (different experiments,
different run-tag vocabulary, not to be mixed).
"""
from __future__ import annotations
from pathlib import Path
import csv
import math

_HERE = Path(__file__).parent

# Semantic-segmentation models only, remote-{cpu,gpu} cases only: read from
# this run tag instead of the default remote-cpu/remote-gpu run (e.g. a
# 12fps-capped rerun). Set either to None to disable that device's override
# (falls back to the base remote-{cpu,gpu} run for every model, including
# segmentation).
SEG_MODELS = {"fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101"}
# SEG_REMOTE_CPU_TAG_OVERRIDE: str | None = "remote-cpu12fps"
# SEG_REMOTE_GPU_TAG_OVERRIDE: str | None = "remote-gpu12fps"

SEG_REMOTE_CPU_TAG_OVERRIDE: str | None = "remote-cpu"
SEG_REMOTE_GPU_TAG_OVERRIDE: str | None = "remote-gpu"

IDLE_SUMMARY_FILE = _HERE / "../../experiments/system-stats/idle/summary.csv"
IDLE_HOST_ROBOT    = "robot-cpu-idle"
IDLE_HOST_EDGE_CPU = "edge-asus-cpu-idle"
IDLE_HOST_EDGE_GPU = "edge-asus-gpu-idle"


def load_idle_power_map() -> dict[str, float]:
    """
    {host: exact_idle_power_w} from idle/summary.csv, deriving power from
    energy_j_from_counters / duration_sec (the hardware-counter figure)
    rather than the sampled power_w_mean column, for consistency with the
    exact cpu/gpu_energy_j_total values this is used to correct.
    """
    path = IDLE_SUMMARY_FILE.resolve()
    if not path.exists():
        print(f"[WARNING] Idle summary not found: {path}")
        return {}
    out: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                e = float(row["energy_j_from_counters"])
                d = float(row["duration_sec"])
                if d > 0:
                    out[row["host"]] = e / d
            except (TypeError, ValueError, KeyError):
                continue
    return out


def idle_host_for(host: str, kind: str) -> str | None:
    """Which idle/summary.csv row backs out this (host, kind)'s always-on
    draw. Returns None for combinations with no idle baseline defined."""
    if host == "robot" and kind == "cpu":
        return IDLE_HOST_ROBOT
    if host == "edge-asus" and kind == "cpu":
        return IDLE_HOST_EDGE_CPU
    if host == "edge-asus" and kind == "gpu":
        return IDLE_HOST_EDGE_GPU
    return None


def subtract_idle(j: float, duration_sec: float, idle_power_w: float | None) -> float:
    """Workload-only Joules: total minus idle_power_w x duration_sec,
    clipped at 0 so measurement noise can't produce a negative energy."""
    if idle_power_w is None or not math.isfinite(idle_power_w) or not math.isfinite(duration_sec):
        return j
    return max(0.0, j - idle_power_w * duration_sec)
