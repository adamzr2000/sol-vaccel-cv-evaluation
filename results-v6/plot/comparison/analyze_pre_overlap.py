#!/usr/bin/env python3
"""
Does vaccel's preprocessing get slower only when camera capture work runs at
the same time?

model_benchmark_ros_v2.py records, per sample, how much of the preprocessing
window overlapped the in-process GStreamer capture work (H264 decode +
videoconvert + the appsink callback): capture_overlap_n / capture_overlap_ms
in benchmark_data.csv. ROS2's go2_camerapublisher runs preprocessing inside
that same capture callback, so decode and preprocessing never overlap there
by construction -- its t_preprocess_ms is the "no overlap" reference.

Prints, per run: preprocessing p50 overall, for samples with no overlap and
with overlap, plus a least-squares fit pre_ms = a + b * overlap_ms (a = the
no-contention cost, b = ms of slowdown per ms of overlapping capture work),
next to the matching ros2 run's t_preprocess_ms p50.

Usage (from results/plot/comparison/):
  python3 analyze_pre_overlap.py <vaccel-run-tag> [<vaccel-run-tag> ...]
  e.g. python3 analyze_pre_overlap.py e2eGSt-diag e2eGSt-diag-ser
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from vaccel_comparison_test import (MODELS, RUN_TAG, ROS_MONITOR_DIR, VACCEL_ROBOT_DIR,
                                    VACCEL_RUN_SUFFIX, find_one, load_json)


def ros_pre_p50(model: str) -> float | None:
    try:
        monitor = load_json(find_one(ROS_MONITOR_DIR, f"monitor_{model}_sol_{RUN_TAG}_*.json"))
    except FileNotFoundError:
        return None
    pre = monitor["summary"].get("t_preprocess_ms")
    return pre["p50"] if pre else None


def analyze(tag: str, model: str) -> None:
    run_dir = VACCEL_ROBOT_DIR / f"{tag}_{model}_{VACCEL_RUN_SUFFIX}"
    csv_path = run_dir / "benchmark_data.csv"
    if not csv_path.is_file():
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows or "capture_overlap_ms" not in rows[0]:
        print(f"{tag:<18} {model:<20} [no capture_overlap fields -- run predates the diagnostic]")
        return
    summary = load_json(run_dir / "benchmark_summary.json")
    pre = np.array([float(r["preprocessing_ms"]) for r in rows])
    ov = np.array([float(r["capture_overlap_ms"]) for r in rows])
    none, some = pre[ov == 0], pre[ov > 0]
    b, a = np.polyfit(ov, pre, 1) if np.ptp(ov) > 0 else (0.0, float(np.median(pre)))
    ros = ros_pre_p50(model)

    def p50(x):
        return f"{np.median(x):7.2f} (n={len(x):4d})" if len(x) else "     -- (n=   0)"

    print(f"{tag:<18} {model:<20} {summary['stream'].get('omp_wait_policy', '?'):<8}"
          f"ser={str(summary['stream'].get('serialize_capture')):<6}"
          f"all {p50(pre)}  no-overlap {p50(none)}  overlap {p50(some)}  "
          f"fit a={a:6.2f} b={b:5.2f}  overlap_ms p50={np.median(ov):5.1f}  "
          f"ros2 pre p50={'n/a' if ros is None else f'{ros:.2f}'}")


def main() -> None:
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        sys.exit(2)
    for tag in tags:
        for model, _ in MODELS:
            analyze(tag, model)


if __name__ == "__main__":
    main()
