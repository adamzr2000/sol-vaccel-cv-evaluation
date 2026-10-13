#!/usr/bin/env python3
"""
test_preprocess_isolation.py (standalone experiment — not part of the benchmark harness)

Root-causes the baseline's periodic 700-1040ms preprocessing spikes seen in
model_benchmark_ros.py's video-model (r3d_18) path. Calls
PyTorchBaselineAdapter.preprocess_frames() directly, in a tight loop, on a
FIXED synthetic clip (no ROS, no live camera, no inference) to find out
whether the spike pattern is intrinsic to preprocess_frames() itself (e.g.
periodic torch threadpool/allocator behavior) or an artifact of the live
serial loop (ROS spin thread contention, camera timing, etc).

Does not touch model_adapter.py / model_benchmark_ros.py / any production
file. Safe to run repeatedly.

Usage (inside the torchvision-ros-app:gpu container):
  python3 /src/experiments/test_preprocess_isolation.py --iters 60
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/src")
from model_adapter import get_model_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r3d_18")
    ap.add_argument("--backend", default="vaccel-remote-sol")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--model-dir", default="models/r3d_18_sol")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    _thread_limit = os.environ.get("OMP_THREAD_LIMIT") or os.environ.get("OMP_NUM_THREADS")
    if _thread_limit:
        torch.set_num_interop_threads(int(_thread_limit))
        torch.set_num_threads(int(_thread_limit))

    # Exact same factory call model_benchmark_ros.py uses -- guarantees this
    # test exercises the identical adapter class/preprocess_frames() code
    # path as the real benchmark runs (SolAdapter/VaccelSolAdapter for
    # vaccel-remote-sol, not the plain-PyTorch baseline adapter).
    adapter = get_model_adapter(args.model, args.backend, args.device)
    adapter.load_model(args.model_dir)

    # Fixed synthetic clip: 16 frames of random RGB uint8, same shape as a
    # real camera frame would produce after cv_bridge + BGR2RGB.
    rng = np.random.default_rng(42)
    clip = [rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8) for _ in range(16)]

    for _ in range(args.warmup):
        adapter.preprocess_frames([fr.copy() for fr in clip])

    times_ms = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        adapter.preprocess_frames([fr.copy() for fr in clip])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        flag = "  <<<< SPIKE" if dt_ms > 100 else ""
        print(f"[{i+1:3d}] preprocess_frames: {dt_ms:7.2f} ms{flag}")

    arr = np.asarray(times_ms)
    print("\n=== STATS (ms) ===")
    print(f"mean={arr.mean():.2f} median={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} max={arr.max():.2f}")
    n_spikes = int((arr > 100).sum())
    print(f"spikes (>100ms): {n_spikes}/{len(arr)}")
    if n_spikes > 0:
        print("=> Spikes reproduce WITHOUT ROS/live-camera/inference => intrinsic to preprocess_frames() itself.")
    else:
        print("=> No spikes in isolation => likely caused by interaction with ROS spin thread / live camera timing, not preprocess_frames() alone.")


if __name__ == "__main__":
    main()
