#!/usr/bin/env python3
"""
test_sol_async_dispatch.py (standalone experiment — not part of the benchmark harness)

Determines empirically whether the SOL vaccel wrapper's split-call pattern
(set_IO() -> run() -> get_output()) is genuinely asynchronous — i.e. run()
returns before the agent/local plugin finishes compute, letting the caller
do other work in the gap before get_output() actually blocks for the
result — or whether run() itself still blocks for the full compute time and
only the *output transfer* is deferred to get_output().

Method: run the SAME model 3 ways and compare total wall-clock time:
  A. predict()                         - current production pattern (single
                                          combined call, baseline timing)
  B. set_IO() -> run() -> get_output()  (no gap)   - split-call overhead alone
  C. set_IO() -> run() -> SLEEP -> get_output()     - split-call with a
                                          deliberate local delay standing in
                                          for "preprocess the next frame"

If run() is genuinely async: C's total time =~ max(compute_time, SLEEP_MS),
i.e. the sleep is absorbed "for free" alongside remote/local compute.
If run() is NOT async (blocks for compute, only defers output fetch): C's
total time =~ B's total time + SLEEP_MS, i.e. the sleep adds on top, no
overlap.

Does not touch model_adapter.py / model_benchmark_ros.py / any production
file. Safe to run repeatedly; writes nothing to disk.

Usage (inside the torchvision-ros-app:gpu container, LD_LIBRARY_PATH set up
the same way run_ros.sh does it):
  python3 /src/experiments/test_sol_async_dispatch.py [--remote] [--sleep-ms 60] [--iters 20]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def load_sol_module(vaccel_dir: Path, class_name: str):
    """Mirrors model_adapter.py's SolAdapter._load_model_from_path dynamic import."""
    py_path = vaccel_dir / f"{class_name}.py"
    spec = importlib.util.spec_from_file_location(class_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/src/models/resnet50_sol/lib_gpu/vaccel")
    ap.add_argument("--class-name", default="sol_resnet50")
    ap.add_argument("--remote", action="store_true", help="use_remote=True (needs VACCEL_RPC_ADDRESS set + an agent reachable)")
    ap.add_argument("--sleep-ms", type=float, default=60.0, help="local 'other work' delay to insert between run() and get_output()")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    vaccel_dir = Path(args.model_dir)
    sol_cls = load_sol_module(vaccel_dir, args.class_name)

    print(f"[setup] model_dir={vaccel_dir} remote={args.remote} sleep_ms={args.sleep_ms} iters={args.iters}")

    def fresh_lib():
        return sol_cls(path=str(vaccel_dir), use_remote=args.remote)

    in_arr = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # --- A: predict() baseline (current production call pattern) ---
    lib_a = fresh_lib()
    for _ in range(args.warmup):
        lib_a.predict(in_arr)
    a_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        lib_a.predict(in_arr)
        a_times.append((time.perf_counter() - t0) * 1000.0)

    # --- B: split-call, no gap ---
    lib_b = fresh_lib()
    lib_b.init()
    out_b = np.zeros((1, 1000), dtype=np.float32)
    lib_b.set_IO([in_arr, out_b, None])
    for _ in range(args.warmup):
        lib_b.run()
        lib_b.get_output()
    b_times = []
    b_run_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        lib_b.run()
        t1 = time.perf_counter()
        lib_b.get_output()
        t2 = time.perf_counter()
        b_run_times.append((t1 - t0) * 1000.0)
        b_times.append((t2 - t0) * 1000.0)

    # --- C: split-call, with a deliberate local-work gap between run() and get_output() ---
    lib_c = fresh_lib()
    lib_c.init()
    out_c = np.zeros((1, 1000), dtype=np.float32)
    lib_c.set_IO([in_arr, out_c, None])
    for _ in range(args.warmup):
        lib_c.run()
        time.sleep(args.sleep_ms / 1000.0)
        lib_c.get_output()
    c_times = []
    c_run_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        lib_c.run()
        t1 = time.perf_counter()
        time.sleep(args.sleep_ms / 1000.0)
        lib_c.get_output()
        t2 = time.perf_counter()
        c_run_times.append((t1 - t0) * 1000.0)
        c_times.append((t2 - t0) * 1000.0)

    def stats(vals):
        a = np.asarray(vals)
        return float(np.mean(a)), float(np.median(a)), float(np.std(a))

    a_mean, a_med, a_std = stats(a_times)
    b_mean, b_med, b_std = stats(b_times)
    brun_mean, _, _ = stats(b_run_times)
    c_mean, c_med, c_std = stats(c_times)
    crun_mean, _, _ = stats(c_run_times)

    print("\n=== RESULTS (ms) ===")
    print(f"A. predict()                          : mean={a_mean:.2f} median={a_med:.2f} std={a_std:.2f}")
    print(f"B. set_IO->run->get_output (no gap)    : mean={b_mean:.2f} median={b_med:.2f} std={b_std:.2f}  (run() alone: {brun_mean:.2f} ms)")
    print(f"C. set_IO->run->SLEEP({args.sleep_ms:.0f}ms)->get_output: mean={c_mean:.2f} median={c_med:.2f} std={c_std:.2f}  (run() alone: {crun_mean:.2f} ms)")

    overhead_vs_sum = c_mean - (b_mean + args.sleep_ms)
    overlap_fraction = 1.0 - max(0.0, (c_mean - b_mean)) / args.sleep_ms if args.sleep_ms > 0 else float("nan")

    print("\n=== VERDICT ===")
    print(f"C_total - (B_total + sleep_ms) = {overhead_vs_sum:+.2f} ms")
    print(f"  ~0   -> no overlap: run() already blocked for compute, sleep is pure add-on (NOT async)")
    print(f"  <<0, i.e. C_total ~= max(B_total, sleep_ms) -> genuine overlap (run() IS async)")
    print(f"Estimated overlap fraction of the inserted sleep: {overlap_fraction*100:.0f}%  (0%=no overlap, 100%=fully hidden)")

    if args.remote:
        print(f"\nrun()-alone duration for B vs C ({brun_mean:.2f} ms vs {crun_mean:.2f} ms) should be similar")
        print("regardless of the later sleep -- if run() itself is slow and already includes the full")
        print("compute+RTT, that's further evidence it's NOT returning early.")


if __name__ == "__main__":
    main()
