#!/usr/bin/env python3
"""
test_concurrent_sessions.py (standalone experiment -- not part of the benchmark harness)

Tier-2 probe: can the SAME vaccel-rpc-agent handle TWO independent, concurrent
vaccel sessions issuing overlapping blocking predict() calls at once, safely
and with a real throughput benefit -- or does it serialize internally (no
throughput gain) or corrupt/crash (unsafe)?

Each of the two threads creates its OWN adapter/session (get_model_adapter),
never shares any Python object with the other thread -- this only tests the
AGENT's behavior under concurrent sessions, not any client-side state.

Method:
  A. SEQUENTIAL baseline: thread1 runs N predict() calls, then thread2 runs
     N predict() calls, one after another. Total wall time = T_seq.
  B. CONCURRENT: both threads run their N predict() calls starting at the
     same time. Total wall time = T_conc.

  T_conc ~= T_seq            -> agent serializes sessions internally (no
                                 overlap benefit, but likely safe)
  T_conc ~= T_seq / 2         -> genuine concurrent execution (the real win
                                 Tier 2 would be chasing)
  crash / exception / wrong  -> NOT safe for concurrent sessions
  per-thread output

Each thread uses a FIXED, distinct input per thread and checks every one of
its own N outputs is identical (same class_id/confidence) -- catches any
cross-session corruption (e.g. resource-id mixups) even if nothing crashes.

Does not touch model_adapter.py / model_benchmark_ros*.py / any production
file. Safe to run repeatedly; writes nothing to disk.

Usage (inside the torchvision-ros-app:gpu container):
  python3 /src/experiments/test_concurrent_sessions.py --iters 20
"""
import argparse
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/src")
from model_adapter import get_model_adapter


def worker(thread_id, model, backend, device, model_dir, input_seed, iters, results, errors, barrier=None):
    try:
        adapter = get_model_adapter(model, backend, device)
        adapter.load_model(model_dir)

        rng = np.random.default_rng(input_seed)
        fixed_frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        input_tensor = adapter.preprocess_frame(fixed_frame)

        # Warmup (not timed, not checked)
        for _ in range(3):
            adapter.infer(input_tensor)

        if barrier is not None:
            barrier.wait()  # line up both threads to start at the same instant

        class_ids = []
        confidences = []
        t0 = time.perf_counter()
        for _ in range(iters):
            raw = adapter.infer(input_tensor)
            result = adapter.postprocess(raw)
            class_id_tensor, prob_tensor = result
            class_ids.append(int(class_id_tensor.item()))
            confidences.append(float(prob_tensor.item()))
        elapsed = time.perf_counter() - t0

        results[thread_id] = {
            "elapsed_sec": elapsed,
            "class_ids": class_ids,
            "confidences": confidences,
            "consistent": len(set(class_ids)) == 1,
        }
    except Exception as e:
        errors[thread_id] = repr(e)


def run_pair(model, backend, device, model_dir, iters, concurrent):
    results, errors = {}, {}
    barrier = threading.Barrier(2) if concurrent else None

    t1 = threading.Thread(target=worker, args=(1, model, backend, device, model_dir, 101, iters, results, errors, barrier))
    t2 = threading.Thread(target=worker, args=(2, model, backend, device, model_dir, 202, iters, results, errors, barrier))

    t0 = time.perf_counter()
    if concurrent:
        t1.start(); t2.start()
        t1.join(timeout=120); t2.join(timeout=120)
    else:
        t1.start(); t1.join(timeout=120)
        t2.start(); t2.join(timeout=120)
    total_wall = time.perf_counter() - t0

    return total_wall, results, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--backend", default="vaccel-remote-sol")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--model-dir", default="models/resnet50_sol")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    # get_model_adapter() expects "cuda"/"cpu" (torch.device strings), not the
    # CLI-facing "gpu"/"cpu" -- benchmark_utils.get_benchmark_config() does this
    # same translation for the real scripts. Missing this silently loads the
    # lib_cpu SOL library even when --device gpu was requested.
    device = "cuda" if args.device == "gpu" else args.device
    args.device = device

    print(f"[setup] model={args.model} backend={args.backend} device={args.device} iters={args.iters}")

    print("\n=== A. SEQUENTIAL (thread1 then thread2) ===")
    t_seq, res_seq, err_seq = run_pair(args.model, args.backend, args.device, args.model_dir, args.iters, concurrent=False)
    for tid in (1, 2):
        if tid in err_seq:
            print(f"  thread{tid}: ERROR: {err_seq[tid]}")
        else:
            r = res_seq[tid]
            print(f"  thread{tid}: elapsed={r['elapsed_sec']:.3f}s consistent_output={r['consistent']} "
                  f"class_ids_seen={sorted(set(r['class_ids']))}")
    print(f"  T_seq (total wall, both threads) = {t_seq:.3f}s")

    print("\n=== B. CONCURRENT (both threads start together) ===")
    t_conc, res_conc, err_conc = run_pair(args.model, args.backend, args.device, args.model_dir, args.iters, concurrent=True)
    for tid in (1, 2):
        if tid in err_conc:
            print(f"  thread{tid}: ERROR: {err_conc[tid]}")
        else:
            r = res_conc[tid]
            print(f"  thread{tid}: elapsed={r['elapsed_sec']:.3f}s consistent_output={r['consistent']} "
                  f"class_ids_seen={sorted(set(r['class_ids']))}")
    print(f"  T_conc (total wall, both threads) = {t_conc:.3f}s")

    print("\n=== VERDICT ===")
    if err_conc:
        print("  CRASH/EXCEPTION under concurrent sessions -> NOT safe. Stop here, do not pursue Tier 2.")
        return
    if not all(res_conc.get(tid, {}).get("consistent", False) for tid in (1, 2)):
        print("  INCONSISTENT per-thread outputs under concurrency -> cross-session corruption. NOT safe.")
        return

    speedup = t_seq / t_conc if t_conc > 0 else float("nan")
    print(f"  No crash, no corruption. Speedup (T_seq / T_conc) = {speedup:.2f}x")
    if speedup > 1.3:
        print("  => Agent genuinely overlaps concurrent sessions. Tier 2 is viable -- worth pursuing further.")
    else:
        print("  => Agent appears to serialize sessions internally (no real overlap). Tier 2 would add complexity")
        print("     and client-side concurrency risk for no throughput benefit. Not worth pursuing further.")


if __name__ == "__main__":
    main()
