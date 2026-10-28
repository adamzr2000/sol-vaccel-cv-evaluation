#!/usr/bin/env python3
"""
Bisects what in the real benchmark loop slows preprocessing down.

vaccel_process_probe.py showed preprocessing is ~55ms in the benchmark process
even after load_model() + infer() -- when run back-to-back. The real loop
differs in three ways; this adds them one at a time (one process, model loaded
once, edge agent required) and times adapter.preprocess_frames/frame each step:

  L0  back-to-back, same frames                 (control, = probe stage D)
  L1  back-to-back, fresh frame copies per call (like get_clip()/get_latest())
  L2  same frames, sleep(gap) between calls     (idle gap, no network)
  L3  same frames, real infer() between calls   (RPC idle gap)
  L4  fresh copies + infer() + postprocess      (real loop minus GStreamer)

gap = median infer() time measured during warmup, so L2 isolates "idle
between calls" from "RPC/network activity" (L3). cpu_mhz = mean core clock
read right after each preprocess call (outside the timed part).

Run inside the run_ros.sh container, edge agent up, go2-driver stopped:
  cd /src && MODEL=r3d_18 BACKEND=vaccel-remote-sol DEVICE=gpu \
    OMP_NUM_THREADS=$(nproc) OMP_THREAD_LIMIT=$(nproc) python3 vaccel_loop_probe.py
"""
import glob
import os
import time

import torch

_thread_limit = os.environ.get("OMP_THREAD_LIMIT") or os.environ.get("OMP_NUM_THREADS")
if _thread_limit:
    torch.set_num_interop_threads(int(_thread_limit))
_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _num_threads > 0:
    torch.set_num_threads(_num_threads)

import numpy as np

from benchmark_utils import get_benchmark_config
from model_adapter import get_model_adapter

ITERS = int(os.environ.get("ITERS", "100"))
FREQ_FILES = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))


def cpu_mhz():
    if FREQ_FILES:
        return float(np.mean([int(open(f).read()) for f in FREQ_FILES])) / 1000.0
    mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    return float(np.mean(mhz)) if mhz else float("nan")


cfg = get_benchmark_config()
adapter = get_model_adapter(cfg["model_arch"], cfg["backend"], cfg["device"])
adapter.load_model(os.path.join("models", cfg["model_arch"]))
is_video = cfg["is_video_model"]

rng = np.random.default_rng(0)
n = 16 if is_video else 1
frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(n)]


def pre(fr):
    return adapter.preprocess_frames(fr) if is_video else adapter.preprocess_frame(fr[0])


inp = pre(frames)
rpc_ms = []
with torch.inference_mode():
    for _ in range(20):
        t0 = time.perf_counter()
        adapter.infer(inp)
        rpc_ms.append((time.perf_counter() - t0) * 1000.0)
gap_s = float(np.median(rpc_ms)) / 1000.0
print(f"torch={torch.__version__} threads={torch.get_num_threads()} model={cfg['model_arch']} "
      f"median infer()={gap_s * 1000:.1f}ms freq_source={'sysfs' if FREQ_FILES else '/proc/cpuinfo'}")


def stage(name, fresh=False, between=None):
    ts, mhz = [], []
    for i in range(ITERS + 10):
        fr = [f.copy() for f in frames] if fresh else frames
        t0 = time.perf_counter()
        x = pre(fr)
        dt = (time.perf_counter() - t0) * 1000.0
        f_now = cpu_mhz()
        if i >= 10:
            ts.append(dt)
            mhz.append(f_now)
        if between:
            between(x)
    a = np.array(ts)
    print(f"{name:42s} p50={np.percentile(a, 50):7.2f}ms p90={np.percentile(a, 90):7.2f}ms "
          f"min={a.min():7.2f}ms  cpu_mhz={np.median(mhz):6.0f}")


def do_infer(x):
    with torch.inference_mode():
        adapter.infer(x)


def do_infer_post(x):
    with torch.inference_mode():
        adapter.postprocess(adapter.infer(x))


stage("L0 back-to-back, same frames")
stage("L1 back-to-back, fresh copies", fresh=True)
stage(f"L2 same frames, sleep {gap_s * 1000:.0f}ms between", between=lambda x: time.sleep(gap_s))
stage("L3 same frames, infer() between", between=do_infer)
stage("L4 fresh copies + infer() + postprocess", fresh=True, between=do_infer_post)
