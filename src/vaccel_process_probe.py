#!/usr/bin/env python3
"""
Pinpoints WHEN preprocessing slows down inside the benchmark process.

Times the same fixed torchvision transform (identical to preproc_microbench.py)
at successive stages of one process, the order model_benchmark_ros_v2.py goes
through, and reports thread count, OpenMP runtimes mapped, and CPU burned
while the main thread sleeps (>0 = background threads spinning):

  A  bare: torch/torchvision only
  B  after importing benchmark_utils + model_adapter
  C  after get_model_adapter() + load_model()   (SOL / vAccel libs loaded)
  D  after INFER_CALLS real adapter.infer() calls (RPC to the edge agent)

Run inside the run_ros.sh container (same env as the real benchmark), with
the edge agent up:
  cd /src && MODEL=r3d_18 BACKEND=vaccel-remote-sol DEVICE=gpu \
    OMP_NUM_THREADS=$(nproc) OMP_THREAD_LIMIT=$(nproc) python3 vaccel_process_probe.py
STOP_AFTER=A|B|C stops early (e.g. to sanity-check without an agent).
"""
import os
import time

import torch

# Same thread setup as model_benchmark_ros_v2.py, done before any parallel work.
_thread_limit = os.environ.get("OMP_THREAD_LIMIT") or os.environ.get("OMP_NUM_THREADS")
if _thread_limit:
    torch.set_num_interop_threads(int(_thread_limit))
_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _num_threads > 0:
    torch.set_num_threads(_num_threads)

import numpy as np
import torchvision.transforms as T

ITERS = int(os.environ.get("ITERS", "100"))
INFER_CALLS = int(os.environ.get("INFER_CALLS", "30"))
STOP_AFTER = os.environ.get("STOP_AFTER", "").upper()
CLK_TCK = os.sysconf("SC_CLK_TCK")

video_tf = T.Compose([
    T.Resize(128, antialias=True),
    T.CenterCrop(112),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=[0.432, 0.394, 0.377], std=[0.228, 0.221, 0.217]),
])
rng = np.random.default_rng(0)
frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(16)]


def yardstick():
    fl = [video_tf(torch.from_numpy(f).permute(2, 0, 1)) for f in frames]
    return torch.stack(fl, dim=1).unsqueeze(0).numpy()


def time_fn(fn, iters=ITERS):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return f"p50={np.percentile(a, 50):7.2f}ms p90={np.percentile(a, 90):7.2f}ms min={a.min():7.2f}ms"


def idle_cpu_cores(seconds=2.0):
    def cpu():
        f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])  # utime + stime (fields 14, 15)
    c0 = cpu()
    time.sleep(seconds)
    return (cpu() - c0) / CLK_TCK / seconds


def omp_libs():
    libs = set()
    for line in open("/proc/self/maps"):
        p = line.split()[-1] if len(line.split()) >= 6 else ""
        if any(k in os.path.basename(p) for k in ("gomp", "iomp", "libomp")):
            libs.add(p)
    return sorted(libs) or ["(none)"]


def report(stage, extra=None):
    n_threads = len(os.listdir("/proc/self/task"))
    idle = idle_cpu_cores()
    print(f"\n[{stage}] threads={n_threads}  cpu_while_idle={idle:.2f} cores")
    print(f"    openmp: {', '.join(omp_libs())}")
    print(f"    yardstick video clip: {time_fn(yardstick)}")
    if extra:
        print(f"    adapter.preprocess_frames: {time_fn(extra)}")


print(f"torch={torch.__version__} threads={torch.get_num_threads()} "
      f"interop={torch.get_num_interop_threads()} cpu_count={os.cpu_count()}")
report("A bare")
if STOP_AFTER == "A":
    raise SystemExit

from benchmark_utils import get_benchmark_config  # noqa: E402
from model_adapter import get_model_adapter  # noqa: E402
report("B imports")
if STOP_AFTER == "B":
    raise SystemExit

cfg = get_benchmark_config()
adapter = get_model_adapter(cfg["model_arch"], cfg["backend"], cfg["device"])
adapter.load_model(os.path.join("models", cfg["model_arch"]))
pre = (lambda: adapter.preprocess_frames(frames)) if cfg["is_video_model"] else \
      (lambda: adapter.preprocess_frame(frames[0]))
print(f"\nloaded {cfg['model_arch']} backend={cfg['backend']} device={cfg['device']}")
report("C load_model", pre)
if STOP_AFTER == "C":
    raise SystemExit

inp = pre()
with torch.inference_mode():
    for _ in range(INFER_CALLS):
        adapter.infer(inp)
report(f"D after {INFER_CALLS} infer()", pre)
