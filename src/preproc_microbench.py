#!/usr/bin/env python3
"""
Isolated preprocessing microbenchmark -- no camera, no GStreamer, no RPC.
Times the exact torchvision transforms both projects use (model_adapter.py /
go2_camerapublisher) on synthetic 1280x720 RGB frames, so the same script can
be run in both containers on the same host and compared directly.

Run in each container (no file copy needed):
  docker exec -i go2-driver      python3 - < preproc_microbench.py
  docker exec -i torchvision-app python3 - < preproc_microbench.py
Optional: THREADS=N to force torch.set_num_threads(N).
"""
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ITERS = int(os.environ.get("ITERS", "200"))
WARMUP = int(os.environ.get("WARMUP", "20"))
if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

video_tf = T.Compose([
    T.Resize(128, antialias=True),
    T.CenterCrop(112),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=[0.432, 0.394, 0.377], std=[0.228, 0.221, 0.217]),
])
image_tf = T.Compose([
    T.ToTensor(),
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

rng = np.random.default_rng(0)
frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(16)]


def video_clip():
    fl = [video_tf(torch.from_numpy(f).permute(2, 0, 1)) for f in frames]
    return torch.stack(fl, dim=1).unsqueeze(0).numpy()


def image_frame():
    return image_tf(frames[0]).unsqueeze(0)


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return np.percentile(a, 50), np.percentile(a, 90), a.min()


print(f"torch={torch.__version__} torchvision={torchvision.__version__} "
      f"torch_threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()} "
      f"cpu_count={os.cpu_count()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
for name, fn in [("video clip (16x112)", video_clip), ("image frame (224)", image_frame)]:
    p50, p90, mn = bench(fn)
    print(f"{name:20s} p50={p50:7.2f}ms  p90={p90:7.2f}ms  min={mn:7.2f}ms")
