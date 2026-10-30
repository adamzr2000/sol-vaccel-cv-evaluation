#!/usr/bin/env python3
"""
Isolates WHY vaccel's preprocessing p50 is higher than ros2's: same process,
same H264 decode pipeline, same frames, same torchvision transforms -- only
the structure differs, like the two real pipelines:

  MODE=V  vaccel-like (model_benchmark_ros_v2.py): GStreamer decodes
          continuously on its own streaming thread (callback: cvtColor +
          store, as GstFrameBuffer._handle_sample); the main loop waits for a
          new frame, copies it, preprocesses (timed), then idles GAP_MS
          (stands in for the RPC -- vaccel_loop_probe.py showed a sleep and a
          real infer() behave the same) and optionally throttles to PERIOD_MS.
  MODE=R  ros2-like (go2_camerapublisher): preprocessing runs INSIDE the
          decode callback, right after cvtColor, so decoding of the next frame
          can't overlap it (single streaming thread, no queue elements).

Needs an RTP/H264 stream on udp://127.0.0.1:${PORT} (see run_pre_contention.sh).
For MODE=V, also records per sample how much capture work (decode +
videoconvert, then the callback) overlapped the preprocessing window, as the
real benchmark's capture_overlap_ms diagnostic does.

Env: MODE=V|R  CATEGORY=image|video  GAP_MS  PERIOD_MS  SAMPLES  PORT
     OMP_WAIT_POLICY / OMP_NUM_THREADS as for the real runs.
     DEC_THREADS=N   avdec_h264 max-threads (1 = all decode on the streaming
                     thread, so the measured decode intervals are exact).
     FREEZE_AFTER=N  MODE=V only: stop the pipeline (no decode at all) after N
                     frames and keep preprocessing the last frames -- isolates
                     the loop's idle pattern from any decode contention.
     SYNC=1          MODE=V only: after the RPC, wait for a frame decoded after
                     it returned (candidate fix: preprocess on arrival, as ros2).
     NICE_DECODE=1   lower the GStreamer streaming thread (decode + convert +
                     callback) to nice 19 on its first buffer (candidate fix).
                     Use with DEC_THREADS=1: libav's own decode threads are
                     created before that and would keep the default priority.
Always on: a 5 Hz sampler records which core the preprocessing thread and the
3 busiest (spinning OpenMP worker) threads last ran on, and reports the
fraction of samples where two of them shared a core (colocated_frac).
"""
import os
import resource
import threading
import time
from collections import deque

import cv2
import gi
import numpy as np
import torch

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

_n = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _n > 0:
    torch.set_num_interop_threads(_n)
    torch.set_num_threads(_n)

import torchvision.transforms as T  # noqa: E402

MODE = os.environ.get("MODE", "V").upper()
CATEGORY = os.environ.get("CATEGORY", "image")
GAP_MS = float(os.environ.get("GAP_MS", "14"))
PERIOD_MS = float(os.environ.get("PERIOD_MS", "0"))
SAMPLES = int(os.environ.get("SAMPLES", "600"))
WARMUP = 30
PORT = int(os.environ.get("PORT", "5600"))
DEC_THREADS = os.environ.get("DEC_THREADS")
FREEZE_AFTER = int(os.environ.get("FREEZE_AFTER", "0"))
SYNC = os.environ.get("SYNC", "0") == "1"
NICE_DECODE = os.environ.get("NICE_DECODE", "0") == "1"
W, H, CLIP = 1280, 720, 16

# Same transforms as model_adapter.py's SolAdapter / go2_camerapublisher.
if CATEGORY == "video":
    tf = T.Compose([T.Resize(128, antialias=True), T.CenterCrop(112),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=[0.432, 0.394, 0.377], std=[0.228, 0.221, 0.217])])
else:
    tf = T.Compose([T.ToTensor(), T.Resize(256, antialias=True), T.CenterCrop(224),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def pre_vaccel(x):
    # model_adapter.py SolAdapter.preprocess_frame / preprocess_frames
    if CATEGORY == "video":
        frames = [tf(torch.from_numpy(f).permute(2, 0, 1)) for f in x]
        return torch.stack(frames, dim=1).unsqueeze(0).numpy()
    return tf(x).unsqueeze(0).numpy()


def pre_ros(x):
    # go2_camerapublisher _preprocess_for_category / _preprocess_video_clip
    if CATEGORY == "video":
        return torch.cat([tf(torch.from_numpy(f).permute(2, 0, 1)).permute(1, 2, 0) for f in x], dim=0)
    return tf(x)


lock = threading.Lock()
state = {"latest": None, "seq": 0}
ring = deque(maxlen=CLIP)
busy = deque(maxlen=256)           # (start_ns, end_ns) capture-work intervals
dec_start = [None]
pre_ms, overlap_ms, decode_ms, mhz, faults = [], [], [], [], []


def minflt():
    # Process-wide minor page faults (all threads: torch's workers write the
    # outputs). Read just outside each timed preprocessing region.
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt
FREQ_FILES = sorted(__import__("glob").glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))


def cpu_mhz():
    # Mean core clock (intel_pstate: measured average since the last read).
    # Read right after each preprocessing, outside its timed region.
    return sum(int(open(f).read()) for f in FREQ_FILES) / len(FREQ_FILES) / 1000.0 if FREQ_FILES else float("nan")
done = threading.Event()


niced = [False]
pre_tid = [threading.get_native_id() if MODE == "V" else None]


def on_decode_input(pad, info):
    if NICE_DECODE and not niced[0]:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        niced[0] = True
    if dec_start[0] is None:
        dec_start[0] = time.perf_counter_ns()
    return Gst.PadProbeReturn.OK


def on_sample(sink):
    entry = time.perf_counter_ns()
    if dec_start[0] is not None:
        decode_ms.append((entry - dec_start[0]) / 1e6)
    with lock:
        busy.append((dec_start[0] or entry, entry))
    dec_start[0] = None
    cb0 = time.perf_counter_ns()
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    ok, mi = buf.map(Gst.MapFlags.READ)
    try:
        rgb = cv2.cvtColor(np.frombuffer(mi.data, np.uint8).reshape(H, W, 3), cv2.COLOR_BGR2RGB)
        if MODE == "R":
            ring.append(rgb)
            if CATEGORY != "video" or len(ring) == CLIP:
                pre_tid[0] = threading.get_native_id()
                f0 = minflt()
                t0 = time.perf_counter()
                out = pre_ros(list(ring) if CATEGORY == "video" else rgb)
                dt = (time.perf_counter() - t0) * 1000.0
                faults.append(minflt() - f0)
                # untimed, as in go2_camerapublisher (after the stamp)
                (out if CATEGORY == "video" else out.permute(1, 2, 0)).contiguous().numpy().astype(np.float32)
                pre_ms.append(dt)
                mhz.append(cpu_mhz())
                if len(pre_ms) >= WARMUP + SAMPLES:
                    done.set()
        else:
            with lock:
                state["latest"] = rgb
                ring.append(rgb)
                state["seq"] += 1
    finally:
        buf.unmap(mi)
        with lock:
            busy.append((cb0, time.perf_counter_ns()))
    return Gst.FlowReturn.OK


Gst.init(None)
pipe = Gst.parse_launch(
    f"udpsrc port={PORT} ! application/x-rtp, media=video, encoding-name=H264, clock-rate=90000 "
    f"! rtph264depay ! h264parse ! avdec_h264 name=dec ! videoconvert "
    f"! video/x-raw,width={W},height={H},format=BGR ! appsink drop=1")
sink = pipe.get_by_name("appsink0")
sink.set_property("emit-signals", True)
sink.set_property("max-buffers", 1)
sink.connect("new-sample", on_sample)
pipe.get_by_name("dec").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_decode_input)
if DEC_THREADS:
    pipe.get_by_name("dec").set_property("max-threads", int(DEC_THREADS))
pipe.set_state(Gst.State.PLAYING)

coloc = []                         # 1 = preprocessing thread or spinners shared a core


def placement_sampler():
    me = threading.get_native_id()
    prev = {}
    while not done.is_set():
        time.sleep(0.2)
        cur = {}
        for tid in os.listdir("/proc/self/task"):
            try:
                f = open(f"/proc/self/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            cur[int(tid)] = (int(f[11]) + int(f[12]), int(f[36]))  # utime+stime ticks, last cpu
        deltas = sorted(((cur[t][0] - prev[t][0], t) for t in cur if t in prev and t != me), reverse=True)
        prev = cur
        spinners = [t for d, t in deltas[:3] if d >= 10]     # >= 50% of a core over 0.2 s (100 Hz ticks)
        pt = pre_tid[0]
        if len(spinners) < 3 or pt is None or pt not in cur:
            continue
        cpus = [cur[t][1] for t in spinners if t != pt] + [cur[pt][1]]
        coloc.append(int(len(set(cpus)) < len(cpus)))


threading.Thread(target=placement_sampler, daemon=True).start()


def overlap(t0, t1):
    with lock:
        iv = list(busy)
    return sum(max(0, min(e, t1) - max(s, t0)) for s, e in iv) / 1e6


if MODE == "V":
    last_seq, last_tick, n, frozen = -1, None, 0, False
    deadline = time.time() + 600
    while n < WARMUP + SAMPLES and time.time() < deadline:
        if PERIOD_MS > 0 and last_tick is not None:
            dt = time.time() - last_tick
            if dt < PERIOD_MS / 1000.0:
                time.sleep(PERIOD_MS / 1000.0 - dt)
        last_tick = time.time()
        with lock:
            seq = state["seq"]
            ready = state["latest"] is not None and (CATEGORY != "video" or len(ring) == CLIP)
        if FREEZE_AFTER and not frozen and seq >= FREEZE_AFTER:
            pipe.set_state(Gst.State.NULL)   # no more decode from here on
            frozen = True
        if not ready or (seq == last_seq and not frozen):
            time.sleep(0.001)
            continue
        last_seq = seq
        with lock:
            x = [f.copy() for f in ring] if CATEGORY == "video" else state["latest"].copy()
        f0 = minflt()
        t0n = time.perf_counter_ns()
        pre_vaccel(x)
        t1n = time.perf_counter_ns()
        faults.append(minflt() - f0)
        mhz.append(cpu_mhz())
        time.sleep(GAP_MS / 1000.0)          # stands in for the RPC
        if SYNC:
            # Only a frame decoded AFTER this point counts as new: the next
            # preprocessing then starts right after a decode finishes, as in
            # ros2 (preprocess on arrival), instead of at a random phase.
            with lock:
                last_seq = state["seq"]
        pre_ms.append((t1n - t0n) / 1e6)
        overlap_ms.append(overlap(t0n, t1n))  # window long past: all overlapping intervals recorded
        n += 1
else:
    done.wait(600)

done.set()
pipe.set_state(Gst.State.NULL)

p = np.array(pre_ms[WARMUP:])
line = (f"MODE={MODE} CATEGORY={CATEGORY} GAP_MS={GAP_MS:g} PERIOD_MS={PERIOD_MS:g} "
        f"DEC_THREADS={DEC_THREADS or 'auto'} FREEZE_AFTER={FREEZE_AFTER} SYNC={int(SYNC)} "
        f"NICE_DECODE={int(NICE_DECODE)} OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND')} "
        f"torch={torch.__version__} threads={torch.get_num_threads()} "
        f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY')} n={len(p)}\n"
        f"  pre p50={np.median(p):7.2f}  p90={np.percentile(p, 90):7.2f}  min={p.min():7.2f}  "
        f"| decode+convert per frame p50={np.median(decode_ms) if decode_ms else float('nan'):5.2f}ms"
        f"  cpu_mhz p50={np.median(mhz[WARMUP:]):5.0f}"
        f"  colocated_frac={np.mean(coloc) if coloc else float('nan'):4.2f} (n={len(coloc)})")
fl = np.array(faults[WARMUP:])
if len(fl) == len(p) and len(p):
    zero, some = p[fl == 0], p[fl > 0]
    line += (f"\n  minflt/pre p50={np.median(fl):6.0f}  frac>0={np.mean(fl > 0):4.2f}  "
             f"pre@0faults p50={np.median(zero) if len(zero) else float('nan'):6.2f} (n={len(zero)})  "
             f"pre@faults p50={np.median(some) if len(some) else float('nan'):6.2f} (n={len(some)})  "
             f"GLIBC_TUNABLES={os.environ.get('GLIBC_TUNABLES')}")
if MODE == "V" and len(overlap_ms) > WARMUP:
    o = np.array(overlap_ms[WARMUP:])
    none, some = p[o == 0], p[o > 0]
    b, a = np.polyfit(o, p, 1) if np.ptp(o) > 0 else (0.0, float(np.median(p)))
    line += (f"\n  overlap_ms p50={np.median(o):5.2f}  frac>0={np.mean(o > 0):4.2f}  "
             f"no-overlap p50={np.median(none) if len(none) else float('nan'):7.2f} (n={len(none)})  "
             f"overlap p50={np.median(some) if len(some) else float('nan'):7.2f} (n={len(some)})  "
             f"fit a={a:6.2f} b={b:5.2f}")
print(line, flush=True)
