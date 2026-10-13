#!/usr/bin/env python3
"""
test_preprocess_live_no_inference.py (standalone experiment)

Second step in root-causing the baseline's periodic 700-1040ms "Pre:" spikes
seen in model_benchmark_ros.py's video-model path. test_preprocess_isolation.py
already proved preprocess_frames() alone (no ROS) never spikes.

This test reproduces the EXACT serial fetch -> dedup-gate -> get_clip() ->
preprocess_frames() sequence against the REAL live camera topic + a real
rclpy spin thread (same subscription/threading setup as model_benchmark_ros.py)
but calls NO inference and touches vaccel/the agent not at all. If the spike
reappears here, it implicates live ROS frame-arrival/spin-thread interaction
(GIL contention, GC pressure from continuous frame churn, etc), not anything
about the vaccel RPC call itself.

Does not touch model_adapter.py / model_benchmark_ros.py / any production
file. Safe to run repeatedly; publishes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/src")
from model_adapter import get_model_adapter
from model_benchmark_ros import RosFrameBuffer  # reuse exact production buffer, unmodified

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import threading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r3d_18")
    ap.add_argument("--backend", default="vaccel-remote-sol")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--model-dir", default="models/r3d_18_sol")
    ap.add_argument("--topic", default="/go2/sensor/camera")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    # NOTE: importing model_benchmark_ros (for RosFrameBuffer) already runs its
    # module-level torch.set_num_interop_threads()/set_num_threads() calls, so
    # this script must not call them again (interop threads can only be set once).
    adapter = get_model_adapter(args.model, args.backend, args.device)
    adapter.load_model(args.model_dir)

    rclpy.init()
    node = rclpy.create_node("preprocess_live_no_inference_test")
    frame_buf = RosFrameBuffer(is_video=True, queue_size=1, video_len=16)
    node.create_subscription(Image, args.topic, frame_buf.callback, qos_profile_sensor_data)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Waiting for first frame...")
    t0 = time.time()
    while rclpy.ok():
        if frame_buf.get_received_count() > 0:
            break
        if time.time() - t0 > 10:
            print("No frames received within 10s.")
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)
            node.destroy_node()
            return
        time.sleep(0.05)

    print("Filling clip ring buffer...")
    while rclpy.ok() and frame_buf.get_clip() is None:
        time.sleep(0.01)

    print(f"Running {args.iters} serial fetch+preprocess iterations (NO inference call)...")
    times_ms = []
    last_seen_stamp_ns = frame_buf.get_latest_stamp_ns()
    i = 0
    stall_guard_t0 = time.time()
    while i < args.iters and rclpy.ok():
        cur_stamp = frame_buf.get_latest_stamp_ns()
        if cur_stamp is None or cur_stamp == last_seen_stamp_ns:
            if time.time() - stall_guard_t0 > 20:
                print("Stalled waiting for a fresh frame >20s, aborting.")
                break
            time.sleep(0.001)
            continue
        last_seen_stamp_ns = cur_stamp
        stall_guard_t0 = time.time()

        clip = frame_buf.get_clip()
        if clip is None:
            time.sleep(0.001)
            continue

        t_pre0 = time.perf_counter()
        adapter.preprocess_frames(clip)
        dt_ms = (time.perf_counter() - t_pre0) * 1000.0
        times_ms.append(dt_ms)
        flag = "  <<<< SPIKE" if dt_ms > 100 else ""
        print(f"[{i+1:3d}] preprocess_frames (live, no inference): {dt_ms:8.2f} ms{flag}")
        i += 1

    rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()

    if times_ms:
        arr = np.asarray(times_ms)
        print("\n=== STATS (ms) ===")
        print(f"mean={arr.mean():.2f} median={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} max={arr.max():.2f}")
        n_spikes = int((arr > 100).sum())
        print(f"spikes (>100ms): {n_spikes}/{len(arr)}")
        if n_spikes > 0:
            print("=> Spikes REPRODUCE with live camera + spin thread, with NO inference call at all.")
            print("   => implicates ROS frame-arrival/spin-thread interaction, NOT the vaccel RPC call.")
        else:
            print("=> No spikes even with live camera + spin thread (no inference) => the spike needs the")
            print("   inference call present too (e.g. GIL handoff pattern specific to the blocking RPC wait).")


if __name__ == "__main__":
    main()
