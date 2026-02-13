#!/usr/bin/env python3
import os
import time
import csv
import json
import threading
from collections import deque

import torch
import numpy as np
import cv2
import logging

import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pathlib import Path
from datetime import datetime, timezone

from model_adapter import get_model_adapter

try:
    from segmentation_utils import COLORS, analyze_segmentation_mask
except ImportError:
    COLORS = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

    def analyze_segmentation_mask(mask):
        return ""

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ==========================================
# HELPER: STATS CALCULATOR
# ==========================================
def calculate_stats(data_list):
    """Calculates detailed statistics for a list of numbers."""
    if not data_list:
        return {k: 0.0 for k in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]}

    data = np.array(data_list)
    return {
        "mean": round(float(np.mean(data)), 4),
        "std":  round(float(np.std(data)), 4),
        "min":  round(float(np.min(data)), 4),
        "max":  round(float(np.max(data)), 4),
        "p25":  round(float(np.percentile(data, 25)), 4),
        "p50":  round(float(np.percentile(data, 50)), 4),
        "p75":  round(float(np.percentile(data, 75)), 4),
        "p90":  round(float(np.percentile(data, 90)), 4),
        "p95":  round(float(np.percentile(data, 95)), 4),
        "p99":  round(float(np.percentile(data, 99)), 4)
    }


# ==========================================
# CONFIGURATION (env-driven like original)
# ==========================================
# Backend: 'stock' (default), 'vaccel-local' (or 'vaccel') or 'vaccel-remote'
BACKEND = os.environ.get("BACKEND", "stock")
if BACKEND not in ["stock", "vaccel", "vaccel-local", "vaccel-remote"]:
    print(f"⚠️  Unknown BACKEND '{BACKEND}', defaulting to 'stock'")
    BACKEND = "stock"

if "remote" in BACKEND:
    print(f"   🔎 VACCEL_RPC_ADDRESS={os.environ.get('VACCEL_RPC_ADDRESS')}")

TARGET_DEVICE = os.environ.get("DEVICE", "cpu").lower()
if TARGET_DEVICE == "gpu":
    DEVICE = "cuda"
    if "remote" in BACKEND:
        TORCH_DEVICE = torch.device("cpu")
    else:
        TORCH_DEVICE = torch.device("cuda")
else:
    DEVICE = "cpu"
    TORCH_DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Host + Model
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "edge").lower()
MODEL_ARCH = os.environ.get("MODEL", "resnet50")

# ---------------------------------------------------------------------------
# Keep EXPORT_RESULTS flag (default false)
# ---------------------------------------------------------------------------
EXPORT_RESULTS = os.environ.get("EXPORT_RESULTS", "false").strip().lower() in ("1", "true", "yes", "y", "on")

# Results dir: same structure as original (model-stats/<HOST>/<run_id>/...)
_BASE_RESULTS = Path(os.environ.get("RESULTS_DIR", "/results/experiments/model-stats"))
RESULTS_DIR = _BASE_RESULTS / HOST

MODELS_DIR = Path("models")
CURRENT_MODEL_DIR = MODELS_DIR / MODEL_ARCH

# Detect model type for behavior + ROS outputs
CORE_MODEL_NAME = MODEL_ARCH.replace("_sol", "")
VIDEO_MODELS = ["mc3_18", "r3d_18", "r2plus1d_18", "swin3d_t", "swin3d_s", "swin3d_b"]
IS_VIDEO_MODEL = CORE_MODEL_NAME in VIDEO_MODELS

if IS_VIDEO_MODEL:
    MODEL_TYPE = "video_classification"
elif CORE_MODEL_NAME in ["resnet50", "mobilenet_v3_large", "swin_t", "swin_s", "swin_v2_b"]:
    MODEL_TYPE = "image_classification"
else:
    MODEL_TYPE = "semantic_segmentation"

frames_per_sample = 16 if IS_VIDEO_MODEL else 1

# ROS-specific (env)
INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "/camera/color/image_raw")

# Publish topics (you can disable any by setting it to empty string)
OUTPUT_MASK_TOPIC = os.environ.get("OUTPUT_MASK_TOPIC", "/benchmark/mask")
OUTPUT_OVERLAY_TOPIC = os.environ.get("OUTPUT_OVERLAY_TOPIC", "/benchmark/overlay")
OUTPUT_CLASS_TOPIC = os.environ.get("OUTPUT_CLASS_TOPIC", "/benchmark/classification")

# Controls to avoid inflating postprocessing numbers
PUBLISH_MASK = os.environ.get("PUBLISH_MASK", "0").strip().lower() in ("1", "true", "yes", "y", "on")
PUBLISH_OVERLAY = os.environ.get("PUBLISH_OVERLAY", "0").strip().lower() in ("1", "true", "yes", "y", "on")
PUBLISH_CLASS = os.environ.get("PUBLISH_CLASS", "1").strip().lower() in ("1", "true", "yes", "y", "on")

# Run control
EXPERIMENT_DURATION_SEC = float(os.environ.get("EXPERIMENT_DURATION_SEC", "30"))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "30"))
MAX_FPS = float(os.environ.get("MAX_FPS", "0"))  # 0 = no throttle

# Frame buffering
QUEUE_SIZE = int(os.environ.get("QUEUE_SIZE", "1"))  # 1=latest-only
VIDEO_BUFFER_LEN = int(os.environ.get("VIDEO_BUFFER_LEN", "16"))

# ---------------------------------------------------------------------------
# Duplicate control (default realistic)
# ---------------------------------------------------------------------------
AVOID_DUPLICATES = os.environ.get("AVOID_DUPLICATES", "1").strip().lower() in ("1", "true", "yes", "y", "on")


class RosFrameBuffer:
    """
    Stores latest frame (RGB) and a ring buffer for video clips.
    We do NOT time ROS waiting; benchmark loop reads whatever is available.
    Also tracks the latest ROS header stamp for "fresh frame" gating + observed camera Hz.
    """
    def __init__(self, is_video: bool, queue_size: int = 1, video_len: int = 16):
        self.is_video = is_video
        self.queue_size = max(1, queue_size)
        self.video_len = max(1, video_len)
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.latest_rgb = None
        self.queue = deque(maxlen=self.queue_size)
        self.video_ring = deque(maxlen=self.video_len)

        self.total_received = 0
        self.latest_stamp_ns = None            # ROS header stamp as int nanoseconds
        self.stamp_history_ns = deque(maxlen=200)  # for observed Hz (cheap, not timed)

    def callback(self, msg: Image):
        try:
            # Convert to BGR OpenCV image; ROS cameras commonly publish bgr8
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # ROS2 builtin_interfaces/Time: sec + nanosec
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        except Exception as e:
            print(f"[ROS] Failed to convert Image: {e}")
            return

        with self.lock:
            self.total_received += 1
            self.latest_rgb = rgb
            self.queue.append(rgb)
            if self.is_video:
                self.video_ring.append(rgb)

            self.latest_stamp_ns = stamp_ns
            self.stamp_history_ns.append(stamp_ns)

    def get_latest(self):
        with self.lock:
            if self.latest_rgb is None:
                return None
            return self.latest_rgb.copy()

    def get_clip(self):
        with self.lock:
            if len(self.video_ring) < self.video_len:
                return None
            return [fr.copy() for fr in list(self.video_ring)]

    def get_received_count(self):
        with self.lock:
            return int(self.total_received)

    def get_latest_stamp_ns(self):
        with self.lock:
            return self.latest_stamp_ns

    def get_observed_hz(self):
        """
        Estimate camera publish rate from header stamps.
        Returns 0.0 if not enough data.
        """
        with self.lock:
            stamps = list(self.stamp_history_ns)

        if len(stamps) < 2:
            return 0.0

        # Use last N-1 deltas; filter non-positive deltas (shouldn't happen, but be safe)
        deltas = []
        for i in range(1, len(stamps)):
            d = stamps[i] - stamps[i - 1]
            if d > 0:
                deltas.append(d)

        if not deltas:
            return 0.0

        mean_delta_ns = float(np.mean(np.array(deltas, dtype=np.float64)))
        if mean_delta_ns <= 0:
            return 0.0

        return 1e9 / mean_delta_ns


def main():
    rclpy.init()
    node = rclpy.create_node("model_benchmark_ros2")

    print(f"\n🚀 STARTING MODEL BENCHMARK (ROS2)")
    print(f"   Backend:   {BACKEND}")
    print(f"   Host:      {HOST}")
    print(f"   Model:     {MODEL_ARCH}")
    print(f"   Type:      {MODEL_TYPE}")
    print(f"   Device:    {TARGET_DEVICE}")
    print(f"   Loading:   {CURRENT_MODEL_DIR}")
    print(f"   Input:     {INPUT_TOPIC}")
    print(f"   Duration:  {EXPERIMENT_DURATION_SEC}s")
    print(f"   Warmup:    {WARMUP_ITERS} iters")
    print(f"   Publish:   class={PUBLISH_CLASS} mask={PUBLISH_MASK} overlay={PUBLISH_OVERLAY}")
    print(f"   Export:    EXPORT_RESULTS={EXPORT_RESULTS}")
    print(f"   Control:   AVOID_DUPLICATES={AVOID_DUPLICATES} MAX_FPS={MAX_FPS}")

    if TORCH_DEVICE.type == "cuda" and not torch.cuda.is_available():
        print("   ❌ GPU was selected but no GPU is available.")
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        adapter = get_model_adapter(MODEL_ARCH, BACKEND, DEVICE)
        adapter.load_model(CURRENT_MODEL_DIR)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Output dirs + run_id logic (same as original)
    run_tag = os.environ.get("RUN_TAG")
    if run_tag:
        prefix = run_tag
    else:
        prefix = time.strftime("%d-%m-%Y_%H-%M-%S")

    local_mode = "gpu" if (TORCH_DEVICE.type == "cuda") else "cpu"
    is_vaccel_remote_run = (HOST == "robot" and BACKEND == "vaccel-remote")
    if is_vaccel_remote_run:
        run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_{HOST}_{local_mode}_target-{TARGET_DEVICE}"
    else:
        run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_{HOST}_{local_mode}"

    run_dir = RESULTS_DIR / run_id

    # Only create output dir if exporting
    if EXPORT_RESULTS:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📂 Output Directory: {run_dir}")
    else:
        print(f"   📂 Output Directory: (disabled) would be {run_dir}")

    # ROS pubs/subs
    frame_buf = RosFrameBuffer(is_video=IS_VIDEO_MODEL, queue_size=QUEUE_SIZE, video_len=VIDEO_BUFFER_LEN)

    _ = node.create_subscription(
        Image,
        INPUT_TOPIC,
        frame_buf.callback,
        qos_profile_sensor_data,
    )

    # Start spinning so callbacks run
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Publishers are optional; create only if enabled AND topic is not empty
    pub_class = None
    pub_mask = None
    pub_overlay = None

    qos_out_class = QoSProfile(depth=10)
    qos_out_img = QoSProfile(depth=1)

    if PUBLISH_CLASS and OUTPUT_CLASS_TOPIC.strip():
        pub_class = node.create_publisher(String, OUTPUT_CLASS_TOPIC, qos_out_class)

    if PUBLISH_MASK and OUTPUT_MASK_TOPIC.strip():
        pub_mask = node.create_publisher(Image, OUTPUT_MASK_TOPIC, qos_out_img)

    if PUBLISH_OVERLAY and OUTPUT_OVERLAY_TOPIC.strip():
        pub_overlay = node.create_publisher(Image, OUTPUT_OVERLAY_TOPIC, qos_out_img)

    bridge = CvBridge()

    # Wait for first frame
    print("   ⏳ Waiting for first frame...")
    t_wait0 = time.time()
    while rclpy.ok():
        fr = frame_buf.get_latest()
        if fr is not None:
            break
        if time.time() - t_wait0 > 10:
            print("   ❌ No frames received within 10s. Check INPUT_TOPIC.")
            node.destroy_node()
            rclpy.shutdown()
            return
        time.sleep(0.05)

    # Warmup
    print(f"   🔥 Warming up ({WARMUP_ITERS} iterations)...")
    warm_done = 0
    while warm_done < WARMUP_ITERS and rclpy.ok():
        if IS_VIDEO_MODEL:
            clip = frame_buf.get_clip()
            if clip is None:
                time.sleep(0.01)
                continue
            inp = adapter.preprocess_frames(clip)
        else:
            fr = frame_buf.get_latest()
            if fr is None:
                time.sleep(0.01)
                continue
            inp = adapter.preprocess_frame(fr)

        try:
            with torch.no_grad():
                _ = adapter.infer(inp)
                if TORCH_DEVICE.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup failed: {e}")

        warm_done += 1

        if MAX_FPS > 0:
            time.sleep(1.0 / MAX_FPS)

    # Capture Start Time (ISO format)
    t_start_dt = datetime.now(timezone.utc)
    t_start_iso = t_start_dt.isoformat()

    # Timing lists
    transport_latencies_ms = []   # NEW: Track Data Age
    preprocessing_latencies_ms = []
    inference_latencies_ms = []
    postprocessing_latencies_ms = []
    total_system_latencies_ms = []
    confidence_scores_list = []
    detailed_sample_records = []

    # Counters/rates (camera hz, processed fps, duplicates)
    processed_samples = 0
    duplicates_skipped = 0
    camera_hz_samples = []  # snapshot observed Hz over time (optional, cheap)

    print("   ⏱️  Running Inference...")

    start_wall = time.time()
    sample_idx = 0
    last_tick = 0.0

    # Gate by ROS header stamp (more correct than received_count if you ever drop/reorder)
    last_seen_stamp_ns = frame_buf.get_latest_stamp_ns()

    while rclpy.ok():
        now = time.time()
        if now - start_wall >= EXPERIMENT_DURATION_SEC:
            break

        # Optional throttle (still useful even with AVOID_DUPLICATES)
        if MAX_FPS > 0:
            dt = now - last_tick
            if dt < (1.0 / MAX_FPS):
                time.sleep((1.0 / MAX_FPS) - dt)
            last_tick = time.time()

        # If avoiding duplicates, wait until a new ROS message arrives (by stamp)
        if AVOID_DUPLICATES:
            cur_stamp = frame_buf.get_latest_stamp_ns()
            if cur_stamp is None or cur_stamp == last_seen_stamp_ns:
                duplicates_skipped += 1
                time.sleep(0.001)
                continue
            last_seen_stamp_ns = cur_stamp

        # Track observed camera Hz (not timed, very cheap)
        hz = frame_buf.get_observed_hz()
        if hz > 0:
            camera_hz_samples.append(hz)

        # Fetch input (do NOT include ROS waiting time in timings)
        if IS_VIDEO_MODEL:
            clip = frame_buf.get_clip()
            if clip is None:
                time.sleep(0.001)
                continue
            input_src = f"frame_{sample_idx:06d}_clip"
        else:
            fr = frame_buf.get_latest()
            if fr is None:
                time.sleep(0.001)
                continue
            input_src = f"frame_{sample_idx:06d}"

        # Sync before total timing if local CUDA
        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # --- CALCULATE TRANSPORT LATENCY (DATA AGE) ---
        frame_stamp_ns = frame_buf.get_latest_stamp_ns()
        current_ros_time_ns = node.get_clock().now().nanoseconds
        if frame_stamp_ns is not None:
            current_transport_ms = (current_ros_time_ns - frame_stamp_ns) / 1_000_000.0
        else:
            current_transport_ms = 0.0

        system_start_time = time.perf_counter()

        # --- PREPROCESS ---
        preprocessing_start_time = time.perf_counter()
        try:
            if IS_VIDEO_MODEL:
                input_tensor = adapter.preprocess_frames(clip)
            else:
                input_tensor = adapter.preprocess_frame(fr)
        except Exception as e:
            print(f"Preprocess failed: {e}")
            continue

        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        preprocessing_end_time = time.perf_counter()

        # --- INFERENCE ---
        inference_start_time = time.perf_counter()
        try:
            with torch.no_grad():
                raw_output = adapter.infer(input_tensor)
        except Exception as e:
            print(f"Infer failed: {e}")
            continue

        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        inference_end_time = time.perf_counter()

        # --- POSTPROCESS (ONLY model postprocess is timed) ---
        postprocessing_start_time = time.perf_counter()

        confidence_score = 0.0
        detected_info = ""
        class_id = -1

        # Prepared for publishing AFTER timing is stopped (not timed)
        to_publish_mask = None           # np.uint8 mono8 mask
        to_publish_overlay_bgr = None    # np.uint8 BGR overlay image
        to_publish_class_str = None      # "class,conf"

        try:
            result = adapter.postprocess(raw_output)

            # --- SEGMENTATION ---
            if isinstance(result, torch.Tensor) and result.ndim >= 2:
                mask_idx = result.numpy().astype(np.uint8)  # (H,W)
                detected_info = analyze_segmentation_mask(mask_idx)

                if PUBLISH_MASK and pub_mask is not None:
                    to_publish_mask = mask_idx

                if PUBLISH_OVERLAY and pub_overlay is not None:
                    base = frame_buf.get_latest()
                    if base is not None:
                        mask_colored = COLORS[mask_idx]  # RGB
                        overlay_rgb = cv2.addWeighted(base, 0.6, mask_colored, 0.4, 0)
                        to_publish_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            # --- CLASSIFICATION / VIDEO ---
            elif isinstance(result, tuple):
                class_id_tensor, prob_tensor = result
                class_id = int(class_id_tensor.item())
                confidence_score = float(prob_tensor.item()) * 100.0

                if hasattr(adapter, "categories") and adapter.categories:
                    class_name = adapter.categories[class_id]
                else:
                    class_name = f"Class {class_id}"

                detected_info = f" -> {class_name} ({confidence_score:.1f}%)"

                if PUBLISH_CLASS and pub_class is not None:
                    to_publish_class_str = f"{class_name},{confidence_score:.2f}"

        except Exception as e:
            print(f"Postprocess failed: {e}")

        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        postprocessing_end_time = time.perf_counter()

        # --- PUBLISH (NOT TIMED) ---
        try:
            if to_publish_class_str is not None:
                pub_class.publish(String(data=to_publish_class_str))

            if to_publish_mask is not None:
                mask_msg = bridge.cv2_to_imgmsg(to_publish_mask, encoding="mono8")
                pub_mask.publish(mask_msg)

            if to_publish_overlay_bgr is not None:
                overlay_msg = bridge.cv2_to_imgmsg(to_publish_overlay_bgr, encoding="bgr8")
                pub_overlay.publish(overlay_msg)

        except Exception as e:
            print(f"Publish failed: {e}")

        system_end_time = time.perf_counter()

        # Durations
        current_pre_ms = (preprocessing_end_time - preprocessing_start_time) * 1000.0
        current_inf_ms = (inference_end_time - inference_start_time) * 1000.0
        current_post_ms = (postprocessing_end_time - postprocessing_start_time) * 1000.0
        current_sys_ms = (system_end_time - system_start_time) * 1000.0

        transport_latencies_ms.append(current_transport_ms)
        preprocessing_latencies_ms.append(current_pre_ms)
        inference_latencies_ms.append(current_inf_ms)
        postprocessing_latencies_ms.append(current_post_ms)
        total_system_latencies_ms.append(current_sys_ms)

        if confidence_score > 0:
            confidence_scores_list.append(confidence_score)

        processed_samples += 1

        print(
            f"    [{sample_idx + 1}] {input_src} "
            f"| Age: {current_transport_ms:.1f}ms "
            f"| Pre: {current_pre_ms:.1f}ms "
            f"| Inf: {current_inf_ms:.1f}ms "
            f"| Post: {current_post_ms:.1f}ms "
            f"| Total: {current_sys_ms:.1f}ms"
            f"{detected_info}"
        )

        detailed_sample_records.append({
            "filename": input_src,
            "transport_ms": round(current_transport_ms, 4),
            "preprocessing_ms": round(current_pre_ms, 4),
            "inference_ms": round(current_inf_ms, 4),
            "postprocessing_ms": round(current_post_ms, 4),
            "e2e_ms": round(current_sys_ms, 4),
            "class_id": class_id,
            "confidence": round(confidence_score, 2),
            "info": str(detected_info)
        })

        sample_idx += 1

    # Stop time
    t_stop_dt = datetime.now(timezone.utc)
    t_stop_iso = t_stop_dt.isoformat()
    wall_duration_sec = max(1e-9, (time.time() - start_wall))

    if not inference_latencies_ms:
        print("❌ No successful inferences recorded.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Stats
    stats_transport = calculate_stats(transport_latencies_ms)
    stats_pre = calculate_stats(preprocessing_latencies_ms)
    stats_inf = calculate_stats(inference_latencies_ms)
    stats_post = calculate_stats(postprocessing_latencies_ms)
    stats_sys = calculate_stats(total_system_latencies_ms)
    stats_conf = calculate_stats(confidence_scores_list)

    avg_inf = stats_inf["mean"]
    avg_sys = stats_sys["mean"]
    inference_fps = (1000.0 / avg_inf) * frames_per_sample if avg_inf > 0 else 0
    system_fps = (1000.0 / avg_sys) * frames_per_sample if avg_sys > 0 else 0

    # Effective throughput + observed camera Hz
    processed_fps = float(processed_samples) / wall_duration_sec if wall_duration_sec > 0 else 0.0
    observed_camera_hz = float(np.mean(camera_hz_samples)) if camera_hz_samples else frame_buf.get_observed_hz()
    received_total = frame_buf.get_received_count()
    dropped_est = max(0, received_total - processed_samples) if AVOID_DUPLICATES else 0
    drop_ratio = (float(dropped_est) / float(received_total)) if received_total > 0 else 0.0

    print(f"\n📊 BENCHMARK SUMMARY ({MODEL_ARCH})")
    print(f"   ---------------------------------------------")
    print(f"   Avg Data Age (Transport): {stats_transport['mean']:.2f} ms")
    print(f"   Avg Preprocessing:  {stats_pre['mean']:.2f} ms")
    print(f"   Avg Inference:      {stats_inf['mean']:.2f} ms (P90: {stats_inf['p90']:.2f})")
    print(f"   Avg Postprocessing: {stats_post['mean']:.2f} ms")
    print(f"   Avg System E2E:     {stats_sys['mean']:.2f} ms")
    print(f"   ---------------------------------------------")
    print(f"   Inference FPS:      {inference_fps:.2f}")
    print(f"   System FPS:         {system_fps:.2f}")
    print(f"   ---------------------------------------------")
    print(f"   Camera Hz (obs):    {observed_camera_hz:.2f}")
    print(f"   Processed FPS:      {processed_fps:.2f}  (fresh frames)")
    if AVOID_DUPLICATES:
        print(f"   Frames received:    {received_total}")
        print(f"   Frames processed:   {processed_samples}")
        print(f"   Drop ratio (est):   {drop_ratio*100.0:.2f}%")
    else:
        print(f"   Note: AVOID_DUPLICATES=false → processed FPS may include repeated frames.")
    print(f"   ---------------------------------------------")

    # Export only if EXPORT_RESULTS=true
    if EXPORT_RESULTS:
        json_output_path = run_dir / "benchmark_summary.json"
        csv_output_path = run_dir / "benchmark_data.csv"

        final_output_data = {
            "run_id": run_id,
            "backend": BACKEND,
            "host": HOST,
            "model": MODEL_ARCH,
            "model_type": MODEL_TYPE,
            "device": TARGET_DEVICE,
            "num_samples": len(inference_latencies_ms),
            "time_window": {
                "start": t_start_iso,
                "stop": t_stop_iso,
                "duration_sec": (t_stop_dt - t_start_dt).total_seconds()
            },
            "frames_per_sample": frames_per_sample,
            "fps": {
                "inference": round(inference_fps, 2),
                "system": round(system_fps, 2),
                "processed": round(processed_fps, 2),
                "camera_observed": round(observed_camera_hz, 2)
            },
            "stream": { 
                "avoid_duplicates": bool(AVOID_DUPLICATES),
                "received_frames": int(received_total),
                "processed_frames": int(processed_samples),
                "drop_ratio": round(drop_ratio, 6),
                "duplicates_skipped_loops": int(duplicates_skipped)
            },
            "transport_ms": stats_transport,
            "preprocessing_ms": stats_pre,
            "inference_ms": stats_inf,
            "postprocessing_ms": stats_post,
            "system_ms": stats_sys,
            "confidence_score": stats_conf
        }

        with open(json_output_path, "w") as f:
            json.dump(final_output_data, f, indent=4)
        print(f"   ✅ JSON Summary saved to {json_output_path}")

        if detailed_sample_records:
            keys = detailed_sample_records[0].keys()
            with open(csv_output_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(detailed_sample_records)
            print(f"   ✅ CSV Data saved to {csv_output_path}")
    else:
        print("   💤 EXPORT_RESULTS=false → skipping JSON/CSV export.")

    print("✅ Done.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()