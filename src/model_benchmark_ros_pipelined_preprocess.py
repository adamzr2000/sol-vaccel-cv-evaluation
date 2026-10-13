#!/usr/bin/env python3
"""
model_benchmark_ros_pipelined_preprocess.py (EXPERIMENTAL — not used by evaluate_models_ros.sh)

Full copy of model_benchmark_ros.py with one structural change: the serial
fetch -> preprocess -> infer -> postprocess -> publish loop is split into a
producer thread (dedup-gate + fetch + preprocess) and the main/consumer loop
(infer + postprocess + publish + bookkeeping, unchanged). Confirmed safe to
do (see src/experiments/test_sol_async_dispatch.py and the GIL investigation
in the repo history): vaccel's blocking torch_model_run()/exec_with_resource()
both release the GIL for the full RPC round trip, so the producer can
preprocess sample N+1 on its own thread while the consumer is still blocked
in infer() for sample N -- hiding preprocessing_ms behind the RTT wait for
vaccel-remote-* backends, without touching vaccel itself or model_adapter.py.

Every per-stage field (preprocessing_ms / inference_ms / postprocessing_ms)
keeps its EXACT original meaning, just measured across two threads instead
of inline. Per-sample system_ms/e2e will NOT shrink (that's expected --
pipelining buys throughput/processed_fps, not single-sample latency,
because preprocessing overlaps the *previous* sample's wait, not its own).
run_id is always prefixed "pipelined_preprocess_" regardless of RUN_TAG, so
results can never land in the same files as the serial baseline's -- or, for
that matter, a future "_pipelined" concurrent-sessions (Tier 2) experiment's,
which is a different, separate mechanism (overlapping multiple in-flight
inference calls) reserved for its own naming so the two never get confused.

Run exactly like model_benchmark_ros.py (same env vars), e.g.:
  BACKEND=vaccel-remote-sol MODEL=resnet50 DEVICE=cpu MAX_SAMPLES=50 \
    python3 /src/model_benchmark_ros_pipelined_preprocess.py
"""
import os
import time
import csv
import json
import queue
import threading
from collections import deque

import torch

_thread_limit = os.environ.get("OMP_THREAD_LIMIT") or os.environ.get("OMP_NUM_THREADS")
if _thread_limit:
    torch.set_num_interop_threads(int(_thread_limit))

import numpy as np
import cv2
import logging
import warnings

import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pathlib import Path

from benchmark_utils import (
    get_benchmark_config,
    get_duration_ms,
    calculate_stats,
    get_utc_timestamps,
    start_docker_monitor, stop_docker_monitor,
    start_system_monitor, stop_system_monitor,
    COLORS, analyze_segmentation_mask,
    COCO_CLASSES, stabilize_torch_compile,
    parse_boolean_env
)

from model_adapter import get_model_adapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _num_threads > 0:
    torch.set_num_threads(_num_threads)

cfg = get_benchmark_config()

BACKEND = cfg["backend"]
TARGET_DEVICE = cfg["target_device"]
DEVICE = cfg["device"]
TORCH_DEVICE = cfg["torch_device"]
HOST = cfg["host"]
MODEL_ARCH = cfg["model_arch"]
CORE_MODEL_NAME = cfg["core_model_name"]
MODEL_TYPE = cfg["model_type"]
IS_VIDEO_MODEL = cfg["is_video_model"]
EXPORT_RESULTS = cfg["export_results"]
RESULTS_DIR = cfg["base_results_dir"] / HOST

if "remote" in BACKEND:
    print(f"   🔎 VACCEL_RPC_ADDRESS={os.environ.get('VACCEL_RPC_ADDRESS')}")

MODELS_DIR = Path("models")
CURRENT_MODEL_DIR = MODELS_DIR / MODEL_ARCH

frames_per_sample = 16 if IS_VIDEO_MODEL else 1

INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "go2/sensor/camera")

OUTPUT_MASK_TOPIC = os.environ.get("OUTPUT_MASK_TOPIC", "/benchmark/mask")
OUTPUT_OVERLAY_TOPIC = os.environ.get("OUTPUT_OVERLAY_TOPIC", "/benchmark/overlay")
OUTPUT_CLASS_TOPIC = os.environ.get("OUTPUT_CLASS_TOPIC", "/benchmark/classification")

PUBLISH_MASK = parse_boolean_env("PUBLISH_MASK", "false")
PUBLISH_OVERLAY = parse_boolean_env("PUBLISH_OVERLAY", "false")
PUBLISH_CLASS = parse_boolean_env("PUBLISH_CLASS", "true")

EXPERIMENT_DURATION_SEC = float(os.environ.get("EXPERIMENT_DURATION_SEC", "30"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "30"))
MAX_FPS = float(os.environ.get("MAX_FPS", "0"))

QUEUE_SIZE = int(os.environ.get("QUEUE_SIZE", "1"))
VIDEO_BUFFER_LEN = int(os.environ.get("VIDEO_BUFFER_LEN", "16"))

AVOID_DUPLICATES = parse_boolean_env("AVOID_DUPLICATES", "true")
TRACK_ROS_DATA_AGE = parse_boolean_env("TRACK_ROS_DATA_AGE", "false")


class RosFrameBuffer:
    """Identical to model_benchmark_ros.py's — thread-safe already (used here
    from both the producer and the ROS spin thread, same as the serial
    script already used it from the spin thread + main thread)."""
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
        self.latest_stamp_ns = None
        self.stamp_history_ns = deque(maxlen=200)

    def callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
        with self.lock:
            stamps = list(self.stamp_history_ns)
        if len(stamps) < 2:
            return 0.0
        deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps)) if stamps[i] - stamps[i - 1] > 0]
        if not deltas:
            return 0.0
        mean_delta_ns = float(np.mean(np.array(deltas, dtype=np.float64)))
        if mean_delta_ns <= 0:
            return 0.0
        return 1e9 / mean_delta_ns

    def reset_counters(self):
        with self.lock:
            self.total_received = 0


def _pipeline_producer(frame_buf, adapter, out_queue, stop_event, node):
    """
    Owns everything the serial loop did BEFORE inference: throttle, dedup
    gate, fetch, ROS-data-age calc, camera-Hz tracking, preprocess. Puts one
    dict per successfully preprocessed sample on out_queue. Runs on its own
    thread so it can preprocess sample N+1 while the consumer is still
    blocked in adapter.infer() for sample N (vaccel's blocking RPC call
    releases the GIL for its whole duration -- confirmed from the compiled
    cffi bindings, see src/experiments/test_sol_async_dispatch.py).
    """
    last_seen_stamp_ns = frame_buf.get_latest_stamp_ns()
    last_tick = 0.0
    idx = 0
    pending_dup_skips = 0
    pending_hz_samples = []

    while not stop_event.is_set():
        try:
            now = time.time()

            if MAX_FPS > 0:
                dt = now - last_tick
                if dt < (1.0 / MAX_FPS):
                    time.sleep((1.0 / MAX_FPS) - dt)
                last_tick = time.time()

            if AVOID_DUPLICATES:
                cur_stamp = frame_buf.get_latest_stamp_ns()
                if cur_stamp is None or cur_stamp == last_seen_stamp_ns:
                    pending_dup_skips += 1
                    time.sleep(0.001)
                    continue
                last_seen_stamp_ns = cur_stamp

            hz = frame_buf.get_observed_hz()
            if hz > 0:
                pending_hz_samples.append(hz)

            if IS_VIDEO_MODEL:
                clip = frame_buf.get_clip()
                if clip is None:
                    time.sleep(0.001)
                    continue
                input_src = f"frame_{idx:06d}_clip"
            else:
                fr = frame_buf.get_latest()
                if fr is None:
                    time.sleep(0.001)
                    continue
                input_src = f"frame_{idx:06d}"

            # Same relative position as the serial script's CUDA-sync-before-timing
            # call (fetch -> sync -> ROS-age calc -> start timing). No-op for
            # vaccel-remote-* (client TORCH_DEVICE is always cpu).
            if TORCH_DEVICE.type == "cuda":
                torch.cuda.synchronize()

            if TRACK_ROS_DATA_AGE:
                frame_stamp_ns = frame_buf.get_latest_stamp_ns()
                current_ros_time_ns = node.get_clock().now().nanoseconds
                current_ros_data_age_ms = (
                    (current_ros_time_ns - frame_stamp_ns) / 1_000_000.0 if frame_stamp_ns is not None else 0.0
                )
            else:
                current_ros_data_age_ms = -1.0

            system_start_time_ns = time.perf_counter_ns()
            preprocessing_start_time_ns = time.perf_counter_ns()
            try:
                if IS_VIDEO_MODEL:
                    input_tensor = adapter.preprocess_frames(clip)
                else:
                    input_tensor = adapter.preprocess_frame(fr)
            except Exception as e:
                print(f"[producer] Preprocess failed: {e}")
                continue
            if TORCH_DEVICE.type == "cuda":
                torch.cuda.synchronize()
            preprocessing_end_time_ns = time.perf_counter_ns()

            item = {
                "input_tensor": input_tensor,
                "input_src": input_src,
                "sample_idx": idx,
                "current_ros_data_age_ms": current_ros_data_age_ms,
                "system_start_time_ns": system_start_time_ns,
                "preprocessing_start_time_ns": preprocessing_start_time_ns,
                "preprocessing_end_time_ns": preprocessing_end_time_ns,
                "duplicates_skipped_delta": pending_dup_skips,
                "camera_hz_samples_delta": pending_hz_samples,
            }
            pending_dup_skips = 0
            pending_hz_samples = []

            while not stop_event.is_set():
                try:
                    out_queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

            idx += 1
        except Exception as e:
            # Catch-all so a single transient error (e.g. a ROS/clock hiccup)
            # can never silently kill this daemon thread -- if it did, the
            # consumer would spin on queue.Empty forever with no internal
            # timeout (the EXPERIMENT_DURATION_SEC fallback is only checked
            # when MAX_SAMPLES is unset). Log and keep going instead.
            print(f"[producer] Unexpected error, continuing: {e}")
            time.sleep(0.01)


def main():
    rclpy.init()
    node = rclpy.create_node("model_benchmark_ros2_pipelined_preprocess")

    print(f"\n🚀 STARTING MODEL BENCHMARK (ROS2, PIPELINED-PREPROCESS/EXPERIMENTAL)")
    print(f"   Backend:   {BACKEND}")
    print(f"   Host:      {HOST}")
    print(f"   Model:     {MODEL_ARCH}")
    print(f"   Type:      {MODEL_TYPE}")
    print(f"   Device:    {TARGET_DEVICE}")
    print(f"   Loading:   {CURRENT_MODEL_DIR}")
    print(f"   Input:     {INPUT_TOPIC}")
    print(f"   Duration:  {EXPERIMENT_DURATION_SEC}s")
    print(f"   Samples:   {MAX_SAMPLES if MAX_SAMPLES > 0 else 'not set (duration governs stop)'}")
    print(f"   Warmup:    {WARMUP_ITERS} iters")
    print(f"   Publish:   class={PUBLISH_CLASS} mask={PUBLISH_MASK} overlay={PUBLISH_OVERLAY}")
    print(f"   Export:    EXPORT_RESULTS={EXPORT_RESULTS}")
    print(f"   Monitors:  RESOURCE_MONITORING={cfg['monitor_resources']}")
    print(f"   Control:   AVOID_DUPLICATES={AVOID_DUPLICATES} MAX_FPS={MAX_FPS}")
    print(f"   Data Age:  TRACK_ROS_DATA_AGE={TRACK_ROS_DATA_AGE}")

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

    run_tag = cfg["run_tag"]
    prefix = run_tag if run_tag else time.strftime("%d-%m-%Y_%H-%M-%S")
    # Always prefixed "pipelined_preprocess_" so this can never collide with
    # (or be mistaken for) a serial-baseline run's output, or a future
    # "_pipelined" Tier-2 concurrent-sessions experiment's output.
    prefix = f"pipelined_preprocess_{prefix}"

    local_mode = "gpu" if (TORCH_DEVICE.type == "cuda") else "cpu"
    is_remote_backend = "remote" in BACKEND
    REMOTE_HOST = os.environ.get("REMOTE_HOST", HOST if is_remote_backend else "")
    is_vaccel_remote_run = is_remote_backend and (REMOTE_HOST != HOST)

    if is_remote_backend:
        run_id = f"{prefix}_{CORE_MODEL_NAME}_{BACKEND}_{REMOTE_HOST}-{TARGET_DEVICE}"
    else:
        run_id = f"{prefix}_{CORE_MODEL_NAME}_{BACKEND}_{local_mode}"

    run_dir = RESULTS_DIR / run_id

    if EXPORT_RESULTS or cfg["monitor_resources"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📂 Output Directory: {run_dir}")
    else:
        print(f"   📂 Output Directory: (disabled) would be {run_dir}")

    frame_buf = RosFrameBuffer(is_video=IS_VIDEO_MODEL, queue_size=QUEUE_SIZE, video_len=VIDEO_BUFFER_LEN)

    _ = node.create_subscription(Image, INPUT_TOPIC, frame_buf.callback, qos_profile_sensor_data)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    pub_class = pub_mask = pub_overlay = None
    qos_out_class = QoSProfile(depth=10)
    qos_out_img = QoSProfile(depth=1)

    if PUBLISH_CLASS and OUTPUT_CLASS_TOPIC.strip():
        pub_class = node.create_publisher(String, OUTPUT_CLASS_TOPIC, qos_out_class)
    if PUBLISH_MASK and OUTPUT_MASK_TOPIC.strip():
        pub_mask = node.create_publisher(Image, OUTPUT_MASK_TOPIC, qos_out_img)
    if PUBLISH_OVERLAY and OUTPUT_OVERLAY_TOPIC.strip():
        pub_overlay = node.create_publisher(Image, OUTPUT_OVERLAY_TOPIC, qos_out_img)

    bridge = CvBridge()

    print("   ⏳ Waiting for first frame...")
    t_wait0 = time.time()
    while rclpy.ok():
        fr = frame_buf.get_latest()
        if fr is not None:
            break
        if time.time() - t_wait0 > 10:
            print("   ❌ No frames received within 10s. Check INPUT_TOPIC.")
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)
            node.destroy_node()
            return
        time.sleep(0.05)

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
            with torch.inference_mode():
                _ = adapter.infer(inp)
                if TORCH_DEVICE.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup failed: {e}")

        warm_done += 1
        if MAX_FPS > 0:
            time.sleep(1.0 / MAX_FPS)

    if BACKEND == "ptc":
        print(f"   🔥 Stabilizing PyTorch Compile...")
        dummy_input = frame_buf.get_clip() if IS_VIDEO_MODEL else frame_buf.get_latest()
        if dummy_input is not None:
            stabilize_torch_compile(
                adapter=adapter,
                sample_input=dummy_input,
                torch_device=TORCH_DEVICE,
                iters=int(os.environ.get("PTC_STABILIZE_ITERS", "20")),
                do_postprocess=False,
            )

    if cfg['monitor_resources']:
        docker_csv_dir = str(Path(cfg['docker_csv_base']) / cfg['host'])
        system_csv_dir = str(Path(cfg['system_csv_base']) / cfg['host'])
        Path(docker_csv_dir).mkdir(parents=True, exist_ok=True)
        Path(system_csv_dir).mkdir(parents=True, exist_ok=True)

        host_net_iface_map = {
            "robot":          "ue0",
            "edge-asus":      "enp130s0",
            "edge-xtreme":    "br10",
            "edge-alienware": "enp8s0",
        }

        if is_remote_backend and not is_vaccel_remote_run:
            local_sys_mode = [TARGET_DEVICE, "net"]
        else:
            local_sys_mode = [local_mode, "net"]
        local_net_iface = host_net_iface_map.get(HOST)

        if cfg['monitor_docker']:
            start_docker_monitor(run_id, cfg['docker_endpoint'], docker_csv_dir, "torchvision-app", "torchvision-app_")
        start_system_monitor(run_id, cfg['system_endpoint'], system_csv_dir, local_sys_mode, local_net_iface)

        if is_vaccel_remote_run:
            vaccel_remote_run_id = run_id
            rem_docker_csv_dir = str(Path(cfg['docker_csv_base']) / REMOTE_HOST)
            rem_system_csv_dir = str(Path(cfg['system_csv_base']) / REMOTE_HOST)
            remote_sys_mode = [cfg['target_device'], "net"]
            remote_net_iface = host_net_iface_map.get(REMOTE_HOST)

            if cfg['monitor_docker']:
                start_docker_monitor(vaccel_remote_run_id, cfg['remote_docker_endpoint'], rem_docker_csv_dir, "torchvision-app-agent", "torchvision-app-agent_")
            start_system_monitor(vaccel_remote_run_id, cfg['remote_system_endpoint'], rem_system_csv_dir, remote_sys_mode, remote_net_iface)

        time.sleep(1.2)

    t_start_dt, t_start_iso = get_utc_timestamps()

    ros_data_age_latencies_ms = []
    preprocessing_latencies_ms = []
    inference_latencies_ms = []
    postprocessing_latencies_ms = []
    total_system_latencies_ms = []
    confidence_scores_list = []
    detailed_sample_records = []

    processed_samples = 0
    duplicates_skipped = 0
    camera_hz_samples = []

    print("   ⏱️  Running Inference (pipelined-preprocess producer/consumer)...")

    frame_buf.reset_counters()

    start_wall = time.time()

    pipeline_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    producer_thread = threading.Thread(
        target=_pipeline_producer,
        args=(frame_buf, adapter, pipeline_queue, stop_event, node),
        daemon=True,
    )
    producer_thread.start()

    while rclpy.ok():
        if MAX_SAMPLES > 0:
            if processed_samples >= MAX_SAMPLES:
                stop_event.set()
                break
        elif time.time() - start_wall >= EXPERIMENT_DURATION_SEC:
            stop_event.set()
            break

        try:
            item = pipeline_queue.get(timeout=0.5)
        except queue.Empty:
            if not producer_thread.is_alive():
                print("   ❌ Producer thread died unexpectedly and the queue is drained. Aborting run.")
                stop_event.set()
                break
            continue

        duplicates_skipped += item["duplicates_skipped_delta"]
        camera_hz_samples.extend(item["camera_hz_samples_delta"])

        input_tensor = item["input_tensor"]
        input_src = item["input_src"]
        sample_idx = item["sample_idx"]
        current_ros_data_age_ms = item["current_ros_data_age_ms"]
        system_start_time_ns = item["system_start_time_ns"]
        preprocessing_start_time_ns = item["preprocessing_start_time_ns"]
        preprocessing_end_time_ns = item["preprocessing_end_time_ns"]

        # --- INFERENCE ---
        inference_start_time_ns = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                raw_output = adapter.infer(input_tensor)
        except Exception as e:
            print(f"Infer failed: {e}")
            continue
        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        inference_end_time_ns = time.perf_counter_ns()

        # --- POSTPROCESS ---
        postprocessing_start_time_ns = time.perf_counter_ns()

        confidence_score = 0.0
        detected_info = ""
        class_id = -1

        to_publish_mask = None
        to_publish_overlay_bgr = None
        to_publish_class_str = None

        try:
            result = adapter.postprocess(raw_output)

            if isinstance(result, torch.Tensor) and result.ndim >= 2:
                mask_idx = result.numpy().astype(np.uint8)
                detected_info = analyze_segmentation_mask(mask_idx)

                if PUBLISH_MASK and pub_mask is not None:
                    to_publish_mask = mask_idx

                if PUBLISH_OVERLAY and pub_overlay is not None:
                    base = frame_buf.get_latest()
                    if base is not None:
                        mask_colored = COLORS[mask_idx]
                        overlay_rgb = cv2.addWeighted(base, 0.6, mask_colored, 0.4, 0)
                        to_publish_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

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

            elif isinstance(result, dict):
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                classes = result.get("classes", [])
                num_objects = len(boxes)

                if num_objects > 0:
                    max_idx = torch.argmax(scores).item()
                    confidence_score = float(scores[max_idx].item()) * 100.0
                    class_id = int(classes[max_idx].item())

                    top_class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Obj_{class_id}"
                    detected_info = f" -> Detected {num_objects} obj (Top: {top_class_name} {confidence_score:.1f}%)"

                    if PUBLISH_CLASS and pub_class is not None:
                        to_publish_class_str = f"{top_class_name},{confidence_score:.2f}"
                else:
                    detected_info = " -> Detected 0 objects"

        except Exception as e:
            print(f"Postprocess failed: {e}")

        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        postprocessing_end_time_ns = time.perf_counter_ns()

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

        system_end_time_ns = time.perf_counter_ns()

        current_pre_ms = get_duration_ms(preprocessing_start_time_ns, preprocessing_end_time_ns)
        current_inf_ms = get_duration_ms(inference_start_time_ns, inference_end_time_ns)
        current_post_ms = get_duration_ms(postprocessing_start_time_ns, postprocessing_end_time_ns)
        current_sys_ms = get_duration_ms(system_start_time_ns, system_end_time_ns)

        ros_data_age_latencies_ms.append(current_ros_data_age_ms)
        preprocessing_latencies_ms.append(current_pre_ms)
        inference_latencies_ms.append(current_inf_ms)
        postprocessing_latencies_ms.append(current_post_ms)
        total_system_latencies_ms.append(current_sys_ms)

        if confidence_score > 0:
            confidence_scores_list.append(confidence_score)

        processed_samples += 1

        print(
            f"   [{sample_idx + 1}] {input_src} "
            f"| ROS Age: {current_ros_data_age_ms:.1f}ms "
            f"| Pre: {current_pre_ms:.1f}ms "
            f"| Inf: {current_inf_ms:.1f}ms "
            f"| Post: {current_post_ms:.1f}ms "
            f"| Total: {current_sys_ms:.1f}ms"
            f"{detected_info}"
        )

        detailed_sample_records.append({
            "filename": input_src,
            "ros_data_age_ms": round(current_ros_data_age_ms, 4),
            "preprocessing_ms": round(current_pre_ms, 4),
            "inference_ms": round(current_inf_ms, 4),
            "postprocessing_ms": round(current_post_ms, 4),
            "e2e_ms": round(current_sys_ms, 4),
            "class_id": class_id,
            "confidence": round(confidence_score, 2),
            "info": str(detected_info)
        })

    stop_event.set()
    producer_thread.join(timeout=2.0)

    t_stop_dt, t_stop_iso = get_utc_timestamps()
    wall_duration_sec = max(1e-9, (time.time() - start_wall))

    if cfg["monitor_resources"]:
        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        time.sleep(0.2)

        if cfg['monitor_docker']:
            stop_docker_monitor(cfg['docker_endpoint'])
        stop_system_monitor(cfg['system_endpoint'])

        if is_vaccel_remote_run:
            if cfg['monitor_docker']:
                stop_docker_monitor(cfg['remote_docker_endpoint'])
            stop_system_monitor(cfg['remote_system_endpoint'])

    if not inference_latencies_ms:
        print("❌ No successful inferences recorded.")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        return

    stats_ros_data_age = calculate_stats(ros_data_age_latencies_ms)
    stats_pre = calculate_stats(preprocessing_latencies_ms)
    stats_inf = calculate_stats(inference_latencies_ms)
    stats_post = calculate_stats(postprocessing_latencies_ms)
    stats_sys = calculate_stats(total_system_latencies_ms)
    stats_conf = calculate_stats(confidence_scores_list)

    avg_inf = stats_inf["mean"]
    avg_sys = stats_sys["mean"]

    inference_fps = (1000.0 / avg_inf) if avg_inf > 0 else 0
    system_fps = (1000.0 / avg_sys) if avg_sys > 0 else 0

    system_fps_list = [(1000.0 / t) for t in total_system_latencies_ms if t > 0]
    inference_fps_list = [(1000.0 / t) for t in inference_latencies_ms if t > 0]

    sys_fps_std = float(np.std(system_fps_list)) if system_fps_list else 0.0
    inf_fps_std = float(np.std(inference_fps_list)) if inference_fps_list else 0.0

    processed_fps = float(processed_samples) / wall_duration_sec if wall_duration_sec > 0 else 0.0
    observed_camera_hz = float(np.mean(camera_hz_samples)) if camera_hz_samples else frame_buf.get_observed_hz()
    received_total = frame_buf.get_received_count()
    dropped_est = max(0, received_total - processed_samples) if AVOID_DUPLICATES else 0
    drop_ratio = (float(dropped_est) / float(received_total)) if received_total > 0 else 0.0

    print(f"\n📊 BENCHMARK SUMMARY ({CORE_MODEL_NAME}, PIPELINED-PREPROCESS)")
    print(f"   ---------------------------------------------")
    if TRACK_ROS_DATA_AGE:
        print(f"   Avg ROS Data Age:   {stats_ros_data_age['mean']:.2f} ms")
    else:
        print(f"   Avg ROS Data Age:   -1 (Disabled)")
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

    if EXPORT_RESULTS:
        json_output_path = run_dir / "benchmark_summary.json"
        csv_output_path = run_dir / "benchmark_data.csv"

        final_output_data = {
            "run_id": run_id,
            "backend": BACKEND,
            "host": HOST,
            "model": CORE_MODEL_NAME,
            "model_type": MODEL_TYPE,
            "device": TARGET_DEVICE,
            "pipeline_mode": "producer_consumer_overlap",
            "num_samples": len(inference_latencies_ms),
            "time_window": {
                "start": t_start_iso,
                "stop": t_stop_iso,
                "duration_sec": (t_stop_dt - t_start_dt).total_seconds()
            },
            "frames_per_sample": frames_per_sample,
            "fps": {
                "inference": round(inference_fps, 2),
                "inference_std": round(inf_fps_std, 2),
                "system": round(system_fps, 2),
                "system_std": round(sys_fps_std, 2),
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
            "ros_data_age_ms": stats_ros_data_age,
            "preprocessing_ms": stats_pre,
            "inference_ms": stats_inf,
            "postprocessing_ms": stats_post,
            "system_ms": stats_sys,
            "confidence_score": stats_conf
        }
        if is_remote_backend:
            final_output_data["remote_host"] = REMOTE_HOST
            final_output_data["local_device"] = local_mode
        if MAX_SAMPLES > 0:
            final_output_data["max_samples"] = MAX_SAMPLES
        if "sol" in BACKEND:
            final_output_data["sol_run_mode"] = int(os.environ.get("SOL_RUN_MODE", "2").strip())

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

    rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()


if __name__ == "__main__":
    main()
