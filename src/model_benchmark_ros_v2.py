#!/usr/bin/env python3
# v2: reads the GO2 camera directly via GStreamer (same UDP/H264 multicast
# pipeline go2_camerapublisher uses) instead of subscribing to a ROS2 Image
# topic published by that separate process. See GstFrameBuffer below for why.
# Everything from frame capture onward (preprocess/infer/postprocess/stats/
# export) is UNCHANGED from model_benchmark_ros.py -- only how frames arrive
# differs, via the same RosFrameBuffer-shaped interface GstFrameBuffer exposes.
import os
import time
import csv
import json
import threading
import warnings
from collections import deque

import torch

# Cap interop thread pool to match OMP_THREAD_LIMIT before any parallel work.
# torch.set_num_interop_threads() must be called before the pool is first used;
# no env var is read by torch 2.8.0 for this — it must be set programmatically.
_thread_limit = os.environ.get("OMP_THREAD_LIMIT") or os.environ.get("OMP_NUM_THREADS")
if _thread_limit:
    torch.set_num_interop_threads(int(_thread_limit))

import numpy as np
import cv2
import logging

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import rclpy
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pathlib import Path

# --- REUSING LOGIC FROM benchmark_utils ---
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

# Honour OMP_NUM_THREADS for PyTorch's intra-op thread pool.
# torch.compile / AOTInductor use at::parallel_for (TBB/pthreads), not OpenMP,
# so OMP_NUM_THREADS alone does not control them — this call does.
_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _num_threads > 0:
    torch.set_num_threads(_num_threads)

# ==========================================
# CENTRALIZED CONFIGURATION (via utils)
# ==========================================
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

# ==========================================
# CAMERA CONFIGURATION — direct GStreamer, no ROS2 topic / go2_camerapublisher
# ==========================================
# Same multicast UDP/H264 source + decode pipeline go2_camerapublisher itself
# builds — read directly, in THIS process, instead of depending on that
# separate process to decode and republish over a ROS2 topic. go2_camerapublisher
# was only ever shared convenience infrastructure for the ros-vision-service
# comparison, never part of vaccel's own real architecture; running it as a
# separate process meant its own continuous H264-decode CPU cost silently
# competed with this script's preprocessing/inference loop for CPU time on the
# robot, inflating and adding variance to measured preprocessing cost. See
# GstFrameBuffer below.
GO2_LAN_PORT = os.environ.get("GO2_LAN_PORT", "enp2s0")
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "1280"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "720"))

# Opt-in (default off): hold the GStreamer streaming thread while the timed
# preprocessing section runs, so camera decode/convert never overlaps it --
# the same serialization go2_camerapublisher gets by design, since its
# preprocessing runs inside its own appsink callback. Isolates CPU contention
# from concurrent decode on the robot's 4 cores (see preproc_microbench.py).
SERIALIZE_CAPTURE = parse_boolean_env("SERIALIZE_CAPTURE", "false")

# Publish topics (you can disable any by setting it to empty string)
OUTPUT_MASK_TOPIC = os.environ.get("OUTPUT_MASK_TOPIC", "/benchmark/mask")
OUTPUT_OVERLAY_TOPIC = os.environ.get("OUTPUT_OVERLAY_TOPIC", "/benchmark/overlay")
OUTPUT_CLASS_TOPIC = os.environ.get("OUTPUT_CLASS_TOPIC", "/benchmark/classification")

# Controls to avoid inflating postprocessing numbers (Now using parse_boolean_env)
PUBLISH_MASK = parse_boolean_env("PUBLISH_MASK", "false")
PUBLISH_OVERLAY = parse_boolean_env("PUBLISH_OVERLAY", "false")
PUBLISH_CLASS = parse_boolean_env("PUBLISH_CLASS", "true")

# Run control
EXPERIMENT_DURATION_SEC = float(os.environ.get("EXPERIMENT_DURATION_SEC", "30"))
# 0 = disabled (default): EXPERIMENT_DURATION_SEC governs stop, unchanged.
# >0: takes priority over EXPERIMENT_DURATION_SEC — loop stops after this many
# processed samples instead, so runs can be compared at a fixed sample count
# rather than a fixed wall-clock window.
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "30"))
MAX_FPS = float(os.environ.get("MAX_FPS", "0"))  # 0 = no throttle

# Frame buffering
QUEUE_SIZE = int(os.environ.get("QUEUE_SIZE", "1"))  # 1=latest-only
VIDEO_BUFFER_LEN = int(os.environ.get("VIDEO_BUFFER_LEN", "16"))

# Duplicate control (default realistic)
AVOID_DUPLICATES = parse_boolean_env("AVOID_DUPLICATES", "true")

# Track ROS Data Age (Disable if using rosbags without use_sim_time)
TRACK_ROS_DATA_AGE = parse_boolean_env("TRACK_ROS_DATA_AGE", "false")


class GstFrameBuffer:
    """
    Reads the GO2 camera's raw H264/UDP multicast stream directly via
    GStreamer — the exact same pipeline go2_camerapublisher uses — instead of
    subscribing to a ROS2 Image topic published by that separate process.

    Exposes the identical interface RosFrameBuffer (model_benchmark_ros.py)
    had: get_latest() / get_clip() / get_received_count() /
    get_latest_stamp_ns() / get_observed_hz() / reset_counters(). Everything
    downstream in main() is unchanged — only how frames arrive differs.

    Why this exists: go2_camerapublisher running as a separate OS process
    meant its own continuous H264-decode CPU work silently competed with this
    script's own preprocessing/inference loop for CPU time on the robot,
    inflating and adding variance to measured preprocessing cost (confirmed:
    this script's own preprocessing_ms showed a long tail down to a minimum
    matching ros-vision-service's own go2_camerapublisher-inline figure almost
    exactly, consistent with occasional scheduling contention rather than a
    genuinely slower computation). Decoding in-process removes that confound.
    We do NOT time GStreamer waiting; the benchmark loop reads whatever is
    latest, same as before.
    """
    def __init__(self, node, is_video: bool, queue_size: int, video_len: int,
                 width: int, height: int, lan_port: str):
        # ROS node used only to timestamp frames (node.get_clock().now()),
        # matching go2_camerapublisher's own stamp semantics for a raw,
        # not-yet-preprocessed frame -- assigned right after capture, no
        # preprocessing delay baked in (this script does its own preprocessing
        # downstream, exactly as before). Set BEFORE set_state(PLAYING) below
        # -- new-sample can fire from a GStreamer-internal thread the instant
        # the pipeline starts, and _on_new_sample needs self._node ready by then.
        self._node = node

        self.is_video = is_video
        self.queue_size = max(1, queue_size)
        self.video_len = max(1, video_len)
        self.width = width
        self.height = height
        self.lock = threading.Lock()

        self.latest_rgb = None
        self.queue = deque(maxlen=self.queue_size)
        self.video_ring = deque(maxlen=self.video_len)

        self.total_received = 0
        self.latest_stamp_ns = None              # robot clock, nanoseconds
        self.stamp_history_ns = deque(maxlen=200)  # for observed Hz (cheap, not timed)

        # Held by _on_new_sample for its entire run, and by the main loop (via
        # hold_capture(), SERIALIZE_CAPTURE only) for the timed preprocessing
        # section -- acquiring it waits out any in-flight frame, then blocks new
        # ones. Uncontended (cheap no-op) when SERIALIZE_CAPTURE is off.
        self._capture_gate = threading.Lock()
        self._capture_held = False

        Gst.init(None)
        gstreamer_str = (
            f"udpsrc address=230.1.1.1 port=1720 multicast-iface={lan_port} "
            f"! application/x-rtp, media=video, encoding-name=H264 "
            f"! rtph264depay ! h264parse ! avdec_h264 ! videoconvert "
            f"! video/x-raw,width={self.width},height={self.height},format=BGR "
            f"! appsink drop=1"
        )
        self.pipeline = Gst.parse_launch(gstreamer_str)
        self.appsink = self.pipeline.get_by_name("appsink0")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_new_sample(self, sink):
        # Blocking on the gate (taken before pull-sample) stalls the whole
        # single-streaming-thread pipeline, decode included. The main loop
        # never holds self.lock while holding the gate, so no deadlock.
        with self._capture_gate:
            return self._handle_sample(sink)

    def _handle_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            bgr = np.frombuffer(map_info.data, dtype=np.uint8).reshape((self.height, self.width, 3))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stamp_ns = (
                self._node.get_clock().now().nanoseconds
                if self._node is not None else time.time_ns()
            )

            with self.lock:
                self.total_received += 1
                self.latest_rgb = rgb
                self.queue.append(rgb)
                if self.is_video:
                    self.video_ring.append(rgb)
                self.latest_stamp_ns = stamp_ns
                self.stamp_history_ns.append(stamp_ns)
        finally:
            buf.unmap(map_info)
        return Gst.FlowReturn.OK

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
        """Estimate camera rate from capture stamps. Returns 0.0 if not enough data."""
        with self.lock:
            stamps = list(self.stamp_history_ns)

        if len(stamps) < 2:
            return 0.0

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

    def reset_counters(self):
        with self.lock:
            self.total_received = 0

    def hold_capture(self):
        self._capture_gate.acquire()
        self._capture_held = True

    def release_capture(self):
        if self._capture_held:
            self._capture_held = False
            self._capture_gate.release()

    def stop(self):
        # Release first: a callback blocked on the gate would otherwise block
        # the streaming thread that set_state(NULL) has to join.
        self.release_capture()
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            print(f"[GstFrameBuffer] pipeline teardown failed (ignored): {e}")


def main():
    rclpy.init()
    node = rclpy.create_node("model_benchmark_ros2_v2")

    print(f"\n🚀 STARTING MODEL BENCHMARK (ROS2 v2 — direct GStreamer input)")
    print(f"   Backend:   {BACKEND}")
    print(f"   Host:      {HOST}")
    print(f"   Model:     {MODEL_ARCH}")
    print(f"   Type:      {MODEL_TYPE}")
    print(f"   Device:    {TARGET_DEVICE}")
    print(f"   Loading:   {CURRENT_MODEL_DIR}")
    print(f"   Input:     udp 230.1.1.1:1720 (iface={GO2_LAN_PORT}, {IMAGE_WIDTH}x{IMAGE_HEIGHT}) — no go2_camerapublisher")
    print(f"   Duration:  {EXPERIMENT_DURATION_SEC}s")
    print(f"   Samples:   {MAX_SAMPLES if MAX_SAMPLES > 0 else 'not set (duration governs stop)'}")
    print(f"   Warmup:    {WARMUP_ITERS} iters")
    print(f"   Publish:   class={PUBLISH_CLASS} mask={PUBLISH_MASK} overlay={PUBLISH_OVERLAY}")
    print(f"   Export:    EXPORT_RESULTS={EXPORT_RESULTS}")
    print(f"   Monitors:  RESOURCE_MONITORING={cfg['monitor_resources']}")
    print(f"   Control:   AVOID_DUPLICATES={AVOID_DUPLICATES} MAX_FPS={MAX_FPS} SERIALIZE_CAPTURE={SERIALIZE_CAPTURE}")
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

    local_mode = "gpu" if (TORCH_DEVICE.type == "cuda") else "cpu"
    is_remote_backend = "remote" in BACKEND
    REMOTE_HOST = os.environ.get("REMOTE_HOST", HOST if is_remote_backend else "")
    is_vaccel_remote_run = is_remote_backend and (REMOTE_HOST != HOST)  # dual-monitoring when agent is on a different host

    if is_remote_backend:
        # client=HOST (always cpu preprocessing), server=REMOTE_HOST running TARGET_DEVICE
        run_id = f"{prefix}_{CORE_MODEL_NAME}_{BACKEND}_{REMOTE_HOST}-{TARGET_DEVICE}"
    else:
        run_id = f"{prefix}_{CORE_MODEL_NAME}_{BACKEND}_{local_mode}"

    run_dir = RESULTS_DIR / run_id

    # Create output dir if exporting OR monitoring (monitors need a place to drop CSVs if local)
    if EXPORT_RESULTS or cfg["monitor_resources"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📂 Output Directory: {run_dir}")
    else:
        print(f"   📂 Output Directory: (disabled) would be {run_dir}")

    # Direct GStreamer input — no ROS2 subscription, no go2_camerapublisher dependency.
    frame_buf = GstFrameBuffer(
        node, is_video=IS_VIDEO_MODEL, queue_size=QUEUE_SIZE, video_len=VIDEO_BUFFER_LEN,
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT, lan_port=GO2_LAN_PORT,
    )

    # Start spinning so publisher-side ROS internals run (no subscription left to service).
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Publishers are optional; create only if enabled AND topic is not empty
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

    # Wait for first frame
    print("   ⏳ Waiting for first frame...")
    t_wait0 = time.time()
    while rclpy.ok():
        fr = frame_buf.get_latest()
        if fr is not None:
            break
        if time.time() - t_wait0 > 10:
            print("   ❌ No frames received within 10s. Check GO2_LAN_PORT / camera multicast stream.")
            # See teardown-order comment above (frame-wait timeout branch).
            frame_buf.stop()
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)
            node.destroy_node()
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
            with torch.inference_mode():
                _ = adapter.infer(inp)
                if TORCH_DEVICE.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup failed: {e}")

        warm_done += 1
        if MAX_FPS > 0:
            time.sleep(1.0 / MAX_FPS)

    # --- torch.compile stabilization (ptc only) ---
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

    # --- START RESOURCE MONITORING ---
    if cfg['monitor_resources']:
        docker_csv_dir = str(Path(cfg['docker_csv_base']) / cfg['host'])
        system_csv_dir = str(Path(cfg['system_csv_base']) / cfg['host'])

        # Ensure local CSV directories exist before contacting collectors
        Path(docker_csv_dir).mkdir(parents=True, exist_ok=True)
        Path(system_csv_dir).mkdir(parents=True, exist_ok=True)

        # Define network interface mapping
        host_net_iface_map = {
            "robot":          "ue0",
            "edge-asus":      "enp130s0",
            "edge-xtreme":    "br10",
            "edge-alienware": "enp8s0",
        }

        # Edge loopback remote: agent uses TARGET_DEVICE on the same host → monitor it
        # Robot remote: robot is CPU-only client; agent GPU is captured by is_vaccel_remote_run block
        # "cpu" always included (dict.fromkeys dedupes in case TARGET_DEVICE/local_mode
        # is itself "cpu") -- matches ros-vision-service, which monitors edge CPU
        # alongside GPU by default rather than GPU-only.
        if is_remote_backend and not is_vaccel_remote_run:
            local_sys_mode = list(dict.fromkeys(["cpu", TARGET_DEVICE, "net"]))
        else:
            local_sys_mode = list(dict.fromkeys([local_mode, "net"]))
        local_net_iface = host_net_iface_map.get(HOST)

        if cfg['monitor_docker']:
            start_docker_monitor(run_id, cfg['docker_endpoint'], docker_csv_dir, "torchvision-app", "torchvision-app_")
        start_system_monitor(run_id, cfg['system_endpoint'], system_csv_dir, local_sys_mode, local_net_iface)

        if is_vaccel_remote_run:
            vaccel_remote_run_id = run_id  # same ID; distinguished by host-named subfolder
            rem_docker_csv_dir = str(Path(cfg['docker_csv_base']) / REMOTE_HOST)
            rem_system_csv_dir = str(Path(cfg['system_csv_base']) / REMOTE_HOST)

            remote_sys_mode = list(dict.fromkeys(["cpu", cfg['target_device'], "net"]))
            remote_net_iface = host_net_iface_map.get(REMOTE_HOST)

            if cfg['monitor_docker']:
                start_docker_monitor(vaccel_remote_run_id, cfg['remote_docker_endpoint'], rem_docker_csv_dir, "torchvision-app-agent", "torchvision-app-agent_")
            start_system_monitor(vaccel_remote_run_id, cfg['remote_system_endpoint'], rem_system_csv_dir, remote_sys_mode, remote_net_iface)

        time.sleep(1.2)  # Prime the monitors

    # Capture Start Time (Using Utils)
    t_start_dt, t_start_iso = get_utc_timestamps()

    # Timing lists
    ros_data_age_latencies_ms = []   # Track Data Age
    preprocessing_latencies_ms = []
    inference_latencies_ms = []
    postprocessing_latencies_ms = []
    total_system_latencies_ms = []
    confidence_scores_list = []
    detailed_sample_records = []

    # Counters/rates (camera hz, processed fps, duplicates)
    processed_samples = 0
    duplicates_skipped = 0
    inference_failed = 0
    camera_hz_samples = []

    print("   ⏱️  Running Inference...")

    # RESET COUNTERS HERE so we don't count warmup/compile dropped frames!
    frame_buf.reset_counters()

    start_wall = time.time()
    sample_idx = 0
    last_tick = 0.0

    # Gate by capture stamp
    last_seen_stamp_ns = frame_buf.get_latest_stamp_ns()

    while rclpy.ok():
        now = time.time()
        if MAX_SAMPLES > 0:
            if processed_samples >= MAX_SAMPLES:
                break
        elif now - start_wall >= EXPERIMENT_DURATION_SEC:
            break

        # Optional throttle
        if MAX_FPS > 0:
            dt = now - last_tick
            if dt < (1.0 / MAX_FPS):
                time.sleep((1.0 / MAX_FPS) - dt)
            last_tick = time.time()

        # Avoid duplicates
        if AVOID_DUPLICATES:
            cur_stamp = frame_buf.get_latest_stamp_ns()
            if cur_stamp is None or cur_stamp == last_seen_stamp_ns:
                duplicates_skipped += 1
                time.sleep(0.001)
                continue
            last_seen_stamp_ns = cur_stamp

        # Track observed camera Hz
        hz = frame_buf.get_observed_hz()
        if hz > 0:
            camera_hz_samples.append(hz)

        # Fetch input
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

        # --- CALCULATE ROS DATA AGE ---
        if TRACK_ROS_DATA_AGE:
            frame_stamp_ns = frame_buf.get_latest_stamp_ns()
            current_ros_time_ns = node.get_clock().now().nanoseconds
            if frame_stamp_ns is not None:
                current_ros_data_age_ms = (current_ros_time_ns - frame_stamp_ns) / 1_000_000.0
            else:
                current_ros_data_age_ms = 0.0
        else:
            current_ros_data_age_ms = -1.0

        system_start_time_ns = time.perf_counter_ns()

        # --- PREPROCESS ---
        # Gate taken before the timer starts so waiting out an in-flight
        # frame isn't counted as preprocessing.
        if SERIALIZE_CAPTURE:
            frame_buf.hold_capture()
        try:
            preprocessing_start_time_ns = time.perf_counter_ns()
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
            preprocessing_end_time_ns = time.perf_counter_ns()
        finally:
            frame_buf.release_capture()

        # --- INFERENCE ---
        inference_start_time_ns = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                raw_output = adapter.infer(input_tensor)
        except Exception as e:
            print(f"Infer failed: {e}")
            inference_failed += 1
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

            # --- SEGMENTATION ---
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

            # --- OBJECT DETECTION ---
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

        system_end_time_ns = time.perf_counter_ns()

        # Durations matched exactly to model_benchmark.py
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

        sample_idx += 1

    # Capture Stop Time (Using Utils)
    t_stop_dt, t_stop_iso = get_utc_timestamps()
    wall_duration_sec = max(1e-9, (time.time() - start_wall))

    # --- STOP RESOURCE MONITORING ---
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
        # See teardown-order comment above (frame-wait timeout branch).
        frame_buf.stop()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        return

    # Stats
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

    # Effective throughput + observed camera Hz
    processed_fps = float(processed_samples) / wall_duration_sec if wall_duration_sec > 0 else 0.0
    observed_camera_hz = float(np.mean(camera_hz_samples)) if camera_hz_samples else frame_buf.get_observed_hz()
    received_total = frame_buf.get_received_count()
    dropped_est = max(0, received_total - processed_samples) if AVOID_DUPLICATES else 0
    drop_ratio = (float(dropped_est) / float(received_total)) if received_total > 0 else 0.0

    print(f"\n📊 BENCHMARK SUMMARY ({CORE_MODEL_NAME})")
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
    print(f"   Inference/RPC failed: {inference_failed}")
    print(f"   ---------------------------------------------")

    # Export only if EXPORT_RESULTS=true
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
                "duplicates_skipped_loops": int(duplicates_skipped),
                "inference_failed": int(inference_failed),
                "frame_source": "gstreamer_direct",
                "serialize_capture": bool(SERIALIZE_CAPTURE)
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

    # See teardown-order comment above (frame-wait timeout branch).
    frame_buf.stop()
    rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()


if __name__ == "__main__":
    main()
