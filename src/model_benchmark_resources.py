import os
import time
import glob
import csv
import json
import torch
import numpy as np
import cv2
import requests
from pathlib import Path
from datetime import datetime, timezone

from model_adapter import get_model_adapter

try:
    from segmentation_utils import COLORS, analyze_segmentation_mask
except ImportError:
    COLORS = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    def analyze_segmentation_mask(mask): return ""

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
# CONFIGURATION
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
# UPDATED HOST LOGIC: Default to 'edge-asus', allow any string (flexible)
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "edge-asus").lower()

# Model: Full folder name
MODEL_ARCH = os.environ.get("MODEL", "resnet50")

# --- DEFAULTS ---
BENCH_NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "1024"))
BENCH_NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS", "64"))

EXPORT_RESULTS = os.environ.get("EXPORT_RESULTS", "true").strip().lower() in ("1", "true", "yes", "y", "on")
EXPORT_OUTPUT_IMAGES = os.environ.get("EXPORT_OUTPUT_IMAGES", "false").strip().lower() in ("1", "true", "yes", "y", "on")

DATA_DIRS = [Path("data/images"), Path("data/videos")]
MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# RESULTS PATHS
# ---------------------------------------------------------------------------
# 1. Model Stats Output
_BASE_RESULTS = Path(os.environ.get("RESULTS_DIR", "/results/experiments/model-stats"))
RESULTS_DIR = _BASE_RESULTS / HOST  # e.g. /results/experiments/model-stats/edge-asus/

# Directory is simply the model name
CURRENT_MODEL_DIR = MODELS_DIR / MODEL_ARCH

CORE_MODEL_NAME = MODEL_ARCH.replace("_sol", "")
VIDEO_MODELS = ["mc3_18", "r3d_18", "r2plus1d_18", "swin3d_t", "swin3d_s", "swin3d_b"]
IS_VIDEO_MODEL = CORE_MODEL_NAME in VIDEO_MODELS

if IS_VIDEO_MODEL:
    MODEL_TYPE = "video_classification"
elif CORE_MODEL_NAME in ["resnet50", "mobilenet_v3_large", "swin_t", "swin_s", "swin_v2_b"]:
    MODEL_TYPE = "image_classification"
else:
    MODEL_TYPE = "semantic_segmentation"

# --- MONITOR CONFIG ---
# 2. Docker & System Stats Output (ensure these are strings for the API)
DOCKER_STATS_ENDPOINT = os.environ.get("DOCKER_STATS_ENDPOINT", "http://10.5.1.20:6000")
_DOCKER_BASE = os.environ.get("DOCKER_STATS_CSV_DIR", "/results/experiments/docker-stats")
DOCKER_STATS_CSV_DIR = str(Path(_DOCKER_BASE) / HOST)

SYSTEM_STATS_ENDPOINT = os.environ.get("SYSTEM_STATS_ENDPOINT", "http://10.5.1.20:6001")
_SYSTEM_BASE = os.environ.get("SYSTEM_STATS_CSV_DIR", "/results/experiments/system-stats")
SYSTEM_STATS_CSV_DIR = str(Path(_SYSTEM_BASE) / HOST)

# --- REMOTE (vAccel agent) MONITOR CONFIG ---
DOCKER_STATS_REMOTE_ENDPOINT = os.environ.get("DOCKER_STATS_REMOTE_ENDPOINT", "http://10.5.1.20:6000")
DOCKER_STATS_REMOTE_CSV_DIR = os.environ.get("DOCKER_STATS_REMOTE_CSV_DIR", str(Path(_DOCKER_BASE) / "edge-asus"))

SYSTEM_STATS_REMOTE_ENDPOINT = os.environ.get("SYSTEM_STATS_REMOTE_ENDPOINT", "http://10.5.1.20:6001")
SYSTEM_STATS_REMOTE_CSV_DIR = os.environ.get("SYSTEM_STATS_REMOTE_CSV_DIR", str(Path(_SYSTEM_BASE) / "edge-asus"))

# ==========================================
# HELPER FUNCTIONS FOR MONITORS
# ==========================================
def start_docker_monitor(run_id, endpoint, csv_dir, container_ref, csv_prefix):
    url = f"{endpoint}/monitor/start"
    payload = {
        "containers": [container_ref],
        "interval": 1.0,
        "csv_dir": csv_dir,
        "stdout": False,
        "csv_names": {
            container_ref: f"{csv_prefix}{run_id}"
        }
    }
    try:
        print(f"   📡 Starting Docker Monitor: {url} ({container_ref})")
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print("   ✅ Docker Monitor Started.")
        else:
            print(f"   ⚠️ Docker Monitor Start Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not contact Docker Monitor: {e}")

def stop_docker_monitor(endpoint):
    url = f"{endpoint}/monitor/stop"
    try:
        print(f"   📡 Stopping Docker Monitor: {url}")
        resp = requests.post(url, timeout=5)
        if resp.status_code == 200:
            print("   ✅ Docker Monitor Stopped.")
        else:
            print(f"   ⚠️ Docker Monitor Stop Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not stop Docker Monitor: {e}")

def start_system_monitor(run_id, endpoint, csv_dir, mode):
    url = f"{endpoint}/monitor/start"
    payload = {
        "interval": 1.0,
        "csv_dir": csv_dir,
        "mode": mode,
        "stdout": False,
        "csv_names": { f"{mode}": f"{run_id}" }
    }
    try:
        print(f"   📡 Starting System Monitor: {url} (Mode: {mode})")
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print("   ✅ System Monitor Started.")
        else:
            print(f"   ⚠️ System Monitor Start Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not contact System Monitor: {e}")

def stop_system_monitor(endpoint):
    url = f"{endpoint}/monitor/stop"
    try:
        print(f"   📡 Stopping System Monitor: {url}")
        resp = requests.post(url, timeout=5)
        if resp.status_code == 200:
            print("   ✅ System Monitor Stopped.")
        else:
            print(f"   ⚠️ System Monitor Stop Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not stop System Monitor: {e}")

def main():
    print(f"\n🚀 STARTING MODEL BENCHMARK (RESOURCES)")
    print(f"   Backend: {BACKEND}")
    print(f"   Host:    {HOST}")
    print(f"   Model:   {MODEL_ARCH}")
    print(f"   Type:    {MODEL_TYPE}")
    print(f"   Device:  {TARGET_DEVICE}")
    print(f"   Loading: {CURRENT_MODEL_DIR}")

    if TORCH_DEVICE.type == "cuda" and not torch.cuda.is_available():
        print("   ❌ GPU was selected but no GPU is available.")
        return

    try:
        adapter = get_model_adapter(MODEL_ARCH, BACKEND, DEVICE)
        adapter.load_model(CURRENT_MODEL_DIR)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # 1. SCAN FILES
    image_files = []
    video_files = []
    for d in DATA_DIRS:
        if d.exists():
            image_files.extend(sorted(glob.glob(str(d / "*.jpg"))))
            video_files.extend(sorted(glob.glob(str(d / "*.mp4"))))

    # 2. SELECTION LOGIC
    files_to_process = []
    is_processing_video_files = False

    if IS_VIDEO_MODEL:
        if video_files:
            print(f"   🎥 Found {len(video_files)} video files.")
            limit = BENCH_NUM_VIDEOS
            files_to_process = video_files[:limit]
            is_processing_video_files = True
            print(f"   ✅ Selected {len(files_to_process)} videos for benchmarking (Limit: {limit}).")
        elif image_files:
            print(f"   ⚠️  No .mp4 videos found, but found {len(image_files)} images.")
            print(f"      Video models need temporal data. Simulating with static image stacking.")
            try:
                # Default to 'y' for automated batch runs if interactive input fails
                choice = input(f"      Do you want to use {BENCH_NUM_IMAGES} images as fake static videos? [y/N]: ").strip().lower()
            except EOFError:
                choice = 'y'

            if choice == 'y':
                files_to_process = image_files[:BENCH_NUM_IMAGES]
                print(f"   ✅ Using {len(files_to_process)} images as fake videos.")
            else:
                print("   ❌ Aborting benchmark.")
                return
        else:
            print("   ❌ No data found.")
            return
    else:
        files_to_process = image_files[:BENCH_NUM_IMAGES]
        if not files_to_process:
             print("   ❌ No images found.")
             return
        print(f"   📸 Using {len(files_to_process)} images.")


    # 3. OUTPUT ID (Run Tag Logic)
    # -------------------------------------------------------------------------
    run_tag = os.environ.get("RUN_TAG")
    if run_tag:
        prefix = run_tag
    else:
        prefix = time.strftime("%d-%m-%Y_%H-%M-%S")
        

    local_mode = "gpu" if (TORCH_DEVICE.type == "cuda") else "cpu"

    is_vaccel_remote_run = (HOST == "robot" and BACKEND == "vaccel-remote")
    if is_vaccel_remote_run:
        # local_mode may always be "cpu" here; add target device to avoid collisions
        run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_{HOST}_{local_mode}_target-{TARGET_DEVICE}"
    else:
        run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_{HOST}_{local_mode}"

    # -------------------------------------------------------------------------

    run_dir = RESULTS_DIR / run_id
    img_out_dir = run_dir / "output_images"

    if EXPORT_RESULTS:
        run_dir.mkdir(parents=True, exist_ok=True)
        if EXPORT_OUTPUT_IMAGES: img_out_dir.mkdir(exist_ok=True)
        print(f"   📂 Output Directory: {run_dir}")

    # 4. WARMUP
    print("   🔥 Warming up (30 iterations)...")
    for i in range(min(30, len(files_to_process))):
        try:
            dummy_tensor = adapter.preprocess(files_to_process[i])
            with torch.no_grad():
                _ = adapter.infer(dummy_tensor)
                if TORCH_DEVICE.type == 'cuda': torch.cuda.synchronize()
        except Exception as e:
            print(f"      Warmup failed on {os.path.basename(files_to_process[i])}: {e}")
            
    # --- START MONITORING (DOCKER + SYSTEM) ---
    start_docker_monitor(
        run_id=run_id,
        endpoint=DOCKER_STATS_ENDPOINT,
        csv_dir=DOCKER_STATS_CSV_DIR,
        container_ref="torchvision-app",
        csv_prefix="torchvision-app_",
    )
    start_system_monitor(
        run_id=run_id,
        endpoint=SYSTEM_STATS_ENDPOINT,
        csv_dir=SYSTEM_STATS_CSV_DIR,
        mode=local_mode
    )

    # Remote vAccel agent
    if is_vaccel_remote_run:
        vaccel_remote_run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_edge-asus_{TARGET_DEVICE}"

        start_docker_monitor(
            run_id=vaccel_remote_run_id,
            endpoint=DOCKER_STATS_REMOTE_ENDPOINT,
            csv_dir=DOCKER_STATS_REMOTE_CSV_DIR,
            container_ref="torchvision-app-agent",
            csv_prefix="torchvision-app-agent_",
        )
        start_system_monitor(
            run_id=vaccel_remote_run_id,
            endpoint=SYSTEM_STATS_REMOTE_ENDPOINT,
            csv_dir=SYSTEM_STATS_REMOTE_CSV_DIR,
            mode=TARGET_DEVICE
        )
    time.sleep(1.2)  # ~1 interval: lets monitors emit a first stable sample (Docker precpu primed, psutil/NVML warmed)
    
    # Capture Start Time (ISO format for alignment with Stats)
    t_start_dt = datetime.now(timezone.utc)
    t_start_iso = t_start_dt.isoformat()
    t_stop_iso = None

    # 5. RUN LOOP
    print("   ⏱️  Running Inference...")

    inference_latencies_ms = []
    preprocessing_latencies_ms = []
    postprocessing_latencies_ms = []
    total_system_latencies_ms = []
    confidence_scores_list = []
    detailed_sample_records = []

    frames_per_sample = 16 if IS_VIDEO_MODEL else 1

    try:
        for i, file_path in enumerate(files_to_process):
            file_name = os.path.basename(file_path)
            stem_name = os.path.splitext(file_name)[0]

            if TORCH_DEVICE.type == 'cuda': torch.cuda.synchronize()

            # --- A. SYSTEM START ---
            system_start_time = time.perf_counter()

            # --- B. PREPROCESSING ---
            preprocessing_start_time = time.perf_counter()
            try:
                input_tensor = adapter.preprocess(file_path)
            except Exception as e:
                print(f"      Skipping {file_name}: {e}")
                continue

            # B. Inference
            if TORCH_DEVICE.type == 'cuda': torch.cuda.synchronize()
            preprocessing_end_time = time.perf_counter()

            # --- C. INFERENCE ---
            inference_start_time = time.perf_counter()
            with torch.no_grad():
                raw_output = adapter.infer(input_tensor)

            if TORCH_DEVICE.type == 'cuda': torch.cuda.synchronize()
            inference_end_time = time.perf_counter()

            # --- D. POSTPROCESSING ---
            postprocessing_start_time = time.perf_counter()

            confidence_score = 0.0
            detected_info = ""
            class_id = -1

            try:
                result = adapter.postprocess(raw_output)

                if isinstance(result, torch.Tensor) and result.ndim >= 2:
                    mask_idx = result.numpy()
                    if EXPORT_OUTPUT_IMAGES:
                        mask_colored = COLORS[mask_idx]
                        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(img_out_dir / f"{i:04d}_pred_{stem_name}.png"), mask_bgr)
                    detected_info = analyze_segmentation_mask(mask_idx)

                elif isinstance(result, tuple):
                    class_id_tensor, prob_tensor = result
                    class_id = int(class_id_tensor.item())
                    confidence_score = float(prob_tensor.item()) * 100

                    if hasattr(adapter, 'categories') and adapter.categories:
                        class_name = adapter.categories[class_id]
                    else:
                        class_name = f"Class {class_id}"

                    detected_info = f" -> {class_name} ({confidence_score:.1f}%)"

                    if EXPORT_OUTPUT_IMAGES:
                        display_img = None
                        if is_processing_video_files:
                            cap = cv2.VideoCapture(str(file_path))
                            ret, frame = cap.read()
                            cap.release()
                            if ret: display_img = frame
                        else:
                            display_img = cv2.imread(file_path)

                        if display_img is not None:
                            display_img = cv2.resize(display_img, (224, 224))
                            label_text = f"{class_name} ({confidence_score:.1f}%)"
                            cv2.rectangle(display_img, (5, 5), (250, 25), (0, 0, 0), -1)
                            cv2.putText(display_img, label_text, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.imwrite(str(img_out_dir / f"{i:04d}_pred_{stem_name}.jpg"), display_img)

            except Exception as e:
                print(f"Error post-processing {file_name}: {e}")

            if TORCH_DEVICE.type == 'cuda': torch.cuda.synchronize()
            postprocessing_end_time = time.perf_counter()

            # --- TOTAL PROCESS END ---
            system_end_time = time.perf_counter()

            # --- CALCULATE RAW DURATIONS (Milliseconds) ---
            current_preprocessing_ms = (preprocessing_end_time - preprocessing_start_time) * 1000.0
            current_inference_ms = (inference_end_time - inference_start_time) * 1000.0
            current_postprocessing_ms = (postprocessing_end_time - postprocessing_start_time) * 1000.0
            current_system_ms = (system_end_time - system_start_time) * 1000.0

            # Store raw data
            preprocessing_latencies_ms.append(current_preprocessing_ms)
            inference_latencies_ms.append(current_inference_ms)
            postprocessing_latencies_ms.append(current_postprocessing_ms)
            total_system_latencies_ms.append(current_system_ms)

            if confidence_score > 0:
                confidence_scores_list.append(confidence_score)

            print(f"    [{i+1}/{len(files_to_process)}] {file_name} "
                  f"| Inf: {current_inference_ms:.1f}ms "
                  f"| Total: {current_system_ms:.1f}ms"
                  f" {detected_info}")
            
            # Store detailed record
            detailed_sample_records.append({
                "filename": file_name,
                "preprocessing_ms": round(current_preprocessing_ms, 4),
                "inference_ms": round(current_inference_ms, 4),
                "postprocessing_ms": round(current_postprocessing_ms, 4),
                "e2e_ms": round(current_system_ms, 4),
                "class_id": class_id,
                "confidence": round(confidence_score, 2),
                "info": str(detected_info)
            })      

    finally:
        # Capture Stop Time
        t_stop_dt = datetime.now(timezone.utc)
        t_stop_iso = t_stop_dt.isoformat()
        
        # STOP MONITORING (DOCKER + SYSTEM)
        if TORCH_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        time.sleep(0.2)  # small buffer to avoid stopping exactly on a sampling boundary; captures last tail activity without adding a full extra interval

        stop_docker_monitor(DOCKER_STATS_ENDPOINT)
        stop_system_monitor(SYSTEM_STATS_ENDPOINT)

        if is_vaccel_remote_run:
            stop_docker_monitor(DOCKER_STATS_REMOTE_ENDPOINT)
            stop_system_monitor(SYSTEM_STATS_REMOTE_ENDPOINT)

    # 6. SUMMARY & SAVE
    if not inference_latencies_ms:
        print("\n❌ No data collected.")
        return

    # --- Metrics Calculations ---
    stats_preprocessing = calculate_stats(preprocessing_latencies_ms)
    stats_inference = calculate_stats(inference_latencies_ms)
    stats_postprocessing = calculate_stats(postprocessing_latencies_ms)
    stats_system = calculate_stats(total_system_latencies_ms)
    stats_confidence = calculate_stats(confidence_scores_list)

    # --- FPS Calculations ---
    avg_inf = stats_inference["mean"]
    avg_sys = stats_system["mean"]
    inference_fps = (1000.0 / avg_inf) * frames_per_sample if avg_inf > 0 else 0
    system_fps = (1000.0 / avg_sys) * frames_per_sample if avg_sys > 0 else 0

    print(f"\n📊 BENCHMARK SUMMARY ({MODEL_ARCH})")
    print(f"   ---------------------------------------------")
    print(f"   Avg Preprocessing:  {stats_preprocessing['mean']:.2f} ms")
    print(f"   Avg Inference:      {stats_inference['mean']:.2f} ms (P90: {stats_inference['p90']:.2f})")
    print(f"   Avg Postprocessing: {stats_postprocessing['mean']:.2f} ms")
    print(f"   Avg System E2E:     {stats_system['mean']:.2f} ms")
    print(f"   ---------------------------------------------")
    print(f"   Inference FPS:      {inference_fps:.2f}")
    print(f"   System FPS:         {system_fps:.2f}")
    print(f"   ---------------------------------------------")

    if EXPORT_RESULTS:
        # 1. Export JSON Summary
        json_output_path = run_dir / "benchmark_summary.json"

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
                "duration_sec": (t_stop_dt - t_start_dt).total_seconds() if t_stop_iso else 0
            },

            "frames_per_sample": frames_per_sample,
            "fps": {
                "inference": round(inference_fps, 2),
                "system": round(system_fps, 2)
            },
            "preprocessing_ms": stats_preprocessing,
            "inference_ms": stats_inference,
            "postprocessing_ms": stats_postprocessing,
            "system_ms": stats_system,
            "confidence_score": stats_confidence
        }

        with open(json_output_path, 'w') as json_file:
            json.dump(final_output_data, json_file, indent=4)
        print(f"   ✅ JSON Summary saved to {json_output_path}")

        # 2. Export CSV Data
        csv_output_path = run_dir / "benchmark_data.csv"

        if detailed_sample_records:
            keys = detailed_sample_records[0].keys()
            with open(csv_output_path, 'w', newline='') as csv_file:
                dict_writer = csv.DictWriter(csv_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(detailed_sample_records)
            print(f"   ✅ CSV Data saved to {csv_output_path}")

if __name__ == "__main__":
    main()
