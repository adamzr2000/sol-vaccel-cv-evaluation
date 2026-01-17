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
from datetime import datetime, timezone  # <--- Added for ISO timestamps

from model_adapter import get_model_adapter

try:
    from segmentation_utils import COLORS, analyze_segmentation_mask
except ImportError:
    COLORS = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    def analyze_segmentation_mask(mask): return ""

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DEVICE = os.environ.get("DEVICE", "cpu").lower()
if INPUT_DEVICE == "gpu":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")

# Backend: 'stock' (default) or 'vaccel'
BACKEND = os.environ.get("BACKEND", "stock")

# Host: 'edge' (default) or 'robot'
HOST = os.environ.get("HOST", "edge").lower()
if HOST not in ["robot", "edge"]:
    print(f"⚠️  Unknown HOST '{HOST}', defaulting to 'edge'")
    HOST = "edge"

# Model: Full folder name
MODEL_ARCH = os.environ.get("MODEL", "resnet50")

# --- DEFAULTS ---
BENCH_NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "1024"))
BENCH_NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS", "64"))

EXPORT_RESULTS = os.environ.get("EXPORT_RESULTS", "true").strip().lower() in ("1", "true", "yes", "y", "on")
EXPORT_OUTPUT_IMAGES = os.environ.get("EXPORT_OUTPUT_IMAGES", "false").strip().lower() in ("1", "true", "yes", "y", "on")

DATA_DIRS = [Path("data/images"), Path("data/videos")]
MODELS_DIR = Path("models")

# Path: Saves to 'model-stats'
RESULTS_DIR = Path("/results/experiments/model-stats")

# Directory is simply the model name
CURRENT_MODEL_DIR = MODELS_DIR / MODEL_ARCH

CORE_MODEL_NAME = MODEL_ARCH.replace("_sol", "")
VIDEO_MODELS = ["mc3_18", "r3d_18"]
IS_VIDEO_MODEL = CORE_MODEL_NAME in VIDEO_MODELS

if IS_VIDEO_MODEL:
    MODEL_TYPE = "video_classification"
elif CORE_MODEL_NAME in ["resnet50", "mobilenet_v3_large", "swin_t"]:
    MODEL_TYPE = "image_classification"
else:
    MODEL_TYPE = "semantic_segmentation"

# --- MONITOR CONFIG ---
DOCKER_STATS_ENDPOINT = os.environ.get("DOCKER_STATS_ENDPOINT", "http://10.5.1.20:6000")
DOCKER_STATS_CSV_DIR = "/results/experiments/docker-stats"

SYSTEM_STATS_ENDPOINT = os.environ.get("SYSTEM_STATS_ENDPOINT", "http://10.5.1.20:6001")
SYSTEM_STATS_CSV_DIR = "/results/experiments/system-stats"


# ==========================================
# HELPER FUNCTIONS FOR MONITORS
# ==========================================
def start_docker_monitor(run_id):
    url = f"{DOCKER_STATS_ENDPOINT}/monitor/start"
    payload = {
        "containers": ["torchvision-app"],
        "interval": 1.0,
        "csv_dir": DOCKER_STATS_CSV_DIR,
        "stdout": False,
        "csv_names": {
            "torchvision-app": f"torchvision-app_{run_id}"
        }
    }
    try:
        print(f"   📡 Starting Docker Monitor: {url}")
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print("   ✅ Docker Monitor Started.")
        else:
            print(f"   ⚠️ Docker Monitor Start Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not contact Docker Monitor: {e}")

def stop_docker_monitor():
    url = f"{DOCKER_STATS_ENDPOINT}/monitor/stop"
    try:
        print(f"   📡 Stopping Docker Monitor...")
        resp = requests.post(url, timeout=5)
        if resp.status_code == 200:
            print("   ✅ Docker Monitor Stopped.")
        else:
            print(f"   ⚠️ Docker Monitor Stop Failed: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Could not stop Docker Monitor: {e}")

def start_system_monitor(run_id):
    url = f"{SYSTEM_STATS_ENDPOINT}/monitor/start"
    
    is_gpu = (DEVICE.type == 'cuda')
    mode = "gpu" if is_gpu else "cpu"

    payload = {
        "interval": 1.0,
        "csv_dir": SYSTEM_STATS_CSV_DIR,
        "mode": mode,
        "stdout": False,
        "csv_names": {
            f"{mode}": f"{run_id}",
        }
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

def stop_system_monitor():
    url = f"{SYSTEM_STATS_ENDPOINT}/monitor/stop"
    try:
        print(f"   📡 Stopping System Monitor...")
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
    print(f"   Device:  {DEVICE}")
    print(f"   Loading: {CURRENT_MODEL_DIR}")

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
        
    # Updated naming convention including HOST
    run_id = f"{prefix}_{MODEL_ARCH}_{BACKEND}_{HOST}_{INPUT_DEVICE}"
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
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
        except Exception as e:
            print(f"      Warmup failed on {os.path.basename(files_to_process[i])}: {e}")
            
    # --- START MONITORING (DOCKER + SYSTEM) ---
    start_docker_monitor(run_id)
    start_system_monitor(run_id)
    time.sleep(1.2)  # ~1 interval: lets monitors emit a first stable sample (Docker precpu primed, psutil/NVML warmed)
    
    # Capture Start Time (ISO format for alignment with Stats)
    t_start_dt = datetime.now(timezone.utc)
    t_start_iso = t_start_dt.isoformat()
    t_stop_iso = None

    # 5. RUN LOOP
    print("   ⏱️  Running Inference...")
    benchmark_records = []
    latencies = []       
    proc_latencies = [] 

    try:
        for f_path in files_to_process:
            file_name = os.path.basename(f_path)
            stem_name = os.path.splitext(file_name)[0]

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            proc_start = time.perf_counter()

            # A. Preprocessing
            try:
                input_tensor = adapter.preprocess(f_path)
            except Exception as e:
                print(f"      Skipping {file_name}: {e}")
                continue

            # B. Inference
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            inf_start = time.perf_counter()

            with torch.no_grad():
                raw_output = adapter.infer(input_tensor)

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            inf_end = time.perf_counter()

            # C. Postprocessing
            confidence_score = 0.0
            detected_info = ""

            try:
                result = adapter.postprocess(raw_output)

                if isinstance(result, torch.Tensor) and result.ndim >= 2:
                    mask_idx = result.numpy()
                    if EXPORT_OUTPUT_IMAGES:
                        mask_colored = COLORS[mask_idx]
                        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(img_out_dir / f"pred_{stem_name}.png"), mask_bgr)
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
                            cap = cv2.VideoCapture(str(f_path))
                            ret, frame = cap.read()
                            cap.release()
                            if ret: display_img = frame
                        else:
                            display_img = cv2.imread(f_path)

                        if display_img is not None:
                            display_img = cv2.resize(display_img, (224, 224))
                            label_text = f"{class_name} ({confidence_score:.1f}%)"
                            cv2.rectangle(display_img, (5, 5), (250, 25), (0, 0, 0), -1)
                            cv2.putText(display_img, label_text, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.imwrite(str(img_out_dir / f"pred_{stem_name}.jpg"), display_img)

            except Exception as e:
                print(f"Error post-processing {file_name}: {e}")

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            proc_end = time.perf_counter()

            latency_ms = (inf_end - inf_start) * 1000
            processing_ms = (proc_end - proc_start) * 1000

            benchmark_records.append({
                "image": file_name, 
                "latency_ms": round(latency_ms, 4),
                "processing_ms": round(processing_ms, 4),
                "confidence_score": round(confidence_score, 2)
            })
            
            latencies.append(latency_ms)
            proc_latencies.append(processing_ms)
            
            print(f"      - {file_name}: Inf={latency_ms:.2f}ms | Tot={processing_ms:.2f}ms {detected_info}")

    finally:
        # Capture Stop Time
        t_stop_dt = datetime.now(timezone.utc)
        t_stop_iso = t_stop_dt.isoformat()
        
        # STOP MONITORING (DOCKER + SYSTEM)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        time.sleep(0.2)  # small buffer to avoid stopping exactly on a sampling boundary; captures last tail activity without adding a full extra interval
        stop_docker_monitor()
        stop_system_monitor()

    # 6. SUMMARY & SAVE
    if latencies:
        def get_stats(data):
            if not data: return {}
            return {
                "mean": np.mean(data),
                "std":  np.std(data),
                "min":  np.min(data),
                "max":  np.max(data),
                "p25":  np.percentile(data, 25),
                "p50":  np.percentile(data, 50),
                "p75":  np.percentile(data, 75),
                "p90":  np.percentile(data, 90),
                "p95":  np.percentile(data, 95),
                "p99":  np.percentile(data, 99)
            }

        stats_inf  = get_stats(latencies)
        stats_proc = get_stats(proc_latencies)
        
        conf_values = [r['confidence_score'] for r in benchmark_records]
        stats_conf = get_stats(conf_values) if conf_values else get_stats([0.0])

        frames_per_sample = 16 if IS_VIDEO_MODEL else 1

        inference_fps = (1000.0 / stats_inf["mean"]) * frames_per_sample
        system_fps    = (1000.0 / stats_proc["mean"]) * frames_per_sample

        if EXPORT_RESULTS:
            with open(run_dir / "benchmark_data.csv", 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['image', 'latency_ms', 'processing_ms', 'confidence_score'])
                writer.writeheader()
                writer.writerows(benchmark_records)

            with open(run_dir / "benchmark_summary.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "backend": BACKEND,
                    "host": HOST,
                    "model": MODEL_ARCH,
                    "model_type": MODEL_TYPE,
                    "device": INPUT_DEVICE,
                    "num_samples": len(latencies),
                    
                    # --- Time Alignment ---
                    "time_window": {
                        "start": t_start_iso,
                        "stop": t_stop_iso,
                        "duration_sec": (t_stop_dt - t_start_dt).total_seconds() if t_stop_iso else 0
                    },
                    
                    "frames_per_sample": frames_per_sample,
                    "fps": {
                        "inference": inference_fps,
                        "system": system_fps
                    },

                    "inference_latency_ms": stats_inf,
                    "processing_latency_ms": stats_proc,
                    "confidence_score": stats_conf

                }, f, indent=2)

        print(f"\n✅ Benchmark Complete.")
        print(f"   📂 Results saved to: {run_dir}" if EXPORT_RESULTS else "   📂 Results export disabled.")
        print(f"   ----------------------------------")
        print(f"   📊 Summary for {BACKEND} / {MODEL_ARCH}:")

        print(f"      Inference Mean: {stats_inf['mean']:.2f} ms (P99: {stats_inf['p99']:.2f} ms)")
        print(f"      Inference FPS:  {inference_fps:.2f} (x{frames_per_sample} frames)" if frames_per_sample > 1 else f"      Inference FPS:  {inference_fps:.2f}")
        print(f"      System Mean:    {stats_proc['mean']:.2f} ms")
        print(f"      System FPS:     {system_fps:.2f}")
        
        if stats_conf['mean'] > 0:
            print(f"      Avg Conf:       {stats_conf['mean']:.1f}%")
            
        print(f"   ----------------------------------")
    else:
        print("\n❌ No data collected.")

if __name__ == "__main__":
    main()
