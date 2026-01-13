import os
import time
import glob
import csv
import json
import torch
import numpy as np
import cv2
from pathlib import Path

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

# Model: Full folder name, e.g., 'resnet50' or 'resnet50_sol'
MODEL_ARCH = os.environ.get("MODEL", "deeplabv3_resnet50")

# Separate limits for Images and Videos
BENCH_NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "64"))
BENCH_NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS", "10"))

EXPORT_RESULTS = os.environ.get("EXPORT_RESULTS", "false").strip().lower() in ("1", "true", "yes", "y", "on")
EXPORT_OUTPUT_IMAGES = os.environ.get("EXPORT_OUTPUT_IMAGES", "false").strip().lower() in ("1", "true", "yes", "y", "on")

DATA_DIRS = [Path("data/images"), Path("data/videos")]
MODELS_DIR = Path("models")
RESULTS_DIR = Path("/results")

# NEW: Directory is simply the model name
CURRENT_MODEL_DIR = MODELS_DIR / MODEL_ARCH

# Helper to check type (strip _sol suffix)
CORE_MODEL_NAME = MODEL_ARCH.replace("_sol", "")
VIDEO_MODELS = ["mc3_18", "r3d_18"]
IS_VIDEO_MODEL = CORE_MODEL_NAME in VIDEO_MODELS

# Determine Model Type String
if IS_VIDEO_MODEL:
    MODEL_TYPE = "video_classification"
elif CORE_MODEL_NAME in ["resnet50", "mobilenet_v3_large"]:
    MODEL_TYPE = "image_classification"
else:
    MODEL_TYPE = "semantic_segmentation"


def main():
    print(f"\n🚀 STARTING BENCHMARK")
    print(f"   Backend: {BACKEND}")
    print(f"   Model:   {MODEL_ARCH}")
    print(f"   Type:    {MODEL_TYPE}")
    print(f"   Device:  {DEVICE}")
    print(f"   Loading: {CURRENT_MODEL_DIR}")

    try:
        # Pass backend (stock/vaccel) and full model name
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

    # 2. INTELLIGENT SELECTION LOGIC
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
                choice = input(f"      Do you want to use {BENCH_NUM_IMAGES} images as fake static videos? [y/N]: ").strip().lower()
            except EOFError:
                choice = 'n'

            if choice == 'y':
                files_to_process = image_files[:BENCH_NUM_IMAGES]
                print(f"   ✅ Using {len(files_to_process)} images as fake videos.")
            else:
                print("   ❌ Aborting benchmark.")
                return
        else:
            print("   ❌ No data found (images or videos).")
            return
    else:
        files_to_process = image_files[:BENCH_NUM_IMAGES]
        if not files_to_process:
             print("   ❌ No images found.")
             return
        print(f"   📸 Using {len(files_to_process)} images.")


    # 3. PREPARE OUTPUT
    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    
    # ID Format: {time}_{model_variant}_{backend}_{device}
    run_id = f"{timestamp}_{MODEL_ARCH}_{BACKEND}_{INPUT_DEVICE}"
    
    run_dir = RESULTS_DIR / run_id
    img_out_dir = run_dir / "output_images"

    if EXPORT_RESULTS:
        run_dir.mkdir(parents=True, exist_ok=True)
        if EXPORT_OUTPUT_IMAGES: img_out_dir.mkdir(exist_ok=True)

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

    # 5. RUN LOOP
    print("   ⏱️  Running Inference...")
    latency_records = []
    latencies = []

    for f_path in files_to_process:
        file_name = os.path.basename(f_path)
        stem_name = os.path.splitext(file_name)[0]

        # A. Preprocessing
        try:
            input_tensor = adapter.preprocess(f_path)
        except Exception as e:
            print(f"      Skipping {file_name}: {e}")
            continue

        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            raw_output = adapter.infer(input_tensor)
        
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000

        detected_info = ""
        try:
            result = adapter.postprocess(raw_output)

            # --- SEGMENTATION ---
            if isinstance(result, torch.Tensor) and result.ndim >= 2:
                mask_idx = result.numpy()
                if EXPORT_OUTPUT_IMAGES:
                    mask_colored = COLORS[mask_idx]
                    mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(img_out_dir / f"pred_{stem_name}.png"), mask_bgr)
                
                detected_info = analyze_segmentation_mask(mask_idx)

            # --- CLASSIFICATION / VIDEO ---
            elif isinstance(result, tuple):
                class_id_tensor, prob_tensor = result
                class_id = int(class_id_tensor.item())
                confidence = float(prob_tensor.item()) * 100 
                
                if hasattr(adapter, 'categories') and adapter.categories:
                    class_name = adapter.categories[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                detected_info = f" -> {class_name} ({confidence:.1f}%)"

                # Visualization using OpenCV
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
                        label_text = f"{class_name} ({confidence:.1f}%)"
                        cv2.rectangle(display_img, (5, 5), (250, 25), (0, 0, 0), -1)
                        cv2.putText(display_img, label_text, (10, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.imwrite(str(img_out_dir / f"pred_{stem_name}.jpg"), display_img)

        except Exception as e:
            print(f"Error post-processing {file_name}: {e}")

        latency_records.append({"image": file_name, "latency_ms": round(latency_ms, 4)})
        latencies.append(latency_ms)
        print(f"      - {file_name}: {latency_ms:.2f} ms{detected_info}")

    # 6. SUMMARY & SAVE
    if latencies:
        mean_lat = np.mean(latencies)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        std_dev = np.std(latencies)
        throughput = 1000 / mean_lat

        if EXPORT_RESULTS:
            with open(run_dir / "latency_results.csv", 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['image', 'latency_ms'])
                writer.writeheader()
                writer.writerows(latency_records)
            
            with open(run_dir / "latency_results_summary.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "backend": BACKEND,
                    "model": MODEL_ARCH,
                    "model_type": MODEL_TYPE,
                    "device": INPUT_DEVICE,
                    "num_samples": len(latencies),
                    "latency_ms": {
                        "mean": mean_lat, 
                        "min": min_lat, 
                        "max": max_lat, 
                        "std": std_dev
                    },
                    "fps": throughput
                }, f, indent=2)

        print(f"\n✅ Benchmark Complete.")
        print(f"   📂 Results saved to: {run_dir}" if EXPORT_RESULTS else "   📂 Results export disabled.")
        print(f"   ----------------------------------")
        print(f"   📊 Summary for {BACKEND} / {MODEL_ARCH}:")
        
        if IS_VIDEO_MODEL:
            print(f"      Mean (Clip):  {mean_lat:.2f} ms")
            print(f"      Clips/Sec:    {throughput:.2f}")
            print(f"      Frames/Sec:   {throughput * 16:.2f}")
        else:
            print(f"      Mean:         {mean_lat:.2f} ms")
            print(f"      FPS:          {throughput:.2f}")

        print(f"      Min:          {min_lat:.2f} ms")
        print(f"      Max:          {max_lat:.2f} ms")
        print(f"      Std:          ±{std_dev:.2f} ms")
        print(f"   ----------------------------------")
    else:
        print("\n❌ No data collected.")

if __name__ == "__main__":
    main()