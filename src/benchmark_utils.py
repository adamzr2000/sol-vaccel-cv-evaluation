import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from datetime import datetime, timezone
import requests

# ==========================================
# CONFIGURATION PARSER
# ==========================================
def parse_boolean_env(var_name, default="false"):
    return os.environ.get(var_name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def get_benchmark_config():
    """Parses environment variables and determines model/device paths."""
    # 1. Backend
    backend = os.environ.get("BACKEND", "stock").strip().lower()
    valid_backends = ["stock", "ptc", "sol", "vaccel-local-sol", "vaccel-remote-sol"]

    if backend not in valid_backends:
        print(f"⚠️  Unknown BACKEND '{backend}', defaulting to 'stock'")
        backend = "stock"

    # 2. Device Routing
    target_device = os.environ.get("DEVICE", "cpu").lower()
    if target_device == "gpu":
        device = "cuda"
        torch_device = torch.device("cpu") if "remote" in backend else torch.device("cuda")
    else:
        device = "cpu"
        torch_device = torch.device("cpu")

    # 3. Model & Host
    host = os.environ.get("HOST", "edge").lower()

    # User provides base name (e.g., 'resnet50'). Strip '_sol' just in case they typed it by habit.
    core_model_name = os.environ.get("MODEL", "resnet50").replace("_sol", "")
    
    # Automatically append '_sol' to the directory path if a SOL-based backend is selected
    if backend in ["sol", "vaccel-local-sol", "vaccel-remote-sol"]:
        model_arch = f"{core_model_name}_sol"
    else:
        model_arch = core_model_name

    # 4. Model Type Determination
    video_models = ["mc3_18", "r3d_18", "r2plus1d_18", "swin3d_t", "swin3d_s", "swin3d_b"]
    detection_models = ["yolov5s"]
    
    is_video_model = core_model_name in video_models
    
    if is_video_model:
        model_type = "video_classification"
    elif core_model_name in ["resnet50", "mobilenet_v3_large", "swin_t", "swin_s", "swin_v2_b"]:
        model_type = "image_classification"
    elif core_model_name in detection_models:
        model_type = "object_detection"
    else:
        model_type = "semantic_segmentation"

    return {
        "backend": backend,
        "target_device": target_device,
        "device": device,
        "torch_device": torch_device,
        "host": host,
        "model_arch": model_arch,           # e.g., 'yolov5s_sol' (Used to load the right folder
        "core_model_name": core_model_name, # e.g., 'yolov5s' (Used for internal logic)
        "model_type": model_type,
        "is_video_model": is_video_model,
        "num_images": int(os.environ.get("NUM_IMAGES", "64")),
        "num_videos": int(os.environ.get("NUM_VIDEOS", "10")),
        "export_results": parse_boolean_env("EXPORT_RESULTS", "false"),
        "export_output_images": parse_boolean_env("EXPORT_OUTPUT_IMAGES", "false"),
        "allow_fake_videos": parse_boolean_env("ALLOW_FAKE_VIDEOS", "0"),
        "base_results_dir": Path(os.environ.get("RESULTS_DIR", "/results/experiments/model-stats")),
        "run_tag": os.environ.get("RUN_TAG"),
        "monitor_resources": parse_boolean_env("RESOURCE_MONITORING", "false"),
        "docker_endpoint": os.environ.get("DOCKER_STATS_ENDPOINT", "http://10.5.1.20:6000"),
        "system_endpoint": os.environ.get("SYSTEM_STATS_ENDPOINT", "http://10.5.1.20:6001"),
        "docker_csv_base": os.environ.get("DOCKER_STATS_CSV_DIR", "/results/experiments/docker-stats"),
        "system_csv_base": os.environ.get("SYSTEM_STATS_CSV_DIR", "/results/experiments/system-stats"),
        "remote_docker_endpoint": os.environ.get("DOCKER_STATS_REMOTE_ENDPOINT", "http://10.5.1.20:6000"),
        "remote_system_endpoint": os.environ.get("SYSTEM_STATS_REMOTE_ENDPOINT", "http://10.5.1.20:6001")
    }

# ==========================================
# TIME & DURATION HELPERS
# ==========================================
def get_duration_ms(start_ns, end_ns):
    """Converts nanosecond start/end timestamps to a millisecond duration."""
    return (end_ns - start_ns) * 1e-6

def get_utc_timestamps():
    """Returns the current UTC datetime object and its ISO 8601 string representation."""
    dt = datetime.now(timezone.utc)
    return dt, dt.isoformat()

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
# RESULT PARSING & VISUALIZATION
# ==========================================
def process_segmentation(result, img_out_dir, stem_name, i, export_images, colors, analyze_fn):
    """Parses segmentation output and optionally saves a colorized mask."""
    mask_idx = result.detach().to("cpu").numpy()
    detected_info = analyze_fn(mask_idx)

    if export_images:
        mask_colored = colors[mask_idx]
        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_out_dir / f"{i:04d}_pred_{stem_name}.png"), mask_bgr)

    return -1, 0.0, detected_info

def process_classification(result, file_path, img_out_dir, stem_name, i, export_images, adapter, is_video):
    """Parses classification output and optionally saves a labeled image frame."""
    class_id_tensor, prob_tensor = result
    class_id = int(class_id_tensor.item())
    confidence_score = float(prob_tensor.item()) * 100

    if hasattr(adapter, 'categories') and adapter.categories:
        class_name = adapter.categories[class_id]
    else:
        class_name = f"Class {class_id}"

    detected_info = f" -> {class_name} ({confidence_score:.1f}%)"

    if export_images:
        display_img = None
        if is_video:
            cap = cv2.VideoCapture(str(file_path))
            ret, frame = cap.read()
            cap.release()
            if ret: display_img = frame
        else:
            display_img = cv2.imread(str(file_path))

        if display_img is not None:
            display_img = cv2.resize(display_img, (224, 224))
            label_text = f"{class_name} ({confidence_score:.1f}%)"
            cv2.rectangle(display_img, (5, 5), (250, 25), (0, 0, 0), -1)
            cv2.putText(display_img, label_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(img_out_dir / f"{i:04d}_pred_{stem_name}.jpg"), display_img)

    return class_id, confidence_score, detected_info

def process_detection(result, file_path, img_out_dir, stem_name, i, export_images, coco_classes):
    """Parses object detection output and optionally saves an image with bounding boxes."""
    boxes = result["boxes"]
    scores = result["scores"]
    classes = result["classes"]
    num_objects = len(boxes)

    if num_objects > 0:
        max_idx = torch.argmax(scores).item()
        confidence_score = float(scores[max_idx].item()) * 100
        class_id = int(classes[max_idx].item())
        top_class_name = coco_classes[class_id] if class_id < len(coco_classes) else str(class_id)
        detected_info = f" -> Detected {num_objects} object(s) (Top: {top_class_name} @ {confidence_score:.1f}%)"
    else:
        confidence_score = 0.0
        class_id = -1
        detected_info = " -> Detected 0 objects"

    if export_images:
        display_img = cv2.imread(str(file_path))
        if display_img is not None:
            display_img = cv2.resize(display_img, (640, 640))
            for idx in range(num_objects):
                x1, y1, x2, y2 = boxes[idx].int().tolist()
                conf = float(scores[idx].item()) * 100
                cls = int(classes[idx].item())
                class_name = coco_classes[cls] if cls < len(coco_classes) else str(cls)

                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"{class_name} {conf:.1f}%", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imwrite(str(img_out_dir / f"{i:04d}_pred_{stem_name}.jpg"), display_img)

    return class_id, confidence_score, detected_info

# ==========================================
# COCO DATASET CLASSES (For YOLOv5)
# ==========================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ==========================================
# 1. COLOR MAP (Standard VOC Colormap)
# ==========================================
COLORS = np.array([
    (0, 0, 0),       # Background
    (128, 0, 0),     # Aeroplane
    (0, 128, 0),     # Bicycle
    (128, 128, 0),   # Bird
    (0, 0, 128),     # Boat
    (128, 0, 128),   # Bottle
    (0, 128, 128),   # Bus
    (128, 128, 128), # Car
    (64, 0, 0),      # Cat
    (192, 0, 0),     # Chair
    (64, 128, 0),    # Cow
    (192, 128, 0),   # Dining Table
    (64, 0, 128),    # Dog
    (192, 0, 128),   # Horse
    (64, 128, 128),  # Motorbike
    (192, 128, 128), # Person
    (0, 64, 0),      # Potted Plant
    (128, 64, 0),    # Sheep
    (0, 192, 0),     # Sofa
    (128, 192, 0),   # Train
    (0, 64, 128)     # TV Monitor
], dtype=np.uint8)

# ==========================================
# 2. CLASS NAMES (Pascal VOC Index 0-20)
# ==========================================
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_position_label(cy, cx, h, w):
    """
    Returns a string like 'Top-Left', 'Center', 'Bottom-Right' based on centroids.
    """
    vert = "Center"
    if cy < h / 3: vert = "Top"
    elif cy > 2 * h / 3: vert = "Bottom"
    
    horz = "" 
    if cx < w / 3: horz = "-Left"
    elif cx > 2 * w / 3: horz = "-Right"
    
    if vert == "Center" and horz == "": return "Center"
    if vert == "Center": return horz.replace("-", "") 
    return f"{vert}{horz}"

def analyze_segmentation_mask(mask_idx):
    """
    Analyzes a segmentation mask (numpy array) and returns a formatted string 
    describing detected objects, their positions, and screen coverage %.
    """
    unique, counts = np.unique(mask_idx, return_counts=True)
    total_pixels = mask_idx.size
    H, W = mask_idx.shape
    
    detected_list = []
    
    for cls_id, count in zip(unique, counts):
        if cls_id == 0: continue # Skip background
        
        # Calculate Percentage
        percentage = (count / total_pixels) * 100
        
        # Heuristic: Ignore objects smaller than 1% of the screen (reduces noise)
        if percentage > 1.0: 
            # Get Name
            class_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else str(cls_id)
            
            # Get Position (Centroid)
            ys, xs = np.where(mask_idx == cls_id)
            cy, cx = np.mean(ys), np.mean(xs)
            pos_str = get_position_label(cy, cx, H, W)
            
            detected_list.append(f"{class_name} ({pos_str}): {int(percentage)}%")
            
    if detected_list:
        return " -> [" + ", ".join(detected_list) + "]"
    else:
        return " -> [background only]"


# ==========================================
# RESOURCE MONITORING HELPERS
# ==========================================

def start_docker_monitor(run_id, endpoint, csv_dir, container_ref, csv_prefix):
    url = f"{endpoint}/monitor/start"
    payload = {
        "containers": [container_ref],
        "interval": 1.0,
        "csv_dir": csv_dir,
        "stdout": False,
        "csv_names": { container_ref: f"{csv_prefix}{run_id}" }
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


def stabilize_torch_compile(
    adapter,
    sample_path: str,
    torch_device: torch.device,
    iters: int = 10,
    do_postprocess: bool = False,
):
    """
    Force torch.compile (Inductor) to finish compiling/caching before timing.
    Should be called ONLY for backend == 'ptc'.

    - Uses one representative sample_path (same input kind as benchmark loop).
    - Runs a few inference iterations and synchronizes so compile work completes.
    """
    print(f"   🧠 Stabilizing torch.compile graphs ({iters} iters)...")

    # Build one fixed-shape input (important for compile stability)
    x = adapter.preprocess(sample_path)

    # Run a few times to trigger compile + cache
    for _ in range(iters):
        with torch.inference_mode():
            y = adapter.infer(x)
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    # Optional: include postprocess once (only if you want it warmed too)
    if do_postprocess:
        _ = adapter.postprocess(y)

    print("   ✅ torch.compile stabilized.")