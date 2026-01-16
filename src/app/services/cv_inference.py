import asyncio
import logging
import time
import traceback
from pathlib import Path
from collections import deque

import cv2
import torch
import numpy as np
from fastapi import FastAPI

# Import your custom adapters
from model_adapter import get_model_adapter

# Import visualization helpers
try:
    from segmentation_utils import COLORS, VOC_CLASSES
except ImportError:
    COLORS = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    VOC_CLASSES = []

# Import shared plumbing
from app.services.image_inference import (
    get_session_data,
    reset_session_metrics,
    add_metric,
    get_metrics,
    capture_video,
    prepare_overlay,
)

logger = logging.getLogger("uvicorn.error")

# =========================================================
# 1. MODEL REGISTRY
# =========================================================
SUPPORTED_MODELS = [
    # --- Classification ---
    {"name": "resnet50",             "label": "resnet50"},
    {"name": "resnet50_sol",         "label": "resnet50_sol"},
    {"name": "mobilenet_v3_large",   "label": "mobilenet_v3_large"},
    {"name": "mobilenet_v3_large_sol","label": "mobilenet_v3_large_sol"},
    
    # --- Segmentation ---
    {"name": "deeplabv3_resnet50",   "label": "deeplabv3_resnet50"},
    {"name": "deeplabv3_resnet50_sol","label": "deeplabv3_resnet50_sol"},
    {"name": "fcn_resnet50",         "label": "fcn_resnet50"},
    {"name": "fcn_resnet50_sol",     "label": "fcn_resnet50_sol"},
    
    # --- Video Action ---
    {"name": "mc3_18",               "label": "mc3_18"},
    {"name": "mc3_18_sol",           "label": "mc3_18_sol"},
    {"name": "r3d_18",               "label": "r3d_18"},
    {"name": "r3d_18_sol",           "label": "r3d_18_sol"},
]

ALLOWED_MODEL_NAMES = {m["name"] for m in SUPPORTED_MODELS}

# =========================================================
# 2. SESSION & STATE MANAGEMENT
# =========================================================
def list_models():
    return SUPPORTED_MODELS

def reset_session_data(app: FastAPI, session_id: str):
    s = get_session_data(app, session_id)
    s.update({
        "model_name": "resnet50",   # Default
        "backend": "stock",         # 'stock' (local) or 'vaccel'
        "device": "cpu",            # 'cpu' or 'gpu' (mapped to cuda)
        "adapter": None,            # active adapter instance
        "frame_buffer": deque(maxlen=16),
        "stop_stream": False,
        "metrics_ready": asyncio.Event(),
        "metrics": {}
    })
    reset_session_metrics(app, session_id)
    return s

def set_model_choice(app: FastAPI, session_id: str, name: str):
    s = get_session_data(app, session_id)
    if name not in ALLOWED_MODEL_NAMES:
        name = "resnet50"
    
    if s.get("model_name") != name:
        logger.info(f"🔄 [Session {session_id[:8]}] Switching Model: {s.get('model_name')} -> {name}")
        s["model_name"] = name
        s["adapter"] = None 
        s["frame_buffer"].clear()

def set_inference_params(app: FastAPI, session_id: str, backend: str, device: str):
    s = get_session_data(app, session_id)
    
    if backend not in ["stock", "vaccel-local", "vaccel-rpc"]: backend = "stock"
    if device not in ["cpu", "gpu"]: device = "cpu"
    
    if s.get("backend") != backend or s.get("device") != device:
        logger.info(f"⚙️ [Session {session_id[:8]}] Updating Params: {s.get('backend')}/{s.get('device')} -> {backend}/{device}")
        s["backend"] = backend
        s["device"] = device
        s["adapter"] = None 
        s["frame_buffer"].clear()

# =========================================================
# 3. MODEL LOADING LOGIC
# =========================================================
def _ensure_model_loaded(app: FastAPI, session_id: str):
    s = get_session_data(app, session_id)
    
    if s.get("adapter") is not None:
        return

    model_choice = s.get("model_name", "resnet50")
    exec_backend = s.get("backend", "stock") 
    device_str = s.get("device", "cpu")
    
    if device_str == "gpu":
        if not "remote" in exec_backend and not torch.cuda.is_available():
            raise RuntimeError("GPU was selected but no GPU is available")

        adapter_device = "cuda"
    else:
        adapter_device = "cpu"

    logger.info(f"Loading: {model_choice} | Exec: {exec_backend} | Device: {adapter_device}")

    try:
        adapter = get_model_adapter(model_choice, exec_backend, adapter_device)
        model_dir = app.state.models_dir / model_choice
        adapter.load_model(str(model_dir))
        s["adapter"] = adapter

    except Exception as e:
        logger.error(f"Model Load Failed: {e}")
        traceback.print_exc()
        s["adapter"] = None

# =========================================================
# 4. VISUALIZATION HELPERS
# =========================================================
def _draw_result(img, result, model_type, adapter):
    # 1. Segmentation with Legend (Top-Right)
    if model_type == "segmentation":
        if isinstance(result, torch.Tensor):
            mask_idx = result.cpu().numpy()
        else:
            mask_idx = result
        
        # --- Draw the Overlay ---
        mask_colored = COLORS[mask_idx]
        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
        
        if mask_bgr.shape[:2] != img.shape[:2]:
             mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        blended = cv2.addWeighted(img, 0.6, mask_bgr, 0.4, 0)

        # --- Draw the Legend ---
        unique_classes = np.unique(mask_idx)
        h, w = blended.shape[:2]
        
        # START POSITION: Top-Right (170px from right edge)
        x_base = w - 170  
        y_offset = 30
        
        for cls_id in unique_classes:
            if cls_id == 0: continue # Skip background

            # Get Color
            color_rgb = COLORS[cls_id]
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            
            label = f"Class {cls_id}" # Default fallback
            if cls_id < len(VOC_CLASSES):
                label = VOC_CLASSES[cls_id].capitalize()
            elif hasattr(adapter, "categories") and adapter.categories and cls_id < len(adapter.categories):
                label = adapter.categories[cls_id]
            
            if len(label) > 12: label = label[:10] + ".."

            # Draw Color Swatch
            cv2.rectangle(blended, (x_base, y_offset - 15), (x_base + 20, y_offset + 5), color_bgr, -1)
            cv2.rectangle(blended, (x_base, y_offset - 15), (x_base + 20, y_offset + 5), (255, 255, 255), 1)
            
            # Draw Text
            cv2.putText(blended, label, (x_base + 25, y_offset + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(blended, label, (x_base + 25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 30

        return blended

    # 2. Classification / Video (Unchanged)
    elif model_type in ["classification", "video_classification"]:
        class_idx, prob = result
        if hasattr(adapter, "categories") and adapter.categories:
            label = adapter.categories[int(class_idx)]
        else:
            label = f"Class {int(class_idx)}"
            
        text = f"{label}: {prob.item()*100:.1f}%"
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h-40), (w, h), (0,0,0), -1)
        cv2.putText(img, text, (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return img
        
    return img

# =========================================================
# 5. INFERENCE LOOP
# =========================================================
async def generate_frame(app: FastAPI, session_id: str, file_path: Path):
    s = get_session_data(app, session_id)
    s["stop_stream"] = False
    
    cap = capture_video(file_path)
    header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    
    try:
        while not s["stop_stream"]:
            _ensure_model_loaded(app, session_id)
            adapter = s.get("adapter")
            
            ret, frame = cap.read()
            if not ret:
                if not isinstance(file_path, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # --- Force Resolution Cap (Max 640x480) ---
            h, w = frame.shape[:2]
            if w > 640 or h > 480:
                frame = cv2.resize(frame, (640, 480))
            # ------------------------------------------

            process_start = time.perf_counter()

            if adapter:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    model_type = getattr(adapter, "model_type", "classification")
                    
                    input_tensor = None
                    
                    if model_type == "video_classification":
                        # --- VIDEO LOGIC ---
                        t = torch.from_numpy(img_rgb).permute(2, 0, 1) 
                        if adapter.transform: 
                            t = adapter.transform(t)
                        
                        s["frame_buffer"].append(t)
                        
                        if len(s["frame_buffer"]) == 16:
                            input_tensor = torch.stack(list(s["frame_buffer"]), dim=1).unsqueeze(0)
                            if s["device"] == "gpu" and hasattr(input_tensor, "to"):
                                input_tensor = input_tensor.to(adapter.device)
                    else:
                        # --- IMAGE LOGIC ---
                        if adapter.transform:
                            input_tensor = adapter.transform(img_rgb).unsqueeze(0)
                        else:
                            input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        
                        if hasattr(input_tensor, "to"):
                            input_tensor = input_tensor.to(adapter.device)

                    if input_tensor is not None:
                        if s["device"] == "gpu" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                        # --- Inference Start ---
                        t0 = time.perf_counter()
                        
                        if hasattr(adapter, "model_name") and "sol" in s.get("model_name", ""): 
                            raw_out = adapter.infer(input_tensor.cpu().numpy())
                        else:
                            raw_out = adapter.infer(input_tensor)
                        
                        if s["device"] == "gpu" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                        # --- Inference End ---
                        inf_ms = (time.perf_counter() - t0) * 1000.0
                        add_metric(app, session_id, "inference_time", inf_ms)
                        
                        result = adapter.postprocess(raw_out)
                        frame = _draw_result(frame, result, model_type, adapter)

                except Exception as e:
                    traceback.print_exc() 
                    logger.error(f"Inference Error: {e}")
                    cv2.putText(frame, "Infer Error", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # --- FPS Calculation ---
            now = time.perf_counter()
            m = get_metrics(app, session_id)
            if m["last_frame_time"]:
                current_fps = 1.0 / max(now - m["last_frame_time"], 1e-6)
                m["fps_history"].append(current_fps)
                add_metric(app, session_id, "fps", sum(m["fps_history"]) / len(m["fps_history"]))
            
            add_metric(app, session_id, "last_frame_time", now)
            add_metric(app, session_id, "processing_time", (now - process_start)*1000)
            
            if s["metrics_ready"] and not s["metrics_ready"].is_set():
                s["metrics_ready"].set()

            frame = prepare_overlay(frame, m)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"".join([header, buf.tobytes(), b"\r\n"])
            
            await asyncio.sleep(0.001)

    finally:
        cap.release()
