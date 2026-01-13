import asyncio
import logging
import time
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
    from segmentation_utils import COLORS
except ImportError:
    COLORS = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

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
# 1. MODEL REGISTRY (Flatted Structure)
# =========================================================
SUPPORTED_MODELS = [
    # --- Classification ---
    {"name": "resnet50",             "label": "ResNet50 (Baseline)"},
    {"name": "resnet50_sol",         "label": "ResNet50 (SOL)"},
    {"name": "mobilenet_v3_large",   "label": "MobileNetV3 (Baseline)"},
    {"name": "mobilenet_v3_large_sol","label": "MobileNetV3 (SOL)"},
    
    # --- Segmentation ---
    {"name": "deeplabv3_resnet50",   "label": "DeepLabV3 (Baseline)"},
    {"name": "deeplabv3_resnet50_sol","label": "DeepLabV3 (SOL)"},
    {"name": "fcn_resnet50",         "label": "FCN ResNet50 (Baseline)"},
    
    # --- Video Action ---
    {"name": "mc3_18",               "label": "MC3_18 (Baseline)"},
    {"name": "mc3_18_sol",           "label": "MC3_18 (SOL)"},
    {"name": "r3d_18",               "label": "R3D_18 (Baseline)"},
    {"name": "r3d_18_sol",           "label": "R3D_18 (SOL)"},
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
    
    # Invalidate if changed
    if s.get("model_name") != name:
        s["model_name"] = name
        s["adapter"] = None 
        s["frame_buffer"].clear()

def set_inference_params(app: FastAPI, session_id: str, backend: str, device: str):
    s = get_session_data(app, session_id)
    
    # Validations
    if backend not in ["stock", "vaccel"]: backend = "stock"
    if device not in ["cpu", "gpu"]: device = "cpu"
    
    # Invalidate if changed
    if s.get("backend") != backend or s.get("device") != device:
        s["backend"] = backend
        s["device"] = device
        s["adapter"] = None 
        # Note: We don't necessarily need to clear frame buffer on device change, 
        # but it's safer to avoid tensor device mismatches.
        s["frame_buffer"].clear()

# =========================================================
# 3. MODEL LOADING LOGIC
# =========================================================
def _parse_model_config(model_choice: str):
    """
    Splits 'resnet50_sol' -> ('resnet50', 'sol')
    Splits 'resnet50'     -> ('resnet50', 'baseline')
    """
    if model_choice.endswith("_sol"):
        return model_choice.replace("_sol", ""), "sol"
    return model_choice, "baseline"

def _ensure_model_loaded(app: FastAPI, session_id: str):
    s = get_session_data(app, session_id)
    
    if s.get("adapter") is not None:
        return

    model_choice = s.get("model_name", "resnet50")
    exec_backend = s.get("backend", "stock") # stock vs vaccel
    device_str = s.get("device", "cpu")
    
    # Map UI device to PyTorch device
    if device_str == "gpu" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
        if device_str == "gpu":
            logger.warning("GPU requested but CUDA not available. Fallback to CPU.")

    logger.info(f"Loading: {model_choice} | Exec: {exec_backend} | Device: {torch_device}")

    try:
        # --- PATH A: STOCK (Local Adapter) ---
        if exec_backend == "stock":
            # 1. Parse architecture and flavor from name
            arch_name, flavor = _parse_model_config(model_choice)
            
            # 2. Instantiate Adapter
            adapter = get_model_adapter(arch_name, flavor, torch_device)
            
            # 3. Load Weights
            # Directory convention: {arch}_{flavor} (e.g. resnet50_sol)
            model_dir = app.state.models_dir / f"{arch_name}_{flavor}"
            adapter.load_model(str(model_dir))
            
            s["adapter"] = adapter

        # --- PATH B: VACCEL (Future Integration) ---
        elif exec_backend == "vaccel":
            # Future: Initialize vAccel resource here
            # s["vaccel_sess"] = ...
            logger.info("vAccel backend selected (Placeholder)")
            pass

    except Exception as e:
        logger.error(f"Model Load Failed: {e}")
        s["adapter"] = None

# =========================================================
# 4. VISUALIZATION HELPERS
# =========================================================
def _draw_result(img, result, model_type, adapter):
    # 1. Segmentation
    if model_type == "segmentation":
        # result is tensor/numpy mask
        if isinstance(result, torch.Tensor):
            mask_idx = result.cpu().numpy()
        else:
            mask_idx = result
            
        mask_colored = COLORS[mask_idx]
        mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
        if mask_bgr.shape[:2] != img.shape[:2]:
             mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(img, 0.6, mask_bgr, 0.4, 0)

    # 2. Classification / Video
    elif model_type in ["classification", "video_classification"]:
        # result is (class_idx, prob)
        class_idx, prob = result
        if hasattr(adapter, "categories") and adapter.categories:
            label = adapter.categories[int(class_idx)]
        else:
            label = f"Class {int(class_idx)}"
            
        text = f"{label}: {float(prob)*100:.1f}%"
        
        # Draw Bottom Bar
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
            # 1. Management
            _ensure_model_loaded(app, session_id)
            adapter = s.get("adapter")
            
            # 2. Capture
            ret, frame = cap.read()
            if not ret:
                if not isinstance(file_path, int): # Loop file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            process_start = time.time()

            # 3. Inference Block
            if adapter:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    model_type = getattr(adapter, "model_type", "classification")
                    
                    # --- PREPROCESS ---
                    input_tensor = None
                    
                    if model_type == "video_classification":
                        # Rolling Buffer Logic
                        frame_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
                        if adapter.transform: frame_tensor = adapter.transform(frame_tensor)
                        
                        s["frame_buffer"].append(frame_tensor)
                        
                        if len(s["frame_buffer"]) == 16:
                            # Stack buffer -> (1, C, 16, H, W)
                            input_tensor = torch.stack(list(s["frame_buffer"]), dim=1).unsqueeze(0)
                            if s["device"] == "gpu" and hasattr(input_tensor, "to"):
                                input_tensor = input_tensor.to(adapter.device)
                    else:
                        # Single Image Logic
                        if adapter.transform:
                            t = torch.from_numpy(img_rgb).permute(2, 0, 1)
                            input_tensor = adapter.transform(t).unsqueeze(0)
                        else:
                            input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        
                        if hasattr(input_tensor, "to"):
                            input_tensor = input_tensor.to(adapter.device)

                    # --- INFER ---
                    if input_tensor is not None:
                        t0 = time.time()
                        
                        # Handle SOL (numpy) vs Baseline (Torch) inputs
                        if hasattr(adapter, "model_name") and "sol" in s.get("model_name", ""): 
                            raw_out = adapter.infer(input_tensor.cpu().numpy())
                        else:
                            raw_out = adapter.infer(input_tensor)
                            
                        result = adapter.postprocess(raw_out)
                        inf_ms = (time.time() - t0) * 1000.0
                        add_metric(app, session_id, "inference_time", inf_ms)
                        
                        # --- VISUALIZE ---
                        frame = _draw_result(frame, result, model_type, adapter)

                except Exception as e:
                    # logger.error(f"Inference Error: {e}") # Reduce log spam
                    cv2.putText(frame, "Infer Error", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # 4. Metrics & Stream
            now = time.time()
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