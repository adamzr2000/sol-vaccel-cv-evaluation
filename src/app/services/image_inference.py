# app/services/image_inference.py
import asyncio
from pathlib import Path
from collections import deque

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ---------- Session helpers ----------
def get_sessions(app: FastAPI):
    return app.state.inference_sessions

def get_session_data(app: FastAPI, session_id: str):
    return app.state.inference_sessions.setdefault(session_id, {})

def get_session_websockets(app: FastAPI, session_id: str):
    return app.state.websockets.setdefault(session_id, set())

# ---------- Metrics helpers ----------
def get_metrics(app: FastAPI, session_id: str):
    s = get_session_data(app, session_id)
    return s.setdefault("metrics", {})

def add_metric(app: FastAPI, session_id: str, key: str, value):
    get_metrics(app, session_id)[key] = value

def reset_session_metrics(app: FastAPI, session_id: str):
    m = get_metrics(app, session_id)
    m["fps"] = 0.0
    m["inference_time"] = 0.0
    m["processing_time"] = 0.0
    m["fps_history"] = deque(maxlen=30)
    m["last_frame_time"] = 0.0

def prepare_overlay(frame, metrics: dict):
    # Draw simple metrics box
    cv2.rectangle(frame, (5, 5), (260, 110), (0, 0, 0), -1)
    
    # Text info
    cv2.putText(frame, f"FPS: {metrics.get('fps', 0.0):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Infer: {metrics.get('inference_time', 0.0):.1f} ms", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Proc: {metrics.get('processing_time', 0.0):.1f} ms", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# ---------- Stream control ----------
def reset_stop_stream(app: FastAPI, session_id: str):
    get_session_data(app, session_id)["stop_stream"] = False

def set_stop_stream(app: FastAPI, session_id: str, value: bool):
    get_session_data(app, session_id)["stop_stream"] = value

# ---------- Video selection / cleanup ----------
def get_session_video(app: FastAPI, session: dict) -> Path:
    upload_dir = Path(app.state.videos_upload_dir)
    videos_dir = Path(app.state.videos_dir)
    uploaded_video = session.get("uploaded_video")
    if uploaded_video:
        file_name = uploaded_video[next(iter(uploaded_video))]
        return upload_dir / file_name
    
    # Fallback
    return videos_dir / "test_video.mp4"

def set_session_video(session: dict, video_name: str, file_path: str):
    session["uploaded_video"] = {video_name: Path(file_path).name}

def set_session_source(session: dict, *, use_camera: bool, camera_id: int = 0):
    session["use_camera"] = bool(use_camera)
    session["camera_id"] = int(camera_id)

def get_session_source(app, session: dict):
    """
    Returns either an int (camera id) or a Path (file) based on session flags.
    """
    use_cam = session.get("use_camera", False)
    if use_cam:
        return int(session.get("camera_id", 0))  # webcam index
    
    # else use uploaded or static fallback
    return get_session_video(app, session)

def delete_session_video(app: FastAPI, session: dict):
    dir_path = Path(app.state.videos_upload_dir)
    uploaded_video = session.get("uploaded_video")
    if uploaded_video:
        file_name = uploaded_video[next(iter(uploaded_video))]
        (dir_path / file_name).unlink(missing_ok=True)
    session.pop("uploaded_video", None)

def delete_all_uploaded_videos(app: FastAPI):
    dir_path = Path(app.state.videos_upload_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    for f in dir_path.iterdir():
        if f.is_file() and not f.name.startswith("."):
            f.unlink(missing_ok=True)

def capture_video(src: int | Path):
    """Open webcam when src is an int (camera index), or a file Path."""
    cap = cv2.VideoCapture(src if isinstance(src, int) else str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")
    
    # Set resolution for webcams (files ignore this)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    # Attempt to use MJPG for faster USB cam reading
    if cap.get(cv2.CAP_PROP_FOURCC) != 0.0:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
    return cap

# ---------- Metrics websocket ----------
async def wait_for_metrics_ready(websocket: WebSocket, session_id: str):
    s = get_session_data(websocket.app, session_id)
    event = s.get("metrics_ready")
    if not event:
        return False
    try:
        await asyncio.wait_for(event.wait(), timeout=10.0)
        event.clear()
        return True
    except asyncio.TimeoutError:
        event.clear()
        try:
            await websocket.send_json({"status": "error", "message": "Metrics not ready"})
        finally:
            await websocket.close()
        return False

async def send_metrics(websocket: WebSocket):
    await websocket.accept()

    # 1. DEBUG: Print available sessions
    all_sessions = list(websocket.app.state.inference_sessions.keys())
    print(f"DEBUG: WS connected. Available sessions: {all_sessions}")

    # 2. Try to find the active session
    # Priority: Cookie -> First active session -> "anonymous"
    session_id = websocket.cookies.get("session_id")
    
    if not session_id or session_id not in websocket.app.state.inference_sessions:
        if all_sessions:
            session_id = all_sessions[0] # Pick the first available one
            print(f"DEBUG: WS using fallback session: {session_id}")
        else:
            session_id = "anonymous"
            print("DEBUG: WS using anonymous (New Session)")

    if not await wait_for_metrics_ready(websocket, session_id):
        print(f"DEBUG: Metrics timeout for {session_id}")
        return

    conns = get_session_websockets(websocket.app, session_id)
    conns.add(websocket)
    try:
        while True:
            m = get_metrics(websocket.app, session_id)
            # DEBUG: Uncomment if you suspect data is zero
            # print(f"DEBUG: Sending metrics for {session_id}: {m.get('fps', 0)}")
            
            await websocket.send_json({
                "status": "success",
                "message": "",
                "metrics": {
                    "fps": m.get("fps", 0.0),
                    "inference_time": m.get("inference_time", 0.0),
                    "processing_time": m.get("processing_time", 0.0),
                },
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("DEBUG: WS Disconnect")
    finally:
        conns.discard(websocket)