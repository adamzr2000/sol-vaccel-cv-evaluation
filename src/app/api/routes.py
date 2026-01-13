from pathlib import Path
from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)

# Shared plumbing (from image_inference)
from app.services.image_inference import (
    reset_stop_stream,
    set_stop_stream,
    delete_session_video,
    reset_session_metrics,
    send_metrics,
    set_session_source,
    get_session_source,
)

# CV-specific logic (New)
from app.services.cv_inference import (
    list_models,
    reset_session_data,
    set_model_choice,
    set_inference_params,
    generate_frame,
)

router = APIRouter()
templates = None  # Injected from main.py


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Redirect to our new main page
    return RedirectResponse(url="/cv-benchmark")


@router.get("/cv-benchmark", response_class=HTMLResponse)
async def cv_benchmark(request: Request):
    models = list_models()
    session_id = request.session.get("id", "anonymous")

    # Reset session state for a fresh start
    reset_session_data(request.app, session_id)

    # Set safe defaults
    default_model = "resnet50"
    request.session["model_name"] = default_model
    set_model_choice(request.app, session_id, default_model)

    return templates.TemplateResponse(
        "cv_benchmark.html",
        {
            "request": request,
            "page_title": "Torchvision",
            "models": models,
        },
    )


@router.post("/set-model")
async def set_model(request: Request):
    data = await request.json()
    session_id = request.session.get("id", "anonymous")
    
    model_name = data.get("name", "resnet50")
    
    # Update Service State
    set_model_choice(request.app, session_id, model_name)

    # Reflect in session cookie
    request.session["model_name"] = model_name
    
    return {"status": "info", "message": f"Model set to '{model_name}'"}


@router.post("/set-params")
async def set_params(request: Request):
    data = await request.json()
    session_id = request.session.get("id", "anonymous")
    
    backend = data.get("backend", "stock")
    device = data.get("device", "cpu")
    
    # Update Service State
    set_inference_params(request.app, session_id, backend, device)
    
    return {"status": "info", "message": f"Updated params: {backend} on {device}"}


@router.post("/set-source")
async def set_source(request: Request):
    """
    Body: { "use_camera": true/false, "camera_id": 0 }
    """
    data = await request.json()
    use_camera = bool(data.get("use_camera", False))
    camera_id = int(data.get("camera_id", 0))
    
    set_session_source(request.session, use_camera=use_camera, camera_id=camera_id)
    
    msg = "Using webcam" if use_camera else "Using video file"
    return {"status": "info", "message": f"{msg} (id={camera_id})"}


@router.get("/video-feed")
async def video_feed(request: Request):
    session_id = request.session.get("id", "anonymous")
    src = get_session_source(request.app, request.session)
    
    reset_stop_stream(request.app, session_id)
    reset_session_metrics(request.app, session_id)

    return StreamingResponse(
        generate_frame(
            app=request.app,
            session_id=session_id,
            file_path=src, 
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/stop-stream")
async def stop_video_stream(request: Request):
    session_id = request.session.get("id", "anonymous")
    set_stop_stream(request.app, session_id, value=True)
    return {"status": "info", "message": "Stream stopped"}


@router.post("/delete-video")
async def delete_video(request: Request):
    delete_session_video(request.app, request.session)
    return {"status": "info", "message": "Video reset to default"}


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await send_metrics(websocket)