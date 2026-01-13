import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI()

# Global state for a simple demo
stream_enabled = True
cap = None


def get_capture():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # webcam 0
    return cap


def mjpeg_generator():
    """
    Generates an MJPEG stream (multipart/x-mixed-replace) from OpenCV frames.
    """
    global stream_enabled

    cap_local = get_capture()

    while True:
        if not stream_enabled:
            # If disabled, just yield nothing and wait a bit
            # (we can't "sleep" async here; OpenCV read loop is fine for a demo)
            # A simple workaround: keep looping but don't emit frames.
            continue

        ok, frame = cap_local.read()
        if not ok:
            continue

        # Encode as JPEG
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Webcam Stream</title>
    <style>
      body { font-family: sans-serif; margin: 24px; }
      .row { display: flex; gap: 16px; align-items: center; }
      img { width: 640px; height: auto; border: 1px solid #444; border-radius: 8px; }
      button { padding: 10px 14px; border-radius: 8px; border: 1px solid #444; cursor: pointer; }
      .status { opacity: 0.8; }
    </style>
  </head>
  <body>
    <h2>Live Webcam</h2>

    <div class="row">
      <button id="toggleBtn">Stop stream</button>
      <span class="status" id="statusText">Streaming: ON</span>
    </div>

    <div style="margin-top: 16px;">
      <img id="cam" src="/video" alt="webcam stream"/>
    </div>

    <script>
      const btn = document.getElementById("toggleBtn");
      const img = document.getElementById("cam");
      const statusText = document.getElementById("statusText");

      let streaming = true;

      btn.onclick = async () => {
        const res = await fetch("/toggle", { method: "POST" });
        const data = await res.json();
        streaming = data.enabled;

        if (streaming) {
          // Force refresh (cache-bust) to resume stream cleanly
          img.src = "/video?t=" + Date.now();
          btn.textContent = "Stop stream";
          statusText.textContent = "Streaming: ON";
        } else {
          // Remove src to stop requests in the browser
          img.src = "";
          btn.textContent = "Start stream";
          statusText.textContent = "Streaming: OFF";
        }
      };
    </script>
  </body>
</html>
"""


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/toggle")
def toggle():
    global stream_enabled
    stream_enabled = not stream_enabled
    return JSONResponse({"enabled": stream_enabled})


@app.on_event("shutdown")
def shutdown():
    global cap
    if cap is not None:
        cap.release()
        cap = None
