# app/main.py
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import os
import re
import glob
import docker
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from monitoring import DockerContainerMonitor

# ---------- Logging ----------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Models ----------
class MessageResponse(BaseModel):
    success: bool = True
    message: str

class StartRequest(BaseModel):
    containers: List[str] = Field(..., description="List of container names/IDs/prefixes")
    interval: float = 1.0
    csv_dir: Optional[str] = Field(None, description="If set, writes one CSV per container in this directory")
    csv_names: Optional[Dict[str, str]] = Field(
        None,
        description="Optional map {container_ref: csv filename or absolute path} to override automatic naming"
    )
    stdout: bool = False
    write_header: bool = True

class StartResponse(BaseModel):
    success: bool = True
    started: List[str]
    skipped: List[str] = []
    message: str

class StatusItem(BaseModel):
    container: str
    running: bool

class StatusResponse(BaseModel):
    monitors: List[StatusItem]

class SampleResponse(BaseModel):
    container: str
    sample: Optional[dict] = None

class WindowTotalsResponse(BaseModel):
    container: str
    totals: Optional[dict] = None

# ---------- App lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.docker = None
    app.state.monitors: Dict[str, DockerContainerMonitor] = {}  # multiple monitors
    try:
        client = docker.from_env()
        client.ping()
        info = client.version()
        logger.info(f"Connected to Docker daemon - Version: {info.get('Version')}")
        app.state.docker = client
        yield
    finally:
        # Stop all running monitors gracefully
        mons: Dict[str, DockerContainerMonitor] = getattr(app.state, "monitors", {})
        for ref, mon in list(mons.items()):
            try:
                if mon.is_running():
                    mon.stop()
            except Exception as e:
                logger.error(f"Error stopping monitor '{ref}' during shutdown: {e}")

app = FastAPI(title="Docker metrics collector API", version="2.0.0", lifespan=lifespan)

# ---------- Helpers ----------
def _ensure_no_conflict(app: FastAPI, ref: str):
    mon = app.state.monitors.get(ref)
    if mon and mon.is_running():
        raise HTTPException(status_code=409, detail={"success": False, "message": f"Monitoring already running for '{ref}'. Stop it first."})

def _sanitize_ref(ref: str) -> str:
    """Make a filename-safe container ref."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in ref)

def _next_run_csv(csv_dir: str, ref: str) -> str:
    safe = _sanitize_ref(ref)
    base = csv_dir.rstrip("/")
    # Find existing <safe>-run*.csv and compute next index
    pattern = os.path.join(base, f"{safe}-run*.csv")
    existing = glob.glob(pattern)
    max_idx = 0
    rx = re.compile(rf"{re.escape(safe)}-run(\d+)\.csv$")
    for path in existing:
        fname = os.path.basename(path)
        m = rx.search(fname)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    next_idx = max_idx + 1
    return os.path.join(base, f"{safe}-run{next_idx}.csv")

def _make_csv_path(csv_dir: Optional[str], ref: str, csv_names: Optional[Dict[str, str]]) -> Optional[str]:
    """
    If csv_dir is set, choose CSV path for this container ref.
    Priority:
      1) req.csv_names[ref] if provided (absolute path allowed; otherwise treated as filename under csv_dir)
      2) fallback to automatic <sanitized_ref>-runN.csv
    """
    if not csv_dir:
        return None

    base = csv_dir.rstrip("/")

    if csv_names and ref in csv_names and csv_names[ref]:
        name = csv_names[ref].strip()

        # Absolute path: use as-is
        if os.path.isabs(name):
            return name

        # Relative: treat as filename inside csv_dir
        if not name.lower().endswith(".csv"):
            name += ".csv"
        return os.path.join(base, name)

    return _next_run_csv(base, ref)


# ---------- Endpoints ----------
@app.post(
    "/monitor/start",
    tags=["Monitoring"],
    summary="Start monitoring for one or more containers",
    response_model=StartResponse,
)
def monitor_start(req: StartRequest):
    started, skipped = [], []
    for ref in req.containers:
        try:
            _ensure_no_conflict(app, ref)
            csv_path = _make_csv_path(req.csv_dir, ref, req.csv_names)
            if csv_path:
                logger.info("CSV path for '%s': %s", ref, csv_path)

            mon = DockerContainerMonitor(
                container_ref=ref,
                interval=req.interval,
                csv_path=csv_path,
                write_header=req.write_header,
                stdout=req.stdout,
            )

            mon.start()
            app.state.monitors[ref] = mon
            started.append(ref)
        except HTTPException:
            skipped.append(ref)
        except Exception as e:
            logger.error(f"monitor_start failed for '{ref}': {e}")
            skipped.append(ref)

    if not started and skipped:
        raise HTTPException(status_code=500, detail={"success": False, "message": "No monitors started.", "skipped": skipped})

    msg = f"Started {len(started)} monitor(s)."
    if skipped:
        msg += f" Skipped {len(skipped)}."
    return StartResponse(started=started, skipped=skipped, message=msg)

@app.post(
    "/monitor/stop",
    tags=["Monitoring"],
    summary="Stop monitoring. If container is omitted, stop all.",
    response_model=MessageResponse,
)
def monitor_stop(container: Optional[str] = Query(default=None, description="Container ref to stop. If omitted, stop all.")):
    mons: Dict[str, DockerContainerMonitor] = app.state.monitors

    if container:
        mon = mons.get(container)
        if not mon or not mon.is_running():
            raise HTTPException(status_code=400, detail={"success": False, "message": f"No active monitoring for '{container}'."})
        try:
            mon.stop()
            mons.pop(container, None)
            return {"success": True, "message": f"Monitoring stopped for '{container}'."}
        except Exception as e:
            logger.error(f"monitor_stop failed for '{container}': {e}")
            raise HTTPException(status_code=500, detail={"success": False, "message": str(e)})
    else:
        any_running = False
        for ref, mon in list(mons.items()):
            try:
                if mon.is_running():
                    mon.stop()
                    any_running = True
            except Exception as e:
                logger.error(f"monitor_stop (all) failed stopping '{ref}': {e}")
        app.state.monitors.clear()
        if not any_running:
            raise HTTPException(status_code=400, detail={"success": False, "message": "No active monitoring to stop."})
        return {"success": True, "message": "All monitoring stopped."}

@app.get(
    "/monitor/status",
    tags=["Monitoring"],
    summary="List monitors and whether they are running",
    response_model=StatusResponse,
)
def monitor_status():
    items = [StatusItem(container=ref, running=mon.is_running()) for ref, mon in app.state.monitors.items()]
    return StatusResponse(monitors=items)

@app.get(
    "/monitor/last",
    tags=["Monitoring"],
    summary="Get last sample for a container",
    response_model=SampleResponse,
)
def monitor_last(container: str = Query(..., description="Container ref")):
    mon = app.state.monitors.get(container)
    if not mon:
        raise HTTPException(status_code=404, detail={"success": False, "message": f"No monitor found for '{container}'."})
    return SampleResponse(container=container, sample=mon.get_last_sample())

@app.get(
    "/monitor/window",
    tags=["Monitoring"],
    summary="Get I/O and network totals since monitoring started (per container)",
    response_model=WindowTotalsResponse,
)
def monitor_window(container: str = Query(..., description="Container ref")):
    mon = app.state.monitors.get(container)
    if not mon:
        raise HTTPException(status_code=404, detail={"success": False, "message": f"No monitor found for '{container}'."})
    return WindowTotalsResponse(container=container, totals=mon.get_window_totals())