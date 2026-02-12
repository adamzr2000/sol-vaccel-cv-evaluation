# app/main.py
import logging
import csv
import threading
import time
import os
import glob
import re
from datetime import datetime, timezone
from typing import Optional, Literal, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pynvml
import psutil

from rapl import discover_rapl_packages, read_energy_uj, compute_watts

# ---------- Logging ----------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Data Models ----------
class StartRequest(BaseModel):
    interval: float = 1.0
    csv_dir: Optional[str] = None
    tag: Optional[str] = "system-stats"
    mode: Literal["cpu", "gpu", "both"] = "both"
    csv_names: Optional[Dict[str, str]] = None  # keys: "cpu", "gpu"
    stdout: bool = False

class StatusResponse(BaseModel):
    is_running: bool
    mode: Optional[str]

class HealthResponse(BaseModel):
    status: str
    gpu_count: int
    cpu_rapl_available: bool

# ---------- Global State ----------
class SystemMonitorState:
    running = False
    thread: Optional[threading.Thread] = None
    stop_event = threading.Event()
    current_mode: Optional[str] = None

state = SystemMonitorState()

# ---------- Helper Functions ----------
def _sanitize_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)

def get_next_run_index(csv_dir: str, tag: str) -> int:
    """Find the next run index (e.g., 1, 2, 3) for the given tag."""
    base = csv_dir.rstrip("/")
    safe_tag = _sanitize_tag(tag)

    pattern = os.path.join(base, f"{safe_tag}-run*")
    existing = glob.glob(pattern)
    max_idx = 0

    # Matches both ...-run1_cpu.csv and ...-run1.csv
    rx = re.compile(rf"{re.escape(safe_tag)}-run(\d+)(_.*)?\.csv$")

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
    return max_idx + 1

def _resolve_csv_path(csv_dir: str, name: str) -> str:
    """
    If name is absolute -> use as-is.
    Else -> treat as filename under csv_dir.
    Ensures .csv suffix.
    """
    name = name.strip()
    if os.path.isabs(name):
        path = name
    else:
        if not name.lower().endswith(".csv"):
            name += ".csv"
        path = os.path.join(csv_dir.rstrip("/"), name)
    return path

def _open_csv_writer(path: str, fieldnames: list[str]):
    """
    Open CSV in append mode; write header only if file is new/empty.
    """
    # Ensure parent dir exists (safe even if already exists)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)

    try:
        # If file is empty, write header
        if f.tell() == 0:
            w.writeheader()
            f.flush()
    except Exception:
        # If tell() fails for any reason, do not break monitoring
        pass

    return f, w

def _get_cpu_temp_c() -> float:
    """
    Prefer package temperature if available; otherwise average cores; fallback to first sensor.
    """
    try:
        temps = psutil.sensors_temperatures() or {}

        # Common Intel / AMD keys
        preferred_keys = ["coretemp", "k10temp"]
        for key in preferred_keys:
            if key in temps and temps[key]:
                entries = temps[key]

                # Try to find a package sensor first
                package = [e.current for e in entries if e.label and "package" in e.label.lower() and e.current is not None]
                if package:
                    return float(package[0])

                # Else average all valid entries
                vals = [e.current for e in entries if e.current is not None]
                if vals:
                    return float(sum(vals) / len(vals))

        # Fallback: first available sensor entry
        for _, entries in temps.items():
            if entries:
                for e in entries:
                    if e.current is not None:
                        return float(e.current)
    except Exception:
        pass

    return 0.0

# ---------- Monitoring Thread ----------
def monitor_loop(
    interval: float,
    csv_dir: Optional[str],
    tag: str,
    run_idx: int,
    mode: str,
    stdout: bool,
    csv_names: Optional[Dict[str, str]] = None
):
    # --- 1) Setup CSV writers based on mode ---
    f_cpu = None
    w_cpu = None
    f_gpu = None
    w_gpu = None

    if csv_dir:
        safe_tag = _sanitize_tag(tag)
        base_path = os.path.join(csv_dir.rstrip("/"), f"{safe_tag}-run{run_idx}")

        # default paths
        cpu_path = f"{base_path}_cpu.csv"
        gpu_path = f"{base_path}_gpu.csv"

        # overrides (optional)
        if csv_names:
            if csv_names.get("cpu"):
                cpu_path = _resolve_csv_path(csv_dir, csv_names["cpu"])
            if csv_names.get("gpu"):
                gpu_path = _resolve_csv_path(csv_dir, csv_names["gpu"])

        try:
            if mode in ("cpu", "both"):
                f_cpu, w_cpu = _open_csv_writer(cpu_path, fieldnames=[
                    "timestamp", "timestamp_iso",
                    "cpu_watts", "cpu_util_percent", "cpu_temp_c"
                ])

            if mode in ("gpu", "both"):
                f_gpu, w_gpu = _open_csv_writer(gpu_path, fieldnames=[
                    "timestamp", "timestamp_iso", "gpu_index", "gpu_name",
                    "power_draw_w", "power_limit_w", "util_gpu_percent",
                    "util_mem_percent", "mem_used_mb", "temp_c"
                ])

            logger.info(f"Logging started. Mode: {mode}. Run Index: {run_idx}")

        except Exception as e:
            logger.error(f"Failed to open CSV files: {e}")
            if f_cpu:
                f_cpu.close()
            if f_gpu:
                f_gpu.close()
            return

    # --- 2) Init Hardware ---
    # CPU: RAPL packages discovery (works even if none found)
    rapl_packages = discover_rapl_packages()
    last_cpu_energy = read_energy_uj(rapl_packages)

    # Prime cpu_percent to avoid first-sample junk
    if mode in ("cpu", "both"):
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # GPU: NVML init (track success)
    gpu_ok = False
    gpu_handles = []
    gpu_names = []
    if mode in ("gpu", "both"):
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            gpu_names = [pynvml.nvmlDeviceGetName(h) for h in gpu_handles]
            gpu_ok = True
        except Exception as e:
            logger.error(f"NVML Init failed: {e}")
            gpu_ok = False

    last_time = time.monotonic()

    # --- 3) Main Loop ---
    while not state.stop_event.is_set():
        now_mono = time.monotonic()
        elapsed = now_mono - last_time

        if elapsed < interval:
            # Sleep smarter: wake up less often but remain responsive to stop_event
            remaining = interval - elapsed
            time.sleep(min(remaining, 0.2))
            continue

        time_delta = elapsed
        last_time = now_mono

        now_dt = datetime.now(timezone.utc)
        ts = int(now_dt.timestamp() * 1000)
        ts_iso = now_dt.isoformat()

        # === CPU BLOCK ===
        if mode in ("cpu", "both"):
            # Power (W) via RAPL (supports multi-package + wrap if max_energy_range_uj available)
            curr_energy = read_energy_uj(rapl_packages)
            cpu_watts = compute_watts(last_cpu_energy, curr_energy, time_delta, rapl_packages)
            last_cpu_energy = curr_energy

            # Utilization (%)
            try:
                cpu_util = psutil.cpu_percent(interval=None)
            except Exception:
                cpu_util = 0.0

            # Temperature (C)
            cpu_temp = _get_cpu_temp_c()

            if w_cpu:
                w_cpu.writerow({
                    "timestamp": ts,
                    "timestamp_iso": ts_iso,
                    "cpu_watts": round(cpu_watts, 2),
                    "cpu_util_percent": cpu_util,
                    "cpu_temp_c": cpu_temp
                })

            if stdout:
                logger.info(f"[CPU] {cpu_watts:.2f} W | Util {cpu_util}% | Temp {cpu_temp:.1f}C")

        # === GPU BLOCK ===
        if mode in ("gpu", "both") and gpu_ok:
            for i, h in enumerate(gpu_handles):
                try:
                    pwr = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

                    row = {
                        "timestamp": ts, "timestamp_iso": ts_iso,
                        "gpu_index": i, "gpu_name": gpu_names[i],
                        "power_draw_w": round(pwr, 2), "power_limit_w": round(limit, 2),
                        "util_gpu_percent": util.gpu, "util_mem_percent": util.memory,
                        "mem_used_mb": round(mem.used / 1024**2, 2),
                        "temp_c": temp
                    }
                    if w_gpu:
                        w_gpu.writerow(row)

                    if stdout:
                        logger.info(f"[GPU {i}] {pwr:.2f} W | Util {util.gpu}% | Temp {temp}C")

                except Exception:
                    # Keep collector resilient (same as your original intent)
                    pass

        # Flush buffers
        if f_cpu:
            f_cpu.flush()
        if f_gpu:
            f_gpu.flush()

    # --- Cleanup ---
    if f_cpu:
        f_cpu.close()
    if f_gpu:
        f_gpu.close()
    if mode in ("gpu", "both") and gpu_ok:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    logger.info("Monitor thread stopped.")

# ---------- API Lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.running:
        state.stop_event.set()
        if state.thread:
            state.thread.join()

app = FastAPI(title="System Stats Collector", lifespan=lifespan)

# ---------- Endpoints ----------
@app.get("/health", response_model=HealthResponse)
def health_check():
    gpu_c = 0
    try:
        pynvml.nvmlInit()
        gpu_c = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # More robust: if any package exists, say True
    rapl_ok = bool(discover_rapl_packages())

    return HealthResponse(status="ok", gpu_count=gpu_c, cpu_rapl_available=rapl_ok)

@app.post("/monitor/start")
def start(req: StartRequest):
    if state.running:
        raise HTTPException(status_code=409, detail=f"Running in mode: {state.current_mode}")

    if req.interval <= 0:
        raise HTTPException(status_code=400, detail="interval must be > 0")

    run_idx = 1
    if req.csv_dir:
        if not os.path.exists(req.csv_dir):
            raise HTTPException(status_code=400, detail="csv_dir does not exist")
        run_idx = get_next_run_index(req.csv_dir, req.tag)

    state.stop_event.clear()
    state.current_mode = req.mode
    state.thread = threading.Thread(
        target=monitor_loop,
        args=(req.interval, req.csv_dir, req.tag, run_idx, req.mode, req.stdout, req.csv_names),
        daemon=True
    )
    state.thread.start()
    state.running = True

    files_created = []
    if req.csv_dir:
        safe_tag = _sanitize_tag(req.tag)
        cpu_name = f"{safe_tag}-run{run_idx}_cpu.csv"
        gpu_name = f"{safe_tag}-run{run_idx}_gpu.csv"

        # apply overrides (report names, not full paths) — keep your existing behavior
        if req.csv_names:
            if req.mode in ("cpu", "both") and req.csv_names.get("cpu"):
                cpu_name = req.csv_names["cpu"] if req.csv_names["cpu"].lower().endswith(".csv") else req.csv_names["cpu"] + ".csv"
            if req.mode in ("gpu", "both") and req.csv_names.get("gpu"):
                gpu_name = req.csv_names["gpu"] if req.csv_names["gpu"].lower().endswith(".csv") else req.csv_names["gpu"] + ".csv"

        if req.mode in ("cpu", "both"):
            files_created.append(cpu_name)
        if req.mode in ("gpu", "both"):
            files_created.append(gpu_name)

    return {
        "success": True,
        "mode": req.mode,
        "files": files_created if req.csv_dir else "None (No Directory)"
    }

@app.post("/monitor/stop")
def stop():
    if not state.running:
        return {"success": False, "message": "Not running"}
    state.stop_event.set()
    if state.thread:
        state.thread.join(timeout=5.0)
    state.running = False
    state.current_mode = None
    return {"success": True}

@app.get("/monitor/status", response_model=StatusResponse)
def status():
    return {"is_running": state.running, "mode": state.current_mode}
