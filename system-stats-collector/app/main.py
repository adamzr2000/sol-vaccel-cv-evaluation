# app/main.py
import logging
import csv
import threading
import time
import os
import glob
import re
from datetime import datetime, timezone
from typing import List, Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pynvml
import psutil

# ---------- Logging ----------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Data Models ----------
class StartRequest(BaseModel):
    interval: float = 1.0
    csv_dir: Optional[str] = None
    tag: Optional[str] = "system-stats"
    mode: Literal["cpu", "gpu", "both"] = "both"  # <--- Select what to monitor
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
def get_next_run_index(csv_dir: str, tag: str) -> int:
    """Finds the next run index (e.g., 1, 2, 3) for the given tag."""
    base = csv_dir.rstrip("/")
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
    
    # We look for any file starting with tag-run
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
                if idx > max_idx: max_idx = idx
            except ValueError: pass
    return max_idx + 1

def read_cpu_energy_uj() -> Optional[int]:
    """Reads the raw CPU energy counter in microjoules."""
    path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return int(f.read().strip())
    except Exception:
        pass
    return None

# ---------- Monitoring Thread ----------
def monitor_loop(interval: float, csv_dir: Optional[str], tag: str, run_idx: int, mode: str, stdout: bool):
    
    # --- 1. setup CSV writers based on mode ---
    f_cpu = None
    w_cpu = None
    f_gpu = None
    w_gpu = None
    
    if csv_dir:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
        base_path = os.path.join(csv_dir.rstrip("/"), f"{safe_tag}-run{run_idx}")

        try:
            # CPU Writer
            if mode in ("cpu", "both"):
                f_cpu = open(f"{base_path}_cpu.csv", "a", newline="")
                w_cpu = csv.DictWriter(f_cpu, fieldnames=[
                    "timestamp", "timestamp_iso", 
                    "cpu_watts", "cpu_util_percent", "cpu_temp_c"
                ])
                w_cpu.writeheader()
                f_cpu.flush()

            # GPU Writer
            if mode in ("gpu", "both"):
                f_gpu = open(f"{base_path}_gpu.csv", "a", newline="")
                w_gpu = csv.DictWriter(f_gpu, fieldnames=[
                    "timestamp", "timestamp_iso", "gpu_index", "gpu_name", 
                    "power_draw_w", "power_limit_w", "util_gpu_percent", 
                    "util_mem_percent", "mem_used_mb", "temp_c"
                ])
                w_gpu.writeheader()
                f_gpu.flush()
                
            logger.info(f"Logging started. Mode: {mode}. Run Index: {run_idx}")
            
        except Exception as e:
            logger.error(f"Failed to open CSV files: {e}")
            if f_cpu: f_cpu.close()
            if f_gpu: f_gpu.close()
            return

    # --- 2. Init Hardware ---
    gpu_handles = []
    gpu_names = []
    if mode in ("gpu", "both"):
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            gpu_names = [pynvml.nvmlDeviceGetName(h) for h in gpu_handles]
        except Exception as e:
            logger.error(f"NVML Init failed: {e}")

    last_time = time.monotonic()
    last_cpu_energy = read_cpu_energy_uj()

    # --- 3. Main Loop ---
    while not state.stop_event.is_set():
        now_mono = time.monotonic()
        if now_mono - last_time < interval:
            time.sleep(0.05)
            continue
        
        # Calculate precise delta for power math
        time_delta = now_mono - last_time
        last_time = now_mono

        now_dt = datetime.now(timezone.utc)
        ts = int(now_dt.timestamp() * 1000)
        ts_iso = now_dt.isoformat()

        # === CPU BLOCK ===
        if mode in ("cpu", "both"):
            # 1. Power
            cpu_watts = 0.0
            curr_energy = read_cpu_energy_uj()
            if last_cpu_energy is not None and curr_energy is not None:
                diff = curr_energy - last_cpu_energy
                if diff < 0: diff = 0 
                if time_delta > 0:
                    cpu_watts = (diff / 1_000_000.0) / time_delta
            last_cpu_energy = curr_energy

            # 2. Utilization
            # interval=None is crucial so it doesn't block!
            cpu_util = psutil.cpu_percent(interval=None) 
            
            # 3. Temperature
            cpu_temp = 0.0
            try:
                # 'coretemp' is common for Intel, 'k10temp' for AMD
                # psutil returns a dictionary of sensors. We take the first valid one.
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    cpu_temp = temps["coretemp"][0].current
                elif "k10temp" in temps:
                    cpu_temp = temps["k10temp"][0].current
                else:
                    # Fallback: grab the first available sensor if specific ones aren't found
                    for name, entries in temps.items():
                        if entries:
                            cpu_temp = entries[0].current
                            break
            except Exception:
                pass

            # Write to CSV
            if w_cpu:
                w_cpu.writerow({
                    "timestamp": ts, 
                    "timestamp_iso": ts_iso, 
                    "cpu_watts": round(cpu_watts, 2),
                    "cpu_util_percent": cpu_util,
                    "cpu_temp_c": cpu_temp
                })
            
            # Log to Stdout
            if stdout:
                logger.info(f"[CPU] {cpu_watts:.2f} W | Util {cpu_util}% | Temp {cpu_temp}C")

        # === GPU BLOCK ===
        if mode in ("gpu", "both"):
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
                    if w_gpu: w_gpu.writerow(row)
                    if stdout:
                        logger.info(f"[GPU {i}] {pwr:.2f} W | Util {util.gpu}% | Temp {temp}C")
                except Exception:
                    pass

        # Flush buffers
        if f_cpu: f_cpu.flush()
        if f_gpu: f_gpu.flush()

    # --- Cleanup ---
    if f_cpu: f_cpu.close()
    if f_gpu: f_gpu.close()
    if mode in ("gpu", "both"):
        try: pynvml.nvmlShutdown()
        except: pass
    logger.info("Monitor thread stopped.")

# ---------- API Lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.running:
        state.stop_event.set()
        if state.thread: state.thread.join()

app = FastAPI(title="System Stats Collector", lifespan=lifespan)

# ---------- Endpoints ----------
@app.get("/health", response_model=HealthResponse)
def health_check():
    gpu_c = 0
    try:
        pynvml.nvmlInit()
        gpu_c = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except: pass
    
    rapl_ok = os.path.exists("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
    
    return HealthResponse(status="ok", gpu_count=gpu_c, cpu_rapl_available=rapl_ok)

@app.post("/monitor/start")
def start(req: StartRequest):
    if state.running:
        raise HTTPException(status_code=409, detail=f"Running in mode: {state.current_mode}")
    
    run_idx = 1
    if req.csv_dir:
        if not os.path.exists(req.csv_dir):
            raise HTTPException(status_code=400, detail="csv_dir does not exist")
        run_idx = get_next_run_index(req.csv_dir, req.tag)

    state.stop_event.clear()
    state.current_mode = req.mode
    state.thread = threading.Thread(
        target=monitor_loop,
        args=(req.interval, req.csv_dir, req.tag, run_idx, req.mode, req.stdout),
        daemon=True
    )
    state.thread.start()
    state.running = True
    
    files_created = []
    if req.csv_dir:
        base = f"{req.tag}-run{run_idx}"
        if req.mode in ("cpu", "both"): files_created.append(f"{base}_cpu.csv")
        if req.mode in ("gpu", "both"): files_created.append(f"{base}_gpu.csv")

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
    if state.thread: state.thread.join(timeout=5.0)
    state.running = False
    state.current_mode = None
    return {"success": True}

@app.get("/monitor/status", response_model=StatusResponse)
def status():
    return {"is_running": state.running, "mode": state.current_mode}