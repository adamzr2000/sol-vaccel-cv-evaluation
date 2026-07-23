# app/main.py
import logging
import csv
import json
import threading
import time
import os
import glob
import re
from datetime import datetime, timezone
from typing import Optional, Literal, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pynvml
import psutil

from rapl import discover_rapl_packages, read_energy_uj, compute_energy_diff_uj

# ---------- Logging ----------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Data Models ----------
class StartRequest(BaseModel):
    interval: float = 1.0
    csv_dir: Optional[str] = None
    tag: Optional[str] = "system-stats"
    # Mode is now a flexible list. Defaults to tracking everything.
    mode: List[Literal["cpu", "gpu", "net"]] = Field(default=["cpu", "gpu", "net"])
    net_interface: Optional[str] = Field(None, description="Optional: specific interface like 'eth0'. If null, tracks system-wide totals.")
    csv_names: Optional[Dict[str, str]] = None  # keys: "cpu", "gpu", "net"
    stdout: bool = False

class StatusResponse(BaseModel):
    is_running: bool
    mode: Optional[List[str]]

class HealthResponse(BaseModel):
    status: str
    gpu_count: int
    cpu_rapl_available: bool

# ---------- Global State ----------
class SystemMonitorState:
    running = False
    thread: Optional[threading.Thread] = None
    stop_event = threading.Event()
    current_mode: Optional[List[str]] = None
    last_energy_totals: Optional[dict] = None

state = SystemMonitorState()

# ---------- Helper Functions ----------
def _sanitize_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)

def get_next_run_index(csv_dir: str, tag: str) -> int:
    base = csv_dir.rstrip("/")
    safe_tag = _sanitize_tag(tag)
    pattern = os.path.join(base, f"{safe_tag}-run*")
    existing = glob.glob(pattern)
    max_idx = 0
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
    name = name.strip()
    if os.path.isabs(name):
        path = name
    else:
        if not name.lower().endswith(".csv"):
            name += ".csv"
        path = os.path.join(csv_dir.rstrip("/"), name)
    return path

def _derive_totals_path(base_path: str, cpu_path: str, gpu_path: str, net_path: str, mode: List[str]) -> str:
    """
    Name the energy-totals JSON after whichever device CSV name the caller
    actually chose (via csv_names), left unmodified - so it stays 1:1 aligned
    with that CSV and never collides with a different run that only differs
    by device (e.g. the same model/backend/tag run once on cpu and once on
    gpu would otherwise derive the same totals filename if the device word
    were stripped). cpu and gpu are never monitored in the same run in this
    project, so no merging/agreement logic between them is needed here.
    Falls back to the tag/run_idx based base_path only when neither cpu nor
    gpu has a path (e.g. mode=["net"] alone).
    """
    for path, kind in ((cpu_path, "cpu"), (gpu_path, "gpu")):
        if kind not in mode or not path:
            continue
        stem = os.path.basename(path)
        if stem.lower().endswith(".csv"):
            stem = stem[:-4]
        return os.path.join(os.path.dirname(path), f"energy_totals_{stem}.json")
    return f"{base_path}_energy_totals.json"

def _open_csv_writer(path: str, fieldnames: list[str]):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    try:
        if f.tell() == 0:
            w.writeheader()
            f.flush()
    except Exception:
        pass
    return f, w

def _get_cpu_temp_c() -> float:
    try:
        temps = psutil.sensors_temperatures() or {}
        preferred_keys = ["coretemp", "k10temp"]
        for key in preferred_keys:
            if key in temps and temps[key]:
                entries = temps[key]
                package = [e.current for e in entries if e.label and "package" in e.label.lower() and e.current is not None]
                if package:
                    return float(package[0])
                vals = [e.current for e in entries if e.current is not None]
                if vals:
                    return float(sum(vals) / len(vals))
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
    mode: List[str],
    stdout: bool,
    net_interface: Optional[str] = None,
    csv_names: Optional[Dict[str, str]] = None
):
    f_cpu, w_cpu = None, None
    f_gpu, w_gpu = None, None
    f_net, w_net = None, None

    if csv_dir:
        safe_tag = _sanitize_tag(tag)
        base_path = os.path.join(csv_dir.rstrip("/"), f"{safe_tag}-run{run_idx}")

        cpu_path = f"{base_path}_cpu.csv"
        gpu_path = f"{base_path}_gpu.csv"
        net_path = f"{base_path}_net.csv"

        if csv_names:
            if csv_names.get("cpu"): cpu_path = _resolve_csv_path(csv_dir, csv_names["cpu"])
            if csv_names.get("gpu"): gpu_path = _resolve_csv_path(csv_dir, csv_names["gpu"])
            if csv_names.get("net"): net_path = _resolve_csv_path(csv_dir, csv_names["net"])

        try:
            if "cpu" in mode:
                f_cpu, w_cpu = _open_csv_writer(cpu_path, fieldnames=[
                    "timestamp", "timestamp_iso", "cpu_watts", "cpu_util_percent", "cpu_temp_c",
                    "cpu_energy_j_cumulative"
                ])
            if "gpu" in mode:
                f_gpu, w_gpu = _open_csv_writer(gpu_path, fieldnames=[
                    "timestamp", "timestamp_iso", "gpu_index", "gpu_name",
                    "power_draw_w", "power_limit_w", "util_gpu_percent",
                    "util_mem_percent", "mem_used_mb", "temp_c",
                    "gpu_energy_j_cumulative"
                ])
            if "net" in mode:
                f_net, w_net = _open_csv_writer(net_path, fieldnames=[
                    "timestamp", "timestamp_iso", "net_rx_mb", "net_tx_mb"
                ])
            logger.info(f"Logging started. Mode flags: {mode}. Run Index: {run_idx}")
        except Exception as e:
            logger.error(f"Failed to open CSV files: {e}")
            if f_cpu: f_cpu.close()
            if f_gpu: f_gpu.close()
            if f_net: f_net.close()
            return

    # Init Hardware
    rapl_packages = discover_rapl_packages()
    last_cpu_energy = read_energy_uj(rapl_packages)
    cpu_energy_uj_total = 0
    cpu_read_failure_warned = False
    if "cpu" in mode:
        try: psutil.cpu_percent(interval=None)
        except Exception: pass

    gpu_ok, gpu_handles, gpu_names = False, [], []
    gpu_energy_start_mj: List[Optional[int]] = []
    if "gpu" in mode:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            gpu_names = [pynvml.nvmlDeviceGetName(h) for h in gpu_handles]
            gpu_ok = True
        except Exception as e:
            logger.error(f"NVML Init failed: {e}")

        for i, h in enumerate(gpu_handles):
            try:
                gpu_energy_start_mj.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            except Exception as e:
                gpu_energy_start_mj.append(None)
                logger.warning(f"[GPU {i}] total energy counter unavailable, gpu_energy_j_total will be omitted: {e}")

    last_net_io = None
    if "net" in mode:
        try:
            if net_interface:
                last_net_io = psutil.net_io_counters(pernic=True).get(net_interface)
            else:
                last_net_io = psutil.net_io_counters(pernic=False)
        except Exception as e:
            logger.error(f"Failed to init network counters: {e}")

    run_start_mono = time.monotonic()
    run_start_iso = datetime.now(timezone.utc).isoformat()
    last_time = run_start_mono

    # Main Loop
    while not state.stop_event.is_set():
        now_mono = time.monotonic()
        elapsed = now_mono - last_time

        if elapsed < interval:
            time.sleep(min(interval - elapsed, 0.2))
            continue

        time_delta = elapsed
        last_time = now_mono

        now_dt = datetime.now(timezone.utc)
        ts = int(now_dt.timestamp() * 1000)
        ts_iso = now_dt.isoformat()

        # CPU BLOCK
        if "cpu" in mode:
            curr_energy = read_energy_uj(rapl_packages)
            diff_uj = compute_energy_diff_uj(last_cpu_energy, curr_energy, rapl_packages)
            if diff_uj is not None:
                cpu_watts = (diff_uj / 1_000_000.0) / time_delta
                cpu_energy_uj_total += diff_uj
            else:
                cpu_watts = 0.0
                if not cpu_read_failure_warned:
                    logger.warning("RAPL energy read failed mid-run - cpu_watts/cpu_energy_j_total will undercount from this point.")
                    cpu_read_failure_warned = True
            last_cpu_energy = curr_energy
            try: cpu_util = psutil.cpu_percent(interval=None)
            except Exception: cpu_util = 0.0
            cpu_temp = _get_cpu_temp_c()
            cpu_energy_j_cumulative = round(cpu_energy_uj_total / 1_000_000.0, 3) if rapl_packages else None

            if w_cpu:
                w_cpu.writerow({"timestamp": ts, "timestamp_iso": ts_iso, "cpu_watts": round(cpu_watts, 2), "cpu_util_percent": cpu_util, "cpu_temp_c": cpu_temp, "cpu_energy_j_cumulative": cpu_energy_j_cumulative})
            if stdout:
                logger.info(f"[CPU] {cpu_watts:.2f} W | Util {cpu_util}% | Temp {cpu_temp:.1f}C")

        # GPU BLOCK
        if "gpu" in mode and gpu_ok:
            for i, h in enumerate(gpu_handles):
                try:
                    pwr = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception as e:
                    pwr = None
                    logger.warning(f"[GPU {i}] power usage unavailable: {e}")

                try:
                    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                except Exception as e:
                    limit = None
                    logger.warning(f"[GPU {i}] power limit unavailable: {e}")

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    util_gpu = util.gpu
                    util_mem = util.memory
                except Exception as e:
                    util_gpu = None
                    util_mem = None
                    logger.warning(f"[GPU {i}] utilization unavailable: {e}")

                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem_used_mb = round(mem.used / 1024**2, 2)
                except Exception as e:
                    mem_used_mb = None
                    logger.warning(f"[GPU {i}] memory info unavailable: {e}")

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception as e:
                    temp = None
                    logger.warning(f"[GPU {i}] temperature unavailable: {e}")

                gpu_energy_j_cumulative = None
                if gpu_energy_start_mj[i] is not None:
                    try:
                        curr_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                        gpu_energy_j_cumulative = round((curr_mj - gpu_energy_start_mj[i]) / 1000.0, 3)
                    except Exception as e:
                        logger.warning(f"[GPU {i}] total energy counter read failed mid-run: {e}")

                if w_gpu:
                    w_gpu.writerow({"timestamp": ts, "timestamp_iso": ts_iso, "gpu_index": i, "gpu_name": gpu_names[i], "power_draw_w": round(pwr, 2) if pwr is not None else None, "power_limit_w": round(limit, 2) if limit is not None else None, "util_gpu_percent": util_gpu, "util_mem_percent": util_mem, "mem_used_mb": mem_used_mb, "temp_c": temp, "gpu_energy_j_cumulative": gpu_energy_j_cumulative})
                if stdout:
                    parts = []
                    if pwr is not None:
                        parts.append(f"{pwr:.2f} W")
                    if util_gpu is not None:
                        parts.append(f"Util {util_gpu}%")
                    if temp is not None:
                        parts.append(f"Temp {temp}C")
                    if parts:
                        logger.info(f"[GPU {i}] " + " | ".join(parts))

        # NET BLOCK
        if "net" in mode and last_net_io:
            try:
                if net_interface:
                    curr_net_io = psutil.net_io_counters(pernic=True).get(net_interface)
                else:
                    curr_net_io = psutil.net_io_counters(pernic=False)
                
                if curr_net_io:
                    rx_mb = (curr_net_io.bytes_recv - last_net_io.bytes_recv) / (1024 * 1024)
                    tx_mb = (curr_net_io.bytes_sent - last_net_io.bytes_sent) / (1024 * 1024)
                    last_net_io = curr_net_io
                    
                    if w_net:
                        w_net.writerow({"timestamp": ts, "timestamp_iso": ts_iso, "net_rx_mb": round(rx_mb, 4), "net_tx_mb": round(tx_mb, 4)})
                    if stdout:
                        logger.info(f"[NET] RX {rx_mb:.4f} MB | TX {tx_mb:.4f} MB")
            except Exception:
                pass

        if f_cpu: f_cpu.flush()
        if f_gpu: f_gpu.flush()
        if f_net: f_net.flush()

    # Cleanup
    if f_cpu: f_cpu.close()
    if f_gpu: f_gpu.close()
    if f_net: f_net.close()

    # Capture any energy since the last completed tick up to this exact stop
    # instant, so cpu_energy_j_total covers the same [start, stop] window as
    # gpu_energy_j_total (which reads its counter fresh here too) rather than
    # only the last completed sampling tick.
    if "cpu" in mode:
        final_energy = read_energy_uj(rapl_packages)
        final_diff_uj = compute_energy_diff_uj(last_cpu_energy, final_energy, rapl_packages)
        if final_diff_uj is not None:
            cpu_energy_uj_total += final_diff_uj
        elif not cpu_read_failure_warned:
            logger.warning("RAPL energy read failed on final stop read - cpu_energy_j_total is missing the last partial interval.")

    run_stop_mono = time.monotonic()
    run_stop_iso = datetime.now(timezone.utc).isoformat()

    # Exact run-total energy: CPU from the accumulated wrap-corrected RAPL
    # deltas, GPU from the onboard energy counter (not from integrating the
    # per-tick power samples), so neither figure depends on sampling cadence.
    totals: dict = {
        "tag": tag,
        "run_idx": run_idx,
        "mode": mode,
        "run_started_iso": run_start_iso,
        "run_stopped_iso": run_stop_iso,
        "duration_sec": round(run_stop_mono - run_start_mono, 3),
    }
    if "cpu" in mode:
        totals["cpu_energy_j_total"] = round(cpu_energy_uj_total / 1_000_000.0, 3) if rapl_packages else None

    if "gpu" in mode and gpu_ok:
        gpu_totals = []
        for i, h in enumerate(gpu_handles):
            entry = {"gpu_index": i, "gpu_name": gpu_names[i], "gpu_energy_j_total": None}
            if gpu_energy_start_mj[i] is not None:
                try:
                    end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                    entry["gpu_energy_j_total"] = round((end_mj - gpu_energy_start_mj[i]) / 1000.0, 3)
                except Exception as e:
                    logger.warning(f"[GPU {i}] failed to read total energy at stop: {e}")
            gpu_totals.append(entry)
        totals["gpu"] = gpu_totals

    state.last_energy_totals = totals
    if csv_dir:
        try:
            totals_path = _derive_totals_path(base_path, cpu_path, gpu_path, net_path, mode)
            with open(totals_path, "w") as tf:
                json.dump(totals, tf, indent=2)
        except Exception as e:
            logger.error(f"Failed to write energy totals JSON: {e}")

    if "gpu" in mode and gpu_ok:
        try: pynvml.nvmlShutdown()
        except Exception: pass
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
    except Exception: pass
    rapl_ok = bool(discover_rapl_packages())
    return HealthResponse(status="ok", gpu_count=gpu_c, cpu_rapl_available=rapl_ok)

@app.post("/monitor/start")
def start(req: StartRequest):
    if state.running:
        raise HTTPException(status_code=409, detail=f"Running in mode: {state.current_mode}")
    if req.interval <= 0:
        raise HTTPException(status_code=400, detail="interval must be > 0")
    if not req.mode:
        raise HTTPException(status_code=400, detail="mode list cannot be empty")

    run_idx = 1
    if req.csv_dir:
        if not os.path.exists(req.csv_dir):
            raise HTTPException(status_code=400, detail="csv_dir does not exist")
        run_idx = get_next_run_index(req.csv_dir, req.tag)

    state.stop_event.clear()
    state.current_mode = req.mode
    state.last_energy_totals = None
    state.thread = threading.Thread(
        target=monitor_loop,
        args=(req.interval, req.csv_dir, req.tag, run_idx, req.mode, req.stdout, req.net_interface, req.csv_names),
        daemon=True
    )
    state.thread.start()
    state.running = True

    files_created = []
    if req.csv_dir:
        safe_tag = _sanitize_tag(req.tag)
        cpu_name = f"{safe_tag}-run{run_idx}_cpu.csv"
        gpu_name = f"{safe_tag}-run{run_idx}_gpu.csv"
        net_name = f"{safe_tag}-run{run_idx}_net.csv"

        if req.csv_names:
            if "cpu" in req.mode and req.csv_names.get("cpu"):
                cpu_name = req.csv_names["cpu"] if req.csv_names["cpu"].lower().endswith(".csv") else req.csv_names["cpu"] + ".csv"
            if "gpu" in req.mode and req.csv_names.get("gpu"):
                gpu_name = req.csv_names["gpu"] if req.csv_names["gpu"].lower().endswith(".csv") else req.csv_names["gpu"] + ".csv"
            if "net" in req.mode and req.csv_names.get("net"):
                net_name = req.csv_names["net"] if req.csv_names["net"].lower().endswith(".csv") else req.csv_names["net"] + ".csv"

        if "cpu" in req.mode: files_created.append(cpu_name)
        if "gpu" in req.mode: files_created.append(gpu_name)
        if "net" in req.mode: files_created.append(net_name)

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
    return {"success": True, "energy_totals": state.last_energy_totals}

@app.get("/monitor/status", response_model=StatusResponse)
def status():
    return {"is_running": state.running, "mode": state.current_mode}