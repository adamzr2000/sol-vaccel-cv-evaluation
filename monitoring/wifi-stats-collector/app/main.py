# app/main.py
#
# Standalone WiFi link monitor -- a dedicated, minimal service for the small
# set of network-perturbation experiments (robot <-wired-> miniPC <-wifi->
# AP <-> edge). Deliberately separate from system-stats-collector/
# docker-stats-collector: this runs on a plain NAT/forwarding box with no
# GPU and nothing RAPL-relevant to measure, so it carries none of those
# dependencies -- just fastapi/uvicorn/pydantic and the `iw` binary.
import logging
import csv
import json
import threading
import time
import os
import glob
import re
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Helpers (generic CSV/tag plumbing, no shared state) ----------
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
        m = rx.search(os.path.basename(path))
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    return max_idx + 1

def _resolve_csv_path(csv_dir: str, name: str) -> str:
    name = name.strip()
    if os.path.isabs(name):
        return name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return os.path.join(csv_dir.rstrip("/"), name)

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

# ---------- WiFi link reading ----------
_MAC_RE = re.compile(r"Connected to ([0-9a-fA-F:]{17})")
_COUNTER_RE = {
    "tx_packets": re.compile(r"tx packets:\s*(\d+)"),
    "tx_retries": re.compile(r"tx retries:\s*(\d+)"),
    "tx_failed":  re.compile(r"tx failed:\s*(\d+)"),
}
_SIGNAL_RE = re.compile(r"^\s*signal:\s*(-?\d+)", re.MULTILINE)

def _detect_ap_mac(iface: str) -> Optional[str]:
    """
    Auto-detect the AP a wireless client interface is currently associated
    to, via `iw dev <iface> link` ("Connected to <mac> (on <iface>)", or
    "Not connected." if unassociated). Returns None on any failure --
    missing iw, nonexistent interface, or not associated.
    """
    try:
        out = subprocess.run(
            ["iw", "dev", iface, "link"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    m = _MAC_RE.search(out.stdout)
    return m.group(1) if m else None

def _read_station(iface: str, mac: str) -> Optional[Dict[str, Optional[int]]]:
    """
    Cumulative tx_packets/tx_retries/tx_failed counters plus instantaneous
    RSSI (signal, dBm -- NOT cumulative, returned as-is) for one known peer,
    via `iw dev <iface> station get <mac>`. Returns None (not zeros) if the
    query fails for any reason -- peer roamed away, iw missing, interface
    gone -- callers treat None as "skip this sample", never as "0 retries".
    """
    try:
        out = subprocess.run(
            ["iw", "dev", iface, "station", "get", mac],
            capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    result: Dict[str, Optional[int]] = {}
    found_any = False
    for key, pattern in _COUNTER_RE.items():
        m = pattern.search(out.stdout)
        if m:
            found_any = True
            result[key] = int(m.group(1))
        else:
            result[key] = None
    if not found_any:
        return None

    sig_m = _SIGNAL_RE.search(out.stdout)
    result["rssi_dbm"] = int(sig_m.group(1)) if sig_m else None
    return result

# ---------- State ----------
class MonitorState:
    running = False
    thread: Optional[threading.Thread] = None
    stop_event = threading.Event()
    last_totals: Optional[dict] = None
    interface: Optional[str] = None
    mac: Optional[str] = None

state = MonitorState()
_lock = threading.Lock()  # guards start/stop against concurrent requests

# ---------- API models ----------
class StartRequest(BaseModel):
    interval: float = 1.0
    csv_dir: Optional[str] = None
    tag: Optional[str] = "wifi-stats"
    wifi_interface: str
    csv_name: Optional[str] = None
    stdout: bool = False

class StatusResponse(BaseModel):
    is_running: bool
    interface: Optional[str] = None
    mac: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    iw_available: bool

# ---------- Monitoring thread ----------
def monitor_loop(
    interval: float,
    csv_dir: Optional[str],
    tag: str,
    run_idx: int,
    wifi_interface: str,
    stdout: bool,
    csv_name: Optional[str],
    mac: str,
):
    f_csv, w_csv = None, None
    csv_path = None

    if csv_dir:
        safe_tag = _sanitize_tag(tag)
        csv_path = os.path.join(csv_dir.rstrip("/"), f"{safe_tag}-run{run_idx}_wifi.csv")
        if csv_name:
            csv_path = _resolve_csv_path(csv_dir, csv_name)
        try:
            f_csv, w_csv = _open_csv_writer(csv_path, fieldnames=[
                "timestamp", "timestamp_iso",
                "tx_packets_cumulative", "tx_retries_cumulative", "tx_failed_cumulative",
                "rssi_dbm",
            ])
            logger.info(f"Logging started. iface={wifi_interface} mac={mac} run_idx={run_idx}")
        except Exception as e:
            logger.error(f"Failed to open CSV file: {e}")
            return

    # Exact-duration start/end snapshots, independent of sampling cadence --
    # the rate total is computed from these two reads, not by summing
    # per-tick deltas, so it doesn't depend on how often we happened to poll.
    start_stats = _read_station(wifi_interface, mac)
    if start_stats is None:
        logger.warning(f"Could not read initial station stats for {mac} on {wifi_interface}.")

    run_start_mono = time.monotonic()
    run_start_iso = datetime.now(timezone.utc).isoformat()
    last_time = run_start_mono

    while not state.stop_event.is_set():
        now_mono = time.monotonic()
        elapsed = now_mono - last_time
        if elapsed < interval:
            time.sleep(min(interval - elapsed, 0.2))
            continue
        last_time = now_mono

        now_dt = datetime.now(timezone.utc)
        ts = int(now_dt.timestamp() * 1000)
        ts_iso = now_dt.isoformat()

        curr = _read_station(wifi_interface, mac)
        if curr is not None:
            if w_csv:
                w_csv.writerow({
                    "timestamp": ts, "timestamp_iso": ts_iso,
                    "tx_packets_cumulative": curr["tx_packets"],
                    "tx_retries_cumulative": curr["tx_retries"],
                    "tx_failed_cumulative": curr["tx_failed"],
                    "rssi_dbm": curr["rssi_dbm"],
                })
                f_csv.flush()
            if stdout:
                logger.info(
                    f"tx_packets={curr['tx_packets']} tx_retries={curr['tx_retries']} "
                    f"tx_failed={curr['tx_failed']} rssi={curr['rssi_dbm']}dBm"
                )
        # A transient None here (brief roam/deassociation) is not fatal --
        # just skip this tick. The totals block re-reads at stop regardless.

    if f_csv:
        f_csv.close()

    end_stats = _read_station(wifi_interface, mac)
    run_stop_mono = time.monotonic()
    run_stop_iso = datetime.now(timezone.utc).isoformat()

    totals: dict = {
        "tag": tag,
        "run_idx": run_idx,
        "interface": wifi_interface,
        "mac": mac,
        "run_started_iso": run_start_iso,
        "run_stopped_iso": run_stop_iso,
        "duration_sec": round(run_stop_mono - run_start_mono, 3),
    }
    if start_stats is not None and end_stats is not None:
        d_packets = end_stats["tx_packets"] - start_stats["tx_packets"]
        d_retries = end_stats["tx_retries"] - start_stats["tx_retries"]
        d_failed = end_stats["tx_failed"] - start_stats["tx_failed"]
        # A dropped+reassociated link resets these counters near zero, which
        # would otherwise show up as a silently wrong negative delta/rate --
        # flag it explicitly instead of reporting nonsense. The per-tick CSV
        # (raw cumulative values, timestamped) is unaffected and still shows
        # exactly where the reset happened.
        reset_detected = d_packets < 0 or d_retries < 0 or d_failed < 0
        totals.update({
            "tx_packets_delta": d_packets,
            "tx_retries_delta": d_retries,
            "tx_failed_delta": d_failed,
            "retry_rate": round(d_retries / d_packets, 6) if d_packets > 0 and not reset_detected else None,
            "counter_reset_suspected": reset_detected,
        })
        if reset_detected:
            logger.warning(
                "Negative delta detected (tx_packets/retries/failed) -- the link likely "
                "dropped and reassociated mid-run, resetting its counters. Deltas/rate "
                "above are not meaningful; check the per-tick CSV for the actual reset point."
            )
    else:
        logger.warning("Could not read start/end station stats -- rate totals omitted.")

    state.last_totals = totals
    if csv_dir and csv_path:
        try:
            base = csv_path[:-4] if csv_path.lower().endswith(".csv") else csv_path
            with open(f"{base}_totals.json", "w") as tf:
                json.dump(totals, tf, indent=2)
        except Exception as e:
            logger.error(f"Failed to write totals JSON: {e}")

    logger.info("Monitor thread stopped.")

# ---------- API lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.running:
        state.stop_event.set()
        if state.thread:
            state.thread.join()

app = FastAPI(title="WiFi Stats Collector", lifespan=lifespan)

# ---------- Endpoints ----------
@app.get("/health", response_model=HealthResponse)
def health_check():
    iw_ok = False
    try:
        iw_ok = subprocess.run(["iw", "--version"], capture_output=True, timeout=2.0).returncode == 0
    except Exception:
        pass
    return HealthResponse(status="ok", iw_available=iw_ok)

@app.post("/start")
def start(req: StartRequest):
    # Whole body under one lock: /start is a check-then-act on state.running,
    # and FastAPI dispatches sync endpoints to a threadpool, so two
    # near-simultaneous requests could otherwise both pass the check before
    # either sets running=True -- two monitor threads racing on one CSV
    # file. Serializing here costs nothing for this low-frequency admin API.
    with _lock:
        if state.running:
            raise HTTPException(status_code=409, detail="WiFi monitor already running")
        if req.interval <= 0:
            raise HTTPException(status_code=400, detail="interval must be > 0")

        mac = _detect_ap_mac(req.wifi_interface)
        if not mac:
            raise HTTPException(
                status_code=400,
                detail=f"'{req.wifi_interface}' is not associated to an AP (or iw/interface not found) "
                       "-- cannot auto-detect peer MAC."
            )

        run_idx = 1
        if req.csv_dir:
            if not os.path.exists(req.csv_dir):
                raise HTTPException(status_code=400, detail="csv_dir does not exist")
            run_idx = get_next_run_index(req.csv_dir, req.tag)

        state.stop_event.clear()
        state.last_totals = None
        state.thread = threading.Thread(
            target=monitor_loop,
            args=(req.interval, req.csv_dir, req.tag, run_idx, req.wifi_interface, req.stdout, req.csv_name, mac),
            daemon=True
        )
        state.thread.start()
        state.running = True
        state.interface = req.wifi_interface
        state.mac = mac

        # Mirror monitor_loop's own path logic exactly (not a re-derivation)
        # so what's reported here can never drift from what actually gets
        # written -- e.g. an absolute csv_name or incidental whitespace
        # previously could have been reported differently than resolved.
        if req.csv_dir:
            if req.csv_name:
                reported_path = _resolve_csv_path(req.csv_dir, req.csv_name)
            else:
                safe_tag = _sanitize_tag(req.tag)
                reported_path = os.path.join(req.csv_dir.rstrip("/"), f"{safe_tag}-run{run_idx}_wifi.csv")
        else:
            reported_path = None

        return {
            "success": True,
            "interface": req.wifi_interface,
            "mac": mac,
            "file": reported_path if reported_path else "None (No Directory)",
        }

@app.post("/stop")
def stop():
    with _lock:
        if not state.running:
            return {"success": False, "message": "Not running"}
        state.stop_event.set()
        if state.thread:
            state.thread.join(timeout=5.0)
        state.running = False
        state.interface = None
        state.mac = None
        return {"success": True, "totals": state.last_totals}

@app.get("/status", response_model=StatusResponse)
def status():
    # Under the lock so a /status call can never observe a half-updated
    # state (running=True but interface/mac still from the previous run,
    # or vice versa) mid-transition in /start or /stop.
    with _lock:
        return {"is_running": state.running, "interface": state.interface, "mac": state.mac}
