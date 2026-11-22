#!/usr/bin/env python3
"""
agent_server.py -- starts a fresh, profiled vAccel agent per model run (edge side).

  POST /start  {"backend": "sol"|"ptc", "device": "cpu"|"gpu", "run_id": "..."}
  POST /stop   body = the run's benchmark_summary.json

Writes results/experiments/agent-profile/<host>/<run_id>.log (raw agent log) and
<run_id>.csv (sample, computation_ms: measured inferences only, warm-up excluded;
sample i = row i of the robot's benchmark_data.csv).

Usage:  python3 agent_server.py [--port 9130] [--host edge-asus]
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
from check_agent_profile import edge_calls  # noqa: E402

CONTAINER = "torchvision-app-agent"  # name used by both run_agent_*.sh
OUT_DIR = HERE / "results" / "experiments" / "agent-profile" / "edge-asus"
REGION = {"sol": "sol_predict", "ptc": "torch_predict+torch_out"}
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
READY_TIMEOUT_S = 120

state = {"proc": None, "run_id": None, "backend": None}


def stop_agent() -> None:
    subprocess.run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    if state["proc"] is not None:
        try:
            state["proc"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            state["proc"].kill()
    state["proc"] = None


def start(req: dict) -> dict:
    backend, device, run_id = req.get("backend"), req.get("device"), req.get("run_id", "")
    if backend not in REGION or device not in ("cpu", "gpu") or not RUN_ID.match(run_id):
        return {"ok": False, "error": f"bad request {req}"}
    stop_agent()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f"{run_id}.log"
    offset = log.stat().st_size if log.exists() else 0
    state["proc"] = subprocess.Popen([str(HERE / f"run_agent_{backend}.sh"), device, "--profile", str(log)],
                                     cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
    state.update(run_id=run_id, backend=backend)
    t0 = time.time()
    while time.time() - t0 < READY_TIMEOUT_S:
        if state["proc"].poll() is not None:
            return {"ok": False, "error": "agent exited during start-up", "log": str(log)}
        if log.exists():
            with open(log, errors="replace") as f:
                f.seek(offset)
                if "Listening on" in f.read():
                    return {"ok": True, "log": str(log), "startup_s": round(time.time() - t0, 1),
                            "load_1min": round(os.getloadavg()[0], 2)}
        time.sleep(0.5)
    stop_agent()
    return {"ok": False, "error": f"agent not listening after {READY_TIMEOUT_S} s", "log": str(log)}


def stop(summary: dict) -> dict:
    run_id, backend = state["run_id"], state["backend"]
    stop_agent()
    if run_id is None:
        return {"ok": True, "note": "no agent was running"}
    state.update(run_id=None, backend=None)
    if summary.get("run_id") not in (None, run_id):
        return {"ok": False, "error": f"summary is for {summary.get('run_id')}, agent ran {run_id}"}
    n, window = summary.get("num_samples"), summary.get("time_window")
    if not n or not window:
        return {"ok": True, "run_id": run_id, "note": "no summary (run failed?) -- log kept, no csv"}
    t0, t1 = (datetime.fromisoformat(window[k]).timestamp() for k in ("start", "stop"))
    calls = edge_calls(OUT_DIR / f"{run_id}.log", REGION[backend], t0, t1, int(n))
    if not calls:
        return {"ok": False, "run_id": run_id, "error": "no measured calls in the log for this run's window"}
    out = OUT_DIR / f"{run_id}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "computation_ms"])
        w.writerows((i, f"{ms:.4f}") for i, ms in calls.items())
    return {"ok": True, "run_id": run_id, "region": REGION[backend], "csv": str(out), "calls": len(calls),
            "expected": int(n), "median_ms": round(statistics.median(calls.values()), 3) if calls else None}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            routes = {"/start": start, "/stop": stop}
            res = routes[self.path](body) if self.path in routes else {"ok": False, "error": "unknown path"}
        except Exception as e:
            res = {"ok": False, "error": repr(e)}
        print(f"[agent-server] {self.path} -> {res}", flush=True)
        data = json.dumps(res).encode()
        self.send_response(200 if res.get("ok") else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="vAccel agent control server (edge).")
    ap.add_argument("--port", type=int, default=9130)
    ap.add_argument("--host", default="edge-asus", help="host name used for the output folder")
    a = ap.parse_args()
    port, OUT_DIR = a.port, OUT_DIR.parent / a.host
    print(f"[agent-server] listening on 0.0.0.0:{port}, output in {OUT_DIR}", flush=True)
    try:
        HTTPServer(("0.0.0.0", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_agent()
