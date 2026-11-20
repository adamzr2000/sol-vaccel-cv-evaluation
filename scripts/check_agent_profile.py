#!/usr/bin/env python3
"""
check_agent_profile.py -- test helper: per-call edge computation from a vAccel
agent profile log (run_agent_*.sh --profile LOG) for one benchmark run, next to
the robot-side call time of the same run.

The agent prints, after every call, cumulative totals per profiling region:
  <date>-<time> - <info> [prof] sol_predict: total_time: <ns> nsec nr_entries: <n>
Per-call time = difference of consecutive totals of a region. Only lines inside
the run's time window (benchmark_summary.json, UTC; the agent logs in UTC via
TZ=UTC) are used, which excludes the warm-up calls.

Usage:
  python3 check_agent_profile.py AGENT_LOG RUN_DIR [--region sol_predict]
    RUN_DIR = folder with benchmark_summary.json and benchmark_data.csv
"""
import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}\.\d+).*?\[prof\] (\S+): total_time: (\d+) nsec "
                  r"nr_entries: (\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def per_call(log: Path, region: str, start: float, stop: float, slack: float = 0.05) -> list:
    """Per-call durations (ms) of `region` whose log line falls inside [start, stop]."""
    prev, out = None, []
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.match(ANSI.sub("", raw))
        if not m or m.group(2) != region:
            continue
        t = datetime.strptime(m.group(1), "%Y.%m.%d-%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        total, n = int(m.group(3)), int(m.group(4))
        if prev and n == prev[1] + 1 and start - slack <= t <= stop + slack:
            out.append((total - prev[0]) / 1e6)
        prev = (total, n)
    return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--region", default="sol_predict",
                    help="sol_predict (SOL; includes the GPU sync). PyTorch plugin regions are torch_input / "
                         "torch_run / torch_out, but torch_run has no CUDA sync and the D2H copy is unprofiled, "
                         "so they do not give the complete GPU computation")
    a = ap.parse_args()

    summary = json.loads((a.run_dir / "benchmark_summary.json").read_text())
    w = summary["time_window"]
    start, stop = (datetime.fromisoformat(w[k]).timestamp() for k in ("start", "stop"))
    with open(a.run_dir / "benchmark_data.csv", newline="") as f:
        call = [float(r["inference_ms"]) for r in csv.DictReader(f)]
    edge = per_call(a.log, a.region, start, stop)

    print(f"run {summary['run_id']}  ({summary['num_samples']} inferences, window {w['duration_sec']:.1f} s)")
    if not edge:
        print(f"[!] no '{a.region}' lines inside the window -- check: agent started with --profile, "
              f"log is from this run, clocks/time zone (agent must log UTC)")
        return
    rows = [("robot: call (send -> output)", call), (f"edge:  {a.region} (agent)", edge)]
    for name, v in rows:
        print(f"  {name:32} n={len(v):5}  median {statistics.median(v):8.2f} ms  "
              f"p25 {pct(v, 25):8.2f}  p75 {pct(v, 75):8.2f}")
    print(f"  => communication = call - computation (medians): "
          f"{statistics.median(call) - statistics.median(edge):.2f} ms")
    if len(edge) != len(call):
        print(f"  [!] {len(edge)} agent calls vs {len(call)} robot calls in the window -- "
              f"window/clock mismatch or extra calls; check before trusting the numbers")


if __name__ == "__main__":
    main()
