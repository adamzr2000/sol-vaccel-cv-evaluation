#!/usr/bin/env python3
"""
check_agent_profile.py -- test helper: per-call edge computation from a vAccel
agent profile log (run_agent_*.sh --profile LOG) for one benchmark run, next to
the robot-side call time of the same run.

The agent prints, after every call, cumulative totals per profiling region:
  <date>-<time> - <info> [prof] sol_predict: total_time: <ns> nsec nr_entries: <n>
Per-call time = difference of consecutive totals of a region. The run's agent
session is the one overlapping its time window (benchmark_summary.json, UTC; the
agent logs in UTC via TZ=UTC); its last num_samples calls are the measured ones
(warm-up/primer calls come first).

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

PROF = re.compile(r"\[prof\] (\S+): total_time: (\d+) nsec nr_entries: (\d+)")
STAMP = re.compile(r"(\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}\.\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SESSION = re.compile(r"Created session \d+")


def per_call(log: Path, region: str, start: float, stop: float, n_samples: int) -> list:
    """{sample index: ms} of `region` for the run's measured calls: the last n_samples calls of the
    agent session overlapping [start, stop]. Garbled/truncated log entries are dropped."""
    sessions, cur, t = [], {}, None
    for raw in log.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if (st := STAMP.search(line)):
            t = datetime.strptime(st.group(1), "%Y.%m.%d-%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        if SESSION.search(line):
            sessions.append(cur)
            cur = {}
            continue
        for m in PROF.finditer(line):
            if m.group(1) != region:
                continue
            total, n = int(m.group(2)), int(m.group(3))
            last = max(cur) if cur else 0
            if n > last and (not cur or total > cur[last][1]):   # else garbled/truncated: drop
                cur[n] = (t, total)
    sessions.append(cur)
    hit = [s for s in sessions
           if any(v[0] is not None and start - 1 <= v[0] <= stop + 1 for v in s.values())]
    if len(hit) != 1:
        print(f"[!] {len(hit)} agent sessions overlap the run window (expected 1)")
        if not hit:
            return {}
    calls = hit[-1]
    last = max(calls)
    first = last - n_samples + 1
    if calls[last][0] < stop - 0.1:   # tail of the session missing from the log: select by time instead
        print(f"[!] last logged call {stop - calls[last][0]:.2f} s before the window stop (log tail lost); "
              f"selecting the calls logged inside the window instead of the last {n_samples}")
        inside = [n for n, v in calls.items() if start <= v[0] <= stop + 0.05]
        first, last = min(inside), max(inside)
    out = {n - first: (calls[n][1] - calls[n - 1][1]) / 1e6 if n > 1 else calls[n][1] / 1e6
           for n in range(first, last + 1) if n in calls and (n == 1 or n - 1 in calls)}
    print(f"  agent session: {max(calls)} calls ({first - 1} warm-up/primer), {n_samples - len(out)} "
          f"measured call(s) without a complete log entry skipped; last call "
          f"{calls[last][0] - stop:+.3f} s vs robot window stop (clock offset + reply time)")
    return out   # {sample index 0..n_samples-1 (= robot benchmark_data.csv row): duration ms}


def edge_calls(log: Path, regions: str, start: float, stop: float, n_samples: int) -> dict:
    """{sample index: ms} of a region, or of a '+'-joined sum of regions (calls present in all)."""
    parts = [per_call(log, r, start, stop, n_samples) for r in regions.split("+")]
    common = set.intersection(*(set(p) for p in parts))
    return {i: sum(p[i] for p in parts) for i in sorted(common)}


def edge_times(log: Path, regions: str, start: float, stop: float, n_samples: int) -> list:
    """Per-call durations (ms), in sample order."""
    return list(edge_calls(log, regions, start, stop, n_samples).values())


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--region", default="sol_predict",
                    help="region, or regions summed per call joined by '+'. SOL: sol_predict (input buffer -> "
                         "output buffer, incl. GPU sync). PyTorch with the eval-patched plugin: "
                         "torch_predict+torch_out (same scope). The stock plugin's torch_run has no GPU sync.")
    a = ap.parse_args()

    summary = json.loads((a.run_dir / "benchmark_summary.json").read_text())
    w = summary["time_window"]
    start, stop = (datetime.fromisoformat(w[k]).timestamp() for k in ("start", "stop"))
    with open(a.run_dir / "benchmark_data.csv", newline="") as f:
        call = [float(r["inference_ms"]) for r in csv.DictReader(f)]
    edge = edge_times(a.log, a.region, start, stop, len(call))

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
    if len(edge) < 0.95 * len(call):
        print(f"  [!] {len(edge)} agent calls vs {len(call)} robot calls in the window -- "
              f"window/clock mismatch or extra calls; check before trusting the numbers")


if __name__ == "__main__":
    main()
