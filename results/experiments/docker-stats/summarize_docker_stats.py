#!/usr/bin/env python3
"""
summarize_docker_stats.py

Summarize docker-stats-collector CSVs for a given RUN_TAG.

Output:
  ./_summary/{run_tag}_overall_resource_usage_per_container.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

METRIC_COLS = [
    "cpu_percent",
    "mem_mb",
    "blk_read_mb",
    "blk_write_mb",
]

OUT_FIELDS = [
    "container",
    "host",
    "device",
    "model",
    "backend",
    "cpu_percent_mean",
    "cpu_percent_std",
    "mem_mb_mean",
    "mem_mb_std",
    "blk_read_mb_mean",
    "blk_read_mb_std",
    "blk_write_mb_mean",
    "blk_write_mb_std",
    "net_rx_mbps",
    "net_tx_mbps",
    "duration_sec",
]


def fmt(x: Optional[float], decimals: int = 6) -> str:
    return "" if x is None else f"{x:.{decimals}f}"


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))  # ddof=0 matches numpy default


def parse_stem(stem: str, run_tag: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Parse:
      stem = "{container}_{RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{DEVICE}"

    Extended (vaccel-remote robot-side disambiguation):
      stem = "{container}_{RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{LOCAL_MODE}_target-{TARGET_DEVICE}"

    Return:
      (container, model, backend, host, device)
    """
    needle = f"_{run_tag}_"
    idx = stem.find(needle)
    if idx < 0:
        return None

    container = stem[:idx]
    remainder = stem[idx + len(needle):]
    parts = remainder.split("_")
    if len(parts) < 4:
        return None

    # Extended pattern:
    # ..._{BACKEND}_{HOST}_{LOCAL_MODE}_target-{TARGET_DEVICE}
    if len(parts) >= 5 and parts[-1].startswith("target-"):
        target = parts[-1]          # "target-gpu"
        local_mode = parts[-2]      # "cpu"
        host = parts[-3]            # "robot"
        backend = parts[-4]         # "vaccel-remote"
        model = "_".join(parts[:-4])
        device = f"{local_mode}_{target}"  # "cpu_target-gpu"
    else:
        device = parts[-1]
        host = parts[-2]
        backend = parts[-3]
        model = "_".join(parts[:-3])

    return container, model, backend, host, device

def discover_run_tags(csv_files: List[Path]) -> List[str]:
    tags = set()
    for p in csv_files:
        parts = p.stem.split("_")
        if len(parts) >= 6:
            tags.add(parts[1])
    return sorted(tags)


def read_metrics_and_duration(csv_path: Path) -> Tuple[Dict[str, List[float]], Optional[float], Optional[float], Optional[float]]:
    values = {k: [] for k in METRIC_COLS}
    t_min: Optional[int] = None
    t_max: Optional[int] = None

    net_rx_first: Optional[float] = None
    net_rx_last: Optional[float] = None
    net_tx_first: Optional[float] = None
    net_tx_last: Optional[float] = None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp")
            if ts:
                try:
                    ts_i = int(float(ts))
                    t_min = ts_i if t_min is None else min(t_min, ts_i)
                    t_max = ts_i if t_max is None else max(t_max, ts_i)
                except ValueError:
                    pass

            # collect mean/std metrics
            for k in METRIC_COLS:
                v = row.get(k)
                if v:
                    try:
                        values[k].append(float(v))
                    except ValueError:
                        pass

            # capture first/last cumulative net values (MB)
            rx = row.get("net_rx_mb")
            if rx:
                try:
                    rx_f = float(rx)
                    if net_rx_first is None:
                        net_rx_first = rx_f
                    net_rx_last = rx_f
                except ValueError:
                    pass

            tx = row.get("net_tx_mb")
            if tx:
                try:
                    tx_f = float(tx)
                    if net_tx_first is None:
                        net_tx_first = tx_f
                    net_tx_last = tx_f
                except ValueError:
                    pass

    duration_sec = None
    if t_min is not None and t_max is not None and t_max >= t_min:
        duration_sec = (t_max - t_min) / 1000.0

    net_rx_delta_mb = None
    if net_rx_first is not None and net_rx_last is not None:
        net_rx_delta_mb = max(0.0, net_rx_last - net_rx_first)

    net_tx_delta_mb = None
    if net_tx_first is not None and net_tx_last is not None:
        net_tx_delta_mb = max(0.0, net_tx_last - net_tx_first)

    return values, duration_sec, net_rx_delta_mb, net_tx_delta_mb



def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-tag")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    cwd = Path(".").resolve()
    csv_files = sorted(cwd.glob("*.csv"))

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs in current directory:\n")
        tags = discover_run_tags(csv_files)
        if not tags:
            print("  (none found)")
        else:
            for t in tags:
                print(f"  - {t}")
        print("\n👉 Usage:")
        print("   python3 summarize_docker_stats.py --run-tag <RUN_TAG>")
        return

    run_tag = args.run_tag
    matched = [p for p in csv_files if f"_{run_tag}_" in p.stem]
    if not matched:
        print(f"❌ No CSV files matched RUN_TAG='{run_tag}'")
        return

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_tag}_overall_resource_usage_per_container.csv"

    # include device in key to avoid overwriting cpu vs gpu runs
    rows: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}

    for csv_path in matched:
        parsed = parse_stem(csv_path.stem, run_tag)
        if not parsed:
            continue

        container, model, backend, host, device = parsed
        metrics, duration_sec, net_rx_delta_mb, net_tx_delta_mb = read_metrics_and_duration(csv_path)

        row: Dict[str, str] = {
            "container": container,
            "host": host,
            "device": device,
            "model": model,
            "backend": backend,
        }

        for k in METRIC_COLS:
            mu, sd = mean_std(metrics[k])
            row[f"{k}_mean"] = fmt(mu, 6)
            row[f"{k}_std"] = fmt(sd, 6)

        row["duration_sec"] = fmt(duration_sec, 3)

        net_rx_mbps = None
        net_tx_mbps = None
        if duration_sec and duration_sec > 0:
            if net_rx_delta_mb is not None:
                net_rx_mbps = (net_rx_delta_mb * 8.0) / duration_sec
            if net_tx_delta_mb is not None:
                net_tx_mbps = (net_tx_delta_mb * 8.0) / duration_sec

        row["net_rx_mbps"] = fmt(net_rx_mbps, 6)
        row["net_tx_mbps"] = fmt(net_tx_mbps, 6)

        key = (container, host, device, model, backend)
        rows[key] = row

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())

    print(f"✅ RUN_TAG '{run_tag}' summarized")
    print(f"📄 Output written to: {out_path}")
    print(f"📊 Containers summarized: {len(rows)}")


if __name__ == "__main__":
    main()
