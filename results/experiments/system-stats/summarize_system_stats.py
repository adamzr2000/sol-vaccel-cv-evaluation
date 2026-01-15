#!/usr/bin/env python3
"""
summarize_system_stats.py

Summarize system-stats-collector CSVs for a given RUN_TAG.

Expected filename pattern (NO .csv):
  {RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{DEVICE}

Notes:
- model may contain underscores and *_sol
- backend/host/device are assumed to be the last 3 tokens

Outputs (written under ./_summary):
  - {run_tag}_overall_cpu_stats.csv
  - {run_tag}_overall_gpu_stats.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CPU_METRICS = ["cpu_watts", "cpu_util_percent", "cpu_temp_c"]
GPU_METRICS = ["power_draw_w", "util_gpu_percent", "util_mem_percent", "mem_used_mb", "temp_c"]


CPU_OUT_FIELDS = [
    "host",
    "model",
    "backend",
    "device",
    "cpu_watts_mean",
    "cpu_watts_std",
    "cpu_util_percent_mean",
    "cpu_util_percent_std",
    "cpu_temp_c_mean",
    "cpu_temp_c_std",
    "duration_sec",
    "num_samples",
]

GPU_OUT_FIELDS = [
    "host",
    "model",
    "backend",
    "device",
    "gpu_count",
    "gpu_names",
    "power_draw_w_mean",
    "power_draw_w_std",
    "util_gpu_percent_mean",
    "util_gpu_percent_std",
    "util_mem_percent_mean",
    "util_mem_percent_std",
    "mem_used_mb_mean",
    "mem_used_mb_std",
    "temp_c_mean",
    "temp_c_std",
    "duration_sec",
    "num_samples",
    "gpu_energy_j",
]


def fmt(x: Optional[float], decimals: int = 6) -> str:
    return "" if x is None else f"{x:.{decimals}f}"


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def parse_stem(stem: str, run_tag: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse:
      stem = "{RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{DEVICE}"

    Return:
      (model, backend, host, device)
    """
    needle = f"{run_tag}_"
    if not stem.startswith(needle):
        return None

    remainder = stem[len(needle):]  # "{MODEL}_{BACKEND}_{HOST}_{DEVICE}"
    parts = remainder.split("_")
    if len(parts) < 4:
        return None

    device = parts[-1]
    host = parts[-2]
    backend = parts[-3]
    model = "_".join(parts[:-3])
    return model, backend, host, device


def discover_run_tags(csv_files: List[Path]) -> List[str]:
    tags = set()
    for p in csv_files:
        parts = p.stem.split("_")
        if len(parts) >= 5:
            tags.add(parts[0])
    return sorted(tags)


def _safe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def read_duration_and_metrics(
    csv_path: Path, metric_cols: List[str]
) -> Tuple[Dict[str, List[float]], Optional[float], int]:
    values = {k: [] for k in metric_cols}
    t_min: Optional[int] = None
    t_max: Optional[int] = None
    n = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            ts = row.get("timestamp")
            if ts:
                try:
                    ts_i = int(float(ts))
                    t_min = ts_i if t_min is None else min(t_min, ts_i)
                    t_max = ts_i if t_max is None else max(t_max, ts_i)
                except Exception:
                    pass

            for k in metric_cols:
                v = _safe_float(row.get(k, ""))
                if v is not None:
                    values[k].append(v)

    duration_sec = None
    if t_min is not None and t_max is not None and t_max >= t_min:
        duration_sec = (t_max - t_min) / 1000.0

    return values, duration_sec, n


def read_gpu_id_info(csv_path: Path) -> Tuple[int, str]:
    gpu_ids = set()
    names = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gi = row.get("gpu_index")
            gn = row.get("gpu_name")
            if gi is None:
                continue
            gi_s = str(gi).strip()
            if not gi_s:
                continue
            gpu_ids.add(gi_s)
            if gn and gi_s not in names:
                names[gi_s] = str(gn).strip()

    gpu_count = len(gpu_ids)

    def _k(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    ordered = sorted(gpu_ids, key=_k)
    joined_names = "; ".join([names.get(i, "") for i in ordered if names.get(i, "")])
    return gpu_count, joined_names


def _print_cpu_summary_table(csv_path: Path) -> None:
    try:
        df = pd.read_csv(csv_path)
        cols = [
            "host", "model", "backend", "device",
            "cpu_watts_mean", "cpu_watts_std",
            "cpu_util_percent_mean", "cpu_util_percent_std",
            "cpu_temp_c_mean", "cpu_temp_c_std",
            "duration_sec", "num_samples",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(["host", "device", "model"]).reset_index(drop=True)

        print("\n===== CPU SUMMARY (sanity check) =====")
        print(df.to_string(index=False))
        print("=====================================\n")
    except Exception as e:
        print(f"⚠️  Could not print CPU summary table: {e}")


def _print_gpu_summary_table(csv_path: Path) -> None:
    try:
        df = pd.read_csv(csv_path)
        cols = [
            "host", "model", "backend", "device",
            "gpu_count", "gpu_names",
            "power_draw_w_mean", "power_draw_w_std",
            "util_gpu_percent_mean", "util_gpu_percent_std",
            "mem_used_mb_mean", "mem_used_mb_std",
            "temp_c_mean", "temp_c_std",
            "duration_sec", "num_samples",
            "gpu_energy_j",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(["host", "device", "model"]).reset_index(drop=True)

        print("\n===== GPU SUMMARY (sanity check) =====")
        print(df.to_string(index=False))
        print("=====================================\n")
    except Exception as e:
        print(f"⚠️  Could not print GPU summary table: {e}")


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
        print("   python3 summarize_system_stats.py --run-tag <RUN_TAG>")
        return

    run_tag = args.run_tag.strip()
    matched = [p for p in csv_files if p.stem.startswith(f"{run_tag}_")]
    if not matched:
        print(f"❌ No CSV files matched RUN_TAG='{run_tag}'")
        return

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)

    cpu_out_path = out_dir / f"{run_tag}_overall_cpu_stats.csv"
    gpu_out_path = out_dir / f"{run_tag}_overall_gpu_stats.csv"

    cpu_rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    gpu_rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    for csv_path in matched:
        parsed = parse_stem(csv_path.stem, run_tag)
        if not parsed:
            continue
        model, backend, host, device = parsed

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []

        is_cpu = "cpu_watts" in header or "cpu_util_percent" in header
        is_gpu = "power_draw_w" in header or "util_gpu_percent" in header

        if is_cpu:
            metrics, duration_sec, n = read_duration_and_metrics(csv_path, CPU_METRICS)
            row: Dict[str, str] = {
                "host": host,
                "model": model,
                "backend": backend,
                "device": device,
            }
            for k in CPU_METRICS:
                mu, sd = mean_std(metrics[k])
                row[f"{k}_mean"] = fmt(mu, 6)
                row[f"{k}_std"] = fmt(sd, 6)
            row["duration_sec"] = fmt(duration_sec, 3)
            row["num_samples"] = str(n)

            key = (host, model, backend, device)
            cpu_rows[key] = row

        elif is_gpu:
            metrics, duration_sec, n = read_duration_and_metrics(csv_path, GPU_METRICS)
            gpu_count, gpu_names = read_gpu_id_info(csv_path)

            row: Dict[str, str] = {
                "host": host,
                "model": model,
                "backend": backend,
                "device": device,
                "gpu_count": str(gpu_count),
                "gpu_names": gpu_names,
            }
            for k in GPU_METRICS:
                mu, sd = mean_std(metrics[k])
                row[f"{k}_mean"] = fmt(mu, 6)
                row[f"{k}_std"] = fmt(sd, 6)

            row["duration_sec"] = fmt(duration_sec, 3)
            row["num_samples"] = str(n)

            p_mean = None
            d_sec = duration_sec
            try:
                p_mean = float(np.mean(np.asarray(metrics["power_draw_w"], dtype=float))) if metrics["power_draw_w"] else None
            except Exception:
                p_mean = None

            gpu_energy_j = None
            if p_mean is not None and d_sec is not None:
                gpu_energy_j = p_mean * d_sec

            row["gpu_energy_j"] = fmt(gpu_energy_j, 3)

            key = (host, model, backend, device)
            gpu_rows[key] = row

    if cpu_rows:
        with cpu_out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CPU_OUT_FIELDS)
            writer.writeheader()
            writer.writerows(cpu_rows.values())
        print(f"📄 CPU summary written to: {cpu_out_path} (rows={len(cpu_rows)})")
        _print_cpu_summary_table(cpu_out_path)
    else:
        print("⚠️  No CPU CSVs found/matched to summarize.")

    if gpu_rows:
        with gpu_out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GPU_OUT_FIELDS)
            writer.writeheader()
            writer.writerows(gpu_rows.values())
        print(f"📄 GPU summary written to: {gpu_out_path} (rows={len(gpu_rows)})")
        _print_gpu_summary_table(gpu_out_path)
    else:
        print("⚠️  No GPU CSVs found/matched to summarize.")

    print("✅ Done.")


if __name__ == "__main__":
    main()
