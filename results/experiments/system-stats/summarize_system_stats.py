#!/usr/bin/env python3
"""
summarize_system_stats.py

Summarize system-stats-collector CSVs for a given RUN_TAG.
Recursively finds CSVs in subfolders (e.g., robot/, edge-asus/).
Uses the PARENT DIRECTORY NAME as the 'host' identifier.

Outputs (written under ./_summary):
  - {run_tag}_overall_cpu_stats_{link}.csv
  - {run_tag}_overall_gpu_stats_{link}.csv
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
    "host", "model", "backend", "device",
    "cpu_watts_mean", "cpu_watts_std",
    "cpu_util_percent_mean", "cpu_util_percent_std",
    "cpu_temp_c_mean", "cpu_temp_c_std",
    "duration_sec", "num_samples",
]

GPU_OUT_FIELDS = [
    "host", "model", "backend", "device",
    "gpu_count", "gpu_names",
    "power_draw_w_mean", "power_draw_w_std",
    "util_gpu_percent_mean", "util_gpu_percent_std",
    "util_mem_percent_mean", "util_mem_percent_std",
    "mem_used_mb_mean", "mem_used_mb_std",
    "temp_c_mean", "temp_c_std",
    "duration_sec", "num_samples",
    "gpu_energy_j",
]


def fmt(x: Optional[float], decimals: int = 6) -> str:
    return "" if x is None else f"{x:.{decimals}f}"


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def parse_stem_and_folder(path: Path, run_tag: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse metadata from filename, BUT take 'host' from the parent folder name.
    """
    stem = path.stem
    needle = f"{run_tag}_"
    if not stem.startswith(needle):
        return None

    host = path.parent.name

    remainder = stem[len(needle):]
    parts = remainder.split("_")

    if len(parts) < 4:
        return None

    # vaccel-remote extended naming
    # Pattern: ..._{BACKEND}_{OLD_HOST}_{LOCAL_MODE}_target-{TARGET}
    if len(parts) >= 5 and parts[-1].startswith("target-"):
        target = parts[-1]
        local_mode = parts[-2]
        backend = parts[-4]
        model = "_".join(parts[:-4])
        device = f"{local_mode}_{target}"
    else:
        device = parts[-1]
        backend = parts[-3]
        model = "_".join(parts[:-3])

    return model, backend, host, device


def discover_run_tags(cwd: Path) -> List[str]:
    tags = set()
    for p in cwd.rglob("*.csv"):
        if "_summary" in p.parts:
            continue
        parts = p.stem.split("_")
        if len(parts) >= 2:
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


def _print_summary_table(csv_path: Path, title: str) -> None:
    try:
        df = pd.read_csv(csv_path)
        cols = [
            "host", "model", "backend", "device",
            "cpu_watts_mean", "cpu_util_percent_mean",
            "power_draw_w_mean", "util_gpu_percent_mean", "gpu_energy_j",
            "duration_sec"
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(["host", "device", "model"]).reset_index(drop=True)

        print(f"\n===== {title} (sanity check) =====")
        print(df.to_string(index=False))
        print("=====================================\n")
    except Exception as e:
        print(f"⚠️  Could not print summary table: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-tag")
    # NEW: which extra vaccel-remote family to include
    ap.add_argument("--link", choices=["5g", "wifi"], default="5g")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    cwd = Path(".").resolve()

    csv_files = sorted(
        p for p in cwd.rglob("*.csv")
        if "failed" not in p.parts and "_summary" not in p.parts
    )

    if args.help:
        print("Usage:")
        print("  ./summarize_system_stats.py --run-tag <tag> [--link 5g|wifi]")
        return

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs (scanned recursively):\n")
        tags = discover_run_tags(cwd)
        if not tags:
            print("  (none found)")
        else:
            for t in tags:
                print(f"  - {t}")
        return

    run_tag = args.run_tag.strip()
    link = args.link.strip()

    # 1) ORIGINAL behavior: include run_tag_*
    base_matched = [
        p for p in csv_files
        if "_summary" not in p.parts and p.name.startswith(f"{run_tag}_")
    ]

    # 2) EXTENSION: include ONLY vaccel-remote in run_tag-{link}_*
    extra_matched = [
        p for p in csv_files
        if "_summary" not in p.parts
        and p.name.startswith(f"{run_tag}-{link}_")
        and "vaccel-remote" in p.stem
    ]

    # Merge + dedup preserving order
    seen = set()
    matched = []
    for p in base_matched + extra_matched:
        if p not in seen:
            matched.append(p)
            seen.add(p)

    if not matched:
        print(f"❌ No CSV files matched RUN_TAG='{run_tag}'")
        return

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)

    # NEW: output naming includes _5g or _wifi
    cpu_out_path = out_dir / f"{run_tag}_overall_cpu_stats_{link}.csv"
    gpu_out_path = out_dir / f"{run_tag}_overall_gpu_stats_{link}.csv"

    cpu_rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    gpu_rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    print(
        f"🔍 Found {len(matched)} files "
        f"(base={len(base_matched)}, extra_vaccel_remote_{link}={len(extra_matched)}). Processing..."
    )

    for csv_path in matched:
        # Parse metadata:
        # - base files: parse with run_tag
        # - extra files: parse with run_tag-link (e.g. run1-wifi)
        parsed = parse_stem_and_folder(csv_path, run_tag)
        if not parsed:
            parsed = parse_stem_and_folder(csv_path, f"{run_tag}-{link}")
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

            row = {
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
                if metrics["power_draw_w"]:
                    p_mean = float(np.mean(np.asarray(metrics["power_draw_w"], dtype=float)))
            except Exception:
                pass

            gpu_energy_j = None
            if p_mean is not None and d_sec is not None:
                gpu_energy_j = p_mean * d_sec

            row["gpu_energy_j"] = fmt(gpu_energy_j, 3)

            key = (host, model, backend, device)
            gpu_rows[key] = row

    # --- Write CPU ---
    if cpu_rows:
        with cpu_out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CPU_OUT_FIELDS)
            writer.writeheader()
            writer.writerows(cpu_rows.values())
        print(f"📄 CPU summary written to: {cpu_out_path} (rows={len(cpu_rows)})")
        _print_summary_table(cpu_out_path, "CPU SUMMARY")
    else:
        print("⚠️  No CPU CSVs found/matched.")

    # --- Write GPU ---
    if gpu_rows:
        with gpu_out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GPU_OUT_FIELDS)
            writer.writeheader()
            writer.writerows(gpu_rows.values())
        print(f"📄 GPU summary written to: {gpu_out_path} (rows={len(gpu_rows)})")
        _print_summary_table(gpu_out_path, "GPU SUMMARY")
    else:
        print("⚠️  No GPU CSVs found/matched.")

    print("✅ Done.")


if __name__ == "__main__":
    main()