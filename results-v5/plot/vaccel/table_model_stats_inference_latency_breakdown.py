#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import csv

from results.plot.plot_config import (
    get_path,
    load_config,
    get_model_type_order,
    get_model_display_name,   # <-- use display names
)

cfg = load_config()
INPUT_FILE = str(get_path("model_summary"))
OUTPUT_FILE = "latency_breakdown_table.csv"

MODEL_TYPE_ORDER = get_model_type_order()

# --- Short, paper-friendly labels ---
VARIANT_DEFINITIONS = [
    {"label": "Robot–CPU (TC)",  "match": {"host": "robot", "backend": "ptc", "device": "cpu"}},
    {"label": "Robot–CPU (SOL)", "match": {"host": "robot", "backend": "sol", "device": "cpu"}},

    {"label": "Edge–CPU (TC)",   "match": {"host": "robot"},
     "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-cpu"]},

    {"label": "Edge–CPU (SOL)",  "match": {"host": "robot"},
     "backend_contains": "vaccel-remote-sol", "run_id_contains": ["target-cpu"]},

    {"label": "Edge–GPU (TC)",   "match": {"host": "robot"},
     "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-gpu"]},

    {"label": "Edge–GPU (SOL)",  "match": {"host": "robot"},
     "backend_contains": "vaccel-remote-sol", "run_id_contains": ["target-gpu"]},
]
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def classify_variant(run: dict) -> str | None:
    run_id = str(run.get("run_id", "")).strip()
    backend = str(run.get("backend", "")).lower().strip()
    host = str(run.get("host", "")).lower().strip()
    device = str(run.get("device", "")).lower().strip()

    for v_def in VARIANT_DEFINITIONS:
        match_criteria = v_def.get("match", {})
        matches = True
        for key, val in match_criteria.items():
            if locals().get(key) != val:
                matches = False
                break
        if not matches:
            continue

        backend_sub = v_def.get("backend_contains")
        if backend_sub and backend_sub not in backend:
            continue

        substrings = v_def.get("run_id_contains", [])
        if substrings and not all(sub in run_id for sub in substrings):
            continue

        return v_def["label"]

    return None


def get_ms_stats(run: dict, key: str):
    d = run.get(key, {}) or {}
    mean = d.get("mean", None)
    std = d.get("std", None)

    if mean is None:
        return (np.nan, np.nan)

    try:
        mean_f = float(mean)
    except Exception:
        mean_f = np.nan

    try:
        std_f = float(std) if std is not None else np.nan
    except Exception:
        std_f = np.nan

    return (mean_f, std_f)


def fmt_pm(mean, std, decimals=1):
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]

    # Collect rows
    rows = []
    present_models = set()

    for r in runs:
        model = str(r.get("model", "")).strip()
        if model not in allowed_models:
            continue

        variant = classify_variant(r)
        if variant is None:
            continue

        sys_mu, sys_sd = get_ms_stats(r, "system_ms")
        if not np.isfinite(sys_mu):
            continue

        pre_mu, pre_sd = get_ms_stats(r, "preprocessing_ms")
        inf_mu, inf_sd = get_ms_stats(r, "inference_ms")
        post_mu, post_sd = get_ms_stats(r, "postprocessing_ms")

        rows.append((
            model, variant,
            sys_mu, sys_sd,
            pre_mu, pre_sd,
            inf_mu, inf_sd,
            post_mu, post_sd
        ))
        present_models.add(model)

    if not rows:
        raise SystemExit("No matching rows found (after MODEL_TYPE_ORDER + variant filtering).")

    base_models = ordered_models(sorted(present_models))

    # Fast lookup by (model, variant)
    row_map = {(m, v): vals for (m, v, *vals) in rows}

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Setting", "Total (ms)", "Pre (ms)", "Inference (ms)", "Post (ms)"])

        for m in base_models:
            for v in VARIANTS:
                vals = row_map.get((m, v))
                if vals is None:
                    continue

                (sys_mu, sys_sd,
                 pre_mu, pre_sd,
                 inf_mu, inf_sd,
                 post_mu, post_sd) = vals

                writer.writerow([
                    get_model_display_name(m),  # <-- paper-friendly display name
                    v,
                    fmt_pm(sys_mu, sys_sd),
                    fmt_pm(pre_mu, pre_sd),
                    fmt_pm(inf_mu, inf_sd),
                    fmt_pm(post_mu, post_sd),
                ])

    print(f"[OK] Exported paper-ready table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()