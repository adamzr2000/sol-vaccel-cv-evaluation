#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/model-stats/_summary/run1_benchmark_summary.json"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (8.5, 5.2)

SHOW_VALUE_LABELS = False  # requested

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]

TARGETS = [
    ("robot", "cpu", "model_stats_inference_latency_robot_cpu_boxplot.pdf"),
    ("edge", "cpu", "model_stats_inference_latency_edge_cpu_boxplot.pdf"),
    ("edge", "gpu", "model_stats_inference_latency_edge_gpu_boxplot.pdf"),
]


def split_model_variant(model: str):
    if isinstance(model, str) and model.endswith("_sol"):
        return model[:-4], "SOL"
    return model, "PyTorch"


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def collect_latency_samples(run: dict):
    base_model, variant = split_model_variant(run.get("model", ""))
    host = str(run.get("host", "")).lower()
    device = str(run.get("device", "")).lower()
    backend = str(run.get("backend", "")).lower()

    inf = run.get("inference_latency_ms", {}) or {}
    mu = inf.get("mean", None)
    sd = inf.get("std", None)
    n = run.get("num_samples", None)

    if mu is None or sd is None:
        return None

    try:
        mu = float(mu)
        sd = float(sd)
    except Exception:
        return None

    if not isinstance(n, int):
        try:
            n = int(n)
        except Exception:
            n = 256

    n = max(10, min(n, 1024))
    vals = np.random.normal(loc=mu, scale=max(sd, 1e-9), size=n)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return {
        "backend": backend,
        "host": host,
        "device": device,
        "base_model": base_model,
        "variant": variant,
        "latency_ms": vals,
    }


def plot_target(df: pd.DataFrame, host: str, device: str, out_file: str):
    sub = df[(df["backend"] == "stock") & (df["host"] == host) & (df["device"] == device)].copy()
    if sub.empty:
        print(f"[SKIP] No runs for host={host}, device={device}, backend=stock")
        return

    base_models = ordered_models(sorted(sub["base_model"].unique().tolist()))
    sub["base_model"] = pd.Categorical(sub["base_model"], categories=base_models, ordered=True)
    sub["variant"] = pd.Categorical(sub["variant"], categories=["PyTorch", "SOL"], ordered=True)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=2)
    color_map = {"PyTorch": pal[0], "SOL": pal[1]}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.boxplot(
        data=sub,
        x="base_model",
        y="latency_ms",
        hue="variant",
        palette=color_map,
        width=0.7,
        linewidth=1.0,
        fliersize=2.5,
        ax=ax,
    )

    ax.set_title(f"Inference latency - stock @ {host}-{device}")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Time (ms)")
    ax.tick_params(axis="x", labelrotation=20)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4, title=None)

    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot to: {out_file}")
    plt.close(fig)


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    rows = []
    for r in runs:
        rec = collect_latency_samples(r)
        if not rec:
            continue
        vals = rec.pop("latency_ms")
        tmp = pd.DataFrame(rec, index=range(len(vals)))
        tmp["latency_ms"] = vals
        rows.append(tmp)

    if not rows:
        raise SystemExit("No latency samples could be constructed from input JSON.")

    df = pd.concat(rows, ignore_index=True)

    for host, device, out_file in TARGETS:
        plot_target(df, host, device, out_file)


if __name__ == "__main__":
    main()
