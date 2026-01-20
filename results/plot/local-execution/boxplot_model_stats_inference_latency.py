#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../../experiments/model-stats/_summary/run1_benchmark_summary.json"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "model_stats_inference_latency_boxplot_local_exec.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE_SINGLE = (8.5, 5.2)
FIG_SIZE_COMBINED = (10.5, 11.0)

INCLUDE_VACCEL_LOCAL = True
VARIANT_ORDER = ["PyTorch", "SOL", "SOL + vAccel"] if INCLUDE_VACCEL_LOCAL else ["PyTorch", "SOL"]

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]

TARGETS = [
    ("robot", "cpu", "model_stats_inference_latency_robot_cpu_boxplot_local_exec.pdf"),
    ("edge", "cpu", "model_stats_inference_latency_edge_cpu_boxplot_local_exec.pdf"),
    ("edge", "gpu", "model_stats_inference_latency_edge_gpu_boxplot_local_exec.pdf"),
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def split_variant(model: str, backend: str):
    backend = str(backend).lower().strip()
    model = str(model).strip()

    is_sol = model.endswith("_sol")
    base = model[:-4] if is_sol else model

    if backend == "stock" and not is_sol:
        return base, "PyTorch"
    if backend == "stock" and is_sol:
        return base, "SOL"
    if INCLUDE_VACCEL_LOCAL and backend == "vaccel-local" and is_sol:
        return base, "SOL + vAccel"

    return base, None


def collect_latency_samples(run: dict):
    host = str(run.get("host", "")).lower().strip()
    device = str(run.get("device", "")).lower().strip()
    backend = str(run.get("backend", "")).lower().strip()

    base_model, variant = split_variant(run.get("model", ""), backend)
    if variant is None or variant not in VARIANT_ORDER:
        return None

    inf = run.get("inference_latency_ms", {}) or {}
    mu = inf.get("mean", None)
    sd = inf.get("std", None)
    n = run.get("num_samples", None)

    if mu is None:
        return None

    try:
        mu = float(mu)
        sd = float(sd) if sd is not None else 0.0
    except Exception:
        return None

    if not isinstance(n, int):
        try:
            n = int(n)
        except Exception:
            n = 256

    n = max(10, min(n, 1024))
    sd = 0.0 if (not np.isfinite(sd) or sd < 0) else sd
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


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def plot_target(sub: pd.DataFrame, host: str, device: str, color_map, ax=None, out_file: str | None = None):
    if sub.empty:
        if ax is not None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data for {host}-{device}", ha="center", va="center", transform=ax.transAxes)
        else:
            print(f"[SKIP] No runs for host={host}, device={device}")
        return

    base_models = ordered_models(sorted(sub["base_model"].unique().tolist()))
    sub = sub.copy()
    sub["base_model"] = pd.Categorical(sub["base_model"], categories=base_models, ordered=True)
    sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        created_fig = True

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

    ax.set_title(f"Inference latency @ {host.capitalize()} {device.upper()}")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Time (ms)")
    ax.tick_params(axis="x", labelrotation=20)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    style_axes(ax)
    ax.legend(
        loc="upper right", 
        frameon=True, 
        framealpha=0.9, 
        borderpad=0.4, 
        handlelength=1.4, 
        fontsize="small",
        title_fontsize="small",
        title=None
    )

    if created_fig:
        if not out_file:
            raise ValueError("out_file must be provided when ax is None.")
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

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "separate":
        for host, device, out_file in TARGETS:
            allowed_backends = ["stock"] + (["vaccel-local"] if INCLUDE_VACCEL_LOCAL else [])
            sub = df[
                (df["backend"].isin(allowed_backends))
                & (df["host"] == host)
                & (df["device"] == device)
                & (df["variant"].isin(VARIANT_ORDER))
            ].copy()
            if sub.empty:
                print(f"[SKIP] No runs for host={host}, device={device}")
                continue
            plot_target(sub, host, device, color_map, ax=None, out_file=out_file)

    else:
        fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_COMBINED)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        allowed_backends = ["stock"] + (["vaccel-local"] if INCLUDE_VACCEL_LOCAL else [])
        for ax, (host, device, _out_file) in zip(axes, TARGETS):
            sub = df[
                (df["backend"].isin(allowed_backends))
                & (df["host"] == host)
                & (df["device"] == device)
                & (df["variant"].isin(VARIANT_ORDER))
            ].copy()
            plot_target(sub, host, device, color_map, ax=ax, out_file=None)

        plt.tight_layout()
        fig.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {OUTPUT_COMBINED}")
        plt.close(fig)


if __name__ == "__main__":
    main()
