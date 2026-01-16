#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

INPUT_FILE = "../experiments/model-stats/_summary/run1_benchmark_summary.json"

# Output control
PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "model_stats_inference_latency_barplot.pdf"  # used when PLOT_MODE == "combined"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE_SINGLE = (8.5, 5.2)     # size for separate plots (one per PDF)
FIG_SIZE_COMBINED = (10.5, 11.0) # size for combined plot (3 panels)

SHOW_VALUE_LABELS = True
SHOW_ERROR_BARS = True  # <- toggle

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]

TARGETS = [
    ("robot", "cpu", "model_stats_inference_latency_robot_cpu_barplot.pdf"),
    ("edge", "cpu", "model_stats_inference_latency_edge_cpu_barplot.pdf"),
    ("edge", "gpu", "model_stats_inference_latency_edge_gpu_barplot.pdf"),
]


def split_model_variant(model: str):
    if isinstance(model, str) and model.endswith("_sol"):
        return model[:-4], "SOL"
    return model, "PyTorch"


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def add_value_labels(ax, xs, ys, yerrs, y_top, show_errors: bool):
    fs = max(8, int(plt.rcParams["font.size"] * 0.6))
    for x, y, e in zip(xs, ys, yerrs):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        err = 0.0
        if show_errors and e is not None and not (isinstance(e, float) and np.isnan(e)):
            err = float(e)
        y_text = y + err + 0.02 * y_top
        ax.text(
            x,
            y_text,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=fs,
            clip_on=False,
            zorder=20,
        )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def extract_rows(runs, host, device):
    sub = [
        r for r in runs
        if str(r.get("backend", "")).lower() == "stock"
        and str(r.get("host", "")).lower() == host
        and str(r.get("device", "")).lower() == device
    ]
    if not sub:
        return []

    rows = []
    for r in sub:
        model = r.get("model", "")
        base_model, variant = split_model_variant(model)
        inf = r.get("inference_latency_ms", {}) or {}
        mean = inf.get("mean", None)
        std = inf.get("std", None)
        if mean is None:
            continue
        rows.append((base_model, variant, float(mean), float(std) if std is not None else np.nan))
    return rows


def plot_latency(ax, rows, host, device, color_map):
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {host}-{device} (stock)", ha="center", va="center", transform=ax.transAxes)
        return

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = ["PyTorch", "SOL"]

    mean_map = {(m, v): np.nan for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}
    for m, v, mu, sd in rows:
        if m in base_models and v in variants:
            mean_map[(m, v)] = mu
            std_map[(m, v)] = sd

    all_means = [mean_map[(m, v)] for m in base_models for v in variants]
    all_stds = [std_map[(m, v)] for m in base_models for v in variants]
    y_max = np.nanmax(np.asarray(all_means, dtype=float) + np.nan_to_num(np.asarray(all_stds, dtype=float), nan=0.0))
    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0

    x = np.arange(len(base_models))
    width = 0.34
    xs_pt = x - width / 2
    xs_sol = x + width / 2

    means_pt = [mean_map[(m, "PyTorch")] for m in base_models]
    std_pt = [std_map[(m, "PyTorch")] for m in base_models]
    means_sol = [mean_map[(m, "SOL")] for m in base_models]
    std_sol = [std_map[(m, "SOL")] for m in base_models]

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    ax.bar(
        xs_pt, means_pt, width=width,
        color=color_map["PyTorch"],
        edgecolor=edgecolor, linewidth=linewidth,
        label="PyTorch", zorder=3,
    )
    ax.bar(
        xs_sol, means_sol, width=width,
        color=color_map["SOL"],
        edgecolor=edgecolor, linewidth=linewidth,
        label="SOL", zorder=3,
    )

    if SHOW_ERROR_BARS:
        ax.errorbar(xs_pt, means_pt, yerr=std_pt, fmt="none", ecolor="black",
                    elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)
        ax.errorbar(xs_sol, means_sol, yerr=std_sol, fmt="none", ecolor="black",
                    elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)

    if SHOW_VALUE_LABELS:
        add_value_labels(ax, xs_pt, means_pt, std_pt, y_lim_top, SHOW_ERROR_BARS)
        add_value_labels(ax, xs_sol, means_sol, std_sol, y_lim_top, SHOW_ERROR_BARS)

    ax.set_title(f"Inference latency - stock @ {host}-{device}")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4)


def plot_separate(runs):
    for host, device, out_file in TARGETS:
        rows = extract_rows(runs, host, device)
        if not rows:
            print(f"[SKIP] No runs for host={host}, device={device}, backend=stock")
            continue

        sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
        pal = sns.color_palette("colorblind", n_colors=2)
        color_map = {"PyTorch": pal[0], "SOL": pal[1]}

        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        plot_latency(ax, rows, host, device, color_map)

        plt.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved plot to: {out_file}")
        plt.close(fig)


def plot_combined(runs):
    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=2)
    color_map = {"PyTorch": pal[0], "SOL": pal[1]}

    # Stacked vertically is usually clearest for long x tick labels
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_COMBINED)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, (host, device, _out_file) in zip(axes, TARGETS):
        rows = extract_rows(runs, host, device)
        plot_latency(ax, rows, host, device, color_map)

    plt.tight_layout()
    fig.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved combined plot to: {OUTPUT_COMBINED}")
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

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "combined":
        plot_combined(runs)
    else:
        plot_separate(runs)


if __name__ == "__main__":
    main()
