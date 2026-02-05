#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../../experiments/model-stats/_summary/run1_benchmark_summary.json"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "model_stats_inference_latency_barplot_local_exec.pdf"

FONT_SCALE = 1.2
SPINES_WIDTH = 1.0
FIG_SIZE_WIDTH = 10.5
FIG_HEIGHT_PER_SUBPLOT = 4.0  # Height allocated for each host/device row

SHOW_VALUE_LABELS = True
SHOW_ERROR_BARS = True

HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH = True

INCLUDE_VACCEL_LOCAL = False
VARIANT_ORDER = ["PyTorch", "SOL", "SOL + vAccel"] if INCLUDE_VACCEL_LOCAL else ["PyTorch", "SOL"]

SMOOTH = False
SMOOTH_WINDOW = 3

MODEL_TYPE_ORDER = [
    "mobilenet_v3_large", "resnet50", "swin_t", "swin_s", "swin_v2_b",
    "swin3d_t", "swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "fcn_resnet50", "fcn_resnet101",
]

# --- HARDCODED TARGETS (Updated for new folder structure) ---
# Format: (host, device, output_filename, legend_loc)
TARGETS = [
    ("robot", "cpu", "model_stats_inference_latency_robot_cpu_barplot.pdf", "upper left"),
    ("edge-asus", "cpu", "model_stats_inference_latency_edge_asus_cpu_barplot.pdf", "upper left"),
    ("edge-asus", "gpu", "model_stats_inference_latency_edge_asus_gpu_barplot.pdf", "upper left"),
    
    ("edge-xtreme", "cpu", "model_stats_inference_latency_edge_xtreme_cpu_barplot.pdf", "upper left"),
    ("edge-xtreme", "gpu", "model_stats_inference_latency_edge_xtreme_gpu_barplot.pdf", "upper left"),
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
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


def add_value_labels(ax, xs, ys, yerrs, y_top, show_errors: bool):
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    for x, y, e in zip(xs, ys, yerrs):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        err = 0.0
        if show_errors and e is not None and not (isinstance(e, float) and np.isnan(e)):
            err = float(e)
        y_text = y + err + 0.02 * y_top
        ax.text(
            x, y_text, f"{y:.0f}",
            ha="left", va="bottom",
            rotation=30,
            rotation_mode="anchor",
            color="black", fontsize=fs,
            clip_on=False, zorder=20,
        )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def moving_average(arr, window: int):
    a = np.asarray(arr, dtype=float)
    if window <= 1:
        return a
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(a, kernel, mode="same")


def extract_rows(runs, host, device):
    host = str(host).lower().strip()
    device = str(device).lower().strip()

    allowed_backends = {"stock"}
    if INCLUDE_VACCEL_LOCAL:
        allowed_backends.add("vaccel-local")

    sub = [
        r for r in runs
        if str(r.get("backend", "")).lower().strip() in allowed_backends
        and str(r.get("host", "")).lower().strip() == host
        and str(r.get("device", "")).lower().strip() == device
    ]
    if not sub:
        return []

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    rows = []
    for r in sub:
        model = r.get("model", "")
        backend = r.get("backend", "")
        base_model, variant = split_variant(model, backend)
        if variant is None or variant not in VARIANT_ORDER:
            continue

        if base_model not in allowed_models:
            continue

        inf = r.get("inference_latency_ms", {}) or {}
        mean = inf.get("mean", None)
        std = inf.get("std", None)
        if mean is None:
            continue

        try:
            mu = float(mean)
        except Exception:
            continue
        try:
            sd = float(std) if std is not None else np.nan
        except Exception:
            sd = np.nan

        rows.append((base_model, variant, mu, sd))

    return rows


def plot_latency(ax, rows, host, device, color_map, leg_loc: str):
    host_u = str(host).strip()
    device_u = str(device).upper().strip()

    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {host_u} ({device_u})", ha="center", va="center", transform=ax.transAxes)
        return

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANT_ORDER

    mean_map = {(m, v): np.nan for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}
    for m, v, mu, sd in rows:
        if m in base_models and v in variants:
            mean_map[(m, v)] = mu
            std_map[(m, v)] = sd

    all_means = np.asarray([mean_map[(m, v)] for m in base_models for v in variants], dtype=float)
    all_stds = np.asarray([std_map[(m, v)] for m in base_models for v in variants], dtype=float)
    y_max = np.nanmax(all_means + np.nan_to_num(all_stds, nan=0.0))
    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0

    x = np.arange(len(base_models))
    if len(variants) == 3:
        width = 0.24
        offsets = {"PyTorch": -width, "SOL": 0.0, "SOL + vAccel": +width}
    else:
        width = 0.34
        offsets = {"PyTorch": -width / 2, "SOL": +width / 2}

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    for v in variants:
        xs = x + offsets[v]
        means = np.asarray([mean_map[(m, v)] for m in base_models], dtype=float)
        stds = np.asarray([std_map[(m, v)] for m in base_models], dtype=float)

        ax.bar(
            xs, means, width=width,
            color=color_map[v],
            edgecolor=edgecolor, linewidth=linewidth,
            label=v, zorder=3,
        )

        if SHOW_ERROR_BARS:
            yerr = np.where(np.isfinite(stds), stds, 0.0)
            mask = np.isfinite(means)
            if np.any(mask) and np.any(yerr[mask] > 0):
                ax.errorbar(
                    xs[mask], means[mask], yerr=yerr[mask],
                    fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10
                )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, means, stds, y_lim_top, SHOW_ERROR_BARS)

        if SMOOTH:
            y = np.asarray(means, dtype=float)
            if np.any(np.isfinite(y)):
                y2 = y.copy()
                if np.any(~np.isfinite(y2)):
                    idx_ok = np.flatnonzero(np.isfinite(y2))
                    if idx_ok.size >= 2:
                        idx_bad = np.flatnonzero(~np.isfinite(y2))
                        y2[idx_bad] = np.interp(idx_bad, idx_ok, y2[idx_ok])
                ax.plot(
                    x, moving_average(y2, SMOOTH_WINDOW),
                    linewidth=1.8, color="black", alpha=0.35, zorder=6
                )

    #ax.set_xlabel("ML Model")
    ax.set_ylabel(f"{host_u}\n{device_u} inference time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.set_ylim(0, y_lim_top)

    if HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH:
        for tick, m in zip(ax.get_xticklabels(), base_models):
            mu_pt = mean_map.get((m, "PyTorch"), np.nan)
            mu_sol = mean_map.get((m, "SOL"), np.nan)
            if np.isfinite(mu_pt) and np.isfinite(mu_sol) and (mu_sol > mu_pt):
                tick.set_color("red")

    style_axes(ax)
    ax.legend(
        loc=leg_loc,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
        title=None,
    )


def plot_separate(runs):
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    for host, device, out_file, leg_loc in TARGETS:
        rows = extract_rows(runs, host, device)
        if not rows:
            print(f"[SKIP] No runs for host={host}, device={device}")
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plot_latency(ax, rows, host, device, color_map, leg_loc)

        plt.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved plot to: {out_file}")
        plt.close(fig)


def plot_combined(runs):
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    num_plots = len(TARGETS)
    if num_plots == 0:
        print("No targets configured.")
        return

    # Dynamic Height Calculation
    total_height = num_plots * FIG_HEIGHT_PER_SUBPLOT
    fig, axes = plt.subplots(num_plots, 1, figsize=(FIG_SIZE_WIDTH, total_height))
    
    if num_plots == 1:
        axes = [axes]

    for ax, (host, device, _out_file, leg_loc) in zip(axes, TARGETS):
        rows = extract_rows(runs, host, device)
        plot_latency(ax, rows, host, device, color_map, leg_loc)

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

    if PLOT_MODE == "combined":
        plot_combined(runs)
    else:
        plot_separate(runs)


if __name__ == "__main__":
    main()