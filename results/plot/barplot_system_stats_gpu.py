#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/system-stats/_summary/run1_overall_gpu_stats.csv"

# Output control
PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "system_stats_gpu.pdf"  # used when PLOT_MODE == "combined"
FIG_SIZE_COMBINED = (10.5, 11.5)  # 3 stacked panels

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE_SINGLE = (9.0, 5.4)

VARIANT_ORDER = ["PyTorch", "SOL"]
LEGEND_LOC = "upper right"

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]

PLOTS = [
    dict(
        out="system_stats_gpu_vram.pdf",
        y="mem_used_mb_mean",
        yerr="mem_used_mb_std",
        ylabel="VRAM (MB)",
        title="Edge GPU RAM (VRAM) utilization",
    ),
    dict(
        out="system_stats_gpu_utilization.pdf",
        y="util_gpu_percent_mean",
        yerr="util_gpu_percent_std",
        ylabel="GPU (%)",
        title="Edge GPU utilization",
    ),
    dict(
        out="system_stats_gpu_power.pdf",
        y="power_draw_w_mean",
        yerr="power_draw_w_std",
        ylabel="Power (W)",
        title="Edge GPU power",
    ),
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
            x, y_text, f"{y:.1f}",
            ha="center", va="bottom",
            color="black", fontsize=fs,
            clip_on=False, zorder=20
        )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def plot_metric(df, y_col, yerr_col, ylabel, title, ax=None, out_file=None):
    """
    If ax is provided -> draw into it (no saving).
    If ax is None -> create a figure, draw, and save to out_file.
    """
    base_models = ordered_models(sorted(df["base_model"].unique().tolist()))
    d = df.copy()
    d["base_model"] = pd.Categorical(d["base_model"], categories=base_models, ordered=True)
    d["variant"] = pd.Categorical(d["variant"], categories=VARIANT_ORDER, ordered=True)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=2)
    color_map = {"PyTorch": pal[0], "SOL": pal[1]}

    mean_map = {(m, v): np.nan for m in base_models for v in VARIANT_ORDER}
    std_map = {(m, v): np.nan for m in base_models for v in VARIANT_ORDER}

    for _, r in d.iterrows():
        m = r["base_model"]
        v = r["variant"]
        mean_map[(m, v)] = float(r[y_col]) if pd.notna(r[y_col]) else np.nan
        std_map[(m, v)] = float(r[yerr_col]) if pd.notna(r[yerr_col]) else np.nan

    all_means = np.asarray([mean_map[(m, v)] for m in base_models for v in VARIANT_ORDER], dtype=float)
    all_stds = np.asarray([std_map[(m, v)] for m in base_models for v in VARIANT_ORDER], dtype=float)
    y_max = np.nanmax(all_means + np.nan_to_num(all_stds, nan=0.0))
    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        created_fig = True

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

    ax.bar(xs_pt, means_pt, width=width, color=color_map["PyTorch"],
           edgecolor=edgecolor, linewidth=linewidth, label="PyTorch", zorder=3)
    ax.bar(xs_sol, means_sol, width=width, color=color_map["SOL"],
           edgecolor=edgecolor, linewidth=linewidth, label="SOL", zorder=3)

    if SHOW_ERROR_BARS:
        ax.errorbar(xs_pt, means_pt, yerr=std_pt, fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)
        ax.errorbar(xs_sol, means_sol, yerr=std_sol, fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)

    if SHOW_VALUE_LABELS:
        add_value_labels(ax, xs_pt, means_pt, std_pt, y_lim_top, SHOW_ERROR_BARS)
        add_value_labels(ax, xs_sol, means_sol, std_sol, y_lim_top, SHOW_ERROR_BARS)

    ax.set_title(title)
    ax.set_xlabel("ML Model")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(loc=LEGEND_LOC, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4)

    if created_fig:
        if not out_file:
            raise ValueError("out_file must be provided when ax is None (separate plot mode).")
        plt.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved plot to: {out_file}")


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path)

    needed = {
        "host", "model", "backend", "device",
        "power_draw_w_mean", "power_draw_w_std",
        "util_gpu_percent_mean", "util_gpu_percent_std",
        "mem_used_mb_mean", "mem_used_mb_std",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    df["host"] = df["host"].astype(str).str.lower().str.strip()
    df["device"] = df["device"].astype(str).str.lower().str.strip()
    df["backend"] = df["backend"].astype(str).str.lower().str.strip()

    df = df[(df["backend"] == "stock") & (df["host"] == "edge") & (df["device"] == "gpu")].copy()
    if df.empty:
        raise SystemExit("No rows after filtering backend='stock', host='edge', device='gpu'.")

    base_variant = df["model"].apply(split_model_variant)
    df["base_model"] = base_variant.apply(lambda t: t[0])
    df["variant"] = base_variant.apply(lambda t: t[1])
    df = df[df["variant"].isin(VARIANT_ORDER)].copy()

    if df.empty:
        raise SystemExit("No rows after parsing variants (PyTorch/SOL).")

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "separate":
        for cfg in PLOTS:
            plot_metric(df, cfg["y"], cfg["yerr"], cfg["ylabel"], cfg["title"], ax=None, out_file=cfg["out"])
        return

    # combined: 3 panels stacked vertically, independent y-scales
    fig, axes = plt.subplots(len(PLOTS), 1, figsize=FIG_SIZE_COMBINED)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, cfg in zip(axes, PLOTS):
        plot_metric(df, cfg["y"], cfg["yerr"], cfg["ylabel"], cfg["title"], ax=ax, out_file=None)

    plt.tight_layout()
    fig.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved combined plot to: {OUTPUT_COMBINED}")


if __name__ == "__main__":
    main()
