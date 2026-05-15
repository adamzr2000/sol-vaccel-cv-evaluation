#!/usr/bin/env python3
"""
barplot_system_stats_power_combined.py

Plots Robot CPU, Edge CPU, and Edge GPU power consumption in a 3x3 grid
grouped by model category. Features independent auto-zooming Y-axes, 
shared X-axes per column, and a unified top-mounted legend.
Produces: system_stats_power_combined.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
cfg = load_config()
CPU_FILE = str(get_path("system_cpu_summary"))
GPU_FILE = str(get_path("system_gpu_summary"))

OUTPUT_FILE = "system_stats_power_combined.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 12.0)  # Taller 3-row layout

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

MODEL_TYPE_ORDER = get_model_type_order()

# --- MODEL CATEGORIES ---
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]

CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# Define variants dynamically based on configuration
VARIANTS = [
    "Local CPU (torch.compile)",          # Index 0
    "Local CPU (SOL)",                    # Index 1
    "Remote CPU (torch.compile)",         # Index 2
    "Remote CPU (SOL)",                   # Index 3
    "Remote GPU (torch.compile)",         # Index 4
    "Remote GPU (SOL)",                   # Index 5
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def classify_robot_cpu_variant(row) -> str | None:
    backend = str(row.get("backend", "")).lower().strip()
    device = str(row.get("device", "")).lower().strip()

    if backend == "ptc" and device == "cpu":
        return VARIANTS[0]
    if backend == "sol" and device == "cpu":
        return VARIANTS[1]

    if "vaccel-remote-torch" in backend:
        if "target-cpu" in device: return VARIANTS[2]
        if "target-gpu" in device: return VARIANTS[4]
    if "vaccel-remote-sol" in backend:
        if "target-cpu" in device: return VARIANTS[3]
        if "target-gpu" in device: return VARIANTS[5]

    return None


def load_robot_cpu_rows(cpu_df: pd.DataFrame):
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_cpu_variant(r)
        if v is None: continue
        rows.append({
            "base_model": str(r["model"]).strip(),
            "variant": v,
            "mean": float(r["cpu_watts_mean"]),
            "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan,
        })
    return rows


def load_edge_cpu_remote_rows(cpu_df: pd.DataFrame):
    rows = []
    for _, r in cpu_df.iterrows():
        host, backend, device = str(r.get("host", "")).lower(), str(r.get("backend", "")).lower(), str(r.get("device", "")).lower()
        if "edge" not in host or device != "cpu": continue
        model = str(r.get("model", "")).strip()

        if "vaccel-remote-torch" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[2], "mean": float(r["cpu_watts_mean"]), "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan})
        elif "vaccel-remote-sol" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[3], "mean": float(r["cpu_watts_mean"]), "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan})
    return rows


def load_edge_gpu_remote_rows(gpu_df: pd.DataFrame):
    rows = []
    for _, r in gpu_df.iterrows():
        host, backend, device = str(r.get("host", "")).lower(), str(r.get("backend", "")).lower(), str(r.get("device", "")).lower()
        if "edge" not in host or device != "gpu": continue
        model = str(r.get("model", "")).strip()

        if "vaccel-remote-torch" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[4], "mean": float(r["power_draw_w_mean"]), "std": float(r["power_draw_w_std"]) if pd.notna(r["power_draw_w_std"]) else np.nan})
        elif "vaccel-remote-sol" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[5], "mean": float(r["power_draw_w_mean"]), "std": float(r["power_draw_w_std"]) if pd.notna(r["power_draw_w_std"]) else np.nan})
    return rows


def hatch_for_variant_label(vlabel: str, row_idx: int) -> str | None:
    if row_idx != 0:
        return None
        
    s = vlabel.lower()
    if "remote cpu" in s: return "///"
    if "remote gpu" in s: return "...."
    return None


def compute_offsets(variants_present, row_idx: int):
    n = len(variants_present)
    width = min(0.12, 0.8 / 6) if row_idx == 0 else min(0.25, 0.8 / n)
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def _apply_model_filter(rows, allowed_models):
    present = sorted({r["base_model"] for r in rows})
    dropped = sorted([m for m in present if m not in allowed_models])
    kept = [r for r in rows if r["base_model"] in allowed_models]
    return kept, dropped


def print_debug_table(panels, base_models):
    print("\n" + "=" * 85)
    print(f"{'DEBUG: PLOTTED POWER CONSUMPTION (W)':^85}")
    print("=" * 85)

    for panel_key, ylabel, rows_data, vars_present in panels:
        print(f"\n--- {panel_key.upper()} ---")
        header = f"{'Model':<22} | {'Variant':<28} | {'Mean (W)':>10} | {'Std (W)':>10}"
        print(header)
        print("-" * 85)

        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        std_map = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
            std_map[(r["base_model"], r["variant"])] = r["std"]

        for m in base_models:
            for v in vars_present:
                mean_val = mean_map[(m, v)]
                std_val = std_map[(m, v)]
                if np.isfinite(mean_val):
                    std_str = f"{std_val:10.3f}" if np.isfinite(std_val) else f"{'N/A':>10}"
                    print(f"{m:<22} | {v:<28} | {mean_val:10.3f} | {std_str}")
    print("\n" + "=" * 85 + "\n")


def plot_combined_power(robot_rows, edge_cpu_rows, edge_gpu_rows):
    base_models = ordered_models(sorted({r["base_model"] for r in robot_rows + edge_cpu_rows + edge_gpu_rows}))
    
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANTS))
    color_map = {v: pal[i] for i, v in enumerate(VARIANTS)}

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        3, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.12, "hspace": 0.10},
    )

    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    panels = [
        ("robot", "Robot CPU\npower consumption(W)", robot_rows, VARIANTS),
        ("edge_cpu", "Edge CPU\npower consumption (W)", edge_cpu_rows, [VARIANTS[2], VARIANTS[3]]),
        ("edge_gpu", "Edge GPU\npower consumption (W)", edge_gpu_rows, [VARIANTS[4], VARIANTS[5]]),
    ]

    # --- PRINT TO CONSOLE BEFORE PLOTTING ---
    print_debug_table(panels, base_models)

    for row_idx, (panel_key, ylabel, rows_data, vars_present) in enumerate(panels):
        
        # Build lookup maps for this specific row
        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        std_map = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
            std_map[(r["base_model"], r["variant"])] = r["std"]

        # Dynamically scale width depending on the row index
        width, offsets = compute_offsets(vars_present, row_idx)

        for col_idx, current_models in enumerate(cat_models):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(current_models))

            # Auto-zoom Y-limit logic per subplot
            cat_means = np.asarray([mean_map[(m, v)] for m in current_models for v in vars_present], dtype=float)
            cat_stds = np.asarray([std_map[(m, v)] for m in current_models for v in vars_present], dtype=float)
            
            y_max = np.nanmax(cat_means + (np.nan_to_num(cat_stds, nan=0.0) if SHOW_ERROR_BARS else 0.0))
            if np.isfinite(y_max) and y_max > 0:
                y_lim_top = y_max * 1.05
                step = 0.5 if y_lim_top >= 5 else 0.2
                y_lim_top = np.ceil(y_lim_top / step) * step
            else:
                y_lim_top = 1.0
            
            ax.set_ylim(0, y_lim_top)

            for v in vars_present:
                xs = x + offsets[v]
                means = np.asarray([mean_map[(m, v)] for m in current_models], dtype=float)
                stds = np.asarray([std_map[(m, v)] for m in current_models], dtype=float)

                ax.bar(
                    xs, means, width=width, color=color_map[v],
                    edgecolor=("black" if SHOW_ERROR_BARS else "none"),
                    linewidth=(1.0 if SHOW_ERROR_BARS else 0.0),
                    hatch=hatch_for_variant_label(v, row_idx), 
                    label=v if (row_idx == 0 and col_idx == 0) else "", zorder=3,
                )

                if SHOW_ERROR_BARS:
                    yerr = np.where(np.isfinite(stds), stds, 0.0)
                    if np.any(yerr > 0):
                        ax.errorbar(
                            xs, means, yerr=yerr, fmt="none", ecolor="black",
                            elinewidth=1.0, capsize=4, capthick=1.0, zorder=10 
                        )

            ax.set_xticks(x)
            
            # Only show model names on the bottom row
            if row_idx == 2:
                ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=25, ha="right")
            else:
                ax.set_xticklabels([])

            style_axes(ax)
            ax.margins(x=0.015)

            if col_idx == 0:
                ax.set_ylabel(ylabel)

    # ---- Manual Layout ----
    CAPTION_OFFSET = 0.09

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.15,
        top=0.88,  
    )

    # Global Legend (Built manually to strip out all hatching)
    clean_handles = [
        mpatches.Patch(
            facecolor=color_map[v], 
            edgecolor=("black" if SHOW_ERROR_BARS else "none"), 
            linewidth=(1.0 if SHOW_ERROR_BARS else 0.0)
        ) 
        for v in VARIANTS
    ]
    
    leg = fig.legend(
        clean_handles, VARIANTS,
        title="Execution via vAccel",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3, 
        frameon=True, 
        framealpha=0.9, 
        borderpad=0.4, 
        handlelength=1.4,
    )

    # Aligned (a)(b)(c) captions below the bottom row
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[2, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[2, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(leg, *caption_artists), 
    )
    print(f"[OK] Saved combined plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    cpu_path = Path(CPU_FILE).resolve()
    gpu_path = Path(GPU_FILE).resolve()
    if not cpu_path.exists(): raise SystemExit(f"CPU CSV not found: {cpu_path}")
    if not gpu_path.exists(): raise SystemExit(f"GPU CSV not found: {gpu_path}")

    cpu_df = pd.read_csv(cpu_path)
    gpu_df = pd.read_csv(gpu_path)

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    robot_rows = load_robot_cpu_rows(cpu_df)
    edge_cpu_rows = load_edge_cpu_remote_rows(cpu_df)
    edge_gpu_rows = load_edge_gpu_remote_rows(gpu_df)

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    robot_rows, _ = _apply_model_filter(robot_rows, allowed_models)
    edge_cpu_rows, _ = _apply_model_filter(edge_cpu_rows, allowed_models)
    edge_gpu_rows, _ = _apply_model_filter(edge_gpu_rows, allowed_models)

    if not robot_rows and not edge_cpu_rows and not edge_gpu_rows:
        raise SystemExit("ERROR: No rows remained after filtering!")

    plt.rcParams['hatch.linewidth'] = 0.5
    plot_combined_power(robot_rows, edge_cpu_rows, edge_gpu_rows)


if __name__ == "__main__":
    main()