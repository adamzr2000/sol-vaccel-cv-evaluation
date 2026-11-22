#!/usr/bin/env python3
"""
barplot_system_stats_energy_per_inference.py

Plots Robot CPU, Edge CPU, and Edge GPU energy per frame in a 3x3 grid
grouped by model category. Features row-normalized auto-zooming Y-axes, 
shared X-axes per column, hidden inner Y-axis labels, and a unified top legend.
Produces: system_stats_energy_per_frame.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from results.plot.plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
cfg = load_config()
CPU_FILE = str(get_path("system_cpu_summary"))
GPU_FILE = str(get_path("system_gpu_summary"))
MODEL_PERF_FILE = "../../experiments/model-stats/_summary/finalPerformance_benchmark_summary_wifi.json"

OUTPUT_FILE = "energy-per-frame.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 12.0)  # Taller 3-row layout

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = False

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
    "Local-CPU (torch.compile)",          # Index 0
    "Local-CPU (SOL)",                    # Index 1
    "Remote-CPU (torch.compile)",         # Index 2
    "Remote-CPU (SOL)",                   # Index 3
    "Remote-GPU (torch.compile)",         # Index 4
    "Remote-GPU (SOL)",                   # Index 5
]

# Energy per frame = (Power_W × duration_sec) / num_samples
# Each sample represents one complete frame processing (pre + inf + post)


def load_model_performance_data(json_path):
    """Load model performance data to get actual frame counts and durations."""
    import json
    with open(json_path) as f:
        data = json.load(f)
    
    # Build lookup: (host, model, backend, device) -> (num_samples, duration_sec)
    perf_map = {}
    for run in data.get('runs', []):
        host = str(run.get('host', '')).lower().strip()
        model = str(run.get('model', '')).lower().strip()
        backend = str(run.get('backend', '')).lower().strip()
        device = str(run.get('device', '')).lower().strip()
        
        num_samples = int(run.get('num_samples', 0))
        duration = float(run.get('time_window', {}).get('duration_sec', 0))
        
        key = (host, model, backend, device)
        perf_map[key] = {'num_samples': num_samples, 'duration_sec': duration}
    
    return perf_map


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


def load_robot_cpu_rows(cpu_df: pd.DataFrame, perf_map: dict):
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_cpu_variant(r)
        if v is None: continue
        
        # Get actual frame count from model performance data
        host = str(r.get("host", "")).lower().strip()
        model = str(r.get("model", "")).lower().strip()
        backend = str(r.get("backend", "")).lower().strip()
        device = str(r.get("device", "")).lower().strip()
        
        # Map system stats device to performance data device
        # System stats: 'cpu_target-cpu', 'cpu_target-gpu' → Perf: 'cpu', 'gpu'
        if 'target-cpu' in device:
            perf_device = 'cpu'
        elif 'target-gpu' in device:
            perf_device = 'gpu'
        else:
            perf_device = device  # For local execution: just 'cpu'
        
        key = (host, model, backend, perf_device)
        perf_data = perf_map.get(key)
        
        if perf_data is None:
            print(f"WARNING: No performance data for {key}, skipping")
            continue
        
        # Calculate energy per frame (J/frame) using ACTUAL frame counts
        power_w = float(r["cpu_watts_mean"])
        duration = perf_data['duration_sec']
        num_samples = perf_data['num_samples']
        
        if num_samples == 0:
            continue
            
        energy_per_frame = (power_w * duration) / num_samples
        
        # Propagate uncertainty: std_energy = std_power * (duration / num_samples)
        power_std = float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan
        energy_std = (power_std * duration / num_samples) if pd.notna(power_std) else np.nan
        
        rows.append({
            "base_model": model,
            "variant": v,
            "mean": energy_per_frame,
            "std": energy_std,
        })
    return rows


def load_edge_cpu_remote_rows(cpu_df: pd.DataFrame, perf_map: dict):
    rows = []
    for _, r in cpu_df.iterrows():
        host, backend, device = str(r.get("host", "")).lower(), str(r.get("backend", "")).lower(), str(r.get("device", "")).lower()
        if "edge" not in host or device != "cpu": continue
        model = str(r.get("model", "")).strip()
        
        # Get actual frame count from model performance data
        # Note: model performance tracks from robot perspective with device as 'cpu' or 'gpu'
        # System stats use 'cpu' for edge CPU
        key = ('robot', model, backend, 'cpu')
        perf_data = perf_map.get(key)
        
        if perf_data is None:
            print(f"WARNING: No performance data for {key}, skipping")
            continue
        
        # Calculate energy per frame using ACTUAL frame counts
        power_w = float(r["cpu_watts_mean"])
        duration = perf_data['duration_sec']
        num_samples = perf_data['num_samples']
        
        if num_samples == 0:
            continue
            
        energy_per_frame = (power_w * duration) / num_samples
        
        power_std = float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan
        energy_std = (power_std * duration / num_samples) if pd.notna(power_std) else np.nan

        if "vaccel-remote-torch" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[2], "mean": energy_per_frame, "std": energy_std})
        elif "vaccel-remote-sol" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[3], "mean": energy_per_frame, "std": energy_std})
    return rows


def load_edge_gpu_remote_rows(gpu_df: pd.DataFrame, perf_map: dict):
    rows = []
    for _, r in gpu_df.iterrows():
        host, backend, device = str(r.get("host", "")).lower(), str(r.get("backend", "")).lower(), str(r.get("device", "")).lower()
        if "edge" not in host or device != "gpu": continue
        model = str(r.get("model", "")).strip()
        
        # Get actual frame count from model performance data
        # Note: model performance tracks from robot perspective with device as 'gpu'
        # System stats use 'gpu' for edge GPU
        key = ('robot', model, backend, 'gpu')
        perf_data = perf_map.get(key)
        
        if perf_data is None:
            print(f"WARNING: No performance data for {key}, skipping")
            continue
        
        # Calculate energy per frame using ACTUAL frame counts
        power_w = float(r["power_draw_w_mean"])
        duration = perf_data['duration_sec']
        num_samples = perf_data['num_samples']
        
        if num_samples == 0:
            continue
            
        energy_per_frame = (power_w * duration) / num_samples
        
        power_std = float(r["power_draw_w_std"]) if pd.notna(r["power_draw_w_std"]) else np.nan
        energy_std = (power_std * duration / num_samples) if pd.notna(power_std) else np.nan

        if "vaccel-remote-torch" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[4], "mean": energy_per_frame, "std": energy_std})
        elif "vaccel-remote-sol" in backend:
            rows.append({"base_model": model, "variant": VARIANTS[5], "mean": energy_per_frame, "std": energy_std})
    return rows
    return rows
    return rows


def hatch_for_variant_label(vlabel: str, row_idx: int) -> str | None:
    if row_idx != 0:
        return None
        
    s = vlabel.lower()
    if "remote cpu" in s: return "///"
    if "remote gpu" in s: return "oooo"
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
    print("\n" + "=" * 90)
    print(f"{'DEBUG: PLOTTED ENERGY PER FRAME (J)':^90}")
    print("=" * 90)

    for panel_key, ylabel, rows_data, vars_present in panels:
        print(f"\n--- {panel_key.upper()} ---")
        header = f"{'Model':<22} | {'Variant':<28} | {'Mean (J)':>12} | {'Std (J)':>12}"
        print(header)
        print("-" * 90)

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
                    std_str = f"{std_val:12.6f}" if np.isfinite(std_val) else f"{'N/A':>12}"
                    print(f"{m:<22} | {v:<28} | {mean_val:12.6f} | {std_str}")
    print("\n" + "=" * 90 + "\n")


def plot_combined_power(robot_rows, edge_cpu_rows, edge_gpu_rows):
    base_models = ordered_models(sorted({r["base_model"] for r in robot_rows + edge_cpu_rows + edge_gpu_rows}))
    
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "out", "ytick.direction": "out"}, font_scale=FONT_SCALE)
    
    # Use the well-known "Paired" palette
    # Pairs of: Light Blue/Dark Blue, Light Green/Dark Green, Light Red/Dark Red
    # Load the full tab20b palette
# Grab the full tab20 palette
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "out", "ytick.direction": "out"}, font_scale=FONT_SCALE)
    
    # Grab the full tab20 palette
# Grab professional palettes
    crest = sns.color_palette("crest", 10)
    flare = sns.color_palette("flare", 10)
    purp  = sns.color_palette("Purples", 10)

    color_map = {
        VARIANTS[0]: crest[6],   # Local CPU (torch)  - Smooth Teal
        VARIANTS[1]: crest[2],   # Local CPU (SOL)    - Deep Ocean Blue
        VARIANTS[2]: flare[6],   # Remote CPU (torch) - Soft Coral/Rose
        VARIANTS[3]: flare[2],   # Remote CPU (SOL)   - Deep Rust/Wine
        VARIANTS[4]: purp[3],    # Remote GPU (torch) - Light Lavender
        VARIANTS[5]: purp[7],    # Remote GPU (SOL)   - Deep Royal Purple
    }
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
        sharey=False,  # Kept as False, but we manually sync them per row below
        gridspec_kw={"width_ratios": widths, "wspace": 0.05, "hspace": 0.10},
    )

    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    panels = [
        ("robot", "Robot CPU\nenergy per frame (J)", robot_rows, VARIANTS),
        ("edge_cpu", "Edge CPU\nenergy per frame (J)", edge_cpu_rows, [VARIANTS[2], VARIANTS[3]]),
        ("edge_gpu", "Edge GPU\nenergy per frame (J)", edge_gpu_rows, [VARIANTS[4], VARIANTS[5]]),
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

        # --- Calculate Y-limit for the ENTIRE row instead of per-subplot ---
        all_row_means = np.asarray([mean_map[(m, v)] for cm in cat_models for m in cm for v in vars_present], dtype=float)
        all_row_stds = np.asarray([std_map[(m, v)] for cm in cat_models for m in cm for v in vars_present], dtype=float)
        
        y_max = np.nanmax(all_row_means + (np.nan_to_num(all_row_stds, nan=0.0) if SHOW_ERROR_BARS else 0.0))
        if np.isfinite(y_max) and y_max > 0:
            y_lim_top = y_max * 1.15
            # Auto-scale step based on magnitude
            if y_lim_top >= 10:
                step = 2.0
            elif y_lim_top >= 1:
                step = 0.5
            else:
                step = 0.1
            row_y_lim_top = np.ceil(y_lim_top / step) * step
        else:
            row_y_lim_top = 0.5

        # Dynamically scale width depending on the row index
        width, offsets = compute_offsets(vars_present, row_idx)

        for col_idx, current_models in enumerate(cat_models):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(current_models))

            # Apply shared row limit here
            ax.set_ylim(0, row_y_lim_top)

            # 1. DRAW BARS AND ERROR BARS
            for v in vars_present:
                xs = x + offsets[v]
                means = np.asarray([mean_map[(m, v)] for m in current_models], dtype=float)
                stds = np.asarray([std_map[(m, v)] for m in current_models], dtype=float)

                ax.bar(
                    xs, means, width=width, color=color_map[v],
                    edgecolor="black", # Solid thin black border
                    linewidth=0.8,     # Very thin
                    label=v if (row_idx == 0 and col_idx == 0) else "", zorder=3,
                )

                # --- ERROR BARS RESTORED HERE ---
                if SHOW_ERROR_BARS:
                    yerr = np.where(np.isfinite(stds), stds, 0.0)
                    if np.any(yerr > 0):
                        ax.errorbar(
                            xs, means, yerr=yerr, fmt="none", ecolor="black",
                            elinewidth=1.0, capsize=4, capthick=1.0, zorder=10 
                        )

            # --- START: ENERGY SAVINGS INDICATOR (ROBOT ROW ONLY) ---
            for i, m in enumerate(current_models):
                
                # 1. Determine Base (Baseline) and Compare (Target) variants based on the row
                if row_idx == 0:
                    local_base_vars = [VARIANTS[0], VARIANTS[1]]  # Baseline candidates: Local CPU torch/SOL
                    compare_vars = VARIANTS[2:]   # Compare vs: All 4 Remotes
                    x_center_compare = i + (offsets[VARIANTS[2]] + offsets[VARIANTS[-1]]) / 2.0
                elif row_idx == 1:
                    base_var = VARIANTS[2]        # Baseline: Remote CPU (torch)
                    compare_vars = [VARIANTS[3]]  # Compare vs: Remote CPU (SOL)
                    x_center_compare = i + offsets[VARIANTS[3]]
                else:
                    base_var = VARIANTS[4]        # Baseline: Remote GPU (torch)
                    compare_vars = [VARIANTS[5]]  # Compare vs: Remote GPU (SOL)
                    x_center_compare = i + offsets[VARIANTS[5]]

                if row_idx == 0:
                    local_vals = {
                        v: mean_map.get((m, v), np.nan)
                        for v in local_base_vars
                        if np.isfinite(mean_map.get((m, v), np.nan)) and mean_map.get((m, v), np.nan) > 0
                    }
                    if not local_vals:
                        continue
                    base_var = max(local_vals, key=local_vals.get)
                    base_val = local_vals[base_var]
                    x_start = i + min(offsets[v] for v in local_base_vars)
                else:
                    base_val = mean_map.get((m, base_var), np.nan)
                    x_start = i + offsets[base_var]
                
                if np.isfinite(base_val) and base_val > 0:
                    x_end = i + offsets[vars_present[-1]] # Extend line to the rightmost bar
                    
                    # 2. Calculate average of the target variants
# 2. Calculate average of the target variants
                    comps = [mean_map[(m, v)] for v in compare_vars if np.isfinite(mean_map[(m, v)])]
                    if comps:
                        avg_comp = np.mean(comps)
                        savings_pct = (base_val - avg_comp) / base_val * 100
                        
                        # ---> ADD THIS LINE HERE <---
                        print(f"Panel: {panel_key:<10} | Model: {m:<22} | Energy Savings: {savings_pct:>6.1f}%")

                        # Check if there is a meaningful difference (either reduction or increase)
                        if abs(savings_pct) > 0.5:
                            is_reduction = savings_pct > 0
                            
                            # Dynamically set colors and symbols based on whether it's better or worse
                            badge_color = "darkgreen" if is_reduction else "darkred"
                            symbol = "↓" if is_reduction else "↑"
                            display_pct = abs(savings_pct)
                            
                            # Draw a colored dot-dash reference line from the baseline bar over the target bars
                            ax.hlines(y=base_val, xmin=x_start, xmax=x_end, 
                                      colors=badge_color, linestyles='-.', linewidth=1.8, alpha=0.75, zorder=4,
                                      label="Savings reference" if (row_idx == 0 and col_idx == 0 and i == 0) else "")
                            
                            # Dynamic offset so the arrowhead doesn't crash into the bar
                            arrow_offset = (base_val - avg_comp) * 0.15 
                            
                            # Draw arrow (automatically points up or down depending on the coordinates)
                            ax.annotate(
                                '', xy=(x_center_compare, avg_comp + arrow_offset), xycoords='data',
                                xytext=(x_center_compare, base_val), textcoords='data',
                                arrowprops=dict(arrowstyle="->", color=badge_color, lw=1.5),
                                zorder=4
                            )
                            
                            # Beautiful "Badge" Tag in the middle of the drop/climb
                            # Auto-nudge it if too close to bar tops/value labels
                            y_center = (base_val + avg_comp) / 2.0
                            y_min, y_max_axis = ax.get_ylim()
                            y_pad = 0.06 * (y_max_axis - y_min)

                            if abs(y_center - base_val) < y_pad or abs(y_center - avg_comp) < y_pad:
                                y_above = max(base_val, avg_comp) + y_pad
                                y_below = min(base_val, avg_comp) - y_pad

                                if y_above <= y_max_axis - 0.02 * (y_max_axis - y_min):
                                    y_center = y_above
                                elif y_below >= y_min + 0.02 * (y_max_axis - y_min):
                                    y_center = y_below
                                else:
                                    y_center = min(max(y_center, y_min + y_pad), y_max_axis - y_pad)

                            ax.text(x_center_compare, y_center, f"{symbol} {display_pct:.0f}%", 
                                    ha="center", va="center", fontsize=11, 
                                    color=badge_color, fontweight="bold", 
                                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=badge_color, lw=1.2, alpha=0.95),
                                    zorder=20)
            # --- END: ENERGY SAVINGS INDICATORS ---
            
            ax.set_xticks(x)
            
            # Only show model names on the bottom row
            if row_idx == 2:
                ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=25, ha="right")
            else:
                ax.set_xticklabels([])

            style_axes(ax)
            ax.margins(x=0.015)

            # ---> Keep Y label on left-most plot, hide on all others
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)

    # ---- Manual Layout ----
    CAPTION_OFFSET = 0.13

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
    print(f"[OK] Saved energy per frame plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    cpu_path = Path(CPU_FILE).resolve()
    gpu_path = Path(GPU_FILE).resolve()
    perf_path = Path(MODEL_PERF_FILE).resolve()
    
    if not cpu_path.exists(): raise SystemExit(f"CPU CSV not found: {cpu_path}")
    if not gpu_path.exists(): raise SystemExit(f"GPU CSV not found: {gpu_path}")
    if not perf_path.exists(): raise SystemExit(f"Model performance JSON not found: {perf_path}")

    cpu_df = pd.read_csv(cpu_path)
    gpu_df = pd.read_csv(gpu_path)
    perf_map = load_model_performance_data(perf_path)
    
    print(f"Loaded {len(perf_map)} model performance entries")

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    robot_rows = load_robot_cpu_rows(cpu_df, perf_map)
    edge_cpu_rows = load_edge_cpu_remote_rows(cpu_df, perf_map)
    edge_gpu_rows = load_edge_gpu_remote_rows(gpu_df, perf_map)

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    robot_rows, _ = _apply_model_filter(robot_rows, allowed_models)
    edge_cpu_rows, _ = _apply_model_filter(edge_cpu_rows, allowed_models)
    edge_gpu_rows, _ = _apply_model_filter(edge_gpu_rows, allowed_models)

    if not robot_rows and not edge_cpu_rows and not edge_gpu_rows:
        raise SystemExit("ERROR: No rows remained after filtering!")

    plot_combined_power(robot_rows, edge_cpu_rows, edge_gpu_rows)


if __name__ == "__main__":
    main()