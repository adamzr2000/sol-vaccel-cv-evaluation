#!/usr/bin/env python3
"""
model_stats_inference_latency_categorized.py

Adapted to match the 3-subplot style (Image, Video, Segmentation)
with independent auto-zooming Y-axes, a stacked breakdown of latency, 
SEPARATE top-mounted legends, and detailed console debug output.
Produces: model_stats_inference_latency_final.pdf
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from plot_config import (
    get_path, 
    load_config, 
    get_model_type_order, 
    get_model_display_name
)

# --- CONFIGURATION ---
cfg = load_config()
INPUT_FILE = str(get_path("model_summary"))
# New file for isolating pure inference on remote targets
REMOTE_INFERENCE_FILE = "../../experiments/model-stats/_summary/finalOverhead_benchmark_summary_wifi.json"
OUTPUT_FILE = "model_stats_inference_latency_final.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 6.0)  # Restored height for separate dual legends

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True
HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH = False

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

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {"label": "Robot-CPU (torch.compile)", "match": {"host": "robot", "backend": "ptc", "device": "cpu"}},
    {"label": "Robot-CPU (SOL)",           "match": {"host": "robot", "backend": "sol", "device": "cpu"}},
    {"label": "Edge-CPU (torch.compile)",  "match": {"host": "robot"}, "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-cpu"]},
    {"label": "Edge-CPU (SOL)",            "match": {"host": "robot"}, "backend_contains": "vaccel-remote-sol",   "run_id_contains": ["target-cpu"]},
    {"label": "Edge-GPU (torch.compile)",  "match": {"host": "robot"}, "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-gpu"]},
    {"label": "Edge-GPU (SOL)",            "match": {"host": "robot"}, "backend_contains": "vaccel-remote-sol",   "run_id_contains": ["target-gpu"]},
]
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def base_model_name(model: str) -> str:
    return str(model).strip()


def classify_variant(run: dict):
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


def get_remote_pure_inference(remote_data, model_name, variant_label):
    """Finds pure remote inference time by matching edge config."""
    is_gpu = "gpu" in variant_label.lower()
    is_sol = "sol" in variant_label.lower()

    target_host = "edge-asus"
    target_device = "gpu" if is_gpu else "cpu"
    target_backends = ["sol"] if is_sol else ["stock", "ptc"]

    for r in remote_data:
        if base_model_name(r.get("model", "")) != model_name: 
            continue
        if r.get("host", "").lower() != target_host: 
            continue
        if r.get("device", "").lower() != target_device: 
            continue
        if r.get("backend", "").lower() not in target_backends: 
            continue

        inf = r.get("inference_ms", {}) or {}
        return float(inf.get("mean", 0.0))
    
    return 0.0 # Fallback if missing


def extract_rows(runs, remote_runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        b_model = base_model_name(r.get("model", ""))
        
        # System (Total)
        sys_data = r.get("system_ms", {}) or {}
        sys_mean = sys_data.get("mean", None)
        sys_std = sys_data.get("std", np.nan)
        
        # Inference (Includes network if remote)
        inf_data = r.get("inference_ms", {}) or {}
        inf_mean = inf_data.get("mean", 0.0)

        if sys_mean is None:
            continue

        try:
            total_f = float(sys_mean)
            total_std_f = float(sys_std) if sys_std is not None else np.nan
            inf_f = float(inf_mean)
        except Exception:
            continue

        # Breakdown calculation
        is_remote = "edge" in variant.lower()
        
        pre_post_f = max(0.0, total_f - inf_f) # Remaining is pre/post

        if is_remote:
            pure_inf_f = get_remote_pure_inference(remote_runs, b_model, variant)
            
            # FIX: Handle CPU variance anomalies where standalone is slower than E2E
            if pure_inf_f > inf_f:
                inference_final = inf_f
                network_f = 0.0
            else:
                inference_final = pure_inf_f
                network_f = inf_f - pure_inf_f
        else:
            network_f = 0.0
            inference_final = inf_f

        rows.append((b_model, variant, inference_final, network_f, pre_post_f, total_std_f))
        
    return rows

def print_debug_table(inf_map, net_map, pre_map, std_map, base_models, variants):
    """Prints a detailed breakdown of the extracted metrics to the console."""
    print("\n" + "=" * 110)
    print(f"{'DEBUG: LATENCY BREAKDOWN (ms)':^110}")
    print("=" * 110)
    
    header = f"{'Model':<20} | {'Variant':<28} | {'Inf':>8} | {'Net':>8} | {'Pre/Post':>8} | {'Total':>8} | {'Std':>8}"
    print(header)
    print("-" * 110)
    
    for m in base_models:
        for v in variants:
            inf = inf_map.get((m, v), 0.0)
            net = net_map.get((m, v), 0.0)
            pre = pre_map.get((m, v), 0.0)
            std = std_map.get((m, v), np.nan)
            
            tot = inf + net + pre
            
            # Only print if this variant actually has data
            if tot > 0 or np.isfinite(std):
                std_str = f"{std:8.2f}" if np.isfinite(std) else f"{'N/A':>8}"
                print(f"{m:<20} | {v:<28} | {inf:8.2f} | {net:8.2f} | {pre:8.2f} | {tot:8.2f} | {std_str}")
        print("-" * 110)
    print("=" * 110 + "\n")


def plot_latency_categorized(rows):
    if not rows:
        print("No matching rows found.")
        return

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print("\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  " + str(dropped) + "\n")

    rows = [(m, v, inf, net, pre, sd) for (m, v, inf, net, pre, sd) in rows if m in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering!")
        return

    base_models = ordered_models(sorted({m for m, _, _, _, _, _ in rows}))
    variants = VARIANTS

    # Dictionaries to store breakdown
    inf_map = {(m, v): 0.0 for m in base_models for v in variants}
    net_map = {(m, v): 0.0 for m in base_models for v in variants}
    pre_map = {(m, v): 0.0 for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}
    
    for m, v, inf, net, pre, sd in rows:
        if m in base_models and v in variants:
            inf_map[(m, v)] = inf
            net_map[(m, v)] = net
            pre_map[(m, v)] = pre
            std_map[(m, v)] = sd

    # Print the calculated values to console
    print_debug_table(inf_map, net_map, pre_map, std_map, base_models, variants)

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "in", "ytick.direction": "in"},
        font_scale=FONT_SCALE,
    )
    
    palette_name = cfg.get("palette", "tab10")
    pal = sns.color_palette(palette_name, n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    # Group models into categories
    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        1, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths},
    )
    if len(cat_models) == 1:
        axes = [axes]

    n_vars = len(variants)
    group_width = 0.8
    bar_width = min(0.2, group_width / n_vars)
    start = -((n_vars - 1) * bar_width) / 2
    offsets = {v: start + i * bar_width for i, v in enumerate(variants)}

    for ax_idx, (ax, current_models) in enumerate(zip(axes, cat_models)):
        x = np.arange(len(current_models))

        # --- Y-limit based on Total System Latency ---
        cat_totals = []
        cat_errs = []
        for m in current_models:
            for v in variants:
                tot = inf_map[(m, v)] + net_map[(m, v)] + pre_map[(m, v)]
                cat_totals.append(tot)
                cat_errs.append(std_map[(m, v)])
                
        cat_totals = np.asarray(cat_totals, dtype=float)
        cat_errs = np.asarray(cat_errs, dtype=float)

        ymax = np.nanmax(cat_totals + (np.nan_to_num(cat_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0))
        y_lim_top = (ymax * 1.10) if np.isfinite(ymax) and ymax > 0 else 1.0
        ax.set_ylim(0, y_lim_top)

        # Draw stacked bars
        for v in variants:
            xs = x + offsets[v]
            
            inf_vals = np.asarray([inf_map[(m, v)] for m in current_models], dtype=float)
            net_vals = np.asarray([net_map[(m, v)] for m in current_models], dtype=float)
            pre_vals = np.asarray([pre_map[(m, v)] for m in current_models], dtype=float)
            yerr = np.asarray([std_map[(m, v)] for m in current_models], dtype=float)
            
            tot_vals = inf_vals + net_vals + pre_vals

            # 1. Base (Inference)
            ax.bar(
                xs, inf_vals,
                width=bar_width, color=color_map[v], edgecolor="black", linewidth=0.8,
                hatch="", zorder=3, label=v if ax_idx == 0 else ""
            )

            # 2. Middle (Network)
            ax.bar(
                xs, net_vals, bottom=inf_vals,
                width=bar_width, color=color_map[v], edgecolor="black", linewidth=1.0,
                hatch="\\\\\\\\", zorder=3
            )
            
            # 3. Top (Pre/Post)
            ax.bar(
                xs, pre_vals, bottom=(inf_vals + net_vals),
                width=bar_width, color=color_map[v], edgecolor="black", linewidth=1.0,
                hatch="......", zorder=3
            )

            if SHOW_ERROR_BARS:
                ax.errorbar(
                    xs, tot_vals, yerr=yerr,
                    fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0,
                    zorder=10,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
        
        if HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH and len(variants) >= 2:
            v0, v1 = variants[0], variants[1] 
            for tick, m in zip(ax.get_xticklabels(), current_models):
                t_tc = inf_map[(m, v0)] + net_map[(m, v0)] + pre_map[(m, v0)]
                t_sol = inf_map[(m, v1)] + net_map[(m, v1)] + pre_map[(m, v1)]
                if t_sol > t_tc:
                    tick.set_color("red")

        style_axes(ax)
        ax.margins(x=0.005)

    # ---- Manual layout ----
    WSPACE = 0.12
    CAPTION_OFFSET = 0.13

    # Adjusted top padding to comfortably fit both legends
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.20,
        top=0.73, 
        wspace=WSPACE,
    )

    # 1. Execution Variants Legend (Top)
    color_handles, color_labels = axes[0].get_legend_handles_labels()
    leg1 = fig.legend(
        color_handles, color_labels,
        title="Execution via vAccel",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )
    
    # 2. Breakdown Legend (Bottom)
    hatch_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Inference'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='\\\\\\\\', label='Network (vAccel RPC)'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='......', label='Pre/Post-processing')
    ]
    hatch_labels = ['Inference', 'Network (vAccel RPC)', 'Pre/Post-processing']
    
    leg2 = fig.legend(
        hatch_handles, hatch_labels,
        title="Latency Breakdown",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    # Aligned (a)(b)(c) captions
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes) - CAPTION_OFFSET
    for ax, cap in zip(axes, cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    # Global y-label
    sy = fig.supylabel("Time (ms)", x=0.02)

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(leg1, leg2, sy, *caption_artists),
    )
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    # Load remote reference file (handles missing file gracefully)
    remote_path = Path(REMOTE_INFERENCE_FILE).resolve()
    remote_data = []
    if remote_path.exists():
        with remote_path.open("r") as f:
            remote_data = json.load(f).get("runs", [])
    else:
        print(f"[WARNING] Remote inference file missing: {REMOTE_INFERENCE_FILE}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    rows = extract_rows(runs, remote_data)
    plt.rcParams['hatch.linewidth'] = 0.5  # Makes hatch lines thin and elegant
    plot_latency_categorized(rows)


if __name__ == "__main__":
    main()