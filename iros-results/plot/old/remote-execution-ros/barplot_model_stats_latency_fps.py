#!/usr/bin/env python3
"""
barplot_combined_fps_latency.py

Combines FPS (top) and Latency Breakdown (middle & bottom) into a 3x3 grid.
Features independent auto-zooming Y-axes per subplot, shared X-axes
per column, and unified top-mounted legends.
Produces: model_stats_combined_final.pdf
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
REMOTE_INFERENCE_FILE = "../../experiments/model-stats/_summary/finalOverhead_benchmark_summary_wifi.json"
OUTPUT_FILE = "model_stats_combined_final.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 14.0)  # Taller figure to accommodate three rows of plots

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
    {"label": "Robot CPU (torch.compile)", "match": {"host": "robot", "backend": "ptc", "device": "cpu"}},
    {"label": "Robot CPU (SOL)",           "match": {"host": "robot", "backend": "sol", "device": "cpu"}},
    {"label": "Edge CPU (torch.compile)",  "match": {"host": "robot"}, "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-cpu"]},
    {"label": "Edge CPU (SOL)",            "match": {"host": "robot"}, "backend_contains": "vaccel-remote-sol",   "run_id_contains": ["target-cpu"]},
    {"label": "Edge GPU (torch.compile)",  "match": {"host": "robot"}, "backend_contains": "vaccel-remote-torch", "run_id_contains": ["target-gpu"]},
    {"label": "Edge GPU (SOL)",            "match": {"host": "robot"}, "backend_contains": "vaccel-remote-sol",   "run_id_contains": ["target-gpu"]},
]
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


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

    return 0.0 


def extract_combined_rows(runs, remote_runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        b_model = base_model_name(r.get("model", ""))

        # FPS Data
        fps_data = r.get("fps", {}) or {}
        fps = fps_data.get("system", None)
        fps_err = fps_data.get("system_std", np.nan)

        # Latency Data
        sys_data = r.get("system_ms", {}) or {}
        sys_mean = sys_data.get("mean", None)
        sys_std = sys_data.get("std", np.nan)

        inf_data = r.get("inference_ms", {}) or {}
        inf_mean = inf_data.get("mean", 0.0)

        if fps is None or sys_mean is None:
            continue

        try:
            fps_f = float(fps)
            fps_err_f = float(fps_err) if fps_err is not None else np.nan
            total_f = float(sys_mean)
            total_std_f = float(sys_std) if sys_std is not None else np.nan
            inf_f = float(inf_mean)
        except Exception:
            continue

        is_remote = "edge" in variant.lower()
        pre_post_f = max(0.0, total_f - inf_f) 

        if is_remote:
            pure_inf_f = get_remote_pure_inference(remote_runs, b_model, variant)
            if pure_inf_f > inf_f:
                inference_final = inf_f
                network_f = 0.0
            else:
                inference_final = pure_inf_f
                network_f = inf_f - pure_inf_f
        else:
            network_f = 0.0
            inference_final = inf_f

        rows.append((b_model, variant, fps_f, fps_err_f, inference_final, network_f, pre_post_f, total_std_f))

    return rows


def compute_offsets(variants_present, row_idx: int):
    n = len(variants_present)
    
    # Custom widths for visual balance across rows
    if row_idx == 0:
        width = 0.12  # Row 0: 6 variants (slim)
    elif row_idx == 1:
        width = 0.35  # Row 1: 2 variants (significantly wider to fill space)
    else:
        width = 0.18  # Row 2: 4 variants (medium)
        
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def plot_combined(rows):
    if not rows:
        print("No matching rows found.")
        return

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _, _, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print("\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  " + str(dropped) + "\n")

    rows = [row for row in rows if row[0] in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering!")
        return

    base_models = ordered_models(sorted({r[0] for r in rows}))
    variants = VARIANTS

    # Maps for FPS
    fps_val_map = {(m, v): np.nan for m in base_models for v in variants}
    fps_err_map = {(m, v): np.nan for m in base_models for v in variants}
    
    # Maps for Latency
    inf_map = {(m, v): 0.0 for m in base_models for v in variants}
    net_map = {(m, v): 0.0 for m in base_models for v in variants}
    pre_map = {(m, v): 0.0 for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}

    for m, v, fps, fps_err, inf, net, pre, lat_std in rows:
        if m in base_models and v in variants:
            fps_val_map[(m, v)] = fps
            fps_err_map[(m, v)] = fps_err
            inf_map[(m, v)] = inf
            net_map[(m, v)] = net
            pre_map[(m, v)] = pre
            std_map[(m, v)] = lat_std

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out"},
        font_scale=FONT_SCALE,
    )

    palette_name = cfg.get("palette", "tab10")
    pal = sns.color_palette(palette_name, n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        3, len(cat_models),  # Increased to 3 rows
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.12, "hspace": 0.08},
    )

    # Ensure axes is always 2D array [row, col]
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        # ==========================================
        # --- ROW 0: FPS (All 6 Variants) ---
        # ==========================================
        ax_fps = axes[0, col_idx]
        row_0_vars = variants
        width_fps, offsets_fps = compute_offsets(row_0_vars, 0)

        cat_fps_vals = []
        for m in current_models:
            for v in row_0_vars:
                cat_fps_vals.append(fps_val_map[(m, v)])
        
        cat_fps_vals = np.asarray(cat_fps_vals, dtype=float)

        ymax_fps = np.nanmax(cat_fps_vals)
        y_lim_top_fps = (ymax_fps * 1.02) if np.isfinite(ymax_fps) and ymax_fps > 0 else 1.0
        ax_fps.set_ylim(0, y_lim_top_fps)

        for v in row_0_vars:
            xs = x + offsets_fps[v]
            vals = np.asarray([fps_val_map[(m, v)] for m in current_models], dtype=float)

            ax_fps.bar(
                xs, vals, width=width_fps, color=color_map[v],
                edgecolor="none", linewidth=0.0,
                label=v if col_idx == 0 else "", zorder=3,
            )

        ax_fps.set_xticks(x)
        ax_fps.set_xticklabels([])
        style_axes(ax_fps)
        ax_fps.margins(x=0.005)
        if col_idx == 0: ax_fps.set_ylabel("Frames per second (FPS)")

        # ==========================================
        # --- ROWS 1 & 2: LATENCY BREAKDOWN ---
        # ==========================================
        # Updated Y-labels mapped perfectly to Row 1 and Row 2
        latency_rows = [
            (1, axes[1, col_idx], variants[0:2], "Local Inference Time (ms)"),   # Row 1: Local only
            (2, axes[2, col_idx], variants[2:6], "Remote Inference Time (ms)")   # Row 2: Remote only
        ]

        for row_idx, ax_lat, vars_present, ylabel in latency_rows:
            width_lat, offsets_lat = compute_offsets(vars_present, row_idx)
            
            cat_totals, cat_lat_errs = [], []
            for m in current_models:
                for v in vars_present:
                    tot = inf_map[(m, v)] + net_map[(m, v)] + pre_map[(m, v)]
                    cat_totals.append(tot)
                    cat_lat_errs.append(std_map[(m, v)])

            cat_totals = np.asarray(cat_totals, dtype=float)
            cat_lat_errs = np.asarray(cat_lat_errs, dtype=float)

            ymax_lat = np.nanmax(cat_totals + (np.nan_to_num(cat_lat_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0))
            y_lim_top_lat = (ymax_lat * 1.02) if np.isfinite(ymax_lat) and ymax_lat > 0 else 1.0
            ax_lat.set_ylim(0, y_lim_top_lat)

            for v in vars_present:
                xs = x + offsets_lat[v]
                inf_vals = np.asarray([inf_map[(m, v)] for m in current_models], dtype=float)
                net_vals = np.asarray([net_map[(m, v)] for m in current_models], dtype=float)
                pre_vals = np.asarray([pre_map[(m, v)] for m in current_models], dtype=float)
                yerr = np.asarray([std_map[(m, v)] for m in current_models], dtype=float)

                tot_vals = inf_vals + net_vals + pre_vals

                # Base (Inference) - No hatch, purely solid color
                ax_lat.bar(
                    xs, inf_vals, width=width_lat, color=color_map[v], edgecolor="black", 
                    linewidth=0.8, hatch="", zorder=3
                )
                # Middle (Network) - Diagonal stripes
                ax_lat.bar(
                    xs, net_vals, bottom=inf_vals, width=width_lat, color=color_map[v], 
                    edgecolor="black", linewidth=1.0, hatch="////", zorder=3
                )
                # Top (Pre/Post) - Dense crosshatch
                ax_lat.bar(
                    xs, pre_vals, bottom=(inf_vals + net_vals), width=width_lat, color=color_map[v], 
                    edgecolor="black", linewidth=1.0, hatch="***", zorder=3
                )

                if SHOW_ERROR_BARS:
                    ax_lat.errorbar(
                        xs, tot_vals, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=0, capthick=0.0, zorder=10,
                    )
            
            ax_lat.set_xticks(x)
            if row_idx == 2:
                # Only the very bottom row gets the model names
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])
                
            style_axes(ax_lat)
            ax_lat.margins(x=0.005)
            if col_idx == 0: ax_lat.set_ylabel(ylabel)


    # ---- Manual Layout ----
    CAPTION_OFFSET = 0.09

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.12,
        top=0.90,  
    )

    color_handles, color_labels = axes[0, 0].get_legend_handles_labels()
    
    hatch_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Inference'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Network (vAccel RPC)'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='***', label='Pre/Post-processing')
    ]
    hatch_labels = ['Inference', 'Network (vAccel RPC)', 'Pre/Post-processing']

    dummy = mpatches.Rectangle((0, 0), 0, 0, alpha=0.0)

    combined_handles = [
        color_handles[0], color_handles[3],
        color_handles[1], color_handles[4],
        color_handles[2], color_handles[5],
        hatch_handles[0], hatch_handles[1],
        hatch_handles[2], dummy
    ]
    
    combined_labels = [
        color_labels[0], color_labels[3],
        color_labels[1], color_labels[4],
        color_labels[2], color_labels[5],
        hatch_labels[0], hatch_labels[1],
        hatch_labels[2], " "
    ]

    leg = fig.legend(
        combined_handles, combined_labels,
        title="Execution via vAccel",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
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
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)

def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

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

    rows = extract_combined_rows(runs, remote_data)
    plt.rcParams['hatch.linewidth'] = 0.5
    plot_combined(rows)


if __name__ == "__main__":
    main()