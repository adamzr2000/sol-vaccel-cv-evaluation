#!/usr/bin/env python3
"""
barplot_combined_fps_latency.py

Combines FPS (top) and Latency Breakdown (middle & bottom) into a 3x3 grid.
Features row-normalized auto-zooming Y-axes across ALL rows, shared X-axes 
per column, hidden inner Y-axis labels for cleanliness, and unified legends.
Produces: model_stats_combined_final.pdf
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns
import matplotlib.colors as mcolors

from plot_config import (
    get_path,
    load_config,
    get_model_type_order,
    get_model_display_name
)

# --- CONFIGURATION ---
_HERE = Path(__file__).parent
cfg = load_config()
INPUT_FILE = str(get_path("model_summary"))
# iso run: edge-asus vaccel-remote-{ptc,sol} inference_ms used as pure-inference baseline for remote breakdown
REMOTE_INFERENCE_FILE = _HERE / "../experiments/model-stats/_summary/iso_benchmark_summary.json"
OUTPUT_FILE = "fps-latency-e2e.pdf"

FONT_SCALE = 1.8
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.7
LATENCY_MAX_TICKS = 5
LATENCY_TICK_FMT = "%.0f"
FIG_SIZE = (18, 15.0)  # 4-row layout

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True
HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH = False
METRIC = "median"           # "median" (p50) | "mean" — controls FPS derivation and latency bars
LOCAL_ROBOT_MODE = "vaccel-rpc" # "legacy"    → ptc/sol backends, iros2 run tag
                            # "vaccel-rpc" → vaccel-remote-*/remote_host=robot (loopback), iso run tag

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
if LOCAL_ROBOT_MODE == "legacy":
    _local_defs = [
        {"label": "Local CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "ptc", "device": "cpu"}},
        {"label": "Local CPU (vAccSOL)",        "match": {"host": "robot", "backend": "sol", "device": "cpu"}},
    ]
else:  # vaccel-rpc: robot local via vaccel-local-* (iso run tag, no remote_host)
    _local_defs = [
        {"label": "Local CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-local-ptc", "device": "cpu"}},
        {"label": "Local CPU (vAccSOL)",        "match": {"host": "robot", "backend": "vaccel-local-sol", "device": "cpu"}},
    ]

VARIANT_DEFINITIONS = _local_defs + [
    {"label": "Remote CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-remote-ptc", "device": "cpu", "remote_host": "edge-asus"}},
    {"label": "Remote CPU (vAccSOL)",        "match": {"host": "robot", "backend": "vaccel-remote-sol", "device": "cpu", "remote_host": "edge-asus"}},
    {"label": "Remote GPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-remote-ptc", "device": "gpu", "remote_host": "edge-asus"}},
    {"label": "Remote GPU (vAccSOL)",        "match": {"host": "robot", "backend": "vaccel-remote-sol", "device": "gpu", "remote_host": "edge-asus"}},
]
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def base_model_name(model: str) -> str:
    return str(model).strip()


def classify_variant(run: dict):
    run_id      = str(run.get("run_id",      "")).strip()
    backend     = str(run.get("backend",     "")).lower().strip()
    host        = str(run.get("host",        "")).lower().strip()
    device      = str(run.get("device",      "")).lower().strip()
    remote_host = str(run.get("remote_host", "")).lower().strip()

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
    """Return inference_ms[METRIC] for pure edge inference (no network).

    Aligned with LOCAL_ROBOT_MODE:
      legacy     → ptc / sol backends (direct, no vAccel RPC layer)
      vaccel-rpc → vaccel-remote-ptc / vaccel-remote-sol edge-asus loopback
    """
    is_gpu = "gpu" in variant_label.lower()
    is_sol = "sol" in variant_label.lower()

    target_host   = "edge-asus"
    target_device = "gpu" if is_gpu else "cpu"
    stat_key      = "p50" if METRIC == "median" else "mean"

    if LOCAL_ROBOT_MODE == "legacy":
        target_backend = "sol" if is_sol else "ptc"
        match_remote_host = None          # legacy runs have no remote_host field
    else:  # vaccel-rpc
        target_backend    = "vaccel-local-sol" if is_sol else "vaccel-local-ptc"
        match_remote_host = None          # vaccel-local-* runs have no remote_host field

    for r in remote_data:
        if base_model_name(r.get("model", "")) != model_name:
            continue
        if r.get("host", "").lower() != target_host:
            continue
        if r.get("device", "").lower() != target_device:
            continue
        if r.get("backend", "").lower() != target_backend:
            continue
        if match_remote_host is not None:
            if r.get("remote_host", "").lower() != match_remote_host:
                continue
        inf = r.get("inference_ms", {}) or {}
        return float(inf.get(stat_key, 0.0))

    return 0.0


def extract_combined_rows(runs, remote_runs):
    stat_key = "p50" if METRIC == "median" else "mean"
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        b_model = base_model_name(r.get("model", ""))

        # Latency data (robot perspective = system_ms e2e)
        sys_data = r.get("system_ms", {}) or {}
        sys_metric = sys_data.get(stat_key, None)
        if sys_metric is None:
            continue

        p25 = float(sys_data.get("p25", sys_metric))
        p75 = float(sys_data.get("p75", sys_metric))
        if METRIC == "median":
            err_lower = float(sys_metric) - p25   # asymmetric lower
            err_upper = p75 - float(sys_metric)   # asymmetric upper
        else:
            std = float(sys_data.get("std", 0.0))
            err_lower = std
            err_upper = std

        inf_data = r.get("inference_ms", {}) or {}
        inf_metric = inf_data.get(stat_key, 0.0)

        # FPS derived from system latency (robot e2e) using the chosen metric
        fps = 1000.0 / float(sys_metric) if float(sys_metric) > 0 else None
        if fps is None:
            continue

        pre_data  = r.get("preprocessing_ms",  {}) or {}
        post_data = r.get("postprocessing_ms", {}) or {}

        try:
            fps_f      = float(fps)
            total_f    = float(sys_metric)
            inf_f      = float(inf_metric) if inf_metric else 0.0
            pre_post_f = float(pre_data.get(stat_key, 0.0)) + float(post_data.get(stat_key, 0.0))
        except Exception:
            continue

        is_remote = "remote" in variant.lower()

        if is_remote:
            pure_inf_f = get_remote_pure_inference(remote_runs, b_model, variant)
            inference_final = pure_inf_f if pure_inf_f > 0 and pure_inf_f <= inf_f else inf_f
            # Network = residual so stack sums exactly to system_ms.p50
            network_f = max(0.0, total_f - inference_final - pre_post_f)
        else:
            network_f       = 0.0
            inference_final = inf_f

        rows.append((b_model, variant, fps_f, inference_final, network_f, pre_post_f, err_lower, err_upper))

    return rows


def compute_offsets(variants_present, row_idx: int):
    n = len(variants_present)
    
    # Custom widths for visual balance across rows
    if row_idx == 0:
        width = 0.12  # Row 0: 6 variants (slim)
    else:
        width = 0.35  # Rows 1-3: 2 variants each (wide)
        
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def print_debug_info(base_models, variants, fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map):
    """Prints a clean summary of the plotted data to the console."""
    W = 118
    print("\n" + "=" * W)
    print(f"{'DEBUG: PLOTTED FPS AND LATENCY (ms)':^{W}}")
    print("=" * W)
    header = (f"{'Model':<22} | {'Variant':<28} | {'FPS':>8} | "
              f"{'Inf(ms)':>8} | {'Pre/Post':>8} | {'Net(ms)':>8} | {'Total(ms)':>10} | {'ErrLo':>7} | {'ErrHi':>7}")
    print(header)
    print("-" * W)
    for m in base_models:
        for v in variants:
            fps   = fps_val_map.get((m, v), np.nan)
            inf   = inf_map.get((m, v), 0.0)
            net   = net_map.get((m, v), 0.0)
            pre   = pre_map.get((m, v), 0.0)
            lo    = lower_map.get((m, v), np.nan)
            hi    = upper_map.get((m, v), np.nan)
            total = inf + pre + net

            if np.isfinite(fps) or inf > 0:
                fps_str   = f"{fps:8.2f}"   if np.isfinite(fps) else "     N/A"
                inf_str   = f"{inf:8.2f}"   if inf   > 0 else "       -"
                pre_str   = f"{pre:8.2f}"   if pre   > 0 else "       -"
                net_str   = f"{net:8.2f}"   if net   > 0 else "       -"
                tot_str   = f"{total:10.2f}"
                lo_str    = f"{lo:7.2f}"    if np.isfinite(lo) else "      -"
                hi_str    = f"{hi:7.2f}"    if np.isfinite(hi) else "      -"
                print(f"{m:<22} | {v:<28} | {fps_str} | {inf_str} | {pre_str} | {net_str} | {tot_str} | {lo_str} | {hi_str}")
    print("\n" + "=" * W + "\n")


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
    
    # Maps for Latency
    inf_map   = {(m, v): 0.0    for m in base_models for v in variants}
    net_map   = {(m, v): 0.0    for m in base_models for v in variants}
    pre_map   = {(m, v): 0.0    for m in base_models for v in variants}
    lower_map = {(m, v): np.nan for m in base_models for v in variants}
    upper_map = {(m, v): np.nan for m in base_models for v in variants}

    for m, v, fps, inf, net, pre, e_lower, e_upper in rows:
        if m in base_models and v in variants:
            fps_val_map[(m, v)] = fps
            inf_map[(m, v)]     = inf
            net_map[(m, v)]     = net
            pre_map[(m, v)]     = pre
            lower_map[(m, v)]   = e_lower
            upper_map[(m, v)]   = e_upper

    # Print out all the statistics to the terminal
    print_debug_info(base_models, variants, fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map)

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )

    # Paired palette: light/dark pairs per deployment scenario
    # Blue = Local CPU, Green = Remote CPU, Red = Remote GPU
    # Light = vAccel+Torch (ptc), Dark = vAccSOL (sol)
    paired_pal = sns.color_palette("Paired", 6)

    color_map = {
        "Local CPU (vAccel + Torch)":  paired_pal[0],  # light blue
        "Local CPU (vAccSOL)":         paired_pal[1],  # dark blue
        "Remote CPU (vAccel + Torch)": paired_pal[2],  # light green
        "Remote CPU (vAccSOL)":        paired_pal[3],  # dark green
        "Remote GPU (vAccel + Torch)": paired_pal[4],  # light red
        "Remote GPU (vAccSOL)":        paired_pal[5],  # dark red
    }

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        4, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.14, "hspace": 0.12},
    )

    # Ensure axes is always 2D array [row, col]
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])

    # --- PRE-CALCULATE PER-PANEL Y-LIMITS (each subplot scales independently) ---
    panel_fps_limits   = {}                    # col_idx -> ylim
    panel_lat_limits   = {}                    # (r_idx, col_idx) -> ylim

    for col_idx, current_models in enumerate(cat_models):
        # Row 0: FPS — scale to the tallest bar in this column
        fps_vals = np.asarray(
            [fps_val_map[(m, v)] for m in current_models for v in variants], dtype=float
        )
        ymax = np.nanmax(fps_vals) if np.any(np.isfinite(fps_vals)) else 1.0
        panel_fps_limits[col_idx] = (ymax * 1.12) if ymax > 0 else 1.0

        # Rows 1–3: Latency — scale to tallest stacked bar + upper error bar
        for r_idx, v_present in [(1, variants[0:2]), (2, variants[2:4]), (3, variants[4:6])]:
            totals = np.asarray(
                [inf_map[(m, v)] + net_map[(m, v)] + pre_map[(m, v)] + upper_map[(m, v)]
                 for m in current_models for v in v_present],
                dtype=float,
            )
            ymax = np.nanmax(totals) if np.any(np.isfinite(totals)) else 1.0
            panel_lat_limits[(r_idx, col_idx)] = (ymax * 1.08) if ymax > 0 else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        # ==========================================
        # --- ROW 0: FPS (All 6 Variants) ---
        # ==========================================
        ax_fps = axes[0, col_idx]
        row_0_vars = variants
        width_fps, offsets_fps = compute_offsets(row_0_vars, 0)

        ax_fps.set_ylim(0, panel_fps_limits[col_idx])

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

        # Hide Y-axis labels for all but the leftmost plot
        if col_idx == 0:
            ax_fps.set_ylabel("Frames rate (FPS)")

        # ==========================================
        # --- ROWS 1–3: LATENCY BREAKDOWN ---
        # ==========================================
        latency_rows = [
            (1, axes[1, col_idx], variants[0:2], "Local CPU\nE2E Latency (ms)"),
            (2, axes[2, col_idx], variants[2:4], "Remote CPU\nE2E Latency (ms)"),
            (3, axes[3, col_idx], variants[4:6], "Remote GPU\nE2E Latency (ms)"),
        ]

        for row_idx, ax_lat, vars_present, ylabel in latency_rows:
            width_lat, offsets_lat = compute_offsets(vars_present, row_idx)
            
            ax_lat.set_ylim(0, panel_lat_limits[(row_idx, col_idx)])
            ax_lat.yaxis.set_major_locator(MaxNLocator(nbins=LATENCY_MAX_TICKS))
            ax_lat.yaxis.set_major_formatter(FormatStrFormatter(LATENCY_TICK_FMT))

            for v in vars_present:
                xs = x + offsets_lat[v]
                inf_vals = np.asarray([inf_map[(m, v)]   for m in current_models], dtype=float)
                net_vals = np.asarray([net_map[(m, v)]   for m in current_models], dtype=float)
                pre_vals = np.asarray([pre_map[(m, v)]   for m in current_models], dtype=float)
                yerr = np.array([
                    [lower_map[(m, v)] for m in current_models],
                    [upper_map[(m, v)] for m in current_models],
                ], dtype=float)

                tot_vals = inf_vals + net_vals + pre_vals

                base_color = color_map[v]
                # Create a lighter, semi-transparent version of the color for the overheads
                light_color = mcolors.to_rgba(base_color, alpha=0.4)

                # Bottom: Inference — solid color
                ax_lat.bar(
                    xs, inf_vals, width=width_lat, facecolor=base_color, edgecolor="black",
                    linewidth=STROKE_WIDTH, zorder=3
                )
                # Middle: Pre/Post-processing — light, dotted hatch
                ax_lat.bar(
                    xs, pre_vals, bottom=inf_vals, width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="..", zorder=3
                )
                # Top: Network + vAccel remoting layer — light, diagonal hatch
                ax_lat.bar(
                    xs, net_vals, bottom=(inf_vals + pre_vals), width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="//", zorder=3
                )

                if SHOW_ERROR_BARS:
                    ax_lat.errorbar(
                        xs, tot_vals, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=4, capthick=1.0, zorder=10,
                    )
            
            ax_lat.set_xticks(x)
            if row_idx == 3:
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])
                
            style_axes(ax_lat)
            ax_lat.margins(x=0.005)

            # Hide Y-axis labels for all but the leftmost plot
            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)



    # ---- Manual Layout ----
    CAPTION_OFFSET = 0.075

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.10,
        top=0.84,
    )

    color_handles, color_labels = axes[0, 0].get_legend_handles_labels()
    
    # Use a neutral gray to demonstrate the latency components in the legend
    legend_base = 'lightgray'
    legend_light = mcolors.to_rgba(legend_base, alpha=0.4)

    hatch_handles = [
        mpatches.Patch(facecolor=legend_base, edgecolor='black', hatch='', label='Inference'),
        mpatches.Patch(facecolor=legend_light, edgecolor='black', hatch='//', label='Network + vAccel remoting layer'),
        mpatches.Patch(facecolor=legend_light, edgecolor='black', hatch='..', label='Pre/Post-processing')
    ]
    hatch_labels = ['Inference', 'Network + vAccel remoting layer', 'Pre/Post-processing']

    # 3 rows × 3 cols (filled left-to-right, top-to-bottom):
    #   Row 1: Local CPU (Torch)   |  Remote CPU (Torch)  |  Remote GPU (Torch)
    #   Row 2: Local CPU (SOL)     |  Remote CPU (SOL)    |  Remote GPU (SOL)
    #   Row 3: Inference | Pre/Post-processing | Network + vAccel remoting layer
    combined_handles = [
        color_handles[0], color_handles[2], color_handles[4],
        color_handles[1], color_handles[3], color_handles[5],
        hatch_handles[0], hatch_handles[2], hatch_handles[1],
    ]
    combined_labels = [
        color_labels[0], color_labels[2], color_labels[4],
        color_labels[1], color_labels[3], color_labels[5],
        hatch_labels[0], hatch_labels[2], hatch_labels[1],
    ]

    leg = fig.legend(
        combined_handles, combined_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fancybox=False,
        edgecolor="black",
    )

    # Aligned (a)(b)(c) captions below the bottom row
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[3, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[3, :], cat_captions):
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
        pad_inches=0.02,
        bbox_extra_artists=(leg, *caption_artists),
    )
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)

def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    remote_path = REMOTE_INFERENCE_FILE.resolve()
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

    # In vaccel-rpc local mode, robot loopback data lives in iso (not iros2)
    if LOCAL_ROBOT_MODE == "vaccel-rpc":
        iso_path = REMOTE_INFERENCE_FILE.resolve()
        if iso_path.exists():
            with iso_path.open() as f:
                runs = runs + json.load(f).get("runs", [])
        else:
            print(f"[WARNING] iso file not found for vaccel-rpc local mode: {iso_path}")

    rows = extract_combined_rows(runs, remote_data)
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    plt.rcParams['hatch.linewidth'] = STROKE_WIDTH
    plot_combined(rows)


if __name__ == "__main__":
    main()