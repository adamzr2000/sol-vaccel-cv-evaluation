#!/usr/bin/env python3
"""
barplot_remote_inference.py

Plots inference latency breakdown (Pre-processing, Inference, Post-processing)
for the edge server runs in REMOTE_INFERENCE_FILE, grouped by model category.

Variants: Edge CPU / Edge GPU × PyTorch (stock) / vAccSOL
Produces: remote_inference_latency.pdf
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns

from plot_config import (
    get_model_type_order,
    get_model_display_name,
)

# --- CONFIGURATION ---
REMOTE_INFERENCE_FILE = "../../experiments/model-stats/_summary/finalOverhead_benchmark_summary_wifi.json"
OUTPUT_FILE = "remote_inference_latency.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.7
LATENCY_MAX_TICKS = 5
LATENCY_TICK_FMT = "%.0f"
FIG_SIZE = (18, 5.5)

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
CAT_SEG   = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# Variants present in the remote-inference file
VARIANT_DEFINITIONS = [
    {"label": "Edge CPU (PyTorch)", "backend": "stock", "device": "cpu"},
    {"label": "Edge CPU (vAccSOL)", "backend": "sol",   "device": "cpu"},
    {"label": "Edge GPU (PyTorch)", "backend": "stock", "device": "gpu"},
    {"label": "Edge GPU (vAccSOL)", "backend": "sol",   "device": "gpu"},
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


def classify_variant(run: dict):
    backend = str(run.get("backend", "")).lower().strip()
    device  = str(run.get("device",  "")).lower().strip()
    host    = str(run.get("host",    "")).lower().strip()

    if host != "edge-asus":
        return None

    for v_def in VARIANT_DEFINITIONS:
        if backend == v_def["backend"] and device == v_def["device"]:
            return v_def["label"]
    return None


def extract_rows(runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        model = str(r.get("model", "")).strip()

        pre  = float((r.get("preprocessing_ms")  or {}).get("mean", 0.0))
        inf  = float((r.get("inference_ms")       or {}).get("mean", 0.0))
        post = float((r.get("postprocessing_ms")  or {}).get("mean", 0.0))
        std  = float((r.get("system_ms")          or {}).get("std",  np.nan))

        rows.append((model, variant, pre, inf, post, std))
    return rows


def compute_offsets(n_variants: int):
    width = 0.18
    center = (n_variants - 1) / 2.0
    offsets = [(i - center) * width for i in range(n_variants)]
    return width, offsets


def plot(rows):
    if not rows:
        print("No rows found — check REMOTE_INFERENCE_FILE.")
        return

    allowed = [m.strip() for m in MODEL_TYPE_ORDER]
    rows = [r for r in rows if r[0] in allowed]
    if not rows:
        print("ERROR: No rows after filtering against MODEL_TYPE_ORDER.")
        return

    base_models = ordered_models(sorted({r[0] for r in rows}))

    pre_map = {(m, v): 0.0   for m in base_models for v in VARIANTS}
    inf_map = {(m, v): 0.0   for m in base_models for v in VARIANTS}
    post_map= {(m, v): 0.0   for m in base_models for v in VARIANTS}
    std_map = {(m, v): np.nan for m in base_models for v in VARIANTS}

    for model, variant, pre, inf, post, std in rows:
        if model in base_models and variant in VARIANTS:
            pre_map[(model, variant)]  = pre
            inf_map[(model, variant)]  = inf
            post_map[(model, variant)] = post
            std_map[(model, variant)]  = std

    # Print debug summary
    print(f"\n{'Model':<26} | {'Variant':<24} | {'Pre(ms)':>8} | {'Inf(ms)':>8} | {'Post(ms)':>8}")
    print("-" * 82)
    for m in base_models:
        for v in VARIANTS:
            inf = inf_map[(m, v)]
            if inf > 0:
                print(f"{m:<26} | {v:<24} | {pre_map[(m,v)]:8.2f} | {inf:8.2f} | {post_map[(m,v)]:8.2f}")
    print()

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )

    # Paired palette: blue pair for CPU, red pair for GPU
    paired_pal = sns.color_palette("Paired", 6)
    color_map = {
        "Edge CPU (PyTorch)": paired_pal[0],  # light blue
        "Edge CPU (vAccSOL)": paired_pal[1],  # dark blue
        "Edge GPU (PyTorch)": paired_pal[4],  # light red
        "Edge GPU (vAccSOL)": paired_pal[5],  # dark red
    }

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        in_cat = [m for m in base_models if m in cat]
        if in_cat:
            cat_models.append(in_cat)
            cat_captions.append(CAT_CAPTIONS[i])

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        1, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.05},
    )
    if len(cat_models) == 1:
        axes = [axes]

    # Global Y-limit across all subplots
    all_totals = [
        pre_map[(m, v)] + inf_map[(m, v)] + post_map[(m, v)]
        for m in base_models for v in VARIANTS
    ]
    y_max = np.nanmax(all_totals)
    y_limit = y_max * 1.08

    width_bar, offsets = compute_offsets(len(VARIANTS))

    for col_idx, (current_models, caption) in enumerate(zip(cat_models, cat_captions)):
        ax = axes[col_idx]
        x = np.arange(len(current_models))

        ax.set_ylim(0, y_limit)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=LATENCY_MAX_TICKS))
        ax.yaxis.set_major_formatter(FormatStrFormatter(LATENCY_TICK_FMT))

        for v_idx, v in enumerate(VARIANTS):
            xs = x + offsets[v_idx]
            pre_vals  = np.array([pre_map[(m, v)]  for m in current_models])
            inf_vals  = np.array([inf_map[(m, v)]  for m in current_models])
            post_vals = np.array([post_map[(m, v)] for m in current_models])
            yerr      = np.array([std_map[(m, v)]  for m in current_models])

            base_color  = color_map[v]
            light_color = mcolors.to_rgba(base_color, alpha=0.4)

            overhead_vals = pre_vals + post_vals

            # Inference (bottom, solid)
            ax.bar(
                xs, inf_vals, width=width_bar,
                facecolor=base_color, edgecolor="black", linewidth=STROKE_WIDTH,
                zorder=3,
                label=v if col_idx == 0 else "",
            )
            # Pre+Post-processing (top, dotted hatch)
            ax.bar(
                xs, overhead_vals, bottom=inf_vals, width=width_bar,
                facecolor=light_color, edgecolor="black", linewidth=STROKE_WIDTH,
                hatch="..", zorder=3,
            )

            if SHOW_ERROR_BARS:
                tot = inf_vals + overhead_vals
                ax.errorbar(
                    xs, tot, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=0, zorder=10,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [get_model_display_name(m) for m in current_models],
            rotation=15, ha="right",
        )
        style_axes(ax)
        ax.margins(x=0.01)

        if col_idx == 0:
            ax.set_ylabel("Latency (ms)")
        else:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.82)

    # --- Legend ---
    color_handles, color_labels = axes[0].get_legend_handles_labels()

    legend_base  = "lightgray"
    legend_light = mcolors.to_rgba(legend_base, alpha=0.4)
    hatch_handles = [
        mpatches.Patch(facecolor=legend_base,  edgecolor="black", hatch="",   label="Inference"),
        mpatches.Patch(facecolor=legend_light, edgecolor="black", hatch="..", label="Pre+Post-processing"),
    ]

    combined_handles = color_handles + hatch_handles
    combined_labels  = color_labels  + [h.get_label() for h in hatch_handles]

    leg = fig.legend(
        combined_handles, combined_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(combined_handles),
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    # Category captions
    CAPTION_OFFSET = 0.07
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes) - CAPTION_OFFSET
    for ax, cap in zip(axes, cat_captions):
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
    remote_path = Path(REMOTE_INFERENCE_FILE).resolve()
    if not remote_path.exists():
        raise SystemExit(f"File not found: {remote_path}")

    with remote_path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("JSON does not contain a non-empty 'runs' list.")

    rows = extract_rows(runs)

    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plt.rcParams["hatch.linewidth"] = STROKE_WIDTH

    plot(rows)


if __name__ == "__main__":
    main()
