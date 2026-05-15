#!/usr/bin/env python3
"""
model_stats_inference_latency_categorized.py

Adapted to match the 3-subplot style (Image, Video, Segmentation)
with independent auto-zooming Y-axes and a top-mounted legend.
Produces: model_stats_inference_latency_final.pdf
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
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
OUTPUT_FILE = "model_stats_inference_latency_final.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 5.5)

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


def extract_rows(runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        inf = r.get("system_ms", {}) or {}
        mean = inf.get("mean", None)
        std = inf.get("std", None)

        if mean is None:
            continue

        try:
            mean_f = float(mean)
            std_f = float(std) if std is not None else np.nan
        except Exception:
            continue

        rows.append((base_model_name(r.get("model", "")), variant, mean_f, std_f))
    return rows


def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.8))
    _, y_top = ax.get_ylim()
    pad_y = y_top * 0.015
    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue
        ax.text(
            x, y + pad_y, f"{y:.1f}",
            ha="center", va="bottom", rotation=90,
            fontsize=fs, color="black", fontweight="bold",
            clip_on=True, zorder=20
        )


def print_debug_table(val_map, base_models, variants):
    print("\n" + "=" * 115)
    print(f"{'DEBUG: EXTRACTED TOTAL SYSTEM LATENCY VALUES (ms)':^115}")
    print("=" * 115)

    header = f"{'Model':<20}"
    for v in variants:
        label = (v[:19] + "...") if len(v) > 22 else v
        header += f" | {label:<20}"
    print(header)
    print("-" * 115)

    for m in base_models:
        row_str = f"{m:<20}"
        for v in variants:
            val = val_map.get((m, v), np.nan)
            if np.isnan(val):
                row_str += f" | {'N/A':<20}"
            else:
                row_str += f" | {val:<20.2f}"
        print(row_str)
    print("=" * 115 + "\n")


def plot_latency_categorized(rows):
    if not rows:
        print("No matching rows found.")
        return

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print("\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  " + str(dropped) + "\n")

    rows = [(m, v, mu, sd) for (m, v, mu, sd) in rows if m in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering!")
        return

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}
    
    for m, v, mu, sd in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = mu
            std_map[(m, v)] = sd

    print_debug_table(val_map, base_models, variants)

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "in", "ytick.direction": "in"},
        font_scale=FONT_SCALE,
    )
    
    # Use seaborn palette defined in cfg or fallback to tab10
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

    if not cat_models:
        print("No category models found after filtering.")
        return

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

        # --- per-category y-limit (auto zoom) ---
        cat_vals = []
        cat_errs = []
        for m in current_models:
            for v in variants:
                cat_vals.append(val_map[(m, v)])
                cat_errs.append(std_map[(m, v)])
                
        cat_vals = np.asarray(cat_vals, dtype=float)
        cat_errs = np.asarray(cat_errs, dtype=float)

        ymax = np.nanmax(
            cat_vals + (np.nan_to_num(cat_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0)
        )
        y_lim_top = (ymax * 1.10) if np.isfinite(ymax) and ymax > 0 else 1.0
        ax.set_ylim(0, y_lim_top)

        for v in variants:
            xs = x + offsets[v]
            vals = np.asarray([val_map[(m, v)] for m in current_models], dtype=float)
            yerr = np.asarray([std_map[(m, v)] for m in current_models], dtype=float)

            ax.bar(
                xs, vals,
                width=bar_width,
                color=color_map[v],
                edgecolor=("black" if SHOW_ERROR_BARS else "none"),
                linewidth=(1.0 if SHOW_ERROR_BARS else 0.0),
                label=v if ax_idx == 0 else "",
                zorder=3,
            )

            if SHOW_ERROR_BARS:
                ax.errorbar(
                    xs, vals, yerr=yerr,
                    fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0,
                    zorder=10,
                )

            if SHOW_VALUE_LABELS:
                add_value_labels(ax, xs, vals)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [get_model_display_name(m) for m in current_models],
            rotation=15, ha="right"
        )
        
        # Highlight SOL slower than PyTorch logic per-axis
        if HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH and len(variants) >= 2:
            v0, v1 = variants[0], variants[1] # Assumes v0=ptc, v1=sol
            for tick, m in zip(ax.get_xticklabels(), current_models):
                mu_tc = val_map.get((m, v0), np.nan)
                mu_sol = val_map.get((m, v1), np.nan)
                if np.isfinite(mu_tc) and np.isfinite(mu_sol) and (mu_sol > mu_tc):
                    tick.set_color("red")

        style_axes(ax)
        ax.margins(x=0.005)

    # ---- Manual layout: reserve space for legend (top) and aligned captions (bottom) ----
    WSPACE = 0.12
    CAPTION_OFFSET = 0.15

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.22,
        top=0.78,
        wspace=WSPACE,
    )

    # Legend outside all axes
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        title="Execution via vAccel",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
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
        bbox_extra_artists=(leg, sy, *caption_artists),
    )
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
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

    rows = extract_rows(runs)
    plot_latency_categorized(rows)


if __name__ == "__main__":
    main()