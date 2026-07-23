#!/usr/bin/env python3
"""
barplot_e2e_fps.py (comparison, paper-facing)

Unified vAccel-vs-ROS2 offloading comparison, latency-only: same as
barplot_e2e_latency_and_fps.py but with the FPS row dropped for a simpler
3-row figure (Local/Remote CPU/Remote GPU E2E latency breakdown only).

Color has exactly ONE meaning everywhere in this figure -- hue = framework
(vAccel=blue, ROS2=orange), shade = backend (light=Torch, solid=SOL).
Scenario doesn't get a color slot -- it's conveyed by which row you're
looking at (each row's own y-axis label names it), which is also a hard
requirement independent of color: Local CPU latency (~700-1200ms) and
Remote GPU latency (~15-130ms) are two orders of magnitude apart and can't
share a y-axis, so latency is row-faceted by scenario regardless.

Within each latency row, framework is additionally split into two blocks
(vAccel bars, then ROS2 bars, separated by a small gap) so the color
distinction is reinforced positionally, not just by hue.

Row order:
  Row 0 - Local CPU  E2E latency breakdown
  Row 1 - Remote CPU E2E latency breakdown
  Row 2 - Remote GPU E2E latency breakdown

Produces: e2e-latency-comparison.pdf
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

_HERE = Path(__file__).parent
_VACCEL_DIR = _HERE.parent / "vaccel"
_ROS2_DIR = _HERE.parent / "ros2"

# vaccel's plot_config resolves its JSON paths relative to cwd (not __file__);
# comparison/ sits at the same depth under results/plot/ as vaccel/ and ros2/,
# so its "../../experiments/..." paths resolve the same from here.
os.chdir(_HERE)

sys.path.insert(0, str(_VACCEL_DIR))
import plot_data_e2e as vaccel_data  # noqa: E402
from plot_config import get_model_display_name  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# loaded by explicit path (not sys.path) since both source dirs have a file
# of this same name -- avoids import collisions with each other and with us.
ros2_mod = _load_module(_ROS2_DIR / "barplot_e2e_latency_and_fps.py", "_ros2_e2e_source")

# ros2_mod.load_rows() reads SEG_REMOTE_GPU_TAG_OVERRIDE from its own file
# (ros2/barplot_e2e_latency_and_fps.py) -- this script inherits whatever's
# set there automatically. ROS2 only, Remote GPU only, applied uniformly
# to every semantic-segmentation model. Set it on ros2_mod directly for a
# run limited to this script:
#
# ros2_mod.SEG_REMOTE_GPU_TAG_OVERRIDE = "remote-gpu12fps"

OUTPUT_FILE = "e2e-latency-comparison.pdf"
FIG_SIZE = (18, 11.5)  # 3-row layout (no FPS row)
FONT_SCALE = 2.4
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.7
LATENCY_MAX_TICKS = 5
LATENCY_TICK_FMT = "%.0f"

FRAMEWORKS = ["ROS2", "vAccel"]
BACKENDS = ["Torch", "SOL"]
SCENARIOS = [
    ("Local CPU", "Local CPU\nLatency (ms)"),
    ("Remote CPU", "Remote CPU\nLatency (ms)"),
    ("Remote GPU", "Remote GPU\nLatency (ms)"),
]

ALLOWED_MODELS = [m.strip() for m in vaccel_data.MODEL_TYPE_ORDER]
CATEGORIES = vaccel_data.CATEGORIES
CAT_CAPTIONS = vaccel_data.CAT_CAPTIONS

# Color: hue = framework, shade = backend. Tint amount validated against
# scripts/validate_palette.py -- don't change without re-running it, smaller
# values collapse the two shades into each other for the orange hue.
FRAMEWORK_HUE = {"vAccel": "#2a78d6", "ROS2": "#eb6834"}
TINT = 0.45


def tint(hex_color: str, amount: float = TINT):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


BACKEND_COLOR = {
    (fw, "Torch"): tint(base) for fw, base in FRAMEWORK_HUE.items()
} | {
    (fw, "SOL"): base for fw, base in FRAMEWORK_HUE.items()
}
# fixed bar order per model-group: framework outer (blocks stay visually
# together), backend inner
GROUP_ORDER = [(fw, be) for fw in FRAMEWORKS for be in BACKENDS]
# "vAccSOL" is the paper's own name for the vAccel+SOL combo; everything
# else stays "<framework> + <backend>"
GROUP_LABEL = {
    (fw, be): "vAccSOL" if (fw, be) == ("vAccel", "SOL") else f"{fw} + {be}"
    for fw, be in GROUP_ORDER
}


def light_overlay(color, alpha: float = 0.4):
    return mcolors.to_rgba(color, alpha=alpha)


def scenario_backend_from_variant(variant: str):
    scenario = variant.split(" (")[0].strip()
    backend = "SOL" if "SOL" in variant else "Torch"
    return scenario, backend


def load_merged_rows():
    """Merge vaccel_data.load_rows() and ros2_mod.load_rows() into a common
    (framework, model, scenario, backend, fps, inf, pre, net, lo, hi) shape.
    Note the two loaders return their (net, pre) columns in different order."""
    merged = []

    for model, variant, fps, inf, net, pre, lo, hi in vaccel_data.load_rows():
        scenario, backend = scenario_backend_from_variant(variant)
        merged.append(("vAccel", model, scenario, backend, fps, inf, pre, net, lo, hi))

    for model, variant, fps, inf, pre, net, lo, hi in ros2_mod.load_rows():
        scenario, backend = scenario_backend_from_variant(variant)
        merged.append(("ROS2", model, scenario, backend, fps, inf, pre, net, lo, hi))

    return merged


def build_maps(merged):
    present = sorted({m for _, m, *_ in merged if m in ALLOWED_MODELS})
    base_models = vaccel_data.ordered_models(present)

    cat_models, cat_captions = [], []
    for cat, cap in zip(CATEGORIES, CAT_CAPTIONS):
        in_cat = [m for m in base_models if m in cat]
        if in_cat:
            cat_models.append(in_cat)
            cat_captions.append(cap)

    keys = [
        (m, s, fw, be)
        for m in base_models
        for s, _ in SCENARIOS
        for fw in FRAMEWORKS
        for be in BACKENDS
    ]
    fps_map = {k: np.nan for k in keys}
    inf_map = {k: 0.0 for k in keys}
    pre_map = {k: 0.0 for k in keys}
    net_map = {k: 0.0 for k in keys}
    lo_map = {k: np.nan for k in keys}
    hi_map = {k: np.nan for k in keys}

    for fw, model, scenario, backend, fps, inf, pre, net, lo, hi in merged:
        k = (model, scenario, fw, backend)
        if k not in fps_map:
            continue
        fps_map[k] = fps
        inf_map[k] = inf
        pre_map[k] = pre
        net_map[k] = net
        lo_map[k] = lo
        hi_map[k] = hi

    return base_models, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map, lo_map, hi_map


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def compute_group_offsets(width: float, boundary_gap: float):
    """4-bar offsets, framework outer / backend inner, with boundary_gap
    inserted between the two framework blocks so vAccel/ROS2 separate
    visually. Shared by both the FPS row's per-scenario groups and the
    latency rows."""
    n = len(GROUP_ORDER)
    boundary = len(BACKENDS)
    positions, x = [], 0.0
    for i in range(n):
        if i == boundary:
            x += boundary_gap
        positions.append(x)
        x += width
    total = positions[-1] + width
    shift = total / 2
    return {i: positions[i] + width / 2 - shift for i in range(n)}, total


def plot_combined(merged):
    if not merged:
        print("No matching rows found.")
        return

    base_models, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map, lo_map, hi_map = build_maps(merged)

    widths = [len(cm) for cm in cat_models]
    fig, axes = plt.subplots(
        3, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20, "hspace": 0.12},
    )
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    lat_bar_w = 0.15
    lat_offsets, lat_group_w = compute_group_offsets(lat_bar_w, boundary_gap=0.0)

    # --- pre-calculate per-panel y-limits (each subplot scales independently) ---
    panel_lat_limits = {}  # (row_idx, col_idx) -> ylim

    for col_idx, current_models in enumerate(cat_models):
        for row_idx, (scenario, _) in enumerate(SCENARIOS):
            totals = np.asarray(
                [
                    inf_map[(m, scenario, fw, be)] + pre_map[(m, scenario, fw, be)]
                    + net_map[(m, scenario, fw, be)] + hi_map[(m, scenario, fw, be)]
                    for m in current_models
                    for fw, be in GROUP_ORDER
                ],
                dtype=float,
            )
            ymax = np.nanmax(totals) if np.any(np.isfinite(totals)) else 1.0
            panel_lat_limits[(row_idx, col_idx)] = (ymax * 1.08) if ymax > 0 else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        # ===================== ROWS 0-2: LATENCY BREAKDOWN =====================
        for row_idx, (scenario, ylabel) in enumerate(SCENARIOS):
            ax_lat = axes[row_idx, col_idx]
            ax_lat.set_ylim(0, panel_lat_limits[(row_idx, col_idx)])
            ax_lat.yaxis.set_major_locator(MaxNLocator(nbins=LATENCY_MAX_TICKS))
            ax_lat.yaxis.set_major_formatter(FormatStrFormatter(LATENCY_TICK_FMT))

            for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                xs = x + lat_offsets[g_idx]
                inf_vals = np.asarray([inf_map[(m, scenario, fw, be)] for m in current_models], dtype=float)
                pre_vals = np.asarray([pre_map[(m, scenario, fw, be)] for m in current_models], dtype=float)
                net_vals = np.asarray([net_map[(m, scenario, fw, be)] for m in current_models], dtype=float)
                yerr = np.array([
                    [lo_map[(m, scenario, fw, be)] for m in current_models],
                    [hi_map[(m, scenario, fw, be)] for m in current_models],
                ], dtype=float)
                tot_vals = inf_vals + pre_vals + net_vals

                base_color = BACKEND_COLOR[(fw, be)]
                overlay = light_overlay(base_color)

                ax_lat.bar(
                    xs, inf_vals, width=lat_bar_w, facecolor=base_color,
                    edgecolor="black", linewidth=STROKE_WIDTH,
                    label=GROUP_LABEL[(fw, be)] if (col_idx == 0 and row_idx == 0) else "",
                    zorder=3,
                )
                ax_lat.bar(
                    xs, pre_vals, bottom=inf_vals, width=lat_bar_w, facecolor=overlay,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="..", zorder=3,
                )
                ax_lat.bar(
                    xs, net_vals, bottom=(inf_vals + pre_vals), width=lat_bar_w, facecolor=overlay,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="//", zorder=3,
                )
                ax_lat.errorbar(
                    xs, tot_vals, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10,
                )

            ax_lat.set_xticks(x)
            if row_idx == len(SCENARIOS) - 1:
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])

            style_axes(ax_lat)
            ax_lat.margins(x=0.01)
            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)

    # ---- Manual layout ----
    CAPTION_OFFSET = 0.075
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.10, top=0.86)

    color_handles, color_labels = axes[0, 0].get_legend_handles_labels()  # from row 0 (Local CPU)
    hatch_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="", label="Inference"),
        mpatches.Patch(facecolor=light_overlay("lightgray"), edgecolor="black", hatch="..", label="Pre/Post-processing"),
        mpatches.Patch(facecolor=light_overlay("lightgray"), edgecolor="black", hatch="//", label="Network + Framework Overhead"),
    ]

    # Two separate legends, not one combined list: fig.legend(ncol=N) fills
    # COLUMN-major (top-to-bottom within a column, then next column), not
    # row-major -- a combined 4-color + 3-hatch list at ncol=4 lands as
    # [vAccel-T, ROS2-T, Inference, Overhead] / [vAccel-S, ROS2-S, Pre-Post, --],
    # not the clean "colors row, hatches row" split the layout wants.
    leg = fig.legend(
        color_handles, color_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black",
    )
    fig.add_artist(leg)
    hatch_leg = fig.legend(
        hatch_handles, [h.get_label() for h in hatch_handles],
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=3, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black",
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[-1, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[-1, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(OUTPUT_FILE, bbox_extra_artists=(leg, hatch_leg, *caption_artists))
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    sns.set_theme(
        context="paper", style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    plt.rcParams["hatch.linewidth"] = STROKE_WIDTH

    merged = load_merged_rows()
    plot_combined(merged)


if __name__ == "__main__":
    main()
