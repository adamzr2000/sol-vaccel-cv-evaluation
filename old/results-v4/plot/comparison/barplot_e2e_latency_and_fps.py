#!/usr/bin/env python3
"""
barplot_e2e_latency_and_fps.py (comparison, paper-facing)

Unified vAccel-vs-ROS2 offloading comparison, compacted to the same 4-row
layout as the two source plots (barplot_e2e_latency_and_fps_full.py has
the exhaustive 6-row backup). The individual vaccel/ and ros2/ plots are
not shown in the paper, so this figure uses its own palette.

Color: hue = deployment scenario (gray=Local CPU, blue=Remote CPU,
red=Remote GPU). Within each hue, 4 shades cover framework x backend:
[ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL], ordered light/dark/mid/
darkest (not a plain light-to-dark ramp -- swapping the middle two steps
widens the gap between adjacent bars, same trick as the palette this
replaced).

Checked against scripts/validate_palette.py (dataviz skill): this is the
first palette in this figure's history where all three hues fully clear
the normal-vision adjacency floor (target >=15) -- gray 24.4, blue 19.1,
red 16.5. Everything tried before this (raw tint/shade, several ColorBrewer
slicings, seaborn light_palette, IBM/muted-diverging alternatives) topped
out around 11-14.

Row order:
  Row 0 - FPS (3 scenario-groups x framework x backend, all data)
  Row 1 - Local CPU  E2E latency breakdown
  Row 2 - Remote CPU E2E latency breakdown
  Row 3 - Remote GPU E2E latency breakdown

Produces: e2e-latency-and-fps-comparison.pdf
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

# Set the override on ros2_mod directly (a plain reassignment here would
# only rebind this file's own name, not ros2_mod.load_rows()'s). ROS2 only,
# Remote GPU only, applied uniformly to every semantic-segmentation model
# (see barplot_e2e_latency_and_fps.py's load_rows() for the override
# logic). None (default) = every model reads remote-gpu_model_stats.json.
# The energy plots (ros2/barplot_energy_consumption.py,
# ros2/barplot_energy_per_frame.py) read this exact same variable name
# from that same file, since energy and FPS/latency come from the same
# underlying benchmark run and must stay in sync.
#
# ros2_mod.SEG_REMOTE_GPU_TAG_OVERRIDE = "remote-gpu12fps"

OUTPUT_FILE = "e2e-latency-and-fps-comparison.pdf"
FIG_SIZE = (18, 15.0)  # 4-row layout, same footprint as the source plots
FONT_SCALE = 2.2
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
# fixed order within one scenario's 4-shade group, lightest to darkest:
# [ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL]
GROUP_ORDER = [(fw, be) for fw in FRAMEWORKS for be in BACKENDS]
# FPS row shows all 3 scenarios x GROUP_ORDER, same scenario order as the
# latency rows below
FPS_VARIANT_ORDER = [(s, fw, be) for s, _ in SCENARIOS for fw, be in GROUP_ORDER]

ALLOWED_MODELS = [m.strip() for m in vaccel_data.MODEL_TYPE_ORDER]
CATEGORIES = vaccel_data.CATEGORIES
CAT_CAPTIONS = vaccel_data.CAT_CAPTIONS

# Scenario hues, 4 steps each, mapped onto GROUP_ORDER [ROS2+Torch,
# ROS2+SOL, vAccel+Torch, vAccSOL]. Steps ordered light/dark/mid/darkest
# (middle two swapped from a plain ramp) to widen the gap between adjacent
# bars. Grayscale + ColorBrewer Blues/Reds, each hue's step 1 skipped
# (too close to white) -- validated against scripts/validate_palette.py
# (dataviz skill) as the first combination in this figure's history to
# fully clear the normal-vision adjacency floor (target >=15) on all three
# hues: gray 24.4, blue 19.1, red 16.5.
SCENARIO_STEPS = {
    "Local CPU":  ["#e0e0e0", "#525252", "#999999", "#000000"],
    "Remote CPU": ["#bdd7e7", "#2171b5", "#6baed6", "#08306b"],
    "Remote GPU": ["#fcae91", "#cb181d", "#fb6a4a", "#67000d"],
}

GROUP_COLOR = {
    (scenario, fw, be): steps[i]
    for scenario, steps in SCENARIO_STEPS.items()
    for i, (fw, be) in enumerate(GROUP_ORDER)
}
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


def compute_offsets(n: int, width: float):
    """n contiguous bars centered on 0."""
    center = (n - 1) / 2.0
    return {i: (i - center) * width for i in range(n)}


def plot_combined(merged):
    if not merged:
        print("No matching rows found.")
        return

    base_models, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map, lo_map, hi_map = build_maps(merged)

    widths = [len(cm) for cm in cat_models]
    fig, axes = plt.subplots(
        4, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20, "hspace": 0.12},
    )
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])

    # FPS row: 3 scenario-groups of 4 bars each, with a gap between groups
    fps_bar_w = 0.05
    fps_scenario_gap = 0.045
    fps_group_w = len(GROUP_ORDER) * fps_bar_w
    fps_span = fps_group_w + fps_scenario_gap
    fps_start = -1.5 * fps_group_w - fps_scenario_gap
    fps_intra_offsets = compute_offsets(len(GROUP_ORDER), fps_bar_w)

    # Latency rows: single scenario, 4 contiguous bars
    lat_bar_w = 0.17
    lat_offsets = compute_offsets(len(GROUP_ORDER), lat_bar_w)

    # --- pre-calculate per-panel y-limits (each subplot scales independently) ---
    panel_fps_limits = {}
    panel_lat_limits = {}  # (row_idx, col_idx) -> ylim

    for col_idx, current_models in enumerate(cat_models):
        fps_vals = np.asarray(
            [
                fps_map[(m, s, fw, be)]
                for m in current_models
                for s, fw, be in FPS_VARIANT_ORDER
            ],
            dtype=float,
        )
        ymax = np.nanmax(fps_vals) if np.any(np.isfinite(fps_vals)) else 1.0
        panel_fps_limits[col_idx] = (ymax * 1.12) if ymax > 0 else 1.0

        for row_idx, (scenario, _) in enumerate(SCENARIOS, start=1):
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

        # ============================= ROW 0: FPS =============================
        ax_fps = axes[0, col_idx]
        ax_fps.set_ylim(0, panel_fps_limits[col_idx])

        for s_idx, (scenario, _) in enumerate(SCENARIOS):
            scenario_center = fps_start + fps_group_w / 2 + s_idx * fps_span
            for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                xs = x + scenario_center + fps_intra_offsets[g_idx]
                vals = np.asarray([fps_map[(m, scenario, fw, be)] for m in current_models], dtype=float)
                ax_fps.bar(
                    xs, vals, width=fps_bar_w, color=GROUP_COLOR[(scenario, fw, be)],
                    edgecolor="black", linewidth=STROKE_WIDTH,
                    label=f"{scenario} ({GROUP_LABEL[(fw, be)]})" if (col_idx == 0 and s_idx == 0) else "",
                    zorder=3,
                )

        ax_fps.set_xticks(x)
        ax_fps.set_xticklabels([])
        style_axes(ax_fps)
        ax_fps.margins(x=0.015)
        if col_idx == 0:
            ax_fps.set_ylabel("Frame rate (FPS)")

        # ===================== ROWS 1-3: LATENCY BREAKDOWN =====================
        for row_idx, (scenario, ylabel) in enumerate(SCENARIOS, start=1):
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

                base_color = GROUP_COLOR[(scenario, fw, be)]
                overlay = light_overlay(base_color)

                ax_lat.bar(
                    xs, inf_vals, width=lat_bar_w, facecolor=base_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, zorder=3,
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
            if row_idx == 3:
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])

            style_axes(ax_lat)
            ax_lat.margins(x=0.01)
            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)

    # ---- Manual layout ----
    CAPTION_OFFSET = 0.075
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.10, top=0.84)

    scenario_labels = [s for s, _ in SCENARIOS]

    color_handles = [
        mpatches.Patch(facecolor=GROUP_COLOR[(scenario, fw, be)], edgecolor="black",
                        label=f"{scenario} ({GROUP_LABEL[(fw, be)]})")
        for fw, be in GROUP_ORDER
        for scenario in scenario_labels
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="", label="Inference"),
        mpatches.Patch(facecolor=light_overlay("lightgray"), edgecolor="black", hatch="..", label="Pre/Post-processing"),
        mpatches.Patch(facecolor=light_overlay("lightgray"), edgecolor="black", hatch="//", label="Network + Framework Overhead"),
    ]

    # matplotlib's fig.legend(ncol=N) fills COLUMN-major (top-to-bottom within
    # a column, then next column) -- not row-major. color_handles is already
    # ordered [(fw,be) outer, scenario inner], so with ncol=4 that lands as:
    #   Col 1: ROS2+Torch  (Local, Remote CPU, Remote GPU)
    #   Col 2: ROS2+SOL    (Local, Remote CPU, Remote GPU)
    #   Col 3: vAccel+Torch(Local, Remote CPU, Remote GPU)
    #   Col 4: vAccSOL     (Local, Remote CPU, Remote GPU)
    # Hatches are a separate 3-item legend so they don't force an uneven
    # 16th slot into the 4-column color grid.
    color_labels = [h.get_label() for h in color_handles]
    leg = fig.legend(
        color_handles, color_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black", fontsize="small",
    )
    fig.add_artist(leg)
    hatch_leg = fig.legend(
        hatch_handles, [h.get_label() for h in hatch_handles],
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.895),
        ncol=3, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black", fontsize="small",
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[3, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[3, :], cat_captions):
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
