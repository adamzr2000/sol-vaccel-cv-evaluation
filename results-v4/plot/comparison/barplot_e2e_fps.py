#!/usr/bin/env python3
"""
barplot_e2e_fps.py (comparison, paper-facing)

Unified vAccel-vs-ROS2 offloading comparison, FPS only: the top row of
barplot_e2e_latency_and_fps.py, on its own. No stacking here (a single FPS
number per bar, not a composition breakdown), so bars have no black edge
and there's no hatch/Inference/Pre-Post/Network legend to carry.

Color: hue = deployment scenario (gray=Local CPU, blue=Remote CPU,
red=Remote GPU). Within each hue, 4 shades cover framework x backend:
[ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL] -- same palette as
barplot_e2e_latency_and_fps.py, validated against scripts/validate_palette.py
(dataviz skill) to fully clear the normal-vision adjacency floor on all
three hues (gray 24.4, blue 19.1, red 16.5).

Produces: e2e-fps-comparison.pdf
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

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

OUTPUT_FILE = "e2e-fps-comparison.pdf"
FIG_SIZE = (18, 7.0)  # 1-row layout
FONT_SCALE = 2.6
SPINES_WIDTH = 1.0

FRAMEWORKS = ["ROS2", "vAccel"]
BACKENDS = ["Torch", "SOL"]
SCENARIOS = ["Local CPU", "Remote CPU", "Remote GPU"]
# fixed order within one scenario's 4-shade group:
# [ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL]
GROUP_ORDER = [(fw, be) for fw in FRAMEWORKS for be in BACKENDS]
FPS_VARIANT_ORDER = [(s, fw, be) for s in SCENARIOS for fw, be in GROUP_ORDER]

# Head-to-head connectors: within each hardware group, a short line directly
# linking ROS2 to vAccel for a given backend -- not a scaling trend across
# hardware, just the framework comparison at that placement. Solid = SOL,
# dashed = Torch. Violet, not used anywhere in the gray/blue/red bar
# palette, so the connector reads clearly against all three hue families
# (including the near-black Local CPU bars, where the previous black line
# used to disappear).
CONNECTOR_COLOR = "#6a3d9a"
CONNECTOR_SPECS = [
    ("SOL", "-", "o"),
    ("Torch", "--", "^"),
]
ROS2_IDX = {be: GROUP_ORDER.index(("ROS2", be)) for be in BACKENDS}
VACCEL_IDX = {be: GROUP_ORDER.index(("vAccel", be)) for be in BACKENDS}

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


def scenario_backend_from_variant(variant: str):
    scenario = variant.split(" (")[0].strip()
    backend = "SOL" if "SOL" in variant else "Torch"
    return scenario, backend


def load_merged_rows():
    """Merge vaccel_data.load_rows() and ros2_mod.load_rows() into a common
    (framework, model, scenario, backend, fps) shape."""
    merged = []

    for model, variant, fps, *_ in vaccel_data.load_rows():
        scenario, backend = scenario_backend_from_variant(variant)
        merged.append(("vAccel", model, scenario, backend, fps))

    for model, variant, fps, *_ in ros2_mod.load_rows():
        scenario, backend = scenario_backend_from_variant(variant)
        merged.append(("ROS2", model, scenario, backend, fps))

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
        for s in SCENARIOS
        for fw in FRAMEWORKS
        for be in BACKENDS
    ]
    fps_map = {k: np.nan for k in keys}

    for fw, model, scenario, backend, fps in merged:
        k = (model, scenario, fw, backend)
        if k in fps_map:
            fps_map[k] = fps

    return base_models, cat_models, cat_captions, fps_map


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

    base_models, cat_models, cat_captions, fps_map = build_maps(merged)

    widths = [len(cm) for cm in cat_models]
    fig, axes = plt.subplots(
        1, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20},
    )
    if len(cat_models) == 1:
        axes = np.array([axes])

    # 3 scenario-groups of 4 bars each, with a gap between groups
    bar_w = 0.065
    scenario_gap = 0.05
    group_w = len(GROUP_ORDER) * bar_w
    span = group_w + scenario_gap
    start = -1.5 * group_w - scenario_gap
    intra_offsets = compute_offsets(len(GROUP_ORDER), bar_w)

    panel_limits = {}
    for col_idx, current_models in enumerate(cat_models):
        vals = np.asarray(
            [fps_map[(m, s, fw, be)] for m in current_models for s, fw, be in FPS_VARIANT_ORDER],
            dtype=float,
        )
        ymax = np.nanmax(vals) if np.any(np.isfinite(vals)) else 1.0
        panel_limits[col_idx] = (ymax * 1.12) if ymax > 0 else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))
        ax = axes[col_idx]
        ax.set_ylim(0, panel_limits[col_idx])

        scenario_centers = [start + group_w / 2 + s_idx * span for s_idx in range(len(SCENARIOS))]

        for s_idx, scenario in enumerate(SCENARIOS):
            for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                xs = x + scenario_centers[s_idx] + intra_offsets[g_idx]
                vals = np.asarray([fps_map[(m, scenario, fw, be)] for m in current_models], dtype=float)
                ax.bar(
                    xs, vals, width=bar_w, color=GROUP_COLOR[(scenario, fw, be)],
                    zorder=3,
                )

        # Head-to-head connectors: within each scenario group, a short line
        # from ROS2 straight to vAccel for a given backend -- the framework
        # comparison at that specific hardware placement, not a cross-hardware
        # trend. Solid=SOL, dashed=Torch (see CONNECTOR_SPECS).
        for s_idx, scenario in enumerate(SCENARIOS):
            for be, ls, marker in CONNECTOR_SPECS:
                for m_idx, model in enumerate(current_models):
                    xs_pair = [
                        x[m_idx] + scenario_centers[s_idx] + intra_offsets[ROS2_IDX[be]],
                        x[m_idx] + scenario_centers[s_idx] + intra_offsets[VACCEL_IDX[be]],
                    ]
                    ys_pair = [
                        fps_map[(model, scenario, "ROS2", be)],
                        fps_map[(model, scenario, "vAccel", be)],
                    ]
                    ax.plot(
                        xs_pair, ys_pair, color=CONNECTOR_COLOR, linestyle=ls, marker=marker,
                        markersize=3.5, linewidth=1.0, zorder=6,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
        style_axes(ax)
        ax.margins(x=0.015)
        if col_idx == 0:
            ax.set_ylabel("Frame rate (FPS)")

    # ---- Manual layout ----
    CAPTION_OFFSET = 0.12
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.80)

    # Built explicitly (not via ax.get_legend_handles_labels(), which would
    # only capture whichever bars happened to carry a label= kwarg) so every
    # scenario x framework x backend combo gets its own legend entry.
    # matplotlib's fig.legend(ncol=N) fills COLUMN-major (top-to-bottom within
    # a column, then next column), so this order lands as:
    #   Col 1: ROS2+Torch (Local, Remote CPU, Remote GPU)
    #   Col 2: ROS2+SOL   (Local, Remote CPU, Remote GPU)
    #   Col 3: vAccel+Torch(Local, Remote CPU, Remote GPU)
    #   Col 4: vAccSOL     (Local, Remote CPU, Remote GPU)
    color_handles = [
        mpatches.Patch(facecolor=GROUP_COLOR[(scenario, fw, be)],
                        label=f"{scenario} ({GROUP_LABEL[(fw, be)]})")
        for fw, be in GROUP_ORDER
        for scenario in SCENARIOS
    ]
    color_labels = [h.get_label() for h in color_handles]

    leg = fig.legend(
        color_handles, color_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black", fontsize="x-small",
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes) - CAPTION_OFFSET
    for ax, cap in zip(axes, cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(OUTPUT_FILE, bbox_extra_artists=(leg, *caption_artists))
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

    merged = load_merged_rows()
    plot_combined(merged)


if __name__ == "__main__":
    main()
