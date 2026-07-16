#!/usr/bin/env python3
"""
barplot_energy_per_frame.py (comparison, paper-facing)

Unified vAccel-vs-ROS2 energy-per-frame comparison. Merges the vaccel/ and
ros2/ energy datasets into one 3-row x 3-column figure. Same STRUCTURE as
barplot_e2e_latency.py / barplot_e2e_latency_and_fps.py (3-hue x 4-shade,
GROUP_ORDER = [ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL]) but a
deliberately different palette family, so this figure reads as visually
distinct from the latency ones rather than a repeat: teal (crest) for
Local CPU, orange/magenta (flare) for Remote CPU, purple (Purples) for
Remote GPU -- extending the hue families both source energy scripts
already used, instead of the gray/blue/red set the latency figures use.
Steps chosen the same way (validated against scripts/validate_palette.py,
dataviz skill; middle two steps swapped to widen the adjacent-bar gap) --
all three hues clear the normal-vision floor (target >=15): teal 18.2,
orange/magenta 19.4, purple 16.4.

Row structure follows both source scripts, which is NOT three symmetric
scenario rows (unlike the latency figure) -- it's three different energy
domains that don't all apply to every scenario:
  Row 0 - Robot CPU (J/frame): the robot's own CPU energy, present in all
          3 scenarios (Local CPU, Remote CPU, Remote GPU all keep the robot
          busy running the vision pipeline) -- 3 scenario-groups x 4 bars,
          same nested layout as the FPS row in barplot_e2e_latency_and_fps.py.
  Row 1 - Edge CPU (J/frame): edge-side CPU energy, only meaningful for the
          Remote CPU scenario (edge CPU is idle for Remote GPU, doesn't
          exist for Local CPU) -- 4 bars, one scenario.
  Row 2 - Edge GPU (J/frame): edge-side GPU energy, only meaningful for the
          Remote GPU scenario -- 4 bars, one scenario.

Not carried over from the source scripts: the per-model Torch->SOL savings
badges (arrows + % labels). That's a lot of annotation complexity the other
comparison figures don't have either; add back if wanted.

Produces: energy-per-frame-comparison.pdf
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

_HERE = Path(__file__).parent
_VACCEL_DIR = _HERE.parent / "vaccel"
_ROS2_DIR = _HERE.parent / "ros2"

# vaccel's plot_config resolves its JSON/CSV paths relative to cwd (not
# __file__); comparison/ sits at the same depth under results/plot/ as
# vaccel/ and ros2/, so its "../../experiments/..." paths resolve the same
# from here.
os.chdir(_HERE)

sys.path.insert(0, str(_VACCEL_DIR))
import plot_data_e2e as vaccel_data  # noqa: E402  (shared CATEGORIES/model order)
from plot_config import get_model_display_name  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# loaded by explicit path (not sys.path) since vaccel/, ros2/, and this file
# all have a barplot_energy_per_frame.py -- avoids import collisions.
vaccel_epf = _load_module(_VACCEL_DIR / "barplot_energy_per_frame.py", "_vaccel_epf_source")
ros2_epf = _load_module(_ROS2_DIR / "barplot_energy_per_frame.py", "_ros2_epf_source")

OUTPUT_FILE = "energy-per-frame-comparison.pdf"
FIG_SIZE = (18, 11.5)  # 3-row layout
FONT_SCALE = 2.2
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.8
ENERGY_MAX_TICKS = 5
ENERGY_TICK_FMT = "%.1f"

FRAMEWORKS = ["ROS2", "vAccel"]
BACKENDS = ["Torch", "SOL"]
SCENARIOS = ["Local CPU", "Remote CPU", "Remote GPU"]
# fixed order within one scenario's 4-shade group:
# [ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL]
GROUP_ORDER = [(fw, be) for fw in FRAMEWORKS for be in BACKENDS]

# Manual switch: draw a head-to-head connector (ROS2 -> vAccel, per backend)
# on the Robot CPU row. Local CPU only, not Remote CPU/GPU -- Robot CPU is
# dominated by Local CPU (the robot does the real work there; Remote CPU/GPU
# leave the robot mostly idle, see the plot), so that's the one comparison
# worth calling out without cluttering every scenario-group.
SHOW_TREND = True
CONNECTOR_COLOR = "#6a3d9a"
CONNECTOR_SCENARIO = "Local CPU"
CONNECTOR_SPECS = [
    ("SOL", "-", "o"),
    ("Torch", "--", "^"),
]
ROS2_IDX = {be: GROUP_ORDER.index(("ROS2", be)) for be in BACKENDS}
VACCEL_IDX = {be: GROUP_ORDER.index(("vAccel", be)) for be in BACKENDS}

# Manual switch: zoom inset on the Robot CPU row, re-drawing just the Remote
# CPU/GPU scenario-groups at their own scale (see the comment where it's
# used, in plot_combined). Toggle off to fall back to the plain axis.
SHOW_ROW0_ZOOM_INSET = True
ZOOM_INSET_SCENARIOS = ["Remote CPU", "Remote GPU"]
ZOOM_INSET_BBOX = [0.55, 0.72, 0.43, 0.24]  # [x0, y0, width, height] in axes-fraction coords
ZOOM_BAR_W = 0.085  # wider than row0_bar_w -- inset only shows 2 scenario-groups, not 3, so there's room
ZOOM_GROUP_GAP = 0.05

ALLOWED_MODELS = [m.strip() for m in vaccel_data.MODEL_TYPE_ORDER]
CATEGORIES = vaccel_data.CATEGORIES
CAT_CAPTIONS = vaccel_data.CAT_CAPTIONS

# Deliberately different hue family from the latency figures' gray/blue/red
# -- teal=Local CPU, orange/magenta=Remote CPU, purple=Remote GPU (extends
# the crest/flare/Purples families both source energy scripts already use).
# 4 shades each, steps ordered light/dark/mid/darkest (middle two swapped
# from a plain ramp) to widen the gap between adjacent bars. Validated
# against scripts/validate_palette.py (dataviz skill): teal 18.2, orange
# 19.4, purple 16.4 (target >=15, all pass).
SCENARIO_STEPS = {
    "Local CPU":  ["#8bc191", "#1d6188", "#4a9a8f", "#29427a"],
    "Remote CPU": ["#eb9973", "#8b3271", "#db565d", "#602969"],
    "Remote GPU": ["#bebfdd", "#6e58a7", "#9390c3", "#4e1c8a"],
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


def load_vaccel_epf_rows():
    """Replicates vaccel/barplot_energy_per_frame.py's main() data path,
    returning (robot_rows, edge_cpu_rows, edge_gpu_rows) -- lists of dicts
    with base_model/variant/mean/std keys."""
    cpu_path = Path(vaccel_epf.CPU_FILE).resolve()
    iso_cpu_path = Path(vaccel_epf.ISO_CPU_FILE).resolve()
    gpu_path = Path(vaccel_epf.GPU_FILE).resolve()

    cpu_df = pd.read_csv(cpu_path)
    if iso_cpu_path.exists():
        cpu_df = pd.concat([cpu_df, pd.read_csv(iso_cpu_path)], ignore_index=True)
    else:
        print(f"[WARNING] ISO CPU file not found -- vAccel local robot rows will be missing: {iso_cpu_path}")
    gpu_df = pd.read_csv(gpu_path)

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    iso_perf = vaccel_epf.build_perf_map(vaccel_epf.ISO_PERF_FILE)
    e2e_perf = vaccel_epf.build_perf_map(vaccel_epf.E2E_PERF_FILE)

    robot_rows = vaccel_epf.load_robot_cpu_rows(cpu_df, iso_perf, e2e_perf)
    edge_cpu_rows = vaccel_epf.load_edge_cpu_remote_rows(cpu_df, e2e_perf)
    edge_gpu_rows = vaccel_epf.load_edge_gpu_remote_rows(gpu_df, e2e_perf)

    robot_rows = vaccel_epf._apply_model_filter(robot_rows, ALLOWED_MODELS)
    edge_cpu_rows = vaccel_epf._apply_model_filter(edge_cpu_rows, ALLOWED_MODELS)
    edge_gpu_rows = vaccel_epf._apply_model_filter(edge_gpu_rows, ALLOWED_MODELS)

    return robot_rows, edge_cpu_rows, edge_gpu_rows


def load_merged_maps():
    """Returns three dicts:
      robot_cpu_map: (model, scenario, fw, be) -> J/frame
      edge_cpu_map:  (model, fw, be) -> J/frame   (Remote CPU scenario only)
      edge_gpu_map:  (model, fw, be) -> J/frame   (Remote GPU scenario only)
    """
    robot_cpu_map, edge_cpu_map, edge_gpu_map = {}, {}, {}

    robot_rows, edge_cpu_rows, edge_gpu_rows = load_vaccel_epf_rows()
    for r in robot_rows:
        scenario, backend = scenario_backend_from_variant(r["variant"])
        robot_cpu_map[(r["base_model"], scenario, "vAccel", backend)] = r["mean"]
    for r in edge_cpu_rows:
        _scenario, backend = scenario_backend_from_variant(r["variant"])
        edge_cpu_map[(r["base_model"], "vAccel", backend)] = r["mean"]
    for r in edge_gpu_rows:
        _scenario, backend = scenario_backend_from_variant(r["variant"])
        edge_gpu_map[(r["base_model"], "vAccel", backend)] = r["mean"]

    frame_counts = ros2_epf.load_frame_counts()
    ros2_map = ros2_epf.load_epf_map(frame_counts)
    for (model, variant, host, kind), epf in ros2_map.items():
        if model not in ALLOWED_MODELS:
            continue
        scenario, backend = scenario_backend_from_variant(variant)
        if host == "robot" and kind == "cpu":
            robot_cpu_map[(model, scenario, "ROS2", backend)] = epf
        elif host == "edge-asus" and kind == "cpu" and scenario == "Remote CPU":
            edge_cpu_map[(model, "ROS2", backend)] = epf
        elif host == "edge-asus" and kind == "gpu" and scenario == "Remote GPU":
            edge_gpu_map[(model, "ROS2", backend)] = epf

    return robot_cpu_map, edge_cpu_map, edge_gpu_map


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def rounded_ylim(y_max: float) -> float:
    """Round y ceiling to a clean step (same convention as both source scripts)."""
    top = y_max * 1.05
    if top >= 10:
        step = 2.0
    elif top >= 1:
        step = 0.5
    elif top >= 0.1:
        step = 0.05
    else:
        step = 0.01
    return float(np.ceil(top / step) * step)


def compute_offsets(n: int, width: float):
    """n contiguous bars centered on 0."""
    center = (n - 1) / 2.0
    return {i: (i - center) * width for i in range(n)}


def plot_combined(robot_cpu_map, edge_cpu_map, edge_gpu_map):
    present = sorted({m for (m, *_rest) in robot_cpu_map})
    base_models = vaccel_data.ordered_models(present)

    cat_models, cat_captions = [], []
    for cat, cap in zip(CATEGORIES, CAT_CAPTIONS):
        in_cat = [m for m in base_models if m in cat]
        if in_cat:
            cat_models.append(in_cat)
            cat_captions.append(cap)

    widths = [len(cm) for cm in cat_models]
    fig, axes = plt.subplots(
        3, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20, "hspace": 0.14},
    )
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    # Row 0 (Robot CPU): 3 scenario-groups of 4 bars each, with a gap between groups
    row0_bar_w = 0.05
    scenario_gap = 0.045
    group_w = len(GROUP_ORDER) * row0_bar_w
    span = group_w + scenario_gap
    start = -1.5 * group_w - scenario_gap
    row0_intra_offsets = compute_offsets(len(GROUP_ORDER), row0_bar_w)

    # Rows 1-2 (Edge CPU / Edge GPU): single scenario, 4 contiguous bars
    row_bar_w = 0.17
    row_offsets = compute_offsets(len(GROUP_ORDER), row_bar_w)

    # Row 0 zoom inset: its own tight 2-group layout (Remote CPU, Remote GPU
    # only), not a reuse of the 3-group row0 spacing -- filling the space
    # that Local CPU's now-omitted group would have taken, so bars are wider
    # instead of leaving a blank gap where Local CPU used to sit.
    zoom_group_w = len(GROUP_ORDER) * ZOOM_BAR_W
    zoom_span = zoom_group_w + ZOOM_GROUP_GAP
    zoom_n = len(ZOOM_INSET_SCENARIOS)
    zoom_start = -((zoom_n - 1) / 2) * zoom_span
    zoom_intra_offsets = compute_offsets(len(GROUP_ORDER), ZOOM_BAR_W)
    # pad must cover the full cluster half-width (all zoom_n groups + gaps),
    # not just one group's span, or the outermost bars clip at the edges
    zoom_total_w = zoom_n * zoom_group_w + (zoom_n - 1) * ZOOM_GROUP_GAP
    zoom_xlim_pad = zoom_total_w / 2 + 0.03

    # --- pre-calculate per-panel y-limits (each subplot scales independently) ---
    panel_ylim = {}  # (row_idx, col_idx) -> ylim
    for col_idx, current_models in enumerate(cat_models):
        vals0 = np.asarray(
            [
                robot_cpu_map.get((m, s, fw, be), np.nan)
                for m in current_models for s in SCENARIOS for fw, be in GROUP_ORDER
            ],
            dtype=float,
        )
        panel_ylim[(0, col_idx)] = rounded_ylim(float(np.nanmax(vals0))) if np.any(np.isfinite(vals0)) else 1.0

        vals1 = np.asarray(
            [edge_cpu_map.get((m, fw, be), np.nan) for m in current_models for fw, be in GROUP_ORDER],
            dtype=float,
        )
        panel_ylim[(1, col_idx)] = rounded_ylim(float(np.nanmax(vals1))) if np.any(np.isfinite(vals1)) else 1.0

        vals2 = np.asarray(
            [edge_gpu_map.get((m, fw, be), np.nan) for m in current_models for fw, be in GROUP_ORDER],
            dtype=float,
        )
        panel_ylim[(2, col_idx)] = rounded_ylim(float(np.nanmax(vals2))) if np.any(np.isfinite(vals2)) else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        # ========================= ROW 0: Robot CPU =========================
        ax0 = axes[0, col_idx]
        ax0.set_ylim(0, panel_ylim[(0, col_idx)])
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=ENERGY_MAX_TICKS))
        ax0.yaxis.set_major_formatter(FormatStrFormatter(ENERGY_TICK_FMT))

        for s_idx, scenario in enumerate(SCENARIOS):
            scenario_center = start + group_w / 2 + s_idx * span
            for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                xs = x + scenario_center + row0_intra_offsets[g_idx]
                vals = np.asarray([robot_cpu_map.get((m, scenario, fw, be), np.nan) for m in current_models], dtype=float)
                ax0.bar(
                    xs, vals, width=row0_bar_w, color=GROUP_COLOR[(scenario, fw, be)],
                    edgecolor="black", linewidth=STROKE_WIDTH, zorder=3,
                )

        # Head-to-head connector (see SHOW_TREND): ROS2 -> vAccel, straight
        # line at the same backend, Local CPU group only.
        if SHOW_TREND:
            s_idx = SCENARIOS.index(CONNECTOR_SCENARIO)
            scenario_center = start + group_w / 2 + s_idx * span
            for be, ls, marker in CONNECTOR_SPECS:
                for m_idx, model in enumerate(current_models):
                    ys_pair = [
                        robot_cpu_map.get((model, CONNECTOR_SCENARIO, "ROS2", be), np.nan),
                        robot_cpu_map.get((model, CONNECTOR_SCENARIO, "vAccel", be), np.nan),
                    ]
                    if not (np.isfinite(ys_pair[0]) and np.isfinite(ys_pair[1])):
                        continue
                    xs_pair = [
                        x[m_idx] + scenario_center + row0_intra_offsets[ROS2_IDX[be]],
                        x[m_idx] + scenario_center + row0_intra_offsets[VACCEL_IDX[be]],
                    ]
                    ax0.plot(
                        xs_pair, ys_pair, color=CONNECTOR_COLOR, linestyle=ls, marker=marker,
                        markersize=3.5, linewidth=1.0, zorder=6,
                    )

        # Zoom inset (see SHOW_ROW0_ZOOM_INSET): Local CPU dwarfs Remote
        # CPU/GPU on this axis by design -- that gap is the real story (the
        # robot's own CPU energy nearly vanishes once work is offloaded) --
        # but it flattens the Remote CPU vs Remote GPU comparison to
        # nothing. Re-draw just those two scenario-groups at their own
        # scale in a corner inset, rather than rescale the main axis.
        if SHOW_ROW0_ZOOM_INSET:
            axins = ax0.inset_axes(ZOOM_INSET_BBOX, zorder=8)
            remote_vals_all = []
            for s_idx, scenario in enumerate(ZOOM_INSET_SCENARIOS):
                scenario_center = zoom_start + s_idx * zoom_span
                for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                    xs = x + scenario_center + zoom_intra_offsets[g_idx]
                    vals = np.asarray([robot_cpu_map.get((m, scenario, fw, be), np.nan) for m in current_models], dtype=float)
                    axins.bar(
                        xs, vals, width=ZOOM_BAR_W, color=GROUP_COLOR[(scenario, fw, be)],
                        edgecolor="black", linewidth=STROKE_WIDTH * 0.7, zorder=3,
                    )
                    remote_vals_all.append(vals)

            remote_max = np.nanmax(np.concatenate(remote_vals_all)) if remote_vals_all else np.nan
            axins.set_ylim(0, rounded_ylim(float(remote_max)) if np.isfinite(remote_max) and remote_max > 0 else 1.0)
            axins.set_xlim(x[0] - zoom_xlim_pad, x[-1] + zoom_xlim_pad)
            axins.set_xticks([])
            axins.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axins.yaxis.set_major_formatter(FormatStrFormatter(ENERGY_TICK_FMT))
            axins.tick_params(axis="y", labelsize="xx-small", pad=1.5)
            axins.set_axisbelow(True)
            axins.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.3)
            axins.set_facecolor("white")
            axins.patch.set_alpha(1.0)
            for spine in axins.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        style_axes(ax0)
        ax0.margins(x=0.015)
        if col_idx == 0:
            ax0.set_ylabel("Robot CPU\n(J/frame)")

        # ================= ROWS 1-2: Edge CPU / Edge GPU =====================
        row_specs = [
            (1, "Remote CPU", edge_cpu_map, "Edge CPU\n(J/frame)"),
            (2, "Remote GPU", edge_gpu_map, "Edge GPU\n(J/frame)"),
        ]
        for row_idx, scenario, data_map, ylabel in row_specs:
            ax = axes[row_idx, col_idx]
            ax.set_ylim(0, panel_ylim[(row_idx, col_idx)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=ENERGY_MAX_TICKS))
            ax.yaxis.set_major_formatter(FormatStrFormatter(ENERGY_TICK_FMT))

            for g_idx, (fw, be) in enumerate(GROUP_ORDER):
                xs = x + row_offsets[g_idx]
                vals = np.asarray([data_map.get((m, fw, be), np.nan) for m in current_models], dtype=float)
                ax.bar(
                    xs, vals, width=row_bar_w, color=GROUP_COLOR[(scenario, fw, be)],
                    edgecolor="black", linewidth=STROKE_WIDTH, zorder=3,
                )

            # Head-to-head connector (see SHOW_TREND): ROS2 -> vAccel, same
            # style as the Robot CPU row. No scenario-group clutter concern
            # here -- each of these rows is already a single scenario.
            if SHOW_TREND:
                for be, ls, marker in CONNECTOR_SPECS:
                    for m_idx, model in enumerate(current_models):
                        ys_pair = [
                            data_map.get((model, "ROS2", be), np.nan),
                            data_map.get((model, "vAccel", be), np.nan),
                        ]
                        if not (np.isfinite(ys_pair[0]) and np.isfinite(ys_pair[1])):
                            continue
                        xs_pair = [
                            x[m_idx] + row_offsets[ROS2_IDX[be]],
                            x[m_idx] + row_offsets[VACCEL_IDX[be]],
                        ]
                        ax.plot(
                            xs_pair, ys_pair, color=CONNECTOR_COLOR, linestyle=ls, marker=marker,
                            markersize=3.5, linewidth=1.0, zorder=6,
                        )

            ax.set_xticks(x)
            if row_idx == 2:
                ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax.set_xticklabels([])

            style_axes(ax)
            ax.margins(x=0.01)
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    # ---- Manual layout ----
    CAPTION_OFFSET = 0.075
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.12, top=0.85)

    # Built explicitly (not via get_legend_handles_labels()) so every
    # scenario x framework x backend combo gets its own legend entry, in an
    # order that lands as a clean 4-col x 3-row grid: fig.legend(ncol=N)
    # fills COLUMN-major (top-to-bottom within a column, then next column).
    #   Col 1: ROS2+Torch  (Local, Remote CPU, Remote GPU)
    #   Col 2: ROS2+SOL    (Local, Remote CPU, Remote GPU)
    #   Col 3: vAccel+Torch(Local, Remote CPU, Remote GPU)
    #   Col 4: vAccSOL     (Local, Remote CPU, Remote GPU)
    color_handles = [
        mpatches.Patch(facecolor=GROUP_COLOR[(scenario, fw, be)], edgecolor="black",
                        label=f"{scenario} ({GROUP_LABEL[(fw, be)]})")
        for fw, be in GROUP_ORDER
        for scenario in SCENARIOS
    ]
    color_labels = [h.get_label() for h in color_handles]

    leg = fig.legend(
        color_handles, color_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black", fontsize="small",
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[2, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[2, :], cat_captions):
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

    robot_cpu_map, edge_cpu_map, edge_gpu_map = load_merged_maps()
    plot_combined(robot_cpu_map, edge_cpu_map, edge_gpu_map)


if __name__ == "__main__":
    main()
