#!/usr/bin/env python3
"""
barplot_network_perturbation.py

Two independent figures/PDFs, not one combined plot:

1. Latency (LATENCY_OUTPUT_FILE): end-to-end per-input latency under
   increasing WiFi background-load levels: vAccel (vaccel-remote-sol) vs
   ROS2 (DDS/RTPS, sol backend) -- both remote GPU (edge-asus). ROS2 is
   shown in two conditions: Resized (robot resizes to ~vaccel's tensor
   resolution before publishing, a transport-only, payload-held-equal
   comparison) and RAW (robot ships the camera's native 1280x720 frame) --
   see network_perturbation_test.py. Bar height = total p50. Error bars are
   asymmetric IQR: lower = p50-p25, upper = p75-p50 -- matches the
   convention already established in results/plot/vaccel/plot_data_e2e.py
   and results/plot/ros2/barplot_e2e_latency_and_fps.py.

2. Energy (ENERGY_OUTPUT_FILE): total energy (robot CPU + edge GPU, stacked)
   for the same models, conditions and load levels -- a fully independent
   plotting path (own data-loading functions, own draw function) from the
   latency figure, sharing only the x-axis layout for visual consistency
   between the two PDFs. No error bars -- energy_totals.json is a single
   hardware-counter total per run, not a distribution.

Load levels are a fixed set (baseline/100/200/300/500/600 Mbps -- 400 Mbps
is skipped, not available for every model/series). vAccel and ROS2 Resized
have data at all six; ROS2 RAW only has data through 300 Mbps -- the
wireless link didn't sustain the raw 1280x720 upload beyond that under
injected load, so its bars simply stop there rather than being estimated.

Produces: network-perturbation-latency-barplot.pdf, network-perturbation-energy-barplot.pdf
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from network_perturbation_test import (
    ROBOT_DIR,
    ROS_MONITOR_DIR,
    ROS_RUNS_RAW,
    ROS_RUNS_RESIZED,
    ROS_SYSTEM_EDGE,
    ROS_SYSTEM_ROBOT,
    SV_SYSTEM_EDGE,
    SV_SYSTEM_ROBOT,
    VACCEL_RUNS,
    find_ros_file,
    load_energy_totals,
    load_json,
    load_summary,
)

LATENCY_OUTPUT_FILE = "network-perturbation-latency-barplot.pdf"
ENERGY_OUTPUT_FILE = "network-perturbation-energy-barplot.pdf"

# Matches the vAccel/ROS2 hue convention already established in
# results/plot/comparison/barplot_e2e_latency.py. Resized is a variant of
# ROS2, not a separate framework, so it reuses the same ROS2 hue with a hatch
# overlay rather than a new categorical color -- keeps "color = framework
# identity" intact and stays distinguishable in grayscale/print.
FRAMEWORK_COLOR = {"vAccel": "#2a78d6", "ROS2": "#eb6834"}
RESIZED_HATCH = "///"
BAR_WIDTH = 0.26

# Row 2 only: edge-GPU segment drawn at reduced alpha of the same framework
# hue, robot-CPU segment at full alpha -- a second, independent encoding
# layered on top of the row-1 framework/hatch encoding, not a new hue (hatch
# is already spoken for by Resized vs RAW).
ENERGY_EDGE_ALPHA = 0.45

# Fixed across all panels -- 400 Mbps intentionally excluded (not available
# for every model/series; see module docstring).
LOAD_LEVELS_FIXED = ["baseline", "100 Mbps", "200 Mbps", "300 Mbps", "500 Mbps", "600 Mbps"]

MODEL_PANELS = [
    ("resnet50", "ResNet-50 SOL — Remote GPU"),
    ("r3d_18", "R3D-18 SOL — Remote GPU"),
    ("deeplabv3_resnet50", "DeepLabV3-ResNet50 SOL — Remote GPU"),
]


def _xticklabels() -> list[str]:
    return [l.replace(" Mbps", "").replace("baseline", "0\n(baseline)") for l in LOAD_LEVELS_FIXED]


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_xticks(np.arange(len(LOAD_LEVELS_FIXED)))
    ax.set_xticklabels(_xticklabels())
    ax.set_xlabel("Background traffic load (Mbps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)


# ==========================================================================
# Row 1: latency
# ==========================================================================

def vaccel_series(model: str) -> tuple[list[int], list[float], list[list[float]]]:
    """Returns (tick_indices, p50s, [lo, hi]) for whichever LOAD_LEVELS_FIXED
    entries this model actually has vaccel data for."""
    tag_by_label = dict(VACCEL_RUNS[model])
    idxs, p50s, lo, hi = [], [], [], []
    for i, label in enumerate(LOAD_LEVELS_FIXED):
        if label not in tag_by_label:
            continue
        run_id = f"{tag_by_label[label]}_{model}_vaccel-remote-sol_edge-asus-gpu"
        sys_ms = load_summary(ROBOT_DIR, run_id)["system_ms"]
        idxs.append(i)
        p50s.append(sys_ms["p50"])
        lo.append(sys_ms["p50"] - sys_ms["p25"])
        hi.append(sys_ms["p75"] - sys_ms["p50"])
    return idxs, p50s, [lo, hi]


def ros2_series(model: str, run_set: dict) -> tuple[list[int], list[float], list[list[float]]]:
    """Same as vaccel_series, but for a ROS2 run_set (RAW or RESIZED) -- RAW
    simply has fewer idxs present (stops at 300 Mbps), which leaves a gap
    rather than a drawn bar at 500/600."""
    tag_by_label = dict(run_set[model])
    idxs, p50s, lo, hi = [], [], [], []
    for i, label in enumerate(LOAD_LEVELS_FIXED):
        if label not in tag_by_label:
            continue
        t_e2e = load_json(find_ros_file(ROS_MONITOR_DIR, "monitor", model, tag_by_label[label]))["summary"]["t_e2e_ms"]
        idxs.append(i)
        p50s.append(t_e2e["p50"])
        lo.append(t_e2e["p50"] - t_e2e["p25"])
        hi.append(t_e2e["p75"] - t_e2e["p50"])
    return idxs, p50s, [lo, hi]


def draw_panel(ax, model: str, title: str) -> None:
    x = np.arange(len(LOAD_LEVELS_FIXED))

    v_idx, v_p50, v_err = vaccel_series(model)
    z_idx, z_p50, z_err = ros2_series(model, ROS_RUNS_RESIZED)
    r_idx, r_p50, r_err = ros2_series(model, ROS_RUNS_RAW)

    # Left-to-right per group: ROS2 Resized, vAccel, ROS2 RAW -- vAccel sits
    # between its two comparison partners; matches legend order.
    pos_resized = x[z_idx] - BAR_WIDTH
    pos_vaccel = x[v_idx]
    pos_raw = x[r_idx] + BAR_WIDTH

    ax.bar(pos_vaccel, v_p50, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR["vAccel"],
           edgecolor="black", linewidth=0.8, zorder=3)
    ax.bar(pos_resized, z_p50, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR["ROS2"],
           edgecolor="black", linewidth=0.8, hatch=RESIZED_HATCH, zorder=3)
    ax.bar(pos_raw, r_p50, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR["ROS2"],
           edgecolor="black", linewidth=0.8, zorder=3)

    for pos, p50, err in ((pos_vaccel, v_p50, v_err), (pos_resized, z_p50, z_err), (pos_raw, r_p50, r_err)):
        ax.errorbar(pos, p50, yerr=err, fmt="none", ecolor="black",
                     elinewidth=1.2, capsize=4, capthick=1.2, zorder=10)

    # Median value labels above each bar's upper error cap, rotated 90 deg at
    # a larger fixed size (still absolute, not font_scale-relative -- these
    # sit close together above adjacent bars, so they can't grow with the
    # rest of the figure's text without colliding; rotating removes the
    # remaining collision risk without needing more horizontal bar spacing).
    # Offset/headroom scaled to this panel's own data range since panels
    # differ by orders of magnitude (e.g. ResNet-50 vs DeepLabV3-ResNet50
    # under heavy load).
    all_series = ((pos_vaccel, v_p50, v_err), (pos_resized, z_p50, z_err), (pos_raw, r_p50, r_err))
    tops = [p + h[1] for _, p50, err in all_series for p, h in zip(p50, zip(*err))]
    data_max = max(tops)
    offset = data_max * 0.03
    for pos, p50, err in all_series:
        for xi, p, hi in zip(pos, p50, err[1]):
            ax.text(xi, p + hi + offset, f"{p:.0f}", ha="center", va="bottom",
                     rotation=90, fontsize=14, zorder=11)
    ax.set_ylim(top=data_max * 1.28)

    _style_axes(ax, title, "Latency per input (ms)")


# ==========================================================================
# Row 2: energy (robot CPU + edge GPU, stacked) -- independent of row 1
# ==========================================================================

def vaccel_energy_series(model: str) -> tuple[list[int], list[float], list[float]]:
    """Returns (tick_indices, robot_cpu_J, edge_gpu_J) for whichever
    LOAD_LEVELS_FIXED entries this model has complete vaccel energy data for."""
    tag_by_label = dict(VACCEL_RUNS[model])
    idxs, robot_j, edge_j = [], [], []
    for i, label in enumerate(LOAD_LEVELS_FIXED):
        if label not in tag_by_label:
            continue
        run_id = f"{tag_by_label[label]}_{model}_vaccel-remote-sol_edge-asus-gpu"
        robot = load_energy_totals(SV_SYSTEM_ROBOT / f"energy_totals_{run_id}.json")
        edge = load_energy_totals(SV_SYSTEM_EDGE / f"energy_totals_{run_id}.json")
        r = robot.get("cpu_energy_j_total") if robot else None
        g = (edge.get("gpu") or [{}])[0].get("gpu_energy_j_total") if edge else None
        if r is None or g is None:
            continue
        idxs.append(i)
        robot_j.append(r)
        edge_j.append(g)
    return idxs, robot_j, edge_j


def ros2_energy_series(model: str, run_set: dict) -> tuple[list[int], list[float], list[float]]:
    """Same as vaccel_energy_series, but for a ROS2 run_set (RAW or RESIZED)."""
    tag_by_label = dict(run_set[model])
    idxs, robot_j, edge_j = [], [], []
    for i, label in enumerate(LOAD_LEVELS_FIXED):
        if label not in tag_by_label:
            continue
        tag = tag_by_label[label]
        monitor_path = find_ros_file(ROS_MONITOR_DIR, "monitor", model, tag)
        ts = monitor_path.stem.split("_")[-1]
        run_id = f"{model}_sol_remote-gpu-{tag}_{ts}"
        robot = load_energy_totals(ROS_SYSTEM_ROBOT / f"energy_totals_{run_id}.json")
        edge = load_energy_totals(ROS_SYSTEM_EDGE / f"energy_totals_{run_id}.json")
        r = robot.get("cpu_energy_j_total") if robot else None
        g = (edge.get("gpu") or [{}])[0].get("gpu_energy_j_total") if edge else None
        if r is None or g is None:
            continue
        idxs.append(i)
        robot_j.append(r)
        edge_j.append(g)
    return idxs, robot_j, edge_j


def _draw_energy_stack(ax, pos, robot_j: list[float], edge_j: list[float], facecolor: str, hatch=None) -> None:
    ax.bar(pos, robot_j, width=BAR_WIDTH, facecolor=facecolor,
           edgecolor="black", linewidth=0.8, hatch=hatch, zorder=3)
    ax.bar(pos, edge_j, width=BAR_WIDTH, bottom=robot_j, facecolor=facecolor,
           edgecolor="black", linewidth=0.8, hatch=hatch, alpha=ENERGY_EDGE_ALPHA, zorder=3)


def draw_energy_panel(ax, model: str, title: str) -> None:
    x = np.arange(len(LOAD_LEVELS_FIXED))

    v_idx, v_robot, v_edge = vaccel_energy_series(model)
    z_idx, z_robot, z_edge = ros2_energy_series(model, ROS_RUNS_RESIZED)
    r_idx, r_robot, r_edge = ros2_energy_series(model, ROS_RUNS_RAW)

    # Display in kJ -- the raw Joule totals are all in the 100s-10000s range,
    # which makes 4-5 digit labels cramped at this bar spacing; kJ keeps
    # enough precision (nothing here is below ~0.1 kJ) while shortening them.
    to_kj = lambda vals: [v / 1000.0 for v in vals]
    v_robot, v_edge = to_kj(v_robot), to_kj(v_edge)
    z_robot, z_edge = to_kj(z_robot), to_kj(z_edge)
    r_robot, r_edge = to_kj(r_robot), to_kj(r_edge)

    # Left-to-right per group: ROS2 Resized, vAccel, ROS2 RAW -- matches
    # draw_panel's order (vAccel between its two comparison partners).
    pos_resized = x[z_idx] - BAR_WIDTH
    pos_vaccel = x[v_idx]
    pos_raw = x[r_idx] + BAR_WIDTH

    _draw_energy_stack(ax, pos_vaccel, v_robot, v_edge, FRAMEWORK_COLOR["vAccel"])
    _draw_energy_stack(ax, pos_resized, z_robot, z_edge, FRAMEWORK_COLOR["ROS2"], hatch=RESIZED_HATCH)
    _draw_energy_stack(ax, pos_raw, r_robot, r_edge, FRAMEWORK_COLOR["ROS2"])

    # Total (robot + edge) value label above each stack, rotated 90 deg at a
    # larger fixed size (still absolute, not font_scale-relative, same
    # collision-avoidance reason as row 1) -- kJ-with-decimals labels are
    # wider strings than row 1's bare-integer ms labels, so upright text at
    # this bar spacing risks colliding with its neighbor; rotating avoids
    # that without needing more horizontal room between bars.
    all_series = ((pos_vaccel, v_robot, v_edge), (pos_resized, z_robot, z_edge), (pos_raw, r_robot, r_edge))
    totals = [r + g for _, robot_j, edge_j in all_series for r, g in zip(robot_j, edge_j)]
    data_max = max(totals)
    offset = data_max * 0.03
    for pos, robot_j, edge_j in all_series:
        for xi, r, g in zip(pos, robot_j, edge_j):
            ax.text(xi, r + g + offset, f"{r + g:.2f}", ha="center", va="bottom",
                     rotation=90, fontsize=14, zorder=11)
    ax.set_ylim(top=data_max * 1.28)

    _style_axes(ax, title, "Total energy (kJ)")


_RESIZE_NOTE = (
    "* Resized upload target: 224×224 for image classification (ResNet-50) and "
    "semantic segmentation (DeepLabV3-ResNet50); 112×112 for video classification (R3D-18)."
)


def _framework_legend(fig):
    handles = [
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["ROS2"], edgecolor="black", hatch=RESIZED_HATCH,
                        label="ROS2 DDS/RTPS — resized to match vAccel payload*"),
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["vAccel"], edgecolor="black", label="vAccel"),
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["ROS2"], edgecolor="black",
                        label="ROS2 DDS/RTPS — native 1280×720 upload"),
    ]
    return fig.legend(handles=handles, loc="upper center", ncol=3,
                       bbox_to_anchor=(0.5, 0.99), frameon=True, edgecolor="black")


def make_latency_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, (model, title) in zip(axes, MODEL_PANELS):
        draw_panel(ax, model, title)

    leg = _framework_legend(fig)
    # Note sits directly below the legend, above the panels -- near the "*"
    # it actually explains, rather than at the bottom of the figure where a
    # reader has to go looking for it. Fixed, absolute size (not font_scale-
    # relative) so it stays compact and doesn't collide with the legend above
    # it or overflow the figure width as the rest of the text scales up.
    note = fig.text(0.5, 0.875, _RESIZE_NOTE, ha="center", fontsize=13, style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 0.85))
    fig.savefig(LATENCY_OUTPUT_FILE, bbox_extra_artists=(leg, note))
    print(f"[OK] Saved plot to: {LATENCY_OUTPUT_FILE}")


def make_energy_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, (model, title) in zip(axes, MODEL_PANELS):
        draw_energy_panel(ax, model, title)

    leg1 = _framework_legend(fig)

    # Second, independent legend for the robot/edge stack encoding -- generic
    # black swatches (the actual fill color is still framework identity, from
    # leg1), since this encodes a different thing (which host) orthogonally
    # to framework/condition.
    energy_handles = [
        mpatches.Patch(facecolor="black", edgecolor="black", alpha=1.0, label="Robot CPU energy"),
        mpatches.Patch(facecolor="black", edgecolor="black", alpha=ENERGY_EDGE_ALPHA, label="Edge GPU energy"),
    ]
    leg2 = fig.legend(handles=energy_handles, loc="upper center", ncol=2,
                       bbox_to_anchor=(0.5, 0.92), frameon=True, edgecolor="black")

    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.savefig(ENERGY_OUTPUT_FILE, bbox_extra_artists=(leg1, leg2))
    print(f"[OK] Saved plot to: {ENERGY_OUTPUT_FILE}")


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=2.6)
    plt.rcParams.update({
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    make_latency_figure()
    make_energy_figure()


if __name__ == "__main__":
    main()
