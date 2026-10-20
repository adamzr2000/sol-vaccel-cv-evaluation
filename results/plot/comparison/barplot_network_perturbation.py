#!/usr/bin/env python3
"""
barplot_network_perturbation.py

Bar comparison of end-to-end per-frame latency under increasing WiFi
background-load levels, vAccel (vaccel-remote-sol) vs ROS2 (DDS/RTPS, sol
backend) -- both remote GPU (edge-asus). One subplot per model.

Bar height = total p50. Error bars are asymmetric IQR: lower = p50-p25,
upper = p75-p50 -- matches the convention already established in
results/plot/vaccel/plot_data_e2e.py and
results/plot/ros2/barplot_e2e_latency_and_fps.py.

Produces: network-perturbation-barplot.pdf
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from network_perturbation_test import (
    LOAD_LEVELS,
    ROBOT_DIR,
    ROS_MONITOR_DIR,
    ROS_RUNS,
    load_json,
    load_summary,
)

OUTPUT_FILE = "network-perturbation-barplot.pdf"

# Matches the vAccel/ROS2 hue convention already established in
# results/plot/comparison/barplot_e2e_latency.py.
FRAMEWORK_COLOR = {"vAccel": "#2a78d6", "ROS2": "#eb6834"}
LOAD_TICK_LABELS = ["0\n(baseline)", "100", "300", "500"]
BAR_WIDTH = 0.38

MODEL_PANELS = [
    ("resnet50", "ResNet-50 SOL — Remote GPU"),
    ("fcn_resnet50", "FCN-ResNet-50 SOL — Remote GPU"),
]


def vaccel_p50_err(model: str) -> tuple[list[float], list[list[float]]]:
    p50s, lo, hi = [], [], []
    for _, tag in LOAD_LEVELS:
        run_id = f"{tag}_{model}_vaccel-remote-sol_edge-asus-gpu"
        sys_ms = load_summary(ROBOT_DIR, run_id)["system_ms"]
        p50s.append(sys_ms["p50"])
        lo.append(sys_ms["p50"] - sys_ms["p25"])
        hi.append(sys_ms["p75"] - sys_ms["p50"])
    return p50s, [lo, hi]


def ros2_p50_err(model: str) -> tuple[list[float], list[list[float]]]:
    p50s, lo, hi = [], [], []
    for _, monitor_file, _ in ROS_RUNS[model]:
        t_e2e = load_json(ROS_MONITOR_DIR / monitor_file)["summary"]["t_e2e_ms"]
        p50s.append(t_e2e["p50"])
        lo.append(t_e2e["p50"] - t_e2e["p25"])
        hi.append(t_e2e["p75"] - t_e2e["p50"])
    return p50s, [lo, hi]


def draw_panel(ax, model: str, title: str) -> None:
    n = len(LOAD_LEVELS)
    x = np.arange(n)
    pos_vaccel = x - BAR_WIDTH / 2
    pos_ros2 = x + BAR_WIDTH / 2

    v_p50, v_err = vaccel_p50_err(model)
    r_p50, r_err = ros2_p50_err(model)

    ax.bar(pos_vaccel, v_p50, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR["vAccel"],
           edgecolor="black", linewidth=0.8, zorder=3)
    ax.bar(pos_ros2, r_p50, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR["ROS2"],
           edgecolor="black", linewidth=0.8, zorder=3)

    ax.errorbar(pos_vaccel, v_p50, yerr=v_err, fmt="none", ecolor="black",
                elinewidth=1.2, capsize=5, capthick=1.2, zorder=10)
    ax.errorbar(pos_ros2, r_p50, yerr=r_err, fmt="none", ecolor="black",
                elinewidth=1.2, capsize=5, capthick=1.2, zorder=10)

    # Median value labels above each bar's upper error cap. Offset/headroom
    # scaled to this panel's own data range since the two panels differ by
    # orders of magnitude (ResNet-50 vs FCN-ResNet-50).
    tops = [p + h[1] for p, h in zip(v_p50, zip(*v_err))] + [p + h[1] for p, h in zip(r_p50, zip(*r_err))]
    data_max = max(tops)
    offset = data_max * 0.025
    for xp, p50, err in ((pos_vaccel, v_p50, v_err), (pos_ros2, r_p50, r_err)):
        for xi, p, hi in zip(xp, p50, err[1]):
            ax.text(xi, p + hi + offset, f"{p:.0f}", ha="center", va="bottom", fontsize="small", zorder=11)
    ax.set_ylim(top=data_max * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels(LOAD_TICK_LABELS)
    ax.set_xlabel("Background traffic load (Mbps)")
    ax.set_ylabel("Latency per frame (ms)")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=2.0)
    plt.rcParams.update({
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (model, title) in zip(axes, MODEL_PANELS):
        draw_panel(ax, model, title)

    handles = [
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["vAccel"], edgecolor="black", label="vAccel"),
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["ROS2"], edgecolor="black", label="ROS2 DDS/RTPS"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99), frameon=True, edgecolor="black")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUTPUT_FILE, bbox_extra_artists=(leg,))
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
