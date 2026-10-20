#!/usr/bin/env python3
"""
boxplot_network_perturbation.py

Box comparison of end-to-end per-frame latency (total p50 + IQR) under
increasing WiFi background-load levels, vAccel (vaccel-remote-sol) vs ROS2
(DDS, sol backend) -- both remote GPU (edge-asus). One subplot per model.

Boxes are built from each run's own precomputed summary stats (q1=p25,
median=p50, q3=p75) via matplotlib's bxp(), since we have per-run percentiles
already (same files network_perturbation_test.py reads) rather than raw
per-sample arrays. Whiskers are NOT extended to min/max -- the box edges (IQR)
are the full visual extent, matching "total p50 + IQR" and nothing wider.

Produces: network-perturbation-boxplot.pdf
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from network_perturbation_test import (
    EDGE_DIR,
    LOAD_LEVELS,
    ROBOT_DIR,
    ROS_MONITOR_DIR,
    ROS_RUNS,
    load_json,
    load_summary,
)

OUTPUT_FILE = "network-perturbation-boxplot.pdf"

# Matches the vAccel/ROS2 hue convention already established in
# results/plot/comparison/barplot_e2e_latency.py.
FRAMEWORK_COLOR = {"vAccel": "#2a78d6", "ROS2": "#eb6834"}
LOAD_TICK_LABELS = ["0\n(baseline)", "100", "300", "500"]

MODEL_PANELS = [
    ("resnet50", "ResNet-50 SOL — Remote GPU"),
    ("fcn_resnet50", "FCN-ResNet-50 SOL — Remote GPU"),
]


def _box_stats(p25: float, p50: float, p75: float) -> dict:
    # No whisker extension beyond q1/q3 -- the box IS the full extent (p50 + IQR).
    return {"med": p50, "q1": p25, "q3": p75, "whislo": p25, "whishi": p75, "fliers": []}


def vaccel_boxes(model: str) -> list[dict]:
    boxes = []
    for _, tag in LOAD_LEVELS:
        run_id = f"{tag}_{model}_vaccel-remote-sol_edge-asus-gpu"
        sys_ms = load_summary(ROBOT_DIR, run_id)["system_ms"]
        boxes.append(_box_stats(sys_ms["p25"], sys_ms["p50"], sys_ms["p75"]))
    return boxes


def ros2_boxes(model: str) -> list[dict]:
    boxes = []
    for _, monitor_file, _ in ROS_RUNS[model]:
        t_e2e = load_json(ROS_MONITOR_DIR / monitor_file)["summary"]["t_e2e_ms"]
        boxes.append(_box_stats(t_e2e["p25"], t_e2e["p50"], t_e2e["p75"]))
    return boxes


def style_box(bp: dict, color: str) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor("black")
    for part in ("whiskers", "caps", "medians"):
        for line in bp[part]:
            line.set_color("black")


def draw_panel(ax, model: str, title: str) -> None:
    n = len(LOAD_LEVELS)
    pos_vaccel = [i * 2 - 0.22 for i in range(n)]
    pos_ros2 = [i * 2 + 0.22 for i in range(n)]

    bp_vaccel = ax.bxp(vaccel_boxes(model), positions=pos_vaccel, widths=0.38, showfliers=False, patch_artist=True)
    bp_ros2 = ax.bxp(ros2_boxes(model), positions=pos_ros2, widths=0.38, showfliers=False, patch_artist=True)
    style_box(bp_vaccel, FRAMEWORK_COLOR["vAccel"])
    style_box(bp_ros2, FRAMEWORK_COLOR["ROS2"])

    ax.set_xticks([i * 2 for i in range(n)])
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
        mpatches.Patch(facecolor=FRAMEWORK_COLOR["ROS2"], edgecolor="black", label="ROS2 DDS"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99), frameon=True, edgecolor="black")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUTPUT_FILE, bbox_extra_artists=(leg,))
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
