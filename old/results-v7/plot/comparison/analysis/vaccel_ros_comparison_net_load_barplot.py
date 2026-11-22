#!/usr/bin/env python3
"""
vaccel_ros_comparison_net_load_barplot.py

Plots the same numbers vaccel_ros_comparison_net_load.py prints to the console
-- ros2 (Fast DDS TCP, CycloneDDS UDP) vs vaccel-remote-sol, remote GPU
(edge-asus), sol backend, 1000 inputs per run, across WiFi background-load
levels -- as one figure, 2 rows x 3 columns:

  row 1: total latency per input (ms, p50), one solid bar per framework,
         asymmetric IQR error bars (no per-stage breakdown -- see
         vaccel_ros_comparison_net_load.py for that).
  row 2: total energy per run (kJ), stacked Robot CPU / Edge CPU / Edge GPU.
  columns: one model each (resnet50 / r3d_18 / deeplabv3_resnet50); x axis =
           background load level, 3 bars per level (ros2 TCP, ros2 UDP, vaccel).

Style (colours, stack encodings, legends, fonts) is imported from
vaccel_ros_comparison_barplot.py so the two figures always match.

Latency outliers: under heavy load Fast DDS TCP's latency reaches seconds (a
send backlog that keeps growing -- see vaccel_ros_comparison_net_load.py),
which would flatten every other bar. A latency bar more than OUTLIER_FACTOR x
the largest other bar at the same level is drawn clipped at the panel top,
with its real value and an arrow printed over it; the y axis is scaled to the
rest. Energy bars are never clipped -- a large total there is real.

A run (or a run's energy files) that doesn't exist leaves its bar out, with a
console warning.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vaccel_ros_comparison_barplot import (
    BAR_WIDTH, ENERGY_COMPONENTS, FONT_SCALE, FRAMEWORK_COLOR, FRAMEWORK_LABEL, FRAMEWORKS,
    VALUE_FONTSIZE, _bar_x, _component_handles, _legend, _stack,
)
from vaccel_ros_comparison_net_load import LOAD_LEVELS, MODELS, collect

OUTPUT_FILE = "vaccel-ros-comparison-net-load-barplot.pdf"
OUTLIER_FACTOR = 3.0

MODEL_TITLE = {
    "resnet50": "ResNet-50\n(image classification)",
    "r3d_18": "R3D-18\n(video classification)",
    "deeplabv3_resnet50": "DeepLabV3-R50\n(semantic segmentation)",
}
LEVEL_TICK = {"baseline": "0", "100 Mbps": "100", "300 Mbps": "300", "500 Mbps": "500", "600 Mbps": "600"}


def load_results() -> dict:
    """{model: {level_label: {framework: contrib | None}}}, framework keys as
    in vaccel_ros_comparison_barplot.FRAMEWORKS."""
    results = {}
    for model, category in MODELS:
        results[model] = {}
        for level_label, level in LOAD_LEVELS:
            entry = {}
            for label, c, missing in collect(model, category, level):
                fw = "vAccel" if label == "vaccel-rpc" else label
                if c is None:
                    print(f"[warn] {fw} {model} {level_label}: missing, bar not drawn ({missing})")
                entry[fw] = c
            results[model][level_label] = entry
    return results


def _outliers(values: dict) -> set:
    """Frameworks whose value is > OUTLIER_FACTOR x the largest other one at
    the same level (e.g. Fast DDS TCP's growing backlog at 500-600 Mbps)."""
    out = set()
    for fw, v in values.items():
        others = [o for f, o in values.items() if f != fw]
        if others and v > OUTLIER_FACTOR * max(others):
            out.add(fw)
    return out


def _finish_panel(ax, bars: list, title: str | None, ylabel: str | None) -> None:
    """bars: [(x, top, label_text, is_outlier)]. Scales y to the non-outlier
    bars, prints each total above its bar (outliers: real value + arrow at
    the clipped top), styles the axes."""
    normal = [top for _, top, _, out in bars if not out]
    if normal:
        data_max = max(normal)
        ax.set_ylim(0, data_max * 1.45)
        offset = data_max * 0.03
        for x, top, text, out in bars:
            if out:
                ax.text(x, data_max * 1.02, f"↑{text}", ha="center", va="bottom", rotation=90,
                        fontsize=VALUE_FONTSIZE, fontweight="bold", zorder=11,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            else:
                ax.text(x, top + offset, text, ha="center", va="bottom", rotation=90,
                        fontsize=VALUE_FONTSIZE, zorder=11)
    ax.set_xticks(np.arange(len(LOAD_LEVELS)))
    ax.set_xticklabels([LEVEL_TICK[l] for l, _ in LOAD_LEVELS])
    ax.set_xlabel("Background traffic load (Mbps)")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)


def draw_latency_panel(ax, model: str, results: dict, title: str, ylabel: str | None) -> None:
    bars = []
    for i, (level_label, _) in enumerate(LOAD_LEVELS):
        level = results[model][level_label]
        present = {fw: c for fw, c in level.items() if c is not None}
        outliers = _outliers({fw: c["total_p50"] for fw, c in present.items()})
        for j, fw in enumerate(FRAMEWORKS):
            c = present.get(fw)
            if c is None:
                continue
            x = _bar_x(i, j)
            total = c["total_p50"]
            ax.bar(x, total, width=BAR_WIDTH, facecolor=FRAMEWORK_COLOR[fw],
                   edgecolor="black", linewidth=0.8, zorder=3)
            lo, hi = max(total - c["p25"], 0.0), max(c["p75"] - total, 0.0)
            ax.errorbar(x, total, yerr=[[lo], [hi]], fmt="none", ecolor="black",
                        elinewidth=1.2, capsize=4, capthick=1.2, zorder=10)
            bars.append((x, total + hi, f"{total:.0f}", fw in outliers))
    _finish_panel(ax, bars, title, ylabel)


def draw_energy_panel(ax, model: str, results: dict, ylabel: str | None) -> None:
    bars = []
    for i, (level_label, _) in enumerate(LOAD_LEVELS):
        level = results[model][level_label]
        totals = {}
        for fw, c in level.items():
            if c is None:
                continue
            parts = [c["energy"][comp[0]] for comp in ENERGY_COMPONENTS]
            if None in parts:
                print(f"[warn] {fw} {model} {level_label}: incomplete energy_totals, energy bar not drawn")
                continue
            totals[fw] = parts
        # No clipping for energy: a large total is real (e.g. a long, lossy
        # CycloneDDS UDP run), not an unbounded backlog like TCP latency.
        for j, fw in enumerate(FRAMEWORKS):
            if fw not in totals:
                continue
            x = _bar_x(i, j)
            total = _stack(ax, x, [p / 1000.0 for p in totals[fw]], FRAMEWORK_COLOR[fw], ENERGY_COMPONENTS)
            bars.append((x, total, f"{total:.2f}", False))
    _finish_panel(ax, bars, None, ylabel)


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "hatch.linewidth": 1.0,
    })

    results = load_results()
    fig, axes = plt.subplots(2, 3, figsize=(40, 21))

    for col, (model, _) in enumerate(MODELS):
        draw_latency_panel(axes[0, col], model, results, MODEL_TITLE.get(model, model),
                           "Median latency\nper input (ms)" if col == 0 else None)
        draw_energy_panel(axes[1, col], model, results,
                          "Total energy (kJ)" if col == 0 else None)

    fw_handles = [mpatches.Patch(facecolor=FRAMEWORK_COLOR[fw], edgecolor="black", label=FRAMEWORK_LABEL[fw])
                  for fw in FRAMEWORKS]
    sup = fig.suptitle("Remote GPU inference using SOL under WiFi background load\n"
                       "ROS2 (Fast DDS TCP / CycloneDDS UDP) vs vAccel, 1000 inputs per run", y=1.065)
    legs = (
        _legend(fig, fw_handles, None, 0.42, 0.965),
        _legend(fig, _component_handles(ENERGY_COMPONENTS), "Energy (bottom row)", 0.88, 0.965, ncol=1),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.2, w_pad=1.0)
    extra = (*legs, sup)
    fig.savefig(OUTPUT_FILE, bbox_extra_artists=extra)
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
