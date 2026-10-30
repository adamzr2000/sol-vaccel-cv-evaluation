#!/usr/bin/env python3
"""
barplot_vaccel_comparison_test.py

Plots the same numbers vaccel_comparison_test.py prints to the console --
ros2 (DDS/RTPS, Fast DDS TCP, sol backend) vs vaccel-remote-sol, remote GPU
(edge-asus), 1000 inputs per run -- as one figure, 2 rows x 3 columns:

  row 1: latency per input (ms), stacked Inference / Pre/Post-processing /
         Network + Framework Overhead, with asymmetric IQR error bars on the
         total (lower = p50-p25, upper = p75-p50).
  row 2: total energy (kJ), stacked Robot CPU / Edge CPU / Edge GPU. No error
         bars -- energy_totals.json is a single counter total per run.
  columns: image classification / video classification / semantic segmentation,
           2 bars per model (ros2, vaccel).

All values come straight from vaccel_comparison_test.py's ros_contrib() /
vaccel_contrib() (same run selection, same p50s, same energy files), so the
plot can never disagree with the console. Stack heights sum exactly to the
console's total: net/fw is the residual total - inference - pre/post.

Color = framework identity, matching barplot_network_perturbation.py /
barplot_e2e_latency.py; the stacked component is a second encoding layered on
top as alpha of the same hue (as barplot_network_perturbation.py's energy
stack) for latency and as hatch pattern for energy, so "blue = vAccel, orange
= ROS2" holds across every paper figure.

Produces: vaccel-comparison-barplot.pdf
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vaccel_comparison_test import MODELS, ros_contrib, vaccel_contrib

OUTPUT_FILE = "vaccel-comparison-barplot.pdf"

FRAMEWORK_COLOR = {"ROS2": "#eb6834", "vAccel": "#2a78d6"}
FRAMEWORKS = ["ROS2", "vAccel"]   # left -> right within each model group
FRAMEWORK_LABEL = {"ROS2": "ROS2 DDS/RTPS", "vAccel": "vAccel"}
BAR_WIDTH = 0.36

# Stacking order bottom -> top: (key, label, alpha, hatch). Latency segments
# are told apart by alpha of the framework hue, energy segments by hatch
# pattern (full hue) -- a different encoding per row, so the two stacks
# can't be confused for each other.
LATENCY_COMPONENTS = [
    ("inference_p50", "Inference", 1.0, None),
    ("prepost_p50", "Pre/Post-processing", 0.6, None),
    ("net_fw_p50", "Network + Framework Overhead", 0.25, None),
]
# Keys are indices into vaccel_comparison_test.energy_row()'s
# (samples, duration_s, robot_cpu_J, edge_cpu_J, edge_gpu_J) tuple.
ENERGY_COMPONENTS = [
    (2, "Robot CPU", 1.0, None),
    (3, "Edge CPU", 1.0, "///"),
    (4, "Edge GPU", 1.0, ".."),
]

FONT_SCALE = 2.9
VALUE_FONTSIZE = 19     # totals above bars (absolute -- 8 bars/panel, see _label_tops)
LEGEND_FONTSIZE = 24

CATEGORY_PANELS = [
    ("image_classification", "Image classification"),
    ("video_classification", "Video classification"),
    ("semantic_segmentation", "Semantic segmentation"),
]

MODEL_LABEL = {
    "resnet50": "ResNet-50",
    "swin_t": "Swin-T",
    "swin_s": "Swin-S",
    "swin_v2_b": "Swin-V2-B",
    "swin3d_s": "Swin3D-S",
    "swin3d_b": "Swin3D-B",
    "r3d_18": "R3D-18",
    "r2plus1d_18": "R(2+1)D-18",
    "deeplabv3_resnet50": "DLV3-R50",
    "deeplabv3_resnet101": "DLV3-R101",
    "fcn_resnet50": "FCN-R50",
    "fcn_resnet101": "FCN-R101",
}

def load_results() -> dict:
    """{model: {"ROS2": contrib | None, "vAccel": contrib | None}} -- None
    when that run is missing (bar simply not drawn, with a console warning)."""
    results = {}
    for model, category in MODELS:
        entry = {}
        for fw, fn in (("ROS2", lambda: ros_contrib(model, category)),
                       ("vAccel", lambda: vaccel_contrib(model, category))):
            try:
                entry[fw] = fn()
            except FileNotFoundError as e:
                print(f"[warn] {fw} {model}: missing, bar not drawn ({e})")
                entry[fw] = None
        results[model] = entry
    return results


def _style_axes(ax, models: list[str], title: str | None, ylabel: str | None) -> None:
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models])
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)


def _stack(ax, x: float, values: list[float], color: str, components: list[tuple]) -> float:
    bottom = 0.0
    for v, (_, _, alpha, hatch) in zip(values, components):
        ax.bar(x, v, width=BAR_WIDTH, bottom=bottom, facecolor=color, alpha=alpha, hatch=hatch,
               edgecolor="black", linewidth=0.8, zorder=3)
        bottom += v
    return bottom


def draw_latency_panel(ax, models: list[str], results: dict, title: str, ylabel: str | None) -> None:
    tops = []
    labels = []
    for i, model in enumerate(models):
        for j, fw in enumerate(FRAMEWORKS):
            c = results[model][fw]
            if c is None:
                continue
            x = i + (j - 0.5) * BAR_WIDTH
            total = _stack(ax, x, [c[comp[0]] for comp in LATENCY_COMPONENTS],
                           FRAMEWORK_COLOR[fw], LATENCY_COMPONENTS)
            # IQR of the total, around the stack top (== console's total p50).
            lo, hi = total - c["p25"], c["p75"] - total
            ax.errorbar(x, total, yerr=[[max(lo, 0.0)], [max(hi, 0.0)]], fmt="none", ecolor="black",
                        elinewidth=1.2, capsize=4, capthick=1.2, zorder=10)
            tops.append(total + max(hi, 0.0))
            labels.append((x, total + max(hi, 0.0), f"{total:.0f}"))
    _label_tops(ax, tops, labels)
    _style_axes(ax, models, title, ylabel)


def draw_energy_panel(ax, models: list[str], results: dict, ylabel: str | None) -> None:
    tops = []
    labels = []
    for i, model in enumerate(models):
        for j, fw in enumerate(FRAMEWORKS):
            c = results[model][fw]
            if c is None:
                continue
            e = c["energy"]
            parts = [e[comp[0]] for comp in ENERGY_COMPONENTS]
            if None in parts:
                print(f"[warn] {fw} {model}: incomplete energy_totals, energy bar not drawn")
                continue
            x = i + (j - 0.5) * BAR_WIDTH
            total = _stack(ax, x, [p / 1000.0 for p in parts], FRAMEWORK_COLOR[fw], ENERGY_COMPONENTS)
            tops.append(total)
            labels.append((x, total, f"{total:.2f}"))
    _label_tops(ax, tops, labels)
    _style_axes(ax, models, None, ylabel)


def _label_tops(ax, tops: list[float], labels: list[tuple]) -> None:
    """Total value above each bar, rotated 90 deg (8 bars per panel sit close
    together -- upright labels would collide), offset/headroom scaled to the
    panel's own range since panels differ ~4x in scale."""
    if not tops:
        return
    data_max = max(tops)
    offset = data_max * 0.03
    for x, y, text in labels:
        ax.text(x, y + offset, text, ha="center", va="bottom", rotation=90, fontsize=VALUE_FONTSIZE, zorder=11)
    ax.set_ylim(0, data_max * 1.32)


def _legend(fig, handles: list, title: str, x: float, y: float):
    return fig.legend(handles=handles, title=title, loc="upper center", ncol=len(handles),
                      bbox_to_anchor=(x, y), frameon=True, edgecolor="black", fancybox=False,
                      fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE,
                      handlelength=1.6, columnspacing=1.2, borderpad=0.35)


def _component_handles(components: list[tuple]) -> list:
    # Gray swatches -- the fill hue is framework identity (framework legend);
    # these only encode which stacked segment is which (alpha or hatch).
    return [mpatches.Patch(facecolor="dimgray", edgecolor="black", alpha=alpha, hatch=hatch, label=label)
            for _, label, alpha, hatch in components]


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "hatch.linewidth": 1.0,
    })

    results = load_results()
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))

    for col, (category, title) in enumerate(CATEGORY_PANELS):
        models = [m for m, cat in MODELS if cat == category]
        draw_latency_panel(axes[0, col], models, results, title,
                           "Latency per input (ms)" if col == 0 else None)
        draw_energy_panel(axes[1, col], models, results,
                          "Total energy (kJ)" if col == 0 else None)

    fw_handles = [mpatches.Patch(facecolor=FRAMEWORK_COLOR[fw], edgecolor="black", label=FRAMEWORK_LABEL[fw])
                  for fw in FRAMEWORKS]
    sup = fig.suptitle("Remote GPU inference using SOL - ROS2 (Fast DDS TCP) vs vAccel, "
                       "1000 inputs per run", y=0.995)
    # Framework legend alone on the first line; the two (wider) component
    # legends side by side on the second -- three in one row don't fit at
    # this font size.
    legs = (
        _legend(fig, fw_handles, None, 0.5, 0.965),
        _legend(fig, _component_handles(LATENCY_COMPONENTS), "Latency (top row)", 0.32, 0.925),
        _legend(fig, _component_handles(ENERGY_COMPONENTS), "Energy (bottom row)", 0.78, 0.925),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.885), h_pad=1.2, w_pad=1.0)
    extra = (*legs, sup)
    fig.savefig(OUTPUT_FILE, bbox_extra_artists=extra)
    fig.savefig(OUTPUT_FILE.replace(".pdf", ".png"), dpi=70, bbox_extra_artists=extra)
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
