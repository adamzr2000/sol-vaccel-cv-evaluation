#!/usr/bin/env python3
"""
exp3_combined.py -- Experiment 3 (ROS 2 vs vAccel, edge GPU, SOL on both) in one figure: the numbers of
exp3-latency-energy.tex, one column per task and one row per metric:

  row 1  inference latency (ms): median, from exp3-latency.csv (total_ms)
  row 2  energy per inference (J): robot + edge, from exp3-energy.csv (run exp3_latency_energy.py first)

2 bars per model (ROS 2 blue, vAccel green, as exp3-latency.pdf), no value labels; task titles on the top
row only, model names on the bottom row only, aligned y labels, one legend on top (as exp1-combined.pdf).

Usage:    python3 exp3_combined.py
Produces: exp3-combined.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (EDGE_COLOR, HERE, LEGEND_STYLE, TITLES, apply_theme, model_ticks, patches, read_rows, save,
                    style_axes)
from exp3_latency import FRAMEWORKS

LAT_CSV, ENERGY_CSV, PDF = HERE / "exp3-latency.csv", HERE / "exp3-energy.csv", HERE / "exp3-combined.pdf"
FIG_SIZE = (20, 11.0)
BAR_WIDTH = 0.36
HEADROOM = 1.1
YLABEL_X = -0.16   # y labels of the first column, in its axes coordinates (aligned rows)


def plot() -> None:
    tasks, lat = read_rows(LAT_CSV, 3)
    _, en = read_rows(ENERGY_CSV, 3)
    rows = [("Inference latency (ms)", {k: float(d["total_ms"]) for k, d in lat.items()}),
            ("Energy per\ninference (J)", {k: float(d["robot_j"]) + float(d["edge_j"]) for k, d in en.items()})]
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=FIG_SIZE,
                             gridspec_kw={"width_ratios": [len(m) for m in tasks.values()], "hspace": 0.12,
                                          "wspace": 0.2, "left": 0.08, "right": 0.988, "top": 0.86, "bottom": 0.13})
    for r, (label, values) in enumerate(rows):
        for c, (task, models) in enumerate(tasks.items()):
            ax = axes[r][c]
            for j, (fw, _, color) in enumerate(FRAMEWORKS):
                for i, m in enumerate(models):
                    if (m, fw) in values:
                        ax.bar(i + (j - 0.5) * BAR_WIDTH, values[(m, fw)], width=BAR_WIDTH, facecolor=color,
                               edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
            style_axes(ax)
            ax.set_ylim(0, max(v for k, v in values.items() if k[0] in models) * HEADROOM)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlim(-0.5, len(models) - 0.5)
            if r == len(rows) - 1:
                model_ticks(ax, models)
            else:
                ax.set_xticks(range(len(models)))
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if r == 0:
                ax.set_title(TITLES.get(task, task))
        axes[r][0].set_ylabel(label)
        axes[r][0].yaxis.set_label_coords(YLABEL_X, 0.5)
    fig.legend(handles=patches([(lbl, c) for _, lbl, c in FRAMEWORKS]), loc="lower center", bbox_to_anchor=(0.53, 0.915),
               ncol=len(FRAMEWORKS), **LEGEND_STYLE)
    save(fig, PDF)


if __name__ == "__main__":
    apply_theme()
    plot()
