#!/usr/bin/env python3
"""
exp3_combined.py -- Experiment 3 (ROS 2 vs vAccel, edge GPU, SOL on both) in one figure: the numbers of
exp3-latency-energy.tex, one column per task and one row per metric:

  row 1  inference latency (ms): median, from exp3-latency.csv (total_ms)
  row 2  energy per inference (J): robot + edge, from exp3-energy.csv (run exp3_latency_energy.py first)

2 bars per model (ROS 2 blue, vAccel orange: light fill + darker border, as exp2-latency.pdf), IQR whiskers on the latency
row only (energy per inference is one value per run), no value labels; task titles on the top
row only, model names on the bottom row only, aligned y labels, one legend on top (as exp1-combined.pdf).

Usage:    python3 exp3_combined.py
Produces: exp3-combined.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (EDGE_COLOR, HERE, LEGEND_STYLE, TITLES, apply_theme, model_ticks, patches, read_rows, save,
                    style_axes)
from exp3_latency import FRAMEWORKS

# bar style (this figure and its companion): light fill + darker border of the same hue -- baseline blue,
# ours orange (pale fills; the borders carry the distinction)
BAR_STYLE = {"base": dict(facecolor="#DCE7F3", edgecolor="#2B74B3"), "ours": dict(facecolor="#FCE8D5", edgecolor="#E15726")}
BAR_EDGE_WIDTH = 2.2   # ~0.6-0.8 pt at the printed size
ROLE = {"ros2": "base", "vaccel": "ours"}

LAT_CSV, ENERGY_CSV, PDF = HERE / "exp3-latency.csv", HERE / "exp3-energy.csv", HERE / "exp3-combined.pdf"
FIG_SIZE = (20, 11.0)
BAR_WIDTH = 0.36
HEADROOM = 1.1
YLABEL_X = -0.16   # y labels of the first column, in its axes coordinates (aligned rows)


def plot() -> None:
    tasks, lat = read_rows(LAT_CSV, 3)
    _, en = read_rows(ENERGY_CSV, 3)
    # (y label, median values, IQR (p25, p75) or None): error bars on the latency row only -- energy per inference
    # is one counter-based value per run, with no per-inference spread
    rows = [("Inference latency (ms)", {k: float(d["total_ms"]) for k, d in lat.items()},
             {k: (float(d["iqr_low_ms"]), float(d["iqr_high_ms"])) for k, d in lat.items()}),
            ("Energy per\ninference (J)", {k: float(d["robot_j"]) + float(d["edge_j"]) for k, d in en.items()}, None)]
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=FIG_SIZE,
                             gridspec_kw={"width_ratios": [len(m) for m in tasks.values()], "hspace": 0.12,
                                          "wspace": 0.2, "left": 0.08, "right": 0.988, "top": 0.86, "bottom": 0.13})
    for r, (label, values, iqr) in enumerate(rows):
        for c, (task, models) in enumerate(tasks.items()):
            ax = axes[r][c]
            for j, (fw, _, _) in enumerate(FRAMEWORKS):
                style = BAR_STYLE[ROLE[fw]]
                for i, m in enumerate(models):
                    if (m, fw) in values:
                        x, v = i + (j - 0.5) * BAR_WIDTH, values[(m, fw)]
                        ax.bar(x, v, width=BAR_WIDTH, linewidth=BAR_EDGE_WIDTH, zorder=3, **style)
                        if iqr is not None:  # IQR (p25-p75) whiskers, as in exp2-latency.pdf
                            lo, hi = iqr[(m, fw)]
                            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=EDGE_COLOR,
                                        elinewidth=2.2, capsize=5, capthick=2.2, zorder=10)
            style_axes(ax, clean=True)
            tops = [iqr[k][1] if iqr is not None else v for k, v in values.items() if k[0] in models]
            ax.set_ylim(0, max(tops) * HEADROOM)
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
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(linewidth=BAR_EDGE_WIDTH, label=lbl, **BAR_STYLE[ROLE[fw]]) for fw, lbl, _ in FRAMEWORKS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.915),
               ncol=len(FRAMEWORKS), **LEGEND_STYLE)
    save(fig, PDF)


if __name__ == "__main__":
    apply_theme()
    plot()
