#!/usr/bin/env python3
"""
exp1_combined.py -- Experiment 1 (vAccSOL): latency, frame rate and energy in one figure, one column per task
and one row per metric (same data and drawing as exp1-latency.pdf, exp1-fps.pdf and exp1-energy.pdf):

  row 1  inference latency (ms): broken linear y axis, Computation (solid) + Communication (hatched),
         IQR error bars                                                                (exp1_latency.draw_bars)
  row 2  frame rate (FPS) = 1000 / median latency                                       (as exp1_latency.plot_fps)
  row 3  energy per inference (J): Robot (solid) + Edge (dotted)                           (as exp1_energy.plot)

Task titles on the top row only, model names on the bottom row only, same bar width and fonts in every
row; no value labels on the bars; one legend row on top: Execution (colours) | Latency (solid / // hatch) |
Energy (solid / dots), the two metrics' parts with different hatches.

Reads exp1-latency.csv and exp1-energy.csv (run exp1_latency.py and exp1_energy.py first).

Usage:    python3 exp1_combined.py
Produces: exp1-combined.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import matplotlib.patches as mpatches

from common import (EDGE_COLOR, HATCH, HATCH_ALPHA, HERE, TARGETS, TITLES, apply_theme, legend_row,
                    lighten, model_ticks, patches, read_rows, save, style_axes)
from exp1_latency import BAR_WIDTH, HEIGHT_RATIOS, LOW_HEADROOM, TOP_HEADROOM

LAT_CSV, ENERGY_CSV, PDF = HERE / "exp1-latency.csv", HERE / "exp1-energy.csv", HERE / "exp1-combined.pdf"
FIG_SIZE = (20, 16.5)
ROW_RATIOS = (3.4, 2.4, 2.4)   # latency (both parts of the broken axis) : frame rate : energy
ROW_SPACE = 0.11               # between metric rows (the broken axis keeps its own, tighter gap)
ENERGY_HATCH = ".."            # edge energy: dots, distinct from the latency's communication hatch (HATCH, //)
HEADROOM = 1.08
YLABEL_X = -0.16               # y labels of the first column, in its axes coordinates (aligned rows)


# bar style, as exp2-latency.pdf / exp3-combined.pdf: pale fill + darker border of the same hue per execution target
# (blue / orange / green as in the Exp. 1 table); the second part of a stacked bar hatched in the border colour
TARGET_STYLE = {"robot_cpu": ("#DCE7F3", "#2B74B3"), "edge_cpu": ("#FCE8D5", "#E15726"), "edge_gpu": ("#D3EEE6", "#2E9B7E")}
LINE = 2.2               # bar borders and error bars (~0.6-0.8 pt at the printed size)
HATCH_LINE = 1.6         # hatch strokes (rcParams hatch.linewidth, this figure only)
BAR_FILL = 0.86          # drawn bar width / slot width (small gap between neighbouring bars)
PART_GRAY = ("#EFEFEF", "#555555")  # neutral fill/border of the Latency / Energy part legends


def x_of(i: int, j: int) -> float:
    return i + (j - 1) * BAR_WIDTH


def stacked(ax, x, target, base, top=0.0, hatch=None):
    """Bar of `target`: base part plain, optional top part hatched (same fill, hatch in the border colour)."""
    fill, edge = TARGET_STYLE[target]
    # slightly narrower than the slot: neighbouring borders no longer overlap into one thick line
    bar = dict(width=BAR_WIDTH * BAR_FILL, facecolor=fill, edgecolor=edge, linewidth=LINE, zorder=3)
    ax.bar(x, base, **bar)
    if top > 0:
        ax.bar(x, top, bottom=base, hatch=hatch, **bar)


def left_break_marks(ax_top, ax_bot) -> None:
    """'//' marks on the left y axis where it is broken (clean style: no right axis)."""
    d, k = 0.025, HEIGHT_RATIOS[1] / HEIGHT_RATIOS[0]
    kw = dict(color=EDGE_COLOR, clip_on=False, linewidth=LINE * 0.8)
    ax_top.plot((-d, d), (-d * k, d * k), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)


def part_legend(solid_label: str, hatched_label: str, hatch: str) -> list:
    """Neutral-gray swatches for the solid and the hatched part of a stacked bar (common.part_patches with
    the row's own hatch)."""
    fill, edge = PART_GRAY
    return [mpatches.Patch(facecolor=fill, edgecolor=edge, linewidth=LINE, label=solid_label),
            mpatches.Patch(facecolor=fill, edgecolor=edge, linewidth=LINE, hatch=hatch, label=hatched_label)]


def latency_row(fig, gs, tasks, values):
    """Broken-axis latency panels; -> [(ax_top, ax_bot)] per task."""
    out = []
    for col, (task, models) in enumerate(tasks.items()):
        sub = gs[col].subgridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=0.07)
        ax_top, ax_bot = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        edge = [values[(m, t)] for m in models for t, _, _ in TARGETS[1:] if (m, t) in values]
        local = [values[(m, "robot_cpu")] for m in models if (m, "robot_cpu") in values]
        low_top = max(d["iqr_high_ms"] for d in edge) * LOW_HEADROOM[False]
        lo, hi = max(low_top, min(d["total_ms"] for d in local) * 0.6), max(d["iqr_high_ms"] for d in local) * TOP_HEADROOM
        ax_bot.set_ylim(0, low_top)
        ax_top.set_ylim(lo, hi)
        for ax in (ax_top, ax_bot):
            for j, (target, _, _) in enumerate(TARGETS):
                for i, m in enumerate(models):
                    d = values.get((m, target))
                    if d is None:
                        continue
                    x = x_of(i, j)
                    stacked(ax, x, target, d["inference_ms"], d["communication_ms"] or 0.0, HATCH)
                    ax.errorbar(x, d["total_ms"], yerr=[[d["total_ms"] - d["iqr_low_ms"]], [d["iqr_high_ms"] - d["total_ms"]]],
                                fmt="none", ecolor=EDGE_COLOR, elinewidth=LINE, capsize=5, capthick=LINE, zorder=10)
            style_axes(ax, clean=True)
            ax.set_xlim(-0.5, len(models) - 0.5)
            ax.set_xticks(range(len(models)))
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_top.set_yticks([t for t in MaxNLocator(nbins=2).tick_values(lo, hi) if lo + 0.08 * (hi - lo) <= t <= hi])
        ax_bot.set_yticks([t for t in MaxNLocator(nbins=5).tick_values(0, low_top) if 0 <= t <= 0.92 * low_top])
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.set_title(TITLES.get(task, task))
        left_break_marks(ax_top, ax_bot)
        out.append((ax_top, ax_bot))
    return out


def fps_row(fig, gs, tasks, values):
    axes = []
    for col, (task, models) in enumerate(tasks.items()):
        ax = fig.add_subplot(gs[col])
        fps = {k: 1000 / v["total_ms"] for k, v in values.items() if k[0] in models}
        top = max(fps.values()) * HEADROOM
        ax.set_ylim(0, top)
        for j, (target, _, _) in enumerate(TARGETS):
            for i, m in enumerate(models):
                if (m, target) in fps:
                    stacked(ax, x_of(i, j), target, fps[(m, target)])
        style_axes(ax, clean=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_xticks(range(len(models)))
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        axes.append(ax)
    return axes


def energy_row(fig, gs, tasks, energy):
    axes = []
    for col, (task, models) in enumerate(tasks.items()):
        ax = fig.add_subplot(gs[col])
        for j, (target, _, _) in enumerate(TARGETS):
            for i, m in enumerate(models):
                if (m, target) in energy:
                    robot, edge = energy[(m, target)]
                    stacked(ax, x_of(i, j), target, robot, edge, ENERGY_HATCH)
        style_axes(ax, clean=True)
        ax.set_ylim(0, max(r + e for k, (r, e) in energy.items() if k[0] in models) * HEADROOM)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.5, len(models) - 0.5)
        model_ticks(ax, models)
        axes.append(ax)
    return axes


def plot() -> None:
    tasks, lat_raw = read_rows(LAT_CSV, 3)
    values = {k: {c: float(v) if v else None for c, v in d.items()} for k, d in lat_raw.items()}
    _, en_raw = read_rows(ENERGY_CSV, 3)
    energy = {k: (float(d["robot_cpu_j"]), float(d["edge_cpu_j"] or 0) + float(d["edge_gpu_j"] or 0))
              for k, d in en_raw.items()}

    plt.rcParams["hatch.linewidth"] = HATCH_LINE
    fig = plt.figure(figsize=FIG_SIZE)
    outer = fig.add_gridspec(3, 1, height_ratios=ROW_RATIOS, hspace=ROW_SPACE,
                             left=0.08, right=0.988, top=0.885, bottom=0.085)
    widths = [len(m) for m in tasks.values()]
    rows = [outer[r].subgridspec(1, len(tasks), width_ratios=widths, wspace=0.2) for r in range(3)]
    lat = latency_row(fig, rows[0], tasks, values)
    fps = fps_row(fig, rows[1], tasks, values)
    en = energy_row(fig, rows[2], tasks, energy)

    # y labels on the first column, at the same x for every row; the latency label is centred on both parts
    # of its broken axis
    top, bot = lat[0]
    span = (top.get_position().y1 - bot.get_position().y0) / bot.get_position().height
    bot.set_ylabel("Inference latency (ms)")
    bot.yaxis.set_label_coords(YLABEL_X, span / 2)
    for ax, label in ((fps[0], "Frame rate (FPS)"), (en[0], "Energy per\ninference (J)")):
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(YLABEL_X, 0.5)

    execution = [mpatches.Patch(facecolor=TARGET_STYLE[t][0], edgecolor=TARGET_STYLE[t][1], linewidth=LINE, label=lbl)
                 for t, lbl, _ in TARGETS]
    legend_row(fig, [("Execution", execution),
                     ("Latency", part_legend("Computation", "Communication", HATCH)),
                     ("Energy", part_legend("Robot", "Edge", ENERGY_HATCH))], y=0.925)
    save(fig, PDF)


if __name__ == "__main__":
    apply_theme()
    plot()
