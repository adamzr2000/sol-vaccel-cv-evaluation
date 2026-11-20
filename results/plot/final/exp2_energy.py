#!/usr/bin/env python3
"""
exp2_energy.py -- Experiment 2: robot energy per inference of AOTInductor vs
SOL, both executed through vAccel, per model and execution target.

Runs (both compilers): Local CPU <- common.LOCAL_RUN_TAG, Edge CPU and Edge GPU
<- common.REMOTE_RUN_TAG. An execution target without runs (Edge CPU before
its rerun under that tag) is left out of the figure.

CSV (exp2-energy.csv): as exp1-energy.csv, per compiler -- measured energy of
robot CPU, edge CPU and edge GPU and the estimated idle share of each, per
inference, plus run duration and number of inferences.

Figure (exp2-energy.pdf): rows = execution target, columns = task; 2 bars per
model (AOTInductor, SOL) = robot CPU energy per inference (measured).

Usage:    python3 exp2_energy.py
Produces: exp2-energy.{csv,pdf}
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (COMPILERS, EDGE_COLOR, HERE, LEGEND_STYLE, REMOTE_RUN_TAG, TARGETS, TITLES, apply_theme,
                    idle_power, model_of, model_ticks, patches, read_rows, save, style_axes, target_runs, write_rows)
from exp1_energy import ENERGY_COLUMNS, energy_values

CSV, PDF = HERE / "exp2-energy.csv", HERE / "exp2-energy.pdf"

FIG_SIZE = (20, 11.0)  # for 3 rows; scaled to the rows with data
BAR_WIDTH = 0.36
HEADROOM = 1.1


def build_csv() -> None:
    idle = idle_power()
    rows = {}
    for target, _, _ in TARGETS:
        tag = None if target == "robot_cpu" else REMOTE_RUN_TAG
        for compiler, *_ in COMPILERS:
            for r in target_runs(target, compiler, tag):
                if (v := energy_values(r, idle)) is not None:
                    rows[(model_of(r), target, compiler)] = v
    write_rows(CSV, ["category", "model", "execution_target", "compiler"] + ENERGY_COLUMNS, rows,
               [(t, c) for t, _, _ in TARGETS for c, *_ in COMPILERS])


def plot() -> None:
    tasks, raw = read_rows(CSV, 4)
    robot = {k: float(d["robot_cpu_j"]) for k, d in raw.items()}
    targets = [t for t in TARGETS if any(k[1] == t[0] for k in robot)]  # rows with data only
    n = len(targets)
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1] * (n + 0.4) / 3.4))
    gs = fig.add_gridspec(n, len(tasks), width_ratios=[len(m) for m in tasks.values()], left=0.095, right=0.988,
                          top=1 - 0.4 / (n + 1), bottom=0.32 / (n + 0.4), wspace=0.2, hspace=0.12)
    for r, (target, row_label, _) in enumerate(targets):
        for col, (task, models) in enumerate(tasks.items()):
            ax = fig.add_subplot(gs[r, col])
            ax.set_ylim(0, max(v for (m, t, _), v in robot.items() if t == target and m in models) * HEADROOM)
            for j, (compiler, _, _, color) in enumerate(COMPILERS):
                for i, m in enumerate(models):
                    if (m, target, compiler) in robot:
                        ax.bar(i + (j - 0.5) * BAR_WIDTH, robot[(m, target, compiler)], width=BAR_WIDTH,
                               facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
            style_axes(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlim(-0.5, len(models) - 0.5)
            if r == n - 1:
                model_ticks(ax, models)
            else:
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(TITLES.get(task, task))
            if col == 0:
                ax.set_ylabel(row_label)
    fig.text(0.008, 0.5, "Energy per inference (J)\nRobot", rotation=90, ha="center", va="center",
             fontsize=plt.rcParams["axes.labelsize"])
    fig.legend(handles=patches([(lbl, c) for _, lbl, _, c in COMPILERS]), loc="lower center",
               bbox_to_anchor=(0.54, 1 - 0.28 / (n + 1)), ncol=len(COMPILERS), **LEGEND_STYLE)
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    apply_theme()
    plot()
