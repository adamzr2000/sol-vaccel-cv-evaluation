#!/usr/bin/env python3
"""
exp2_latency.py -- Experiment 2: inference latency of AOTInductor vs SOL, both
executed through vAccel, per model and execution target.

Pure computation (no network, no pre/post-processing): median inference_ms of
runs where the model executes on the device itself (backend vaccel-local-*):
  Local CPU  robot runs of common.LOCAL_RUN_TAG
  Edge CPU   edge runs of common.EDGE_INFERENCE_RUN_TAG, device cpu
  Edge GPU   same, device gpu
iqr_low/high_ms = 25th / 75th percentile.

Figure: rows = execution target, columns = task; 2 bars per model
(AOTInductor, SOL) with IQR error bars; linear y axis per panel.

Usage:    python3 exp2_latency.py
Produces: exp2-latency.{csv,pdf}
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (COMPILERS, EDGE_COLOR, EDGE_HOST, HERE, LEGEND_STYLE, LOCAL_RUN_TAG, TARGETS, TITLES,
                    apply_theme, edge_runs, load_runs, model_of, model_ticks, patches, read_rows, save, style_axes,
                    write_rows)

CSV, PDF = HERE / "exp2-latency.csv", HERE / "exp2-latency.pdf"
COLUMNS = ["category", "model", "execution_target", "compiler", "inference_ms", "iqr_low_ms", "iqr_high_ms"]

FIG_SIZE = (20, 11.0)
BAR_WIDTH = 0.36
HEADROOM = 1.1


def build_csv() -> None:
    runs = {"robot": [r for r in load_runs(LOCAL_RUN_TAG) if r.get("host", "").lower() == "robot"],
            EDGE_HOST: edge_runs()}
    rows = {}
    for target, _, _ in TARGETS:
        host, device = ("robot" if target == "robot_cpu" else EDGE_HOST), target[-3:]
        for compiler, _, backend, _ in COMPILERS:
            picked = [r for r in runs[host] if (r.get("device", "").lower(), r.get("backend", "").lower())
                      == (device, f"vaccel-local-{backend}")]
            print(f"[source] {target:<9} {compiler:<4} <- {host} ({len(picked)} runs)")
            for r in picked:
                inf = r["inference_ms"]
                rows[(model_of(r), target, compiler)] = [f"{float(inf[k]):.2f}" for k in ("p50", "p25", "p75")]
    write_rows(CSV, COLUMNS, rows, [(t, c) for t, _, _ in TARGETS for c, *_ in COMPILERS])


def plot() -> None:
    tasks, raw = read_rows(CSV, 4)
    values = {k: {c: float(v) for c, v in d.items()} for k, d in raw.items()}
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(len(TARGETS), len(tasks), width_ratios=[len(m) for m in tasks.values()],
                          left=0.085, right=0.988, top=0.885, bottom=0.11, wspace=0.2, hspace=0.12)
    for r, (target, row_label, _) in enumerate(TARGETS):
        for col, (task, models) in enumerate(tasks.items()):
            ax = fig.add_subplot(gs[r, col])
            ax.set_ylim(0, max(d["iqr_high_ms"] for (m, t, _), d in values.items()
                               if t == target and m in models) * HEADROOM)
            for j, (compiler, _, _, color) in enumerate(COMPILERS):
                for i, m in enumerate(models):
                    d = values.get((m, target, compiler))
                    if d is None:
                        continue
                    x, v = i + (j - 0.5) * BAR_WIDTH, d["inference_ms"]
                    ax.bar(x, v, width=BAR_WIDTH, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
                    ax.errorbar(x, v, yerr=[[v - d["iqr_low_ms"]], [d["iqr_high_ms"] - v]], fmt="none",
                                ecolor=EDGE_COLOR, elinewidth=1.4, capsize=4, capthick=1.4, zorder=10)
            style_axes(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlim(-0.5, len(models) - 0.5)
            if r == len(TARGETS) - 1:
                model_ticks(ax, models)
            else:
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(TITLES.get(task, task))
            if col == 0:
                ax.set_ylabel(row_label)
    fig.text(0.008, 0.5, "Inference latency (ms)", rotation=90, ha="center", va="center",
             fontsize=plt.rcParams["axes.labelsize"])
    fig.legend(handles=patches([(lbl, c) for _, lbl, _, c in COMPILERS]), loc="lower center",
               bbox_to_anchor=(0.54, 0.925), ncol=len(COMPILERS), **LEGEND_STYLE)
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    apply_theme()
    plot()
