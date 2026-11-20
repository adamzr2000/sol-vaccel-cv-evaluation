#!/usr/bin/env python3
"""
exp1_network.py -- Experiment 1 (vAccSOL): network throughput of the robot
while offloading inference, per task, split into uplink and downlink.

Throughput = traffic of the robot's network interface over a run, from the
per-second counters in system-stats/robot/{run_id}_net.csv (MB per sample):
  uplink   = transmitted by the robot (input tensors to the edge)
  downlink = received by the robot (model outputs from the edge)
in Mbit/s = sum over the run x 8 / duration (the first row only holds the
counters at start and is not counted). Throughput = data per inference x
inference rate of the run.

Runs: Edge CPU <- common.REMOTE_CPU_RUN_TAG, Edge GPU <- common.REMOTE_RUN_TAG,
except Edge GPU segmentation, taken (whole runs, both directions) from the
uncapped runs "e2eGSt" (2026-10-29): the REMOTE_RUN_TAG segmentation runs were
capped at 7 inferences/s. "e2eGSt" predates the --malloc-tunables fix, which
affects pre-processing time, not the traffic per inference.

CSVs: exp1-network.csv (per model and target; total_std_mbit_s = std of the
total over the run's full 1 s samples) and exp1-network-task.csv (per task and
target: mean over the task's models -- all exchange tensors of the same size --
and the std over all pooled 1 s samples).

Figure (exp1-network.pdf): Edge GPU only (PLOT_TARGETS), one stacked bar per
task (mean): Uplink (dark, bottom) + Downlink (light, top). Stacked because
Wi-Fi is a shared half-duplex medium: the total is what loads the link.

Usage:    python3 exp1_network.py
Produces: exp1-network.csv, exp1-network-task.csv, exp1-network.pdf
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import (EDGE_COLOR, HERE, LEGEND_STYLE, SYSTEM_STATS, TARGETS, TASKS, apply_theme, model_of, patches,
                    save, style_axes, target_runs, write_rows)

CSV, TASK_CSV, PDF = HERE / "exp1-network.csv", HERE / "exp1-network-task.csv", HERE / "exp1-network.pdf"
OFFLOAD_TARGETS = [t for t, _, _ in TARGETS[1:]]   # edge_cpu, edge_gpu
PLOT_TARGETS = ["edge_gpu"]
# (target, task) -> run tag used instead of the target's default (whole runs, both directions)
TAG_OVERRIDE = {("edge_gpu", "semantic_segmentation"): "e2eGSt"}  # uncapped segmentation runs

# (CSV column, legend label, colour): blue-grey pair, distinct in grayscale, no clash with the target colours
DIRECTIONS = [("uplink_mbit_s", "Uplink (robot → edge)", "#4F6D8F"),
              ("downlink_mbit_s", "Downlink (edge → robot)", "#B7C7D9")]
TASK_LABEL = {"image_classification": "Image\nClassification",
              "video_action_recognition": "Video Action\nRecognition",
              "semantic_segmentation": "Semantic\nSegmentation"}
FIG_SIZE = (10.5, 6.4)
BAR_WIDTH = 0.55
HEADROOM = 1.1


def throughput(run_id: str):
    """(uplink Mbit/s, downlink Mbit/s, duration s, [total Mbit/s of each full 1 s sample]), or None."""
    path = SYSTEM_STATS / "robot" / f"{run_id}_net.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 4:
        return None
    seconds = (int(rows[-1]["timestamp"]) - int(rows[0]["timestamp"])) / 1000.0
    tx = sum(float(r["net_tx_mb"]) for r in rows[1:])
    rx = sum(float(r["net_rx_mb"]) for r in rows[1:])
    samples = []  # full intervals only: skip the start row and the final (partial) row
    for prev, cur in zip(rows[1:-2], rows[2:-1]):
        dt = (int(cur["timestamp"]) - int(prev["timestamp"])) / 1000.0
        samples.append((float(cur["net_tx_mb"]) + float(cur["net_rx_mb"])) * 8 / dt)
    return tx * 8 / seconds, rx * 8 / seconds, seconds, samples


def build_csv() -> None:
    task_of = {m: task for task, _, models in TASKS for m in models}
    data = {}  # (model, target) -> throughput(...)
    for target in OFFLOAD_TARGETS:
        overrides = {task: tag for (t, task), tag in TAG_OVERRIDE.items() if t == target}
        runs = [r for r in target_runs(target) if task_of.get(model_of(r)) not in overrides]
        for task, tag in overrides.items():
            runs += [r for r in target_runs(target, tag=tag) if task_of.get(model_of(r)) == task]
        for r in runs:
            if (t := throughput(r["run_id"])) is None:
                print(f"[warn] {r['run_id']}: no network counters, skipped")
            else:
                data[(model_of(r), target)] = t

    rows = {k: [f"{up:.2f}", f"{down:.2f}", f"{np.std(s):.2f}", f"{sec:.1f}"] for k, (up, down, sec, s) in data.items()}
    write_rows(CSV, ["category", "model", "execution_target", "uplink_mbit_s", "downlink_mbit_s", "total_std_mbit_s",
                     "duration_s"], rows, [(t,) for t in OFFLOAD_TARGETS])

    with open(TASK_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "execution_target", "num_models", "uplink_mbit_s", "downlink_mbit_s", "total_mbit_s",
                    "total_std_mbit_s"])
        for task, _, models in TASKS:
            for target in OFFLOAD_TARGETS:
                present = [m for m in models if (m, target) in data]
                if present:
                    up = float(np.mean([data[(m, target)][0] for m in present]))
                    down = float(np.mean([data[(m, target)][1] for m in present]))
                    pooled = [v for m in present for v in data[(m, target)][3]]
                    w.writerow([task, target, len(present)] + [f"{v:.2f}" for v in (up, down, up + down, np.std(pooled))])
    print(f"[OK] Saved table to: {TASK_CSV.name}")


def plot() -> None:
    with open(TASK_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    targets = [t for t in OFFLOAD_TARGETS if t in PLOT_TARGETS and any(r["execution_target"] == t for r in rows)]
    rows = [r for r in rows if r["execution_target"] in targets]
    top = max(float(r["total_mbit_s"]) for r in rows) * HEADROOM
    label = {t: lbl for t, lbl, _ in TARGETS}

    fig, axes = plt.subplots(1, len(targets), figsize=(FIG_SIZE[0] * (1 + 0.7 * (len(targets) - 1)), FIG_SIZE[1]),
                             sharey=True, squeeze=False)
    bar = dict(width=BAR_WIDTH, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
    for ax, target in zip(axes[0], targets):
        tasks = [r for r in rows if r["execution_target"] == target]
        x = np.arange(len(tasks))
        up = np.array([float(t["uplink_mbit_s"]) for t in tasks])
        ax.bar(x, up, facecolor=DIRECTIONS[0][2], **bar)
        ax.bar(x, [float(t["downlink_mbit_s"]) for t in tasks], bottom=up, facecolor=DIRECTIONS[1][2], **bar)
        style_axes(ax)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t["category"]] for t in tasks])
        if len(targets) > 1:
            ax.set_title(label[target])
    axes[0][0].set_ylabel("Network throughput\n(Mbit/s)")
    # legend inside the (first) panel, top segment listed first
    axes[0][0].legend(handles=patches([(lbl, c) for _, lbl, c in reversed(DIRECTIONS)]), loc="upper left",
                      **LEGEND_STYLE)
    fig.tight_layout()
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    apply_theme()
    plot()
