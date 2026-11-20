#!/usr/bin/env python3
"""
exp3_latency.py -- Experiment 3: inference latency of vAccel vs ROS 2 for
offloading to the edge GPU (SOL on both), per model.

Same latency definition for both frameworks: from sending the preprocessed
input on the robot until its model output is received, pre-/post-processing
excluded (median over the run's inputs, IQR = 25th-75th percentile):
  vAccel  duration of the vAccel inference call (robot run of
          common.REMOTE_RUN_TAG) -- the Edge GPU bars of exp1_latency.py
  ROS 2   per input: (result received - input published) - post-processing
          time, from the robot monitor (ros-vision-service, CycloneDDS over
          UDP, run tag common.ROS_RUN_TAG). The input is stamped right after
          pre-processing, just before publishing; the receive time is taken
          after post-processing, hence its subtraction.
Computation = median pure SOL inference on the edge GPU: the isolated vAccel
edge run (common.EDGE_INFERENCE_RUN_TAG) for vAccel, the ROS 2 inference
node's own measurement (edge pipeline file) for ROS 2.
Communication = latency - computation (network + middleware).

Figure: one panel per task, 2 stacked bars per model (ROS 2, vAccel):
Computation (solid) + Communication (hatched), IQR error bars; linear y axis.

Usage:    python3 exp3_latency.py
Produces: exp3-latency.{csv,pdf}
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import (EDGE_COLOR, HATCH, HATCH_ALPHA, HERE, ROS_MODEL_STATS, ROS_RUN_TAG, TARGETS, TASKS, TITLES,
                    apply_theme, edge_runs, legend_row, lighten, model_of, model_ticks, part_patches, patches,
                    read_rows, save, style_axes, target_runs, write_rows)

CSV, PDF = HERE / "exp3-latency.csv", HERE / "exp3-latency.pdf"
COLUMNS = ["category", "model", "framework", "inference_ms", "communication_ms", "total_ms", "iqr_low_ms",
           "iqr_high_ms"]
# (framework, legend label, colour): ROS 2 as gray baseline; vAccel in the Exp. 1 Edge GPU colour (same data)
FRAMEWORKS = [("ros2", "ROS 2", "#BDBDBD"), ("vaccel", "vAccel", dict((t, c) for t, _, c in TARGETS)["edge_gpu"])]

FIG_SIZE = (20, 6.8)
BAR_WIDTH = 0.36
HEADROOM = 1.1


def latest(pattern: str, folder):
    """Newest file matching pattern (one run per model is expected; warns otherwise)."""
    files = sorted(folder.glob(pattern))
    if len(files) > 1:
        print(f"[warn] {len(files)} files for {pattern}; using the newest, {files[-1].name}")
    return files[-1] if files else None


def ros2_values(model: str):
    """[inference, communication, total, p25, p75] (ms) of the ROS 2 run of `model`, or None."""
    mon = latest(f"monitor_{model}_sol_{ROS_RUN_TAG}_*.json", ROS_MODEL_STATS / "robot")
    pip = latest(f"pipeline_{model}_sol_{ROS_RUN_TAG}_*.json", ROS_MODEL_STATS / "edge-asus")
    if mon is None or pip is None:
        print(f"[warn] {model}: no ROS 2 monitor/pipeline file, skipped")
        return None
    records = json.loads(mon.read_text())["records"]
    latency = np.array([(r["t_result_received_ros_at_ns"] - r["t_image_on_bus_ros_at_ns"]) / 1e6
                        - r.get("t_postprocess_ms", 0.0) for r in records])
    inference = float(json.loads(pip.read_text())["summary"]["t_inference_ms"]["p50"])
    p25, p50, p75 = np.percentile(latency, [25, 50, 75])
    return [inference, p50 - inference, p50, p25, p75]


def build_csv() -> None:
    task_models = [m for _, _, models in TASKS for m in models]
    print(f"[source] ros2      <- {ROS_RUN_TAG} ({ROS_MODEL_STATS.parents[2].name})")
    rows = {(m, "ros2"): v for m in task_models if (v := ros2_values(m)) is not None}

    edge = {model_of(r): float(r["inference_ms"]["p50"]) for r in edge_runs()
            if (r.get("device", "").lower(), r.get("backend", "").lower()) == ("gpu", "vaccel-local-sol")}
    for r in target_runs("edge_gpu"):
        call, model = r["inference_ms"], model_of(r)
        if model not in edge:
            print(f"[warn] {model}: no isolated edge inference, skipped")
            continue
        total = float(call["p50"])
        rows[(model, "vaccel")] = [edge[model], total - edge[model], total, float(call["p25"]), float(call["p75"])]

    rows = {k: [f"{x:.1f}" for x in v] for k, v in rows.items()}
    write_rows(CSV, COLUMNS, rows, [(f,) for f, _, _ in FRAMEWORKS])


def plot() -> None:
    tasks, raw = read_rows(CSV, 3)
    values = {k: {c: float(v) for c, v in d.items()} for k, d in raw.items()}
    fig, axes = plt.subplots(1, len(tasks), figsize=FIG_SIZE, gridspec_kw={"width_ratios": [len(m) for m in tasks.values()]})
    for ax, (task, models) in zip(axes, tasks.items()):
        ax.set_ylim(0, max(d["iqr_high_ms"] for (m, _), d in values.items() if m in models) * HEADROOM)
        for j, (fw, _, color) in enumerate(FRAMEWORKS):
            for i, m in enumerate(models):
                d = values.get((m, fw))
                if d is None:
                    continue
                x = i + (j - 0.5) * BAR_WIDTH
                bar = dict(width=BAR_WIDTH, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
                ax.bar(x, d["inference_ms"], facecolor=color, **bar)
                ax.bar(x, d["communication_ms"], bottom=d["inference_ms"], hatch=HATCH,
                       facecolor=lighten(color, HATCH_ALPHA), **bar)
                ax.errorbar(x, d["total_ms"], yerr=[[d["total_ms"] - d["iqr_low_ms"]], [d["iqr_high_ms"] - d["total_ms"]]],
                            fmt="none", ecolor=EDGE_COLOR, elinewidth=1.6, capsize=5, capthick=1.6, zorder=10)
        style_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.5, len(models) - 0.5)
        model_ticks(ax, models)
        ax.set_title(TITLES.get(task, task))
    axes[0].set_ylabel("Inference latency (ms)")
    fig.subplots_adjust(left=0.07, right=0.988, top=0.78, bottom=0.2, wspace=0.18)
    legend_row(fig, [("Framework", patches([(lbl, c) for _, lbl, c in FRAMEWORKS])),
                     ("Latency contribution", part_patches("Computation", "Communication"))], y=0.87)
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    apply_theme()
    plot()
