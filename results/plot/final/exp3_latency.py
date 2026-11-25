#!/usr/bin/env python3
"""
exp3_latency.py -- Experiment 3: inference latency of vAccel vs ROS 2 for
offloading to the edge GPU (SOL on both), per model.

Same latency definition for both frameworks: from sending the preprocessed
input on the robot until its model output is received, pre-/post-processing
excluded (median over the run's inputs, IQR = 25th-75th percentile):
  vAccel  duration of the vAccel inference call (robot run of
          common.REMOTE_RUN_TAG) -- the Edge GPU row of exp1_latency.py
  ROS 2   per input, robot monitor and edge pipeline records joined by the input
          stamp (ros-vision-service, CycloneDDS over UDP, RELIABLE QoS, run tag
          common.ROS_RUN_TAG): result received - input published, both on the
          robot; the receive time is taken when the result callback starts,
          before the robot's post-processing (which vAccel's call time also
          excludes). Older monitor files took it after post-processing, so
          their t_postprocess_ms is subtracted.
Computation = median SOL inference on the edge GPU, measured on the edge in the
same run: inside the vAccel agent (common.agent_computation) for vAccel, by the
ROS 2 inference node (t_inference_ns) for ROS 2.
Communication = latency - computation (network + middleware), from the rounded
values so that the parts add up to the printed latency.

LaTeX table (exp3-latency.tex): ROS 2 and vAccSOL side by side, Comp. / Comm. /
Latency with the equivalent frame rate, in the layout of exp1-latency.tex.

Figure: one panel per task, 2 stacked bars per model (ROS 2, vAccel; SOL on both):
Computation (solid) + Communication (hatched), IQR error bars; linear y axis.

Usage:    python3 exp3_latency.py
Produces: exp3-latency.{csv,pdf,tex}
"""
import csv
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import (BASELINE_COLOR, OURS_COLOR, EDGE_COLOR, HATCH, HATCH_ALPHA, HERE, LEGEND_STYLE, MODEL_STATS, ROS_MODEL_STATS, ROS_RUN_TAG, TARGETS, TASKS,
                    TEX_COLOUR, TITLES, agent_computation, apply_theme, latency_tex, lighten, model_of, model_ticks,
                    part_patches, patches, read_rows, save, style_axes, target_runs, write_rows)

CSV, PDF, TEX = HERE / "exp3-latency.csv", HERE / "exp3-latency.pdf", HERE / "exp3-latency.tex"
COLUMNS = ["category", "model", "framework", "inference_ms", "communication_ms", "total_ms", "iqr_low_ms",
           "iqr_high_ms"]
# (framework, legend label, colour): baseline orange, ours green (common.BASELINE_COLOR / OURS_COLOR)
FRAMEWORKS = [("ros2", "ROS 2", BASELINE_COLOR), ("vaccel", "vAccel", OURS_COLOR)]

FIG_SIZE = (20, 6.8)
BAR_WIDTH = 0.36
HEADROOM = 1.1


def latest(pattern: str, folder):
    """Newest file matching pattern (one run per model is expected; warns otherwise)."""
    files = sorted(folder.glob(pattern))
    if len(files) > 1:
        print(f"[warn] {len(files)} files for {pattern}; using the newest, {files[-1].name}")
    return files[-1] if files else None


def split(latency: list, computation: list) -> list:
    """[computation, communication, total, p25, p75] (ms): medians rounded to 0.1 ms, communication =
    total - computation of the rounded values (parts add up to the printed latency)."""
    total, comp = round(statistics.median(latency), 1), round(statistics.median(computation), 1)
    q = statistics.quantiles(latency, n=4, method="inclusive")
    return [comp, total - comp, total, q[0], q[2]]


def ros2_values(model: str):
    """split() of the ROS 2 run of `model` (records joined by input stamp), or None."""
    mon = latest(f"monitor_{model}_sol_{ROS_RUN_TAG}_*.json", ROS_MODEL_STATS / "robot")
    pip = latest(f"pipeline_{model}_sol_{ROS_RUN_TAG}_*.json", ROS_MODEL_STATS / "edge-asus")
    if mon is None or pip is None:
        print(f"[warn] {model}: no ROS 2 monitor/pipeline file, skipped")
        return None
    edge = {r["stamp_ns"]: r for r in json.loads(pip.read_text())["records"]}
    monitor = json.loads(mon.read_text())
    pairs = [(r, edge[r["stamp_ns"]]) for r in monitor["records"] if r["stamp_ns"] in edge]
    # New monitor JSONs (metadata e2e_stop = "result_received") stamp the reception at the result
    # callback's entry, before postprocessing; older ones stamped it after, so subtract it there.
    at_entry = monitor["metadata"].get("e2e_stop") == "result_received"
    latency = [(r["t_result_received_ros_at_ns"] - r["t_image_on_bus_ros_at_ns"]) / 1e6
               - (0.0 if at_entry else r["t_postprocess_ms"]) for r, _ in pairs]
    return split(latency, [e["t_inference_ns"] / 1e6 for _, e in pairs])


def vaccel_values(run: dict):
    """split() of a vAccel Edge GPU run (call durations of the robot + agent profile), or None."""
    calls = agent_computation(run)
    lat = [float(r["inference_ms"]) for r in csv.DictReader(open(MODEL_STATS / "robot" / run["run_id"] / "benchmark_data.csv"))]
    return split(lat, calls) if calls else None


def build_csv() -> None:
    task_models = [m for _, _, models in TASKS for m in models]
    print(f"[source] ros2      <- {ROS_RUN_TAG} ({ROS_MODEL_STATS.parents[2].name})")
    rows = {(m, "ros2"): v for m in task_models if (v := ros2_values(m)) is not None}
    for r in target_runs("edge_gpu"):
        if (v := vaccel_values(r)) is None:
            print(f"[warn] {r['run_id']}: no agent profile, skipped")
            continue
        rows[(model_of(r), "vaccel")] = v
    rows = {k: [f"{x:.1f}" for x in v] for k, v in rows.items()}
    write_rows(CSV, COLUMNS, rows, [(f,) for f, _, _ in FRAMEWORKS])


TEX_GROUPS = [("ros2", "ROS 2", "vsROS", "E6E6E6", ["inference_ms", "communication_ms", "total_ms"]),
              ("vaccel", "vAccel", *TEX_COLOUR["edge_gpu"], ["inference_ms", "communication_ms", "total_ms"])]
TEX_CAPTION = ("Exp.~3: Inference latency (ms) measured on the robot for remote inference on the edge GPU with "
               "SOL-compiled models, offloaded via ROS 2 (gray) and vAccel (green). Latency is divided into "
               "computation (Comp.) and communication (Comm.), the latter covering data transfer and middleware "
               "processing. Values are medians, with the equivalent frame rate in parentheses.")


def write_tex() -> None:
    latency_tex(CSV, TEX, TEX_GROUPS, TEX_CAPTION, "tab:exp3-latency", "exp3_latency.py")


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
    fig.subplots_adjust(left=0.07, right=0.988, top=0.8, bottom=0.2, wspace=0.18)
    handles = patches([(lbl, c) for _, lbl, c in FRAMEWORKS]) + part_patches("Computation", "Communication")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.875), ncol=len(handles),
               columnspacing=1.6, **LEGEND_STYLE)  # one row, no titles
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    write_tex()
    apply_theme()
    plot()
