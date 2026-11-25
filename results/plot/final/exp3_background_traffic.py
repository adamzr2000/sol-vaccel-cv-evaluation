#!/usr/bin/env python3
"""
exp3_background_traffic.py -- Experiment 3, background traffic: inference latency of ROS 2 vs vAccel (edge GPU,
SOL on both) under background traffic on the robot's link, per model and traffic load.

Same latency definition as Exp. 3 (exp3_latency.py), over the run's results:
  vAccel  call duration on the robot (benchmark_data.csv) of the run tagged e2e-net-load-{load}M
  ROS 2   result received - input published, both on the robot, of the run tagged remote-gpu-net-load-{load}M
          (monitor and edge pipeline records joined by input stamp; older monitor files stamped reception after
          post-processing, so their t_postprocess_ms is subtracted -- see exp3_latency.ros2_values)
Load 0 (no background traffic) = the runs of exp3-latency-energy.tex (common.ROS_RUN_TAG / REMOTE_RUN_TAG).
Result rate (results/s) = results / time from the first input to the last result (ROS 2) or / the measured
loop (vAccel).

LaTeX table (exp3-background-traffic.tex, single column): one row per load, per model ROS 2 | vAccel medians;
the better of each pair shaded (light green, as in Exp. 2/3); "--" where a framework has no run at that load.

Figures: only the loads measured for both frameworks (plot_loads).
Figure (exp3-background-traffic.pdf): one panel per model, latency vs background traffic; per framework the
median (line + markers at the measured loads) and the IQR (p25-p75, light band); colours as the other comparison
figures (common.BASELINE_COLOR / OURS_COLOR).
Box-plot figure (exp3-background-traffic-box.pdf): the same data as box plots (box = IQR, whiskers =
p5-p95, outliers not drawn), loads as evenly spaced categories.

Usage:    python3 exp3_background_traffic.py
Produces: exp3-background-traffic.{csv,tex,pdf}, exp3-background-traffic-box.pdf
"""
import csv
import json
import statistics
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (BASELINE_COLOR, EDGE_COLOR, HERE, LEGEND_STYLE, MODEL_STATS, OURS_COLOR, REMOTE_RUN_TAG,
                    ROS_MODEL_STATS, ROS_RUN_TAG, TEX_COLOUR, TEX_MODEL, _iso_s, apply_theme, load_runs, model_of,
                    patches, save, style_axes)
from exp3_latency import latest

CSV, TEX, PDF, BOX_PDF = (HERE / "exp3-background-traffic.csv", HERE / "exp3-background-traffic.tex",
                          HERE / "exp3-background-traffic.pdf", HERE / "exp3-background-traffic-box.pdf")
MODELS = ["resnet50", "r3d_18", "deeplabv3_resnet50"]
LOADS = [0, 100, 300, 400, 500, 550, 600]           # Mbit/s of background traffic; 0 = none (Exp. 3 runs)
FRAMEWORKS = [("ros2", "ROS~2"), ("vaccel", "vAccel")]
ROS_TAG, VACCEL_TAG = "remote-gpu-net-load-{}M", "e2e-net-load-{}M"
TEX_BETTER = TEX_COLOUR["edge_gpu"]
CAPTION = ("Exp.~3: Inference latency (ms) under background traffic for remote inference of SOL-optimized models "
           "on the edge GPU via ROS~2 and vAccel. The better of the two is shaded; a dash marks a load not measured.")
# figure
FIG_SIZE = (20, 6.2)
PLOT_FRAMEWORKS = [("ros2", "ROS 2", BASELINE_COLOR), ("vaccel", "vAccel", OURS_COLOR)]
BAND_ALPHA = 0.18
def plot_loads(model: str, values: dict) -> list:
    """Loads drawn for `model` in the figures: only those measured for both frameworks, so every point is a direct
    comparison (the table/CSV keep the single-framework runs, as "--" pairs)."""
    return [load for load in LOADS if all((model, f, load) in values for f, _, _ in PLOT_FRAMEWORKS)]


def tag(template: str, load: int) -> str:
    """Run tag of a load. 0 (no background traffic): the runs of exp3-latency-energy.tex (common.ROS_RUN_TAG /
    common.REMOTE_RUN_TAG), so that this table's first row equals that table's latencies."""
    if load == 0:
        return ROS_RUN_TAG if template == ROS_TAG else REMOTE_RUN_TAG
    return template.format(load)


def ros2_run(model: str, load: int):
    """(latencies ms, results/s) of the ROS 2 run of `model` at `load`, or None."""
    run_tag = tag(ROS_TAG, load)
    mon = latest(f"monitor_{model}_sol_{run_tag}_*.json", ROS_MODEL_STATS / "robot")
    if mon is None:
        return None
    pip = latest(f"pipeline_{model}_sol_{run_tag}_*.json", ROS_MODEL_STATS / "edge-asus")
    if pip is None:
        # the edge file's tag can be garbled (the R3D-18 550M run was saved as "...net-load-\x0cM_..."): take the
        # pipeline of the same model started within 5 s of the monitor (the join by input stamp checks it)
        start = datetime.strptime(mon.stem.rsplit("_", 1)[1], "%Y%m%dT%H%M%S")
        near = [p for p in (ROS_MODEL_STATS / "edge-asus").glob(f"pipeline_{model}_sol_*.json")
                if abs((datetime.strptime(p.stem.rsplit("_", 1)[1], "%Y%m%dT%H%M%S") - start).total_seconds()) <= 5]
        if len(near) != 1:
            return None
        pip = near[0]
        print(f"[info] {mon.name}: edge pipeline matched by start time: {pip.name!r}")
    monitor = json.loads(mon.read_text())
    edge = {r["stamp_ns"] for r in json.loads(pip.read_text())["records"]}
    at_entry = monitor["metadata"].get("e2e_stop") == "result_received"
    rec = [r for r in monitor["records"] if r["stamp_ns"] in edge]
    if not rec:
        return None
    lat = [(r["t_result_received_ros_at_ns"] - r["t_image_on_bus_ros_at_ns"]) / 1e6
           - (0.0 if at_entry else r["t_postprocess_ms"]) for r in rec]
    span = (max(r["t_result_received_ros_at_ns"] for r in rec) - min(r["t_image_on_bus_ros_at_ns"] for r in rec)) / 1e9
    return lat, len(rec) / span


def vaccel_run(model: str, load: int):
    """(latencies ms, results/s) of the vAccel edge-GPU run of `model` at `load`, or None."""
    runs = [r for r in load_runs(tag(VACCEL_TAG, load)) if model_of(r) == model and "gpu" in r["run_id"]]
    if not runs:
        return None
    run = runs[-1]
    with open(MODEL_STATS / "robot" / run["run_id"] / "benchmark_data.csv", newline="") as f:
        lat = [float(r["inference_ms"]) for r in csv.DictReader(f)]
    w = run["time_window"]
    return lat, len(lat) / (_iso_s(w["stop"]) - _iso_s(w["start"]))


def build_csv() -> dict:
    """{(model, framework, load): {median, p25, p75, results, rate}}; also written to exp3-background-traffic.csv."""
    values = {}
    for m in MODELS:
        for f, _ in FRAMEWORKS:
            for load in LOADS:
                run = (ros2_run if f == "ros2" else vaccel_run)(m, load)
                if run is None:
                    print(f"[info] {m} {f} {load} Mbit/s: no run")
                    continue
                lat, rate = run
                q = statistics.quantiles(lat, n=4, method="inclusive")
                values[(m, f, load)] = dict(median=statistics.median(lat), p25=q[0], p75=q[2], results=len(lat),
                                            rate=rate, latencies=lat)
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "framework", "load_mbps", "median_ms", "p25_ms", "p75_ms", "results", "results_per_s"])
        for (m, f, load), v in values.items():
            w.writerow([m, f, load, f"{v['median']:.1f}", f"{v['p25']:.1f}", f"{v['p75']:.1f}", v["results"],
                        f"{v['rate']:.2f}"])
    print(f"[OK] Saved table to: {CSV.name}")
    return values


def write_tex(values: dict) -> None:
    def printed(m, f, load):
        v = values.get((m, f, load))
        return None if v is None else f"{round(v['median'], 1):.1f}"

    def cell(m, f, load):
        x = printed(m, f, load)
        if x is None:
            return "--"
        other = printed(m, "vaccel" if f == "ros2" else "ros2", load)
        better = other is not None and float(x) < float(other)
        return f"\\cellcolor{{{TEX_BETTER[0]}}}{x}" if better else x

    # rows = traffic loads, column groups = models (ROS 2 | vAccel): fits one column
    n_cols = 1 + 2 * len(MODELS)
    spec = "r" * (2 * len(MODELS))
    lines = ["% Requires: \\usepackage[table]{xcolor}, booktabs",
             "% Generated by exp3_background_traffic.py -- do not edit by hand.",
             f"\\definecolor{{{TEX_BETTER[0]}}}{{HTML}}{{{TEX_BETTER[1]}}}",
             "\\begin{table}[!t]", f"\\caption{{{CAPTION}}}", "\\label{tab:exp3-background-traffic}", "\\centering",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{@{{}}l{spec}}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TEX_MODEL[m]}}}" for m in MODELS) + " \\\\",
             "".join(f"\\cmidrule({'l' if i == len(MODELS) - 1 else 'lr'}){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(MODELS))),
             "\\shortstack[l]{Traffic\\\\(Mbit/s)}" + "".join(f" & \\multicolumn{{1}}{{c}}{{{t}}}" for _ in MODELS for _, t in FRAMEWORKS) + " \\\\",
             "\\midrule"]
    for load in LOADS:
        lines.append(" & ".join([str(load)] + [cell(m, f, load) for m in MODELS for f, _ in FRAMEWORKS]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved LaTeX table to: {TEX.name}")


def plot(values: dict) -> None:
    """One panel per model: median latency vs background traffic (line + markers) and IQR (band) per framework."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=FIG_SIZE)
    for ax, m in zip(axes, MODELS):
        top = 0.0
        for f, _, color in PLOT_FRAMEWORKS:
            pts = [(load, values[(m, f, load)]) for load in plot_loads(m, values) if (m, f, load) in values]
            if not pts:
                continue
            x = [load for load, _ in pts]
            ax.fill_between(x, [v["p25"] for _, v in pts], [v["p75"] for _, v in pts], color=color, alpha=BAND_ALPHA,
                            linewidth=0, zorder=2)
            ax.plot(x, [v["median"] for _, v in pts], color=color, linewidth=2.4, zorder=3)
            ax.plot(x, [v["median"] for _, v in pts], linestyle="none", marker="o", markersize=10, color=color, zorder=4)
            top = max(top, max(v["p75"] for _, v in pts))
        style_axes(ax)
        ax.set_ylim(0, top * 1.08)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        last = max(plot_loads(m, values))  # each panel ends just after its last compared load
        ax.set_xlim(-25, last + 25)
        ax.set_xticks(range(0, last + 1, 100))  # regular axis ticks; the markers show the measured loads
        ax.set_title(TEX_MODEL[m])
    axes[0].set_ylabel("Inference latency (ms)")
    fig.subplots_adjust(left=0.07, right=0.988, top=0.8, bottom=0.17, wspace=0.2)
    fig.supxlabel("Background traffic (Mbit/s)", y=0.035, fontsize=plt.rcParams["axes.labelsize"])  # shared
    handles = patches([(lbl, c) for _, lbl, c in PLOT_FRAMEWORKS])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.885), ncol=len(handles), **LEGEND_STYLE)
    save(fig, PDF)


BOX_WIDTH = 0.36
BOX_WHISKERS = (5, 95)  # percentiles; outliers beyond them not drawn


def plot_box(values: dict) -> None:
    """Same data as plot() as box plots: per model and load one ROS 2 | vAccel box pair; box = IQR, line = median,
    whiskers = BOX_WHISKERS percentiles (outliers not drawn); loads as evenly spaced categories."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=FIG_SIZE)
    for ax, m in zip(axes, MODELS):
        loads = plot_loads(m, values)  # categories of this panel (no empty slots)
        for j, (f, _, color) in enumerate(PLOT_FRAMEWORKS):
            pts = [(i, values[(m, f, load)]["latencies"]) for i, load in enumerate(loads) if (m, f, load) in values]
            if not pts:
                continue
            ax.boxplot([lat for _, lat in pts], positions=[i + (j - 0.5) * BOX_WIDTH for i, _ in pts],
                       widths=BOX_WIDTH * 0.9, whis=BOX_WHISKERS, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8),
                       medianprops=dict(color=EDGE_COLOR, linewidth=1.6), whiskerprops=dict(color=EDGE_COLOR),
                       capprops=dict(color=EDGE_COLOR), zorder=3)
        style_axes(ax)
        ax.set_xticks(range(len(loads)))
        ax.set_xticklabels([str(load) for load in loads])
        ax.set_xlim(-0.6, len(loads) - 0.4)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(TEX_MODEL[m])
    axes[0].set_ylabel("Inference latency (ms)")
    fig.subplots_adjust(left=0.07, right=0.988, top=0.8, bottom=0.17, wspace=0.2)
    fig.supxlabel("Background traffic (Mbit/s)", y=0.035, fontsize=plt.rcParams["axes.labelsize"])
    fig.legend(handles=patches([(lbl, c) for _, lbl, c in PLOT_FRAMEWORKS]), loc="lower center",
               bbox_to_anchor=(0.53, 0.885), ncol=len(PLOT_FRAMEWORKS), **LEGEND_STYLE)
    save(fig, BOX_PDF)


if __name__ == "__main__":
    values = build_csv()
    write_tex(values)
    apply_theme()
    plot(values)
    plot_box(values)
