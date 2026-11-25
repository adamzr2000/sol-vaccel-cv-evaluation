#!/usr/bin/env python3
"""
exp3_latency_energy.py -- Experiment 3: ROS 2 vs vAccel for remote inference on the edge GPU (SOL on both):
one single-column LaTeX table with the inference latency and the energy per inference (robot + edge)
per model, ROS 2 | vAccel side by side.

Latency (ms)  Comm. and Total from exp3-latency.csv (run exp3_latency.py first), as printed there.
Layout: per quantity ROS 2 | vAccel side by side; the lower total (latency, robot + edge energy) of
each pair shaded in one colour (as the faster compiler in exp2-latency.tex); ties not shaded.
Energy (J)    per inference; Robot = robot CPU, Robot + Edge = robot + edge CPU + edge GPU, 2 decimals,
              total from the printed values (as exp1_energy.tex_energies):
  vAccel  the Edge GPU row of exp1-energy.csv (run exp1_energy.py first)
  ROS 2   same method as common.device_energy: hardware counter totals (energy_totals JSON: RAPL robot
          CPU, RAPL edge CPU, NVML edge GPU) of the ROS 2 run exp3_latency.py uses, minus the energy outside
          the measured window (per-second power samples), divided by the number of results; window = first
          input published .. last result received over the monitor's records.
Written to exp3-energy.csv (J per inference, unrounded, + number of results and window).

Usage:    python3 exp3_latency_energy.py
Produces: exp3-energy.csv, exp3-latency-energy.tex
"""
import csv
import json
from pathlib import Path

from common import (HERE, ROS_MODEL_STATS, ROS_RUN_TAG, TEX_COLOUR, TEX_INDENT, TEX_MODEL, TEX_TASK, _energy_totals,
                    _integrate, _iso_s, read_rows, write_rows)
from exp3_latency import latest

LAT_CSV, ENERGY_CSV = HERE / "exp3-latency.csv", HERE / "exp1-energy.csv"
OUT_CSV, TEX = HERE / "exp3-energy.csv", HERE / "exp3-latency-energy.tex"
ROS_SYSTEM_STATS = ROS_MODEL_STATS.parent / "system-stats"
FRAMEWORKS = [("ros2", "ROS 2"), ("vaccel", "vAccel")]
# the lower total of each ROS 2 / vAccel pair: one highlight, the light green of Exp. 2 (faster compiler)
TEX_BETTER = TEX_COLOUR["edge_gpu"]
CAPTION = ("Exp.~3: Inference latency (ms) and energy per inference (J) for remote inference of SOL-optimized "
           "models on the edge GPU via ROS~2 and vAccel. The better of the two is shaded.")


def _ros_samples(host: str, kind: str, run: str) -> list:
    """[(sample time s, W)] of a ROS 2 run's per-second monitor file ({kind}_{run}.csv)."""
    col, power = ("cpu_watts" if kind == "cpu" else "power_draw_w"), {}
    with open(ROS_SYSTEM_STATS / host / f"{kind}_{run}.csv", newline="") as f:
        for r in csv.DictReader(f):
            t = int(r["timestamp"]) / 1000.0
            power[t] = power.get(t, 0.0) + float(r[col])
    return sorted(power.items())


def _ros_device(host: str, kind: str, run: str, a: float, b: float) -> float:
    """Counter energy (J) of a ROS 2 run's device within [a, b] (common.device_energy's method)."""
    path = ROS_SYSTEM_STATS / host / f"energy_totals_{run}.json"
    total, _ = _energy_totals(path, kind)
    meta = json.loads(path.read_text())
    s0, s1 = _iso_s(meta["run_started_iso"]), _iso_s(meta["run_stopped_iso"])
    samples = _ros_samples(host, kind, run)
    return max(total - _integrate(samples, s0, s1, s0, a) - _integrate(samples, s0, s1, b, s1), 0.0)


def ros2_energy(model: str):
    """(robot J, edge J, results, window s) per inference of the ROS 2 run of `model`, or None."""
    mon = latest(f"monitor_{model}_sol_{ROS_RUN_TAG}_*.json", ROS_MODEL_STATS / "robot")
    if mon is None:
        return None
    rec, run = json.loads(mon.read_text())["records"], mon.stem[len("monitor_"):]
    a = min(r["t_image_on_bus_ros_at_ns"] for r in rec) / 1e9
    b = max(r["t_result_received_ros_at_ns"] for r in rec) / 1e9
    try:
        robot = _ros_device("robot", "cpu", run, a, b)
        edge = _ros_device("edge-asus", "cpu", run, a, b) + _ros_device("edge-asus", "gpu", run, a, b)
    except (FileNotFoundError, TypeError):
        print(f"[warn] {run}: no energy counters/power samples, skipped")
        return None
    return robot / len(rec), edge / len(rec), len(rec), b - a


def build_csv() -> dict:
    """{(model, framework): (robot J, edge J)} per inference; also written to exp3-energy.csv."""
    tasks, _ = read_rows(LAT_CSV, 3)
    _, vac = read_rows(ENERGY_CSV, 3)
    rows, energy = {}, {}
    for models in tasks.values():
        for m in models:
            if (r := ros2_energy(m)) is not None:
                energy[(m, "ros2")] = r[:2]
                rows[(m, "ros2")] = [f"{r[0]:.4f}", f"{r[1]:.4f}", r[2], f"{r[3]:.1f}"]
            if (d := vac.get((m, "edge_gpu"))) is not None:
                e = (float(d["robot_cpu_j"]), float(d["edge_cpu_j"]) + float(d["edge_gpu_j"] or 0))
                energy[(m, "vaccel")] = e
                rows[(m, "vaccel")] = [f"{e[0]:.4f}", f"{e[1]:.4f}", d["num_inferences"], d["duration_s"]]
    write_rows(OUT_CSV, ["category", "model", "framework", "robot_j", "edge_j", "num_inferences", "window_s"],
               rows, [(f,) for f, _ in FRAMEWORKS])
    return energy


def write_tex(energy: dict) -> None:
    tasks, lat = read_rows(LAT_CSV, 3)

    def cells(m, f):
        """(comm, total latency, robot J, robot + edge J) as printed; None where missing."""
        d, e = lat.get((m, f)), energy.get((m, f))
        robot, edge = (round(e[0], 2), round(e[1], 2)) if e else (None, None)
        return (d["communication_ms"] if d else None, d["total_ms"] if d else None,
                f"{robot:.2f}" if e else None, f"{robot + edge:.2f}" if e else None)

    v = {(m, f): cells(m, f) for ms in tasks.values() for m in ms for f, _ in FRAMEWORKS}
    # single column: the two totals only (inference latency; robot + edge energy), ROS 2 | vAccel side by side.
    # Comm. and the robot-only energy are in exp3-latency.csv / exp3-energy.csv (for the text): computation is
    # the same SOL model on the same GPU, and the robot energy is the same with both middlewares.
    quantities = [(1, "\\shortstack{Inference\\\\latency (ms)}", True),
                  (3, "\\shortstack{Energy per\\\\inference (J)}", True)]

    def cell(m, f, i, compare):
        """Printed value; for the totals, shaded when lower than the other framework's (a tie is not)."""
        x = v[(m, f)][i]
        if x is None:
            return "--"
        other = v[(m, "vaccel" if f == "ros2" else "ros2")][i]
        better = compare and other is not None and float(x) < float(other)
        return f"\\cellcolor{{{TEX_BETTER[0]}}}{x}" if better else x

    # four equal-width centred columns: within each metric every value has the same number of digits
    # (latency xx.x, energy x.xx), so centring keeps the decimal points aligned; equal widths, at least as
    # wide as the two-line group headers, keep the ROS 2 / vAccel columns symmetric; \cellcolor is reliable
    # here, unlike inside siunitx S columns
    spec = ">{\\centering\\arraybackslash}p{3.3em}" * (2 * len(quantities))
    for i, h, _ in quantities:
        if len({len(x[i]) for x in v.values() if x[i] is not None}) > 1:  # e.g. a latency >= 100 ms after a rerun
            print(f"[warn] {h}: values with different digit counts -- centred columns no longer align the decimals")
    lines = ["% Requires: \\usepackage[table]{xcolor}, booktabs, array",
             "% Generated by exp3_latency_energy.py from exp3-latency.csv and exp3-energy.csv -- do not edit by hand.",
             f"\\definecolor{{{TEX_BETTER[0]}}}{{HTML}}{{{TEX_BETTER[1]}}}"]
    lines += ["\\begin{table}[!t]", f"\\caption{{{CAPTION}}}", "\\label{tab:exp3-latency-energy}", "\\centering",
              "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}", f"\\begin{{tabular}}{{@{{}}l{spec}}}", "\\toprule",
              "".join(f" & \\multicolumn{{2}}{{c}}{{{h}}}" for _, h, _ in quantities) + " \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
              "Model" + "".join(f" & \\multicolumn{{1}}{{c}}{{{t}}}" for _ in quantities for _, t in FRAMEWORKS) + " \\\\"]
    for task, models in tasks.items():
        lines += ["\\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{TEX_TASK[task]}}}}} \\\\"]
        for m in models:
            row = [TEX_INDENT + TEX_MODEL[m]] + [cell(m, f, i, shaded) for i, _, shaded in quantities for f, _ in FRAMEWORKS]
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved LaTeX table to: {TEX.name}")


if __name__ == "__main__":
    write_tex(build_csv())
