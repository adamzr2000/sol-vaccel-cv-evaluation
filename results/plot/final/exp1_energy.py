#!/usr/bin/env python3
"""
exp1_energy.py -- Experiment 1 (vAccSOL): robot and edge energy per inference per
model, for each execution target (Local CPU / Edge CPU / Edge GPU).

Energies are exact hardware-counter totals (RAPL CPU package, NVML GPU board;
idle power included) over the measured inference loop -- the monitors' extra
~1.5 s before/after it removed via the per-second power samples (see
common.device_energy) -- divided by the number of inferences of the run.

CSV (exp1-energy.csv) -- everything recorded, per inference, for all cases:
  robot_cpu_j, edge_cpu_j, edge_gpu_j   measured energy of each device
  *_idle_j                              estimated idle share of it = idle power x
                                        the device's run duration / inferences
  duration_s (measured loop), num_inferences
  robot_reduction_pct                   edge targets: reduction of the robot's energy per
                                        inference vs. Local CPU, in % (negative = increase)
Devices not recorded in a run are left empty (the edge in local runs, the
edge GPU in Edge CPU runs).

Figure (exp1-energy.pdf): one panel per task, 3 bars per model (Local CPU / Edge CPU / Edge GPU) =
energy per inference, stacked: the robot's part solid in the target's colour, the edge's part
(edge CPU, + edge GPU for the edge GPU) hatched on top; local inference has no edge part.

LaTeX table (exp1-energy.tex): energy per inference (J, 2 decimals) from exp1-energy.csv --
Local CPU: robot; Edge CPU / Edge GPU: robot, edge (edge CPU, + edge GPU for Edge GPU) and
total = robot + edge, all from the printed values.

Usage:    python3 exp1_energy.py
Produces: exp1-energy.{csv,pdf,tex}
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (DEVICES, EDGE_COLOR, HATCH, HATCH_ALPHA, HERE, legend_row, TARGETS, TEX_COLOUR, TEX_INDENT, TEX_MODEL, TEX_TASK, TITLES,
                    apply_theme, device_energy, idle_power, lighten, model_of, model_ticks, part_patches, patches, read_rows, save, style_axes,
                    target_runs, write_rows)

CSV, PDF, TEX = HERE / "exp1-energy.csv", HERE / "exp1-energy.pdf", HERE / "exp1-energy.tex"
ENERGY_COLUMNS = ([f"{d}_j" for d in DEVICES] + [f"{d}_idle_j" for d in DEVICES]
                  + ["duration_s", "num_inferences"])

FIG_SIZE = (20, 6.8)
BAR_WIDTH = 0.28
HEADROOM = 1.08


def energy_values(run: dict, idle: dict):
    """ENERGY_COLUMNS of one run ('' = device not recorded), or None without robot energy."""
    n = int(run.get("num_samples") or 0)
    measured = {dev: device_energy(run, dev) for dev in DEVICES}  # over the measured loop
    if n <= 0 or measured["robot_cpu"] is None:
        print(f"[warn] {run.get('run_id')}: no robot energy or sample count, skipped")
        return None
    return ([f"{measured[d][0] / n:.4f}" if measured[d] else "" for d in DEVICES]
            + [f"{idle[d] * measured[d][1] / n:.4f}" if measured[d] else "" for d in DEVICES]
            + [f"{measured['robot_cpu'][1]:.1f}", n])


def build_csv() -> None:
    idle = idle_power()
    print("[idle] " + ", ".join(f"{d} {w:.2f} W" for d, w in idle.items()))
    rows = {}
    for target, _, _ in TARGETS:
        for r in target_runs(target):
            if (v := energy_values(r, idle)) is not None:
                rows[(model_of(r), target)] = v
    for (model, target), v in rows.items():
        local = rows.get((model, "robot_cpu"))
        robot, local_robot = float(v[0]), float(local[0]) if local else None
        v.append(f"{(1 - robot / local_robot) * 100:.1f}" if target != "robot_cpu" and local_robot else "")
    write_rows(CSV, ["category", "model", "execution_target"] + ENERGY_COLUMNS + ["robot_reduction_pct"], rows,
               [(t,) for t, _, _ in TARGETS])


TEX_CAPTION = ("Exp.~1: Energy consumed per inference (J) by the robot and the edge during local and remote "
               "inference.")


def tex_energies(raw: dict) -> dict:
    """{(model, target): (robot, edge, total)} in J rounded to 2 decimals, total = robot + edge as printed
    (edge None for Local CPU)."""
    out = {}
    for (m, t), d in raw.items():
        robot = round(float(d["robot_cpu_j"]), 2)
        edge = None if t == "robot_cpu" else round(float(d["edge_cpu_j"]) + float(d["edge_gpu_j"] or 0), 2)
        out[(m, t)] = (robot, edge, robot + (edge or 0))
    return out


def write_tex() -> None:
    tasks, raw = read_rows(CSV, 3)
    e = tex_energies(raw)
    width = lambda vals: max(len(f"{v:.2f}".split(".")[0]) for v in vals)  # siunitx integer digits
    groups = [("robot_cpu", "Local CPU"), ("edge_cpu", "Edge CPU"), ("edge_gpu", "Edge GPU")]
    spec = (f">{{\\columncolor{{{TEX_COLOUR['robot_cpu'][0]}}}}}"
            f"S[table-format={width(v[0] for k, v in e.items() if k[1] == 'robot_cpu')}.2]")
    for t, _ in groups[1:]:
        vals = [v for k, v in e.items() if k[1] == t]
        last = "[\\tabcolsep][0pt]" if t == groups[-1][0] else ""  # shading stops at the table's end
        spec += (f"S[table-format={width(v[0] for v in vals)}.2]S[table-format={width(v[1] for v in vals)}.2]"
                 f">{{\\columncolor{{{TEX_COLOUR[t][0]}}}{last}}}S[table-format={width(v[2] for v in vals)}.2]")
    lines = ["% Requires: \\usepackage[table]{xcolor}, booktabs, array, siunitx",
             "% Generated by exp1_energy.py from exp1-energy.csv -- do not edit by hand."]
    lines += [f"\\definecolor{{{n}}}{{HTML}}{{{rgb}}}" for n, rgb in TEX_COLOUR.values()]
    lines += ["\\begin{table*}[!t]", f"\\caption{{{TEX_CAPTION}}}", "\\label{tab:exp1-energy}", "\\centering",
              "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}", f"\\begin{{tabular}}{{@{{}}l{spec}@{{}}}}",
              "\\toprule",
              " & \\multicolumn{1}{c}{Local CPU} & \\multicolumn{3}{c}{Edge CPU} & \\multicolumn{3}{c}{Edge GPU} \\\\",
              "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
              "Model & {Robot} & {Robot} & {Edge} & {Total} & {Robot} & {Edge} & {Total} \\\\"]
    for task, models in tasks.items():
        lines += ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{TEX_TASK[task]}}}}} \\\\"]
        for m in models:
            local = e.get((m, "robot_cpu"))
            cells = [TEX_INDENT + TEX_MODEL[m], f"{local[0]:.2f}" if local else "{--}"]
            for t, _ in groups[1:]:
                cells += [f"{x:.2f}" for x in e[(m, t)]] if (m, t) in e else ["{--}"] * 3
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved LaTeX table to: {TEX.name}")


def plot() -> None:
    """One panel per task, 3 bars per model (Local CPU / Edge CPU / Edge GPU): energy per inference, the robot's
    part solid in the target's colour, the edge's part (edge CPU, + edge GPU for the edge GPU) hatched on top."""
    tasks, raw = read_rows(CSV, 3)
    robot = {k: float(d["robot_cpu_j"]) for k, d in raw.items()}
    edge = {k: float(d["edge_cpu_j"] or 0) + float(d["edge_gpu_j"] or 0) for k, d in raw.items()}
    fig, axes = plt.subplots(1, len(tasks), figsize=FIG_SIZE, gridspec_kw={"width_ratios": [len(m) for m in tasks.values()]})
    for ax, (task, models) in zip(axes, tasks.items()):
        for j, (target, _, color) in enumerate(TARGETS):
            for i, m in enumerate(models):
                if (m, target) not in robot:
                    continue
                x, bar = i + (j - 1) * BAR_WIDTH, dict(width=BAR_WIDTH, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
                ax.bar(x, robot[(m, target)], facecolor=color, **bar)
                if edge[(m, target)] > 0:
                    ax.bar(x, edge[(m, target)], bottom=robot[(m, target)], hatch=HATCH,
                           facecolor=lighten(color, HATCH_ALPHA), **bar)
        style_axes(ax)
        ax.set_ylim(0, max(robot[k] + edge[k] for k in robot if k[0] in models) * HEADROOM)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.5, len(models) - 0.5)
        model_ticks(ax, models)
        ax.set_title(TITLES.get(task, task))
    axes[0].set_ylabel("Energy per inference (J)")
    fig.subplots_adjust(left=0.07, right=0.988, top=0.78, bottom=0.19, wspace=0.21)
    # two titled legends side by side, as in exp1-latency.pdf
    legend_row(fig, [("Execution", patches([(lbl, c) for _, lbl, c in TARGETS])),
                     ("Energy contribution", part_patches("Robot", "Edge"))], y=0.85)
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    write_tex()
    apply_theme()
    plot()
