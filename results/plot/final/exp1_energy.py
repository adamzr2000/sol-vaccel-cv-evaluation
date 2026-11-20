#!/usr/bin/env python3
"""
exp1_energy.py -- Experiment 1 (vAccSOL): robot energy per inference per
model, for each execution target (Local CPU / Edge CPU / Edge GPU).

Energies are exact hardware-counter totals per run (RAPL CPU package, NVML GPU
board; idle power included), divided by the number of inferences of the run.

CSV (exp1-energy.csv) -- everything recorded, per inference, for all cases:
  robot_cpu_j, edge_cpu_j, edge_gpu_j   measured energy of each device
  *_idle_j                              estimated idle share of it = idle power x
                                        the device's run duration / inferences
  duration_s, num_inferences
Devices not recorded in a run are left empty (the edge in local runs, the
edge GPU in Edge CPU runs).

Figure (exp1-energy.pdf): one panel per task, 3 bars per model = robot CPU
energy per inference (measured), coloured by execution target.

Usage:    python3 exp1_energy.py
Produces: exp1-energy.{csv,pdf}
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (DEVICES, EDGE_COLOR, HERE, LEGEND_STYLE, TARGETS, TITLES, apply_theme, device_energy, idle_power,
                    model_of, model_ticks, patches, read_rows, save, style_axes, target_runs, write_rows)

CSV, PDF = HERE / "exp1-energy.csv", HERE / "exp1-energy.pdf"
ENERGY_COLUMNS = ([f"{d}_j" for d in DEVICES] + [f"{d}_idle_j" for d in DEVICES]
                  + ["duration_s", "num_inferences"])

FIG_SIZE = (20, 6.8)
BAR_WIDTH = 0.28
HEADROOM = 1.08


def energy_values(run: dict, idle: dict):
    """ENERGY_COLUMNS of one run ('' = device not recorded), or None without robot energy."""
    n = int(run.get("num_samples") or 0)
    measured = {dev: device_energy(run["run_id"], dev) for dev in DEVICES}
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
    write_rows(CSV, ["category", "model", "execution_target"] + ENERGY_COLUMNS, rows, [(t,) for t, _, _ in TARGETS])


def plot() -> None:
    tasks, raw = read_rows(CSV, 3)
    robot = {k: float(d["robot_cpu_j"]) for k, d in raw.items()}
    fig, axes = plt.subplots(1, len(tasks), figsize=FIG_SIZE, gridspec_kw={"width_ratios": [len(m) for m in tasks.values()]})
    for ax, (task, models) in zip(axes, tasks.items()):
        for j, (target, _, color) in enumerate(TARGETS):
            for i, m in enumerate(models):
                if (m, target) in robot:
                    ax.bar(i + (j - 1) * BAR_WIDTH, robot[(m, target)], width=BAR_WIDTH, facecolor=color,
                           edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
        style_axes(ax)
        ax.set_ylim(0, max(v for k, v in robot.items() if k[0] in models) * HEADROOM)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-0.5, len(models) - 0.5)
        model_ticks(ax, models)
        ax.set_title(TITLES.get(task, task))
    axes[0].set_ylabel("Energy per inference (J)\nRobot")
    fig.subplots_adjust(left=0.07, right=0.988, top=0.83, bottom=0.19, wspace=0.21)
    fig.legend(handles=patches([(lbl, c) for _, lbl, c in TARGETS]), loc="lower center", bbox_to_anchor=(0.53, 0.89),
               ncol=len(TARGETS), **LEGEND_STYLE)
    save(fig, PDF)


if __name__ == "__main__":
    build_csv()
    apply_theme()
    plot()
