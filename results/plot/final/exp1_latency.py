#!/usr/bin/env python3
"""
exp1_latency.py -- Experiment 1 (vAccSOL): inference latency and equivalent frame rate per model.

Latency = duration of the vAccel inference call measured on the robot: from
sending the input tensor until the output is received (pre-/post-processing
excluded), with inference on the robot CPU, the edge CPU or the edge GPU.

  total_ms          median call duration (inference_ms p50 of the robot run)
  inference_ms      edge: median computation per call, profiled inside the vAccel
                    agent during the same run (common.agent_computation); robot_cpu: = total_ms
  communication_ms  total_ms - inference_ms (network + vAccel remoting; 0 local), from the
                    rounded values so that computation + communication = latency in the tables
  iqr_low/high_ms   25th / 75th percentile of the call duration
  speedup_vs_local  edge targets: Local CPU total_ms / this total_ms
  inference_fps     equivalent frame rate (fps) = 1000 / total_ms, i.e. the inverse of the
                    median latency: the maximum rate of the inference call alone

Latency figure (exp1-latency.pdf): one panel per task, 3 stacked bars per model
(Local CPU / Edge CPU / Edge GPU; Computation solid + Communication hatched), IQR
error bars, and above each edge bar its latency reduction factor vs. Local CPU
("x7.3"; --no-factors: off). Broken linear y axis: the lower part is scaled to
the edge bars, the upper part shows where the 5-70x taller Local CPU bars end.

Equivalent frame rate figure (exp1-fps.pdf): one panel per task, 3 bars per model =
inference_fps, its value inside the top of each edge bar (white) and above each
Local CPU bar (--no-values: off).

LaTeX table (exp1-latency.tex): the medians of exp1-latency.csv, copied as written
there; Local CPU latency, Edge CPU and Edge GPU computation / communication / latency,
each latency followed by its equivalent frame rate in parentheses (as in exp1-fps.pdf).

Usage:    python3 exp1_latency.py [--no-factors] [--no-values]
Produces: exp1-latency.{csv,pdf,tex}, exp1-fps.pdf
"""
import argparse
import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (EDGE_COLOR, HATCH, HATCH_ALPHA, HERE, LEGEND_STYLE, TARGETS, TEX_COLOUR, TITLES,
                    agent_computation, apply_theme, fmt_fps, latency_tex, legend_row, lighten, model_of, model_ticks,
                    part_patches, patches, read_rows, save, style_axes, target_runs, write_rows)

CSV, PDF, FPS_PDF, TEX = (HERE / "exp1-latency.csv", HERE / "exp1-latency.pdf", HERE / "exp1-fps.pdf",
                         HERE / "exp1-latency.tex")
COLUMNS = ["category", "model", "execution_target", "inference_ms", "communication_ms", "total_ms",
           "iqr_low_ms", "iqr_high_ms", "speedup_vs_local", "inference_fps"]

FIG_SIZE = (20, 7.4)
BAR_WIDTH = 0.28
SHOW_FACTORS = True       # latency reduction vs. Local CPU above the edge bars (--no-factors: off)
LOW_HEADROOM = {True: 1.5, False: 1.1}  # lower part top = x highest edge error bar; room for the factor labels
FACTOR_FONTSIZE = 21      # "x7.3" above the edge bars (bold)
TOP_HEADROOM = 1.12       # upper part top = x highest Local CPU error bar
HEIGHT_RATIOS = (1, 2.4)  # upper : lower part of the broken axis

# equivalent frame rate figure
FPS_FIG_SIZE = (20, 6.2)
FPS_BAR_WIDTH = 0.3
FPS_HEADROOM = 1.05       # y top = x highest bar
FPS_FONT_GROWTH = 1.15    # fonts x this vs. the common theme
VALUE_FONTSIZE = 17       # value on each bar
SHOW_VALUES = True        # frame rate values on the bars (--no-values: off)


def build_csv() -> None:
    rows = {}  # (model, target) -> [inference, communication, total, p25, p75]
    for target, _, _ in TARGETS:
        for r in target_runs(target):
            call, model = r["inference_ms"], model_of(r)
            total = float(call["p50"])
            inference = total if target == "robot_cpu" else statistics.median(agent_computation(r) or [0.0])
            if not 0 < inference <= total:
                print(f"[warn] {model} {target}: no valid edge inference ({inference}), skipped")
                continue
            inference, total = round(inference, 1), round(total, 1)  # parts add up to the printed total
            rows[(model, target)] = [inference, total - inference, total, float(call["p25"]), float(call["p75"])]
    local_total = {m: v[2] for (m, t), v in rows.items() if t == "robot_cpu"}
    formatted = {}
    for (model, target), v in rows.items():
        speedup = f"{local_total[model] / v[2]:.2f}" if target != "robot_cpu" and model in local_total else ""
        formatted[(model, target)] = [f"{x:.1f}" for x in v] + [speedup, f"{1000 / v[2]:.1f}"]
    write_rows(CSV, COLUMNS, formatted, [(t,) for t, _, _ in TARGETS])


def draw_bars(ax, models, values, factors: bool) -> None:
    """All bars of one part of a panel; factors: also print the edge bars'
    latency reduction vs. Local CPU (only in the lower part, where the edge bars end)."""
    for j, (target, _, color) in enumerate(TARGETS):
        for i, m in enumerate(models):
            d = values.get((m, target))
            if d is None:
                continue
            x = i + (j - 1) * BAR_WIDTH
            bar = dict(width=BAR_WIDTH, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
            ax.bar(x, d["inference_ms"], facecolor=color, **bar)
            if d["communication_ms"] > 0:
                ax.bar(x, d["communication_ms"], bottom=d["inference_ms"], hatch=HATCH,
                       facecolor=lighten(color, HATCH_ALPHA), **bar)
            ax.errorbar(x, d["total_ms"], yerr=[[d["total_ms"] - d["iqr_low_ms"]], [d["iqr_high_ms"] - d["total_ms"]]],
                        fmt="none", ecolor=EDGE_COLOR, elinewidth=1.6, capsize=5, capthick=1.6, zorder=10)
            if factors and target != "robot_cpu" and d.get("speedup_vs_local"):
                f = d["speedup_vs_local"]
                ax.text(x, d["iqr_high_ms"] + ax.get_ylim()[1] * 0.025, f"\u00d7{f:.0f}" if f >= 10 else f"\u00d7{f:.1f}",
                        rotation=90, ha="center", va="bottom", fontsize=FACTOR_FONTSIZE, fontweight="bold",
                        color=EDGE_COLOR, zorder=11)


def break_marks(ax_top, ax_bot) -> None:
    """'//' marks on both y spines where the axis is broken (same slope in both parts)."""
    d, k = 0.025, HEIGHT_RATIOS[1] / HEIGHT_RATIOS[0]
    kw = dict(color=EDGE_COLOR, clip_on=False, linewidth=1.2)
    for x0 in (0, 1):
        ax_top.plot((x0 - d, x0 + d), (-d * k, d * k), transform=ax_top.transAxes, **kw)
        ax_bot.plot((x0 - d, x0 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)


def plot_latency(show_factors: bool = SHOW_FACTORS) -> None:
    tasks, raw = read_rows(CSV, 3)
    values = {k: {c: float(v) if v else None for c, v in d.items()} for k, d in raw.items()}
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, len(tasks), height_ratios=HEIGHT_RATIOS, width_ratios=[len(m) for m in tasks.values()],
                          hspace=0.07, wspace=0.2, left=0.07, right=0.988, top=0.82, bottom=0.18)

    for col, (task, models) in enumerate(tasks.items()):
        ax_top, ax_bot = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        edge = [values[(m, t)] for m in models for t, _, _ in TARGETS[1:] if (m, t) in values]
        local = [values[(m, "robot_cpu")] for m in models if (m, "robot_cpu") in values]
        low_top = max(d["iqr_high_ms"] for d in edge) * LOW_HEADROOM[show_factors]
        lo, hi = (max(low_top, min(d["total_ms"] for d in local) * 0.6),
                  max(d["iqr_high_ms"] for d in local) * TOP_HEADROOM)
        ax_bot.set_ylim(0, low_top)
        ax_top.set_ylim(lo, hi)
        for ax in (ax_top, ax_bot):
            draw_bars(ax, models, values, factors=show_factors and ax is ax_bot)
            style_axes(ax)
            ax.set_xlim(-0.5, len(models) - 0.5)
        # no tick within 8% of the break (it would collide with the break marks)
        ax_top.set_yticks([t for t in MaxNLocator(nbins=2).tick_values(lo, hi) if lo + 0.08 * (hi - lo) <= t <= hi])
        ax_bot.set_yticks([t for t in MaxNLocator(nbins=5).tick_values(0, low_top) if 0 <= t <= 0.92 * low_top])
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.set_xticks(range(len(models)))
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        model_ticks(ax_bot, models)
        ax_top.set_title(TITLES.get(task, task))
        break_marks(ax_top, ax_bot)

    fig.text(0.008, 0.5, "Inference latency (ms)", rotation=90, ha="center", va="center",
             fontsize=plt.rcParams["axes.labelsize"])
    legend_row(fig, [("Execution", patches([(lbl, c) for _, lbl, c in TARGETS])),
                     ("Latency contribution", part_patches("Computation", "Communication"))], y=0.875)
    save(fig, PDF)


# LaTeX table
# (execution target, header, colour name, colour, CSV columns left to right; the last is the latency,
# followed in the table by its equivalent frame rate in parentheses)
TEX_GROUPS = [(t, title, *TEX_COLOUR[t], cols) for t, title, cols in
              [("robot_cpu", "Local CPU", ["total_ms"]),
               ("edge_cpu", "Edge CPU", ["inference_ms", "communication_ms", "total_ms"]),
               ("edge_gpu", "Edge GPU", ["inference_ms", "communication_ms", "total_ms"])]]
TEX_CAPTION = ("Exp.~1: Inference latency (ms) measured on the robot for models executed via vAccSOL. We compare "
               "local inference on the robot CPU (blue) with remote inference on the edge CPU (orange) and GPU "
               "(green). Remote latency is divided into computation (Comp.) and communication (Comm.), the latter "
               "covering data transfer and vAccel processing. Values are medians, with the equivalent frame rate "
               "in parentheses.")


def write_tex() -> None:
    latency_tex(CSV, TEX, TEX_GROUPS, TEX_CAPTION, "tab:exp1-latency", "exp1_latency.py")


def plot_fps(show_values: bool = SHOW_VALUES) -> None:
    tasks, raw = read_rows(CSV, 3)
    fps = {k: float(d["inference_fps"]) for k, d in raw.items()}
    sizes = ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
    with plt.rc_context({k: plt.rcParams[k] * FPS_FONT_GROWTH for k in sizes}):
        fig, axes = plt.subplots(1, len(tasks), figsize=FPS_FIG_SIZE,
                                 gridspec_kw={"width_ratios": [len(m) for m in tasks.values()]})
        for ax, (task, models) in zip(axes, tasks.items()):
            top = max(v for k, v in fps.items() if k[0] in models) * FPS_HEADROOM
            ax.set_ylim(0, top)
            for j, (target, _, color) in enumerate(TARGETS):
                for i, m in enumerate(models):
                    if (m, target) not in fps:
                        continue
                    x, v, local = i + (j - 1) * FPS_BAR_WIDTH, fps[(m, target)], target == "robot_cpu"
                    ax.bar(x, v, width=FPS_BAR_WIDTH, facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)
                    if show_values:  # Local CPU bars are too short for the value: above them
                        ax.text(x, v + top * 0.012 if local else v - top * 0.015, fmt_fps(v), ha="center",
                                va="bottom" if local else "top", fontsize=VALUE_FONTSIZE,
                                color=EDGE_COLOR if local else "white", zorder=11)
            style_axes(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlim(-0.5, len(models) - 0.5)
            model_ticks(ax, models)
            ax.set_title(TITLES.get(task, task))
        axes[0].set_ylabel("Frame rate (FPS)")
        fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.2, wspace=0.17)
        fig.legend(handles=patches([(lbl, c) for _, lbl, c in TARGETS]), loc="lower center",
                   bbox_to_anchor=(0.53, 0.895), ncol=len(TARGETS), **LEGEND_STYLE)
        save(fig, FPS_PDF)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 1 inference latency and equivalent frame rate: CSV + plots.")
    ap.add_argument("--no-factors", action="store_true", help="omit the latency-reduction labels on the edge bars")
    ap.add_argument("--no-values", action="store_true", help="omit the frame rate values on the bars")
    args = ap.parse_args()
    build_csv()
    write_tex()
    apply_theme()
    plot_latency(show_factors=not args.no_factors)
    plot_fps(show_values=not args.no_values)
