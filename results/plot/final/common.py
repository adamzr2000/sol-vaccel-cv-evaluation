"""
common.py -- shared settings and helpers of the final paper figures (Exp. 1, Exp. 2).

Everything the figure scripts in this folder need lives here: where the
measurements are, which runs feed which execution target, the model/task
order, how runs are read, and the figure style. The scripts write their CSV
and PDF next to themselves (this folder).

All data is read straight from the per-run files written during the
experiments -- no aggregation/summarize script has to be run first:
  model-stats/{robot,edge-asus}/{tag}_*/benchmark_summary.json   run statistics
  system-stats/{robot,edge-asus}/energy_totals_{run_id}.json      energy counters
  system-stats/robot/{run_id}_net.csv                             network counters
  system-stats/idle/energy_totals_{device}-idle.json              idle runs
  agent-profile/edge-asus/{run_id}.csv                            edge computation per call (vAccel agent)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------ data paths

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1] / "experiments"            # results/experiments
MODEL_STATS = EXPERIMENTS / "model-stats"
SYSTEM_STATS = EXPERIMENTS / "system-stats"
IDLE_DIR = SYSTEM_STATS / "idle"                          # energy_totals_{device}-idle.json
AGENT_PROFILE = EXPERIMENTS / "agent-profile" / "edge-asus"  # {run_id}.csv: sample,computation_ms

# ------------------------------------------------------------------ run tags

LOCAL_RUN_TAG = "iso"               # local inference on the robot (Local CPU)
REMOTE_RUN_TAG = "e2e"              # offloaded inference, edge GPU
REMOTE_CPU_RUN_TAG = "e2e"          # offloaded inference, edge CPU
EDGE_INFERENCE_RUN_TAG = "iso"      # isolated inference on the edge (pure computation time)
EDGE_HOST = "edge-asus"

# ROS 2 (Exp. 3): ros-vision-service next to this repository; CycloneDDS (UDP), SOL on the edge GPU
ROS_MODEL_STATS = HERE.parents[3] / "ros-vision-service" / "results" / "experiments" / "model-stats"
ROS_RUN_TAG = "remote-gpu"   # RELIABLE + KEEP_LAST 1 QoS on all topics (2026-11-23)

# ------------------------------------------------------------------ tasks and models

# (task key, title, models in paper order: CNNs before transformers, small to large)
TASKS = [
    ("image_classification", "Image Classification", ["resnet50", "swin_t", "swin_s", "swin_v2_b"]),
    ("video_action_recognition", "Video Action Recognition", ["r3d_18", "r2plus1d_18", "swin3d_s", "swin3d_b"]),
    ("semantic_segmentation", "Semantic Segmentation",
     ["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101"]),
]
TITLES = {key: title for key, title, _ in TASKS}
MODEL_NAME = {
    "resnet50": "ResNet-50", "swin_t": "Swin-T", "swin_s": "Swin-S", "swin_v2_b": "SwinV2-B",
    "r3d_18": "R3D-18", "r2plus1d_18": "R(2+1)D-18", "swin3d_s": "Swin3D-S", "swin3d_b": "Swin3D-B",
    "fcn_resnet50": "FCN-R50", "fcn_resnet101": "FCN-R101",
    "deeplabv3_resnet50": "DLv3-R50", "deeplabv3_resnet101": "DLv3-R101",
}
# LaTeX tables: model rows indented under their italic task row
TEX_INDENT = "\\hspace{1em}"
# names in the LaTeX tables (more room than on the figure axes)
TEX_TASK = {"image_classification": "Image classification", "video_action_recognition": "Video action recognition",
            "semantic_segmentation": "Semantic segmentation"}
TEX_MODEL = {"resnet50": "ResNet-50", "swin_t": "Swin-T", "swin_s": "Swin-S", "swin_v2_b": "SwinV2-B",
             "r3d_18": "R3D-18", "r2plus1d_18": "R(2+1)D-18", "swin3d_s": "Swin3D-S", "swin3d_b": "Swin3D-B",
             "fcn_resnet50": "FCN-ResNet-50", "fcn_resnet101": "FCN-ResNet-101",
             "deeplabv3_resnet50": "DeepLabV3-ResNet-50", "deeplabv3_resnet101": "DeepLabV3-ResNet-101"}

# ------------------------------------------------------------------ execution targets

# (execution target, legend label, colour shared by all paper figures)
TARGETS = [
    ("robot_cpu", "Local CPU", "#D5E4F3"),
    ("edge_cpu", "Edge CPU", "#E0892B"),
    ("edge_gpu", "Edge GPU", "#2E9B7E"),
]
# LaTeX tables: (colour name, light tint of the target colour) for the shaded columns of each target
TEX_COLOUR = {"robot_cpu": ("vsLocal", "E3EDF8"), "edge_cpu": ("vsCPULight", "FBE1B8"),
              "edge_gpu": ("vsGPULight", "D3EEE6")}
TARGET_TAG = {"robot_cpu": LOCAL_RUN_TAG, "edge_cpu": REMOTE_CPU_RUN_TAG, "edge_gpu": REMOTE_RUN_TAG}

# (compiler, legend label, backend suffix, colour) -- Exp. 2
COMPILERS = [
    ("aoti", "vAccel + PyTorch", "ptc", "#BDBDBD"),  # PyTorch AOTInductor .pt2 model, vAccel Torch plugin
    ("sol", "vAccSOL", "sol", "#3A7CA5"),         # SOL-compiled library, vAccel Exec operation
]

# ------------------------------------------------------------------ runs


def load_runs(tag: str) -> list:
    """All runs of `tag`: the per-run summaries model-stats/{robot,EDGE_HOST}/{tag}_*/
    benchmark_summary.json, written by the benchmark itself at the end of each run."""
    runs = []
    for host in ("robot", EDGE_HOST):
        for p in sorted((MODEL_STATS / host).glob(f"{tag}_*/benchmark_summary.json")):
            run = json.loads(p.read_text())
            if run.get("run_id") != p.parent.name:  # the folder name is the run id
                print(f"[warn] {p}: run_id {run.get('run_id')!r} does not match its folder, skipped")
                continue
            runs.append(run)
    return runs


def target_runs(target: str, compiler: str = "sol", tag: str | None = None) -> list:
    """vAccel runs on the robot of one execution target and compiler (from `tag`,
    default the target's run tag)."""
    backend = f"vaccel-{'local' if target == 'robot_cpu' else 'remote'}-{dict((c, b) for c, _, b, _ in COMPILERS)[compiler]}"
    device = "gpu" if target == "edge_gpu" else "cpu"
    tag = tag or TARGET_TAG[target]
    runs = [r for r in load_runs(tag)
            if str(r.get("host", "")).lower() == "robot" and str(r.get("backend", "")).lower() == backend
            and str(r.get("device", "")).lower() == device
            and (target == "robot_cpu" or str(r.get("remote_host", "")).lower() == EDGE_HOST)]
    print(f"[source] {target:<9} {compiler:<4} <- {tag} ({len(runs)} runs)")
    return runs


def edge_runs() -> list:
    """Isolated inference runs on the edge host (pure computation time)."""
    return [r for r in load_runs(EDGE_INFERENCE_RUN_TAG) if str(r.get("host", "")).lower() == EDGE_HOST]


def agent_computation(run: dict) -> list:
    """Edge computation time (ms) of each measured call of a remote run, profiled
    inside the vAccel agent during that run; [] if not recorded."""
    path = AGENT_PROFILE / f"{run['run_id']}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [float(r["computation_ms"]) for r in csv.DictReader(f)]


def model_of(run: dict) -> str:
    return str(run.get("model", "")).strip()


def write_rows(path: Path, header: list, rows: dict, keys: list) -> None:
    """CSV with header + one line per key (model, *rest) in task/model order:
    [task, model, *rest, *rows[key]] for every key present in rows."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for task, _, models in TASKS:
            for m in models:
                for rest in keys:
                    if (m, *rest) in rows:
                        w.writerow([task, m, *rest, *rows[(m, *rest)]])
    print(f"[OK] Saved table to: {path.name}")


def read_rows(path: Path, n_keys: int):
    """-> ({task: [models]} in file order, {(model, *n_keys - 2 more key columns): {column: value}})"""
    tasks, values = {}, {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        key_cols = reader.fieldnames[:n_keys]
        for row in reader:
            models = tasks.setdefault(row[key_cols[0]], [])
            if row[key_cols[1]] not in models:
                models.append(row[key_cols[1]])
            values[tuple(row[c] for c in key_cols[1:])] = {k: v for k, v in row.items() if k not in key_cols}
    return tasks, values


# ------------------------------------------------------------------ LaTeX latency tables

TEX_LATENCY_HEADER = {"inference_ms": "Comp.", "communication_ms": "Comm.", "total_ms": "Latency"}


def fmt_fps(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def latency_tex(csv_path: Path, tex_path: Path, groups: list, caption: str, label: str, script: str) -> None:
    """Latency table (Exp. 1 / Exp. 3 layout): per group (key, title, colour name, colour, CSV columns),
    the columns Comp. / Comm. / Latency as written in csv_path (no re-rounding), the latency shaded
    and followed by its equivalent frame rate (1000 / printed latency, "(33 fps)")."""
    tasks, raw = read_rows(csv_path, 3)
    n_cols = 1 + sum(len(cols) + 1 for *_, cols in groups)
    spec, top, rules, sub, col = "", [], [], [], 2
    for target, title, cname, _, cols in groups:
        for c in cols:
            # siunitx S column width: integer digits of the widest value, 1 decimal
            digits = max(len(raw[(m, target)][c].split(".")[0]) for m in TEX_MODEL if (m, target) in raw)
            if c == "total_ms":  # shaded latency + its frame rate, set close to it in small grey text
                # fps colour: no left overhang (it would paint over the latency's last digit), none on the
                # right at the table's end either
                right = "0pt" if target == groups[-1][0] else "\\tabcolsep"
                spec += (f">{{\\columncolor{{{cname}}}}}S[table-format={digits}.1]@{{\\,}}"
                         f">{{\\columncolor{{{cname}}}[0pt][{right}]\\scriptsize\\color{{gray}}}}l")
            else:
                spec += f"S[table-format={digits}.1]"
            sub.append(f"{{{TEX_LATENCY_HEADER[c]}}}")
        sub.append("")  # frame rate column: the unit is in each cell ("(33 fps)")
        width = len(cols) + 1
        top.append(f"\\multicolumn{{{width}}}{{c}}{{{title}}}")
        rules.append(f"\\cmidrule({'lr' if target != groups[-1][0] else 'l'}){{{col}-{col + width - 1}}}")
        col += width
    lines = ["% Requires: \\usepackage[table]{xcolor}, booktabs, array, siunitx",
             f"% Generated by {script} from {csv_path.name} -- do not edit by hand."]
    lines += [f"\\definecolor{{{cname}}}{{HTML}}{{{rgb}}}" for _, _, cname, rgb, _ in groups]
    lines += ["\\begin{table*}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering",
              "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}", f"\\begin{{tabular}}{{@{{}}l{spec}@{{}}}}",
              "\\toprule", " & ".join(["", *top]) + " \\\\", "".join(rules),
              " & ".join(["Model", *sub]) + " \\\\"]
    for task, models in tasks.items():
        lines += ["\\midrule", f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{TEX_TASK[task]}}}}} \\\\"]
        for m in models:
            cells = [TEX_INDENT + TEX_MODEL[m]]
            for target, _, _, _, cols in groups:
                row = raw.get((m, target))
                cells += [row[c] if row else "{--}" for c in cols]
                cells.append(f"({fmt_fps(1000 / float(row['total_ms']))}~fps)" if row else "")
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved LaTeX table to: {tex_path.name}")



# ------------------------------------------------------------------ energy

# device -> (energy_totals host dir, energy kind, idle run name)
DEVICES = {
    "robot_cpu": ("robot", "cpu", "robot-cpu-idle"),
    "edge_cpu": ("edge-asus", "cpu", "edge-asus-cpu-idle"),
    "edge_gpu": ("edge-asus", "gpu", "edge-asus-gpu-idle"),
}


def _energy_totals(path: Path, kind: str):
    """(energy J, duration s) of the CPU or GPU counter in an energy_totals JSON, or None."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    energy = d.get("cpu_energy_j_total") if kind == "cpu" else \
        sum(g.get("gpu_energy_j_total") or 0.0 for g in d.get("gpu") or []) or None
    return (float(energy), float(d["duration_sec"])) if energy is not None else None


def _power_samples(run_id: str, host: str, kind: str) -> list:
    """[(sample time s, W)] of the run's per-second monitor file: CPU = cpu_watts
    (counter energy of the preceding interval / its length), GPU = power_draw_w
    summed over GPUs. [] if not recorded."""
    col, power = ("cpu_watts" if kind == "cpu" else "power_draw_w"), {}
    # The monitor's file name varies between runs ({run_id}_cpu.csv, {run_id}.csv,
    # {run_id}_gpu.csv): pick the file by its columns, not its name.
    for name in (f"{run_id}_{kind}.csv", f"{run_id}.csv"):
        path = SYSTEM_STATS / host / name
        if path.exists() and col in path.open().readline().strip().split(","):
            break
    else:
        return []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            t = int(r["timestamp"]) / 1000.0
            power[t] = power.get(t, 0.0) + float(r[col])
    return sorted(power.items())


def _integrate(samples: list, t_start: float, t_end: float, a: float, b: float) -> float:
    """Energy (J) in [a, b] from per-interval power: sample i holds the power of
    (t_{i-1}, t_i], the first interval starting at t_start; after the last sample
    (up to t_end) its power is carried on."""
    energy, prev = 0.0, t_start
    for t, p in samples + [(t_end, samples[-1][1])]:
        energy += p * max(0.0, min(t, b) - max(prev, a))
        prev = t
    return energy


def _iso_s(iso: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(iso).timestamp()


def device_energy(run: dict, device: str):
    """(energy J, duration s) of `device` over the measured inference loop of
    `run` (its time_window), or None if not recorded.

    The energy monitors start ~1.2 s before the loop (warm-up of the
    collectors) and stop ~0.2 s after it. The loop's share is the exact counter
    total minus the energy before and after the loop, taken from the
    per-second power samples. Falls back to the whole monitor window (with a
    warning) if those samples or the loop window are missing."""
    host, kind, _ = DEVICES[device]
    path = SYSTEM_STATS / host / f"energy_totals_{run['run_id']}.json"
    total = _energy_totals(path, kind)
    if total is None:
        return None
    window = run.get("time_window") or {}
    samples = _power_samples(run["run_id"], host, kind)
    if not samples or "start" not in window or "stop" not in window:
        print(f"[warn] {run['run_id']} {device}: no power samples/loop window, energy over the whole monitor window")
        return total
    meta = json.loads(path.read_text())
    mon_start, mon_stop = _iso_s(meta["run_started_iso"]), _iso_s(meta["run_stopped_iso"])
    loop_start, loop_stop = _iso_s(window["start"]), _iso_s(window["stop"])
    outside = (_integrate(samples, mon_start, mon_stop, mon_start, loop_start)
               + _integrate(samples, mon_start, mon_stop, loop_stop, mon_stop))
    return max(total[0] - outside, 0.0), loop_stop - loop_start


def idle_power() -> dict:
    """{device: idle W} = counter energy / duration of a separate run without workload."""
    power = {}
    for dev, (_, kind, idle_name) in DEVICES.items():
        e = _energy_totals(IDLE_DIR / f"energy_totals_{idle_name}.json", kind)
        if e is None:
            raise SystemExit(f"missing idle measurement: {IDLE_DIR}/energy_totals_{idle_name}.json")
        power[dev] = e[0] / e[1]
    return power


# ------------------------------------------------------------------ figure style

FONT_SCALE = 2.6
EDGE_COLOR = "#262626"             # bar borders, error bars, legend swatches
HATCH_ALPHA, HATCH = 0.45, "//"    # lighter, hatched part of a stacked bar
LEGEND_STYLE = dict(frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4, fancybox=False,
                    edgecolor="black")


def apply_theme() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE,
                  rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"})
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "hatch.linewidth": 0.7})


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)


def model_ticks(ax, models: list) -> None:
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_NAME.get(m, m) for m in models], rotation=20, ha="right", rotation_mode="anchor")


def lighten(color, strength: float):
    """Opaque mix of `color` with white (1 = colour, 0 = white); unlike an alpha,
    grid lines behind the bar don't show through."""
    return tuple(strength * c + (1 - strength) for c in mcolors.to_rgb(color))


def patches(entries) -> list:
    """Legend patches for [(label, colour), ...]."""
    return [mpatches.Patch(facecolor=c, edgecolor=EDGE_COLOR, label=lbl) for lbl, c in entries]


def part_patches(solid_label: str, hatched_label: str) -> list:
    """Neutral-gray legend patches for the solid and the hatched part of a stacked bar."""
    gray = "#8C8C8C"
    return [mpatches.Patch(facecolor=gray, edgecolor=EDGE_COLOR, label=solid_label),
            mpatches.Patch(facecolor=lighten(gray, HATCH_ALPHA), edgecolor=EDGE_COLOR, hatch=HATCH,
                           label=hatched_label)]


def legend_row(fig, groups, y: float, gap: float = 0.012) -> None:
    """Titled legends side by side, centred over the panels at height y.
    groups = [(title, handles), ...]."""
    legends = [fig.legend(handles=h, title=t, loc="lower left", ncol=len(h), columnspacing=1.2, **LEGEND_STYLE)
               for t, h in groups]
    renderer = fig.canvas.get_renderer()
    widths = [leg.get_window_extent(renderer).width / fig.bbox.width for leg in legends]
    axes_x = [x for ax in fig.axes for x in (ax.get_position().x0, ax.get_position().x1)]
    x = (min(axes_x) + max(axes_x)) / 2 - (sum(widths) + gap * (len(widths) - 1)) / 2
    for leg, w in zip(legends, widths):
        leg.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += w + gap


def save(fig, path: Path) -> None:
    # No bbox_extra_artists: it would replace the default artists of the tight
    # bounding box and crop texts/marks drawn outside the axes.
    fig.savefig(path)
    print(f"[OK] Saved plot to: {path.name}")
    plt.close(fig)
