#!/usr/bin/env python3
"""
barplot_energy_per_frame.py

3-row × 3-column figure:
  Row 0 – Robot CPU energy per frame (6 variants: local-cpu, remote-cpu, remote-gpu × Torch/SOL)
  Row 1 – Edge GPU energy per frame during remote-CPU inference (Torch + SOL)
  Row 2 – Edge GPU energy per frame during remote-GPU inference (Torch + SOL)

energy_per_frame_J = total_energy_J / n_processed_frames
  total_energy_J:      cpu_energy_j / gpu_energy_j from system-stats _summary JSONs
  n_processed_frames:  t_app_ms["n"] from model-stats _summary JSONs

Note: for remote-CPU (row 1) the edge-asus GPU is idle during inference; its
power draw (~8–9 W) represents baseline system energy, not active GPU work.
For remote-GPU (row 2) the GPU is actively used (24–34 W).

Run from results/plot/:
  python3 barplot_energy_per_frame.py
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns


# ── paths ──────────────────────────────────────────────────────────────────────
PLOT_DIR    = Path(__file__).parent.parent
STATS_DIR   = PLOT_DIR.parent / "experiments" / "system-stats" / "ros2" / "_summary"
MODEL_DIR   = PLOT_DIR.parent / "experiments" / "model-stats" / "ros2" / "_summary"
OUTPUT_FILE = "./energy-per-frame.pdf"

# ── plot settings ──────────────────────────────────────────────────────────────
FONT_SCALE   = 2.2
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.8
ENERGY_TICKS = 5
ENERGY_FMT   = "%.1f"
FIG_SIZE     = (18, 11.0)

# ── model categories ───────────────────────────────────────────────────────────
CAT_IMAGE = ["swin_v2_b", "swin_t", "swin_s", "resnet50"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "r3d_18", "r2plus1d_18"]
CAT_SEG   = ["fcn_resnet101", "fcn_resnet50", "deeplabv3_resnet101", "deeplabv3_resnet50"]
CATEGORIES   = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]
MODEL_ORDER = [*CAT_IMAGE, *CAT_VIDEO, *CAT_SEG]

MODEL_LABELS = {
    "swin_v2_b":           "SwinV2-B",
    "swin_t":              "Swin-T",
    "swin_s":              "Swin-S",
    "resnet50":            "ResNet50",
    "swin3d_s":            "Swin3D-S",
    "swin3d_b":            "Swin3D-B",
    "r3d_18":              "R3D-18",
    "r2plus1d_18":         "R(2+1)D-18",
    "fcn_resnet101":       "FCN-R101",
    "fcn_resnet50":        "FCN-R50",
    "deeplabv3_resnet101": "DLv3-R101",
    "deeplabv3_resnet50":  "DLv3-R50",
}

# ── variants ───────────────────────────────────────────────────────────────────
VARIANT_DEFS = [
    {"label": "Local CPU (ROS2 + Torch)",  "backend": "ptc", "run_tag": "local-cpu"},
    {"label": "Local CPU (ROS2 + SOL)",    "backend": "sol", "run_tag": "local-cpu"},
    {"label": "Remote CPU (ROS2 + Torch)", "backend": "aoti", "run_tag": "remote-cpu"},
    {"label": "Remote CPU (ROS2 + SOL)",   "backend": "sol", "run_tag": "remote-cpu"},
    {"label": "Remote GPU (ROS2 + Torch)", "backend": "aoti", "run_tag": "remote-gpu"},
    {"label": "Remote GPU (ROS2 + SOL)",   "backend": "sol", "run_tag": "remote-gpu"},
]
VARIANTS     = [v["label"] for v in VARIANT_DEFS]
CPU_VARIANTS = VARIANTS[2:4]
GPU_VARIANTS = VARIANTS[4:6]

# ── row definitions ────────────────────────────────────────────────────────────
# (row_idx, variants_to_plot, y_label, host_filter, kind_filter, bar_width)
ROBOT_BAR_W = 0.13   # row 0: 6 variants
EDGE_BAR_W  = 0.35   # rows 1–2: 2 variants each

ROW_CONFIGS = [
    (0, VARIANTS,     "Robot CPU \n(J/frame)",     "robot",     "cpu", ROBOT_BAR_W),
    (1, CPU_VARIANTS, "Edge CPU \n(J/frame)", "edge-asus", "cpu", EDGE_BAR_W),
    (2, GPU_VARIANTS, "Edge GPU \n(J/frame)", "edge-asus", "gpu", EDGE_BAR_W),
]
NROWS = len(ROW_CONFIGS)


# ── helpers ────────────────────────────────────────────────────────────────────

def _classify(backend: str, run_tag: str) -> str | None:
    for vd in VARIANT_DEFS:
        if vd["backend"] == backend and vd["run_tag"] == run_tag:
            return vd["label"]
    return None


def _ordered(models: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(set(models), key=lambda m: (rank.get(m, 10_000), m))


def _offsets(variants_present: list[str], bar_w: float) -> dict[str, float]:
    center = (len(variants_present) - 1) / 2.0
    return {v: (i - center) * bar_w for i, v in enumerate(variants_present)}


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


# ── data ───────────────────────────────────────────────────────────────────────

def load_frame_counts() -> dict[tuple, int]:
    """
    Returns {(model, backend, run_tag): n_frames_pipeline} from model-stats summaries.
    Uses n_frames_pipeline — frames processed and recorded at the inference node side.
    """
    counts: dict[tuple, int] = {}
    for tag in ("local-cpu", "remote-cpu", "remote-gpu"):
        path = MODEL_DIR / f"{tag}_model_stats.json"
        if not path.exists():
            print(f"[WARN] Missing model-stats: {path}")
            continue
        for run in json.loads(path.read_text())["runs"]:
            model   = run.get("model", "")
            backend = run.get("backend", "")
            rt      = run.get("run_tag", tag)
            n = run.get("n_frames_pipeline")
            if n:
                counts[(model, backend, rt)] = int(n)
    return counts


def load_epf_map(frame_counts: dict[tuple, int]) -> dict[tuple, float]:
    """
    Returns {(model, variant_label, host, kind): energy_per_frame_J}.

    EPF = (mean_power_W × duration_sec) / n_frames_pipeline
        = total_energy_J_over_experiment / frames_processed_at_inference_node
    """
    epf_map: dict[tuple, float] = {}

    for tag in ("local-cpu", "remote-cpu", "remote-gpu"):
        path = STATS_DIR / f"{tag}_system_stats.json"
        if not path.exists():
            print(f"[WARN] Missing system-stats: {path}")
            continue
        for run in json.loads(path.read_text())["runs"]:
            model   = run["model"]
            backend = run["backend"]
            rt      = run["run_tag"]
            label   = _classify(backend, rt)
            if not label or model not in MODEL_ORDER:
                continue
            n = frame_counts.get((model, backend, rt))
            if not n:
                print(f"[WARN] No frame count for ({model}, {backend}, {rt}) — skipped")
                continue
            for host, host_data in run["hosts"].items():
                for kind, e_key in [("cpu", "cpu_energy_j"), ("gpu", "gpu_energy_j")]:
                    if kind not in host_data:
                        continue
                    ej = host_data[kind].get(e_key)
                    if ej is not None:
                        epf_map[(model, label, host, kind)] = float(ej) / n

    return epf_map


# ── plot ───────────────────────────────────────────────────────────────────────

def _rounded_ylim(y_max: float) -> float:
    """Round y ceiling to a clean step."""
    top = y_max * 1.05
    if top >= 10:
        step = 2.0
    elif top >= 1:
        step = 0.5
    elif top >= 0.1:
        step = 0.05
    else:
        step = 0.01
    return float(np.ceil(top / step) * step)


def plot(epf_map: dict[tuple, float]) -> None:
    all_models = _ordered([m for (m, *_) in epf_map if m in MODEL_ORDER])
    if not all_models:
        print("No data to plot.")
        return

    sns.set_theme(
        context="paper", style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )

    crest = sns.color_palette("crest",   10)
    flare = sns.color_palette("flare",   10)
    purp  = sns.color_palette("Purples", 10)
    color_map = {
        VARIANTS[0]: crest[6],   # local-cpu  torch – dark teal
        VARIANTS[1]: crest[2],   # local-cpu  sol   – light teal
        VARIANTS[2]: flare[6],   # remote-cpu torch – dark orange
        VARIANTS[3]: flare[2],   # remote-cpu sol   – light orange
        VARIANTS[4]: purp[3],    # remote-gpu torch – light purple
        VARIANTS[5]: purp[7],    # remote-gpu sol   – dark purple
    }

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        m_in = [m for m in all_models if m in cat]
        if m_in:
            cat_models.append(m_in)
            cat_captions.append(CAT_CAPTIONS[i])

    ncols  = len(cat_models)
    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        NROWS, ncols, figsize=FIG_SIZE, sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.32, "hspace": 0.14},
    )
    if ncols == 1:
        axes = np.array([[axes[i]] for i in range(NROWS)])

    # Per-panel y-limit (row × col independent)
    panel_ylim: dict[tuple, float] = {}
    for r_idx, v_set, _, host_f, kind_f, _ in ROW_CONFIGS:
        for col_idx, current_models in enumerate(cat_models):
            vals = np.array([
                epf_map.get((m, v, host_f, kind_f), np.nan)
                for m in current_models for v in v_set
            ], dtype=float)
            panel_ylim[(r_idx, col_idx)] = (
                _rounded_ylim(float(np.nanmax(vals))) if np.any(np.isfinite(vals)) else 1.0
            )

    for r_idx, v_set, ylabel, host_f, kind_f, bar_w in ROW_CONFIGS:
        mean_map = {(m, v): epf_map.get((m, v, host_f, kind_f), np.nan)
                    for cm in cat_models for m in cm for v in v_set}

        for col_idx, current_models in enumerate(cat_models):
            ax  = axes[r_idx, col_idx]
            x   = np.arange(len(current_models))
            off = _offsets(v_set, bar_w)

            ax.set_ylim(0, panel_ylim[(r_idx, col_idx)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=ENERGY_TICKS))
            ax.yaxis.set_major_formatter(FormatStrFormatter(ENERGY_FMT))

            for v in v_set:
                first_bar = True
                for xi, m in enumerate(current_models):
                    val = mean_map[(m, v)]
                    if not np.isfinite(val):
                        continue
                    label = v if (r_idx == 0 and col_idx == 0 and first_bar) else ""
                    ax.bar(x[xi] + off[v], val, width=bar_w,
                           color=color_map[v], edgecolor="black",
                           linewidth=STROKE_WIDTH, zorder=3, label=label)
                    first_bar = False

            # ── savings indicator ──────────────────────────────────────────────
            for i, m in enumerate(current_models):
                if r_idx == 0:
                    local_vars   = [VARIANTS[0], VARIANTS[1]]
                    compare_vars = VARIANTS[2:]
                    x_cmp        = i + (off[VARIANTS[2]] + off[VARIANTS[-1]]) / 2.0
                    x_start      = i + min(off[v] for v in local_vars)
                    local_vals   = [mean_map[(m, v)] for v in local_vars
                                    if np.isfinite(mean_map[(m, v)]) and mean_map[(m, v)] > 0]
                    if not local_vals:
                        continue
                    base_val = float(np.mean(local_vals))
                else:
                    base_var     = v_set[0]
                    compare_vars = [v_set[1]]
                    x_cmp        = i + off[v_set[1]]
                    x_start      = i + off[base_var]
                    base_val     = mean_map[(m, base_var)]

                if not (np.isfinite(base_val) and base_val > 0):
                    continue

                comps = [mean_map[(m, v)] for v in compare_vars
                         if np.isfinite(mean_map[(m, v)])]
                if not comps:
                    continue

                avg_comp    = float(np.mean(comps))
                savings_pct = (base_val - avg_comp) / base_val * 100.0
                print(f"Row {r_idx} | {m:<22} | savings: {savings_pct:+.1f}%")

                if abs(savings_pct) <= 0.5:
                    continue

                is_reduction = savings_pct > 0
                badge_color  = "darkgreen" if is_reduction else "darkred"
                symbol       = "↓" if is_reduction else "↑"
                x_end        = i + off[v_set[-1]]

                ax.hlines(
                    y=base_val, xmin=x_start, xmax=x_end,
                    colors=badge_color, linestyles="-.", linewidth=STROKE_WIDTH,
                    alpha=0.75, zorder=4,
                )

                arrow_offset = (base_val - avg_comp) * 0.15
                ax.annotate(
                    "", xy=(x_cmp, avg_comp + arrow_offset),
                    xytext=(x_cmp, base_val),
                    arrowprops=dict(arrowstyle="->", color=badge_color, lw=1.5),
                    zorder=4,
                )

                y_center  = (base_val + avg_comp) / 2.0
                y_min_ax, y_max_ax = ax.get_ylim()
                y_pad = 0.06 * (y_max_ax - y_min_ax)
                if abs(y_center - base_val) < y_pad or abs(y_center - avg_comp) < y_pad:
                    y_above = max(base_val, avg_comp) + y_pad
                    y_below = min(base_val, avg_comp) - y_pad
                    if y_above <= y_max_ax - 0.02 * (y_max_ax - y_min_ax):
                        y_center = y_above
                    elif y_below >= y_min_ax + 0.02 * (y_max_ax - y_min_ax):
                        y_center = y_below
                    else:
                        y_center = min(max(y_center, y_min_ax + y_pad), y_max_ax - y_pad)

                ax.text(
                    x_cmp, y_center, f"{symbol} {abs(savings_pct):.0f}%",
                    ha="center", va="center",
                    color=badge_color, fontweight="bold",
                    fontsize=plt.rcParams["font.size"] * 0.65,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=badge_color, lw=1.2, alpha=0.95),
                    zorder=20,
                )

            ax.set_xticks(x)
            if r_idx == NROWS - 1:
                ax.set_xticklabels(
                    [MODEL_LABELS.get(m, m) for m in current_models],
                    rotation=15, ha="right",
                )
            else:
                ax.set_xticklabels([])

            ax.margins(x=0.005)
            _style(ax)
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    # ── Legend ─────────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.86)

    color_handles = [mpatches.Patch(facecolor=color_map[v], edgecolor="none") for v in VARIANTS]
    savings_handle = Line2D([0], [0], color="darkgreen", linestyle="-.",
                            linewidth=STROKE_WIDTH, alpha=0.75)
    all_handles = color_handles + [savings_handle]
    all_labels  = list(VARIANTS) + ["Energy savings"]

    leg = fig.legend(
        all_handles, all_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=4, frameon=True, framealpha=0.9, borderpad=0.4,
        fancybox=False, handlelength=1.4, edgecolor="black", prop={"size": plt.rcParams["font.size"] * 0.8}
    )
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_boxstyle("square", pad=0.3)

    # ── Captions ───────────────────────────────────────────────────────────────
    CAPTION_OFFSET = 0.08
    y_caption = min(ax.get_position().y0 for ax in axes[NROWS - 1, :]) - CAPTION_OFFSET
    caption_artists = []
    for ax, cap in zip(axes[NROWS - 1, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        t = fig.text(0.5 * (bbox.x0 + bbox.x1), y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(
        OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.02,
        bbox_extra_artists=(leg, *caption_artists),
    )
    print(f"[OK] Saved → {OUTPUT_FILE}")
    plt.close(fig)


def print_summary(epf_map: dict[tuple, float]) -> None:
    print(f"\n{'model':26} {'variant':36} {'host':12} {'kind':4} {'J/frame':>10}")
    print("-" * 95)
    for (model, variant, host, kind), epf in sorted(
        epf_map.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        print(f"{model:26} {variant:36} {host:12} {kind:4} {epf:10.4f}")
    print()


def main() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })
    plt.rcParams["hatch.linewidth"] = STROKE_WIDTH

    frame_counts = load_frame_counts()
    epf_map      = load_epf_map(frame_counts)
    print_summary(epf_map)
    plot(epf_map)


if __name__ == "__main__":
    main()
