#!/usr/bin/env python3
"""
barplot_fps_latency_e2e.py

4-row × 3-column figure for paper:
  Row 0 – FPS (Local CPU + Remote GPU × Torch/SOL — 4 bars)
  Row 1 – Local CPU  E2E latency breakdown (Torch + SOL)
  Row 2 – Remote CPU E2E latency breakdown (Torch + SOL)
  Row 3 – Remote GPU E2E latency breakdown (Torch + SOL)

Latency stack (bottom → top):
  Inference  |  Pre/Post-processing  |  Network + DDS
  (sums exactly to p50(t_e2e_ms))

Run from results/plot/:
  python3 barplot_fps_latency_e2e.py
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns


# ── paths ──────────────────────────────────────────────────────────────────────
SUMMARY_DIR = Path(__file__).parent.parent.parent / "experiments" / "model-stats" / "ros2" / "_summary"
OUTPUT_FILE = Path(__file__).parent / "e2e-latency-and-fps.pdf"

# ── plot settings ──────────────────────────────────────────────────────────────
FONT_SCALE      = 2.2
SPINES_WIDTH    = 1.0
STROKE_WIDTH    = 0.7
LATENCY_TICKS   = 5
LATENCY_FMT     = "%.0f"
FIG_SIZE        = (18, 15.0)
SHOW_ERROR_BARS = True

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
VARIANTS       = [v["label"] for v in VARIANT_DEFS]
LOCAL_VARIANTS = VARIANTS[0:2]
CPU_VARIANTS   = VARIANTS[2:4]
GPU_VARIANTS   = VARIANTS[4:6]
FPS_VARIANTS   = VARIANTS  # all 6 variants in FPS row

# Semantic-segmentation models only: read Remote GPU FPS/latency from an
# alternate summary file (e.g. a fixed-FPS rerun, see
# ../../experiments/model-stats/ros2/_summary/) instead of the default
# remote-gpu_model_stats.json, applied uniformly to every model in CAT_SEG.
# None (default) = every model reads "remote-gpu" as before.
#
# Single source of truth: barplot_energy_consumption.py and
# barplot_energy_per_frame.py import this exact variable (not a copy of
# their own), since FPS/latency and energy come from the same underlying
# benchmark run and must stay in sync -- set it once here (or via its
# comparison/ alias) and both energy plots follow.
SEG_REMOTE_GPU_TAG_OVERRIDE: str | None = None
# SEG_REMOTE_GPU_TAG_OVERRIDE = "remote-gpu12fps"

# ── row definitions ────────────────────────────────────────────────────────────
LAT_ROW_CONFIGS = [
    (1, LOCAL_VARIANTS, "Local CPU\nLatency (ms)"),
    (2, CPU_VARIANTS,   "Remote CPU\nLatency (ms)"),
    (3, GPU_VARIANTS,   "Remote GPU\nLatency (ms)"),
]
NROWS = 4

# ── bar widths ─────────────────────────────────────────────────────────────────
FPS_BAR_W = 0.13   # row 0: 6 variants
BAR_W     = 0.35   # latency rows: 2 variants each


# ── helpers ────────────────────────────────────────────────────────────────────

def _classify(run: dict) -> str | None:
    b = run.get("backend", "").lower()
    t = run.get("run_tag", "").lower()
    for vd in VARIANT_DEFS:
        if vd["backend"] == b and vd["run_tag"] == t:
            return vd["label"]
    return None


def _classify_as(run: dict, forced_tag: str) -> str | None:
    """Classify using forced_tag instead of the run's own run_tag -- for
    override files whose run_tag (e.g. "remote-gpu12fps") doesn't match
    VARIANT_DEFS but should still be read as that canonical variant."""
    b = run.get("backend", "").lower()
    for vd in VARIANT_DEFS:
        if vd["backend"] == b and vd["run_tag"] == forced_tag:
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

def _row_from_run(r: dict, variant: str) -> tuple | None:
    """Build one (model, variant, fps, inf_ms, prepost_ms, net_ms, e2e_lower,
    e2e_upper) row from a raw model-stats run dict, or None if unusable."""
    model = r.get("model", "")
    if model not in MODEL_ORDER:
        return None

    inf_ms   = float((r.get("t_inference_ms")   or {}).get("p50", 0.0))
    pre_ms   = float((r.get("t_preprocess_ms")  or {}).get("p50", 0.0))
    post_ms  = float((r.get("t_postprocess_ms") or {}).get("p50", 0.0))
    e2e      = r.get("t_e2e_ms") or {}
    e2e_p50  = float(e2e.get("p50", 0.0))
    e2e_p25  = float(e2e.get("p25", 0.0))
    e2e_p75  = float(e2e.get("p75", 0.0))

    if e2e_p50 == 0.0:
        return None

    fps        = 1000.0 / e2e_p50
    prepost_ms = pre_ms + post_ms
    # Network absorbs the remainder so inf+pp+net = p50(e2e) exactly.
    # The small median non-additivity gap folds into the network layer.
    net_ms     = max(0.0, e2e_p50 - inf_ms - prepost_ms)
    e2e_lower  = e2e_p50 - e2e_p25
    e2e_upper  = e2e_p75 - e2e_p50

    return (model, variant, fps, inf_ms, prepost_ms, net_ms, e2e_lower, e2e_upper)


def load_rows() -> list[tuple]:
    """
    Returns list of (model, variant, fps, inf_ms, prepost_ms, net_ms, e2e_lower, e2e_upper).
    All bar heights use median (p50). Stack sums exactly to p50(t_e2e_ms):
      inf_ms     = p50(t_inference_ms)
      prepost_ms = p50(t_preprocess_ms) + p50(t_postprocess_ms)
      net_ms     = p50(t_e2e_ms) - inf_ms - prepost_ms   (network + DDS; absorbs the
                   small median non-additivity gap from the pipeline stages)
    Error bars are asymmetric IQR: lower=p50-p25, upper=p75-p50.
    """
    rows = []
    for tag in ("local-cpu", "remote-cpu", "remote-gpu"):
        for r in json.loads((SUMMARY_DIR / f"{tag}_model_stats.json").read_text())["runs"]:
            model = r.get("model", "")
            # Segmentation models read their Remote GPU rows from the
            # override file below instead, when one is configured.
            if tag == "remote-gpu" and model in CAT_SEG and SEG_REMOTE_GPU_TAG_OVERRIDE:
                continue
            variant = _classify(r)
            if variant is None:
                continue
            row = _row_from_run(r, variant)
            if row is not None:
                rows.append(row)

    # Segmentation Remote GPU override: pull every segmentation model's
    # rows from the configured summary file instead, classified as the
    # normal "Remote GPU" variant regardless of that file's own run_tag
    # (e.g. a "remote-gpu12fps"-tagged rerun still reads as "Remote GPU").
    if SEG_REMOTE_GPU_TAG_OVERRIDE:
        ov_path = SUMMARY_DIR / f"{SEG_REMOTE_GPU_TAG_OVERRIDE}_model_stats.json"
        if not ov_path.exists():
            print(f"[WARN] Missing override model-stats: {ov_path}")
        else:
            for r in json.loads(ov_path.read_text())["runs"]:
                if r.get("model", "") not in CAT_SEG:
                    continue
                variant = _classify_as(r, "remote-gpu")
                if variant is None:
                    continue
                row = _row_from_run(r, variant)
                if row is not None:
                    rows.append(row)

    return rows


# ── main plot ──────────────────────────────────────────────────────────────────

def plot(rows: list[tuple]) -> None:
    if not rows:
        print("No matching rows found.")
        return

    base_models = _ordered([r[0] for r in rows])

    fps_map   = {(m, v): np.nan for m in base_models for v in VARIANTS}
    inf_map   = {(m, v): 0.0    for m in base_models for v in VARIANTS}
    pp_map    = {(m, v): 0.0    for m in base_models for v in VARIANTS}
    net_map   = {(m, v): 0.0    for m in base_models for v in VARIANTS}
    lower_map = {(m, v): np.nan for m in base_models for v in VARIANTS}
    upper_map = {(m, v): np.nan for m in base_models for v in VARIANTS}

    for model, variant, fps, inf_ms, prepost_ms, net_ms, e2e_lower, e2e_upper in rows:
        if model in base_models and variant in VARIANTS:
            fps_map[  (model, variant)] = fps
            inf_map[  (model, variant)] = inf_ms
            pp_map[   (model, variant)] = prepost_ms
            net_map[  (model, variant)] = net_ms
            lower_map[(model, variant)] = e2e_lower
            upper_map[(model, variant)] = e2e_upper

    sns.set_theme(
        context="paper", style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )

    # Paired 12-color: blue=local-cpu, green=remote-cpu, red=remote-gpu
    paired = sns.color_palette("Paired", 12)
    color_map = {
        VARIANTS[0]: paired[0],   # local-cpu  torch – light blue
        VARIANTS[1]: paired[1],   # local-cpu  sol   – dark  blue
        VARIANTS[2]: paired[2],   # remote-cpu torch – light green
        VARIANTS[3]: paired[3],   # remote-cpu sol   – dark  green
        VARIANTS[4]: paired[4],   # remote-gpu torch – light red
        VARIANTS[5]: paired[5],   # remote-gpu sol   – dark  red
    }

    # Build per-category column groups
    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        m_in = [m for m in base_models if m in cat]
        if m_in:
            cat_models.append(m_in)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    ncols  = len(cat_models)
    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        NROWS, ncols, figsize=FIG_SIZE, sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20, "hspace": 0.12},
    )
    if ncols == 1:
        axes = np.array([[axes[i]] for i in range(NROWS)])

    for col_idx, current_models in enumerate(cat_models):
        # Per-column y-limits — each category panel uses its own scale
        col_fps = np.array([fps_map[(m, v)] for m in current_models for v in FPS_VARIANTS], dtype=float)
        fps_ylim = float(np.nanmax(col_fps)) * 1.08 if np.any(np.isfinite(col_fps)) else 1.0

        lat_ylim = {}
        for r_idx, v_set, _ in LAT_ROW_CONFIGS:
            if r_idx == 1:
                tops = np.array(
                    [inf_map[(m, v)] + pp_map[(m, v)] + upper_map[(m, v)]
                     for m in current_models for v in v_set], dtype=float
                )
            else:
                tops = np.array(
                    [inf_map[(m, v)] + pp_map[(m, v)] + net_map[(m, v)] + upper_map[(m, v)]
                     for m in current_models for v in v_set], dtype=float
                )
            lat_ylim[r_idx] = float(np.nanmax(tops)) * 1.08 if np.any(np.isfinite(tops)) else 1.0

        x = np.arange(len(current_models))

        # ── Row 0: FPS ─────────────────────────────────────────────────────────
        ax0 = axes[0, col_idx]
        ax0.set_ylim(0, fps_ylim)
        off0 = _offsets(FPS_VARIANTS, FPS_BAR_W)

        for v in FPS_VARIANTS:
            vals = np.array([fps_map[(m, v)] for m in current_models], dtype=float)
            ax0.bar(x + off0[v], vals, width=FPS_BAR_W, color=color_map[v],
                    edgecolor="black", linewidth=STROKE_WIDTH,
                    label=v if col_idx == 0 else "", zorder=3)

        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.margins(x=0.005)
        _style(ax0)
        if col_idx == 0:
            ax0.set_ylabel("Frame rate (FPS)")

        # ── Rows 1–3: Latency breakdown ────────────────────────────────────────
        for r_idx, v_set, ylabel in LAT_ROW_CONFIGS:
            ax_lat = axes[r_idx, col_idx]
            off    = _offsets(v_set, BAR_W)
            ax_lat.set_ylim(0, lat_ylim[r_idx])
            ax_lat.yaxis.set_major_locator(MaxNLocator(nbins=LATENCY_TICKS))
            ax_lat.yaxis.set_major_formatter(FormatStrFormatter(LATENCY_FMT))

            for v in v_set:
                xs       = x + off[v]
                inf_vals = np.array([inf_map[(m, v)] for m in current_models], dtype=float)
                pp_vals  = np.array([pp_map[ (m, v)] for m in current_models], dtype=float)
                if r_idx == 1:
                    tot_vals = inf_vals + pp_vals
                else:
                    net_vals = np.array([net_map[(m, v)] for m in current_models], dtype=float)
                    tot_vals = inf_vals + pp_vals + net_vals
                yerr     = np.array([
                    [lower_map[(m, v)] for m in current_models],
                    [upper_map[(m, v)] for m in current_models],
                ], dtype=float)

                base_c  = color_map[v]
                light_c = mcolors.to_rgba(base_c, alpha=0.4)

                ax_lat.bar(xs, inf_vals, width=BAR_W,
                           facecolor=base_c, edgecolor="black",
                           linewidth=STROKE_WIDTH, zorder=3)
                ax_lat.bar(xs, pp_vals, bottom=inf_vals, width=BAR_W,
                           facecolor=light_c, edgecolor="black",
                           linewidth=STROKE_WIDTH, hatch="..", zorder=3)
                if r_idx != 1:
                    ax_lat.bar(xs, net_vals, bottom=(inf_vals + pp_vals), width=BAR_W,
                               facecolor=light_c, edgecolor="black",
                               linewidth=STROKE_WIDTH, hatch="//", zorder=3)

                if SHOW_ERROR_BARS:
                    ax_lat.errorbar(xs, tot_vals, yerr=yerr, fmt="none",
                                    ecolor="black", elinewidth=1.0,
                                    capsize=4, capthick=1.0, zorder=10)

            ax_lat.set_xticks(x)
            if r_idx == NROWS - 1:
                ax_lat.set_xticklabels(
                    [MODEL_LABELS.get(m, m) for m in current_models],
                    rotation=20, ha="right",
                )
            else:
                ax_lat.set_xticklabels([])

            ax_lat.margins(x=0.005)
            _style(ax_lat)
            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)

    # ── Legend ─────────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.10, top=0.84)

    legend_base  = "lightgray"
    legend_light = mcolors.to_rgba(legend_base, alpha=0.4)

    # Build color patches manually — remote-cpu only appears in latency rows,
    # not in the FPS row, so we can't rely on axes[0,0].get_legend_handles_labels().
    color_handles = [mpatches.Patch(facecolor=color_map[v], edgecolor="black", label=v)
                     for v in VARIANTS]
    hatch_handles = [
        mpatches.Patch(facecolor=legend_base,  edgecolor="black", hatch="",   label="Inference"),
        mpatches.Patch(facecolor=legend_light, edgecolor="black", hatch="..", label="Pre/Post-processing"),
        mpatches.Patch(facecolor=legend_light, edgecolor="black", hatch="//", label="Network + ROS2 overhead"),
    ]

    # 3-col × 3-row legend (matplotlib fills row-by-row):
    #   [Local Torch]      [Remote CPU Torch]  [Remote GPU Torch]
    #   [Local SOL]        [Remote CPU SOL]    [Remote GPU SOL]
    #   [Inference]        [Pre/Post]          [Network + DDS]
    combined_handles = [
        color_handles[0], color_handles[2], color_handles[4],
        color_handles[1], color_handles[3], color_handles[5],
        hatch_handles[0], hatch_handles[1], hatch_handles[2],
    ]
    combined_labels = [
        VARIANTS[0], VARIANTS[2], VARIANTS[4],
        VARIANTS[1], VARIANTS[3], VARIANTS[5],
        "Inference", "Pre/Post-processing", "Network + ROS2 overhead",
    ]

    leg = fig.legend(
        combined_handles, combined_labels,
        title=None, loc="upper center", bbox_to_anchor=(0.5, 0.95),
        ncol=3, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4,
        fancybox=False, edgecolor="black"
    )

    # ── Captions ───────────────────────────────────────────────────────────────
    CAPTION_OFFSET = 0.075
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


def print_summary(rows: list[tuple]) -> None:
    hdr = f"{'model':26} {'variant':36} {'FPS':>7} {'inf_ms(p50)':>12} {'prepost_ms':>11} {'net_ms':>9} {'e2e_p50':>9} {'e2e_lo':>8} {'e2e_hi':>8}"
    print(hdr)
    print("-" * len(hdr))
    for model, variant, fps, inf_ms, prepost_ms, net_ms, e2e_lo, e2e_hi in sorted(
        rows, key=lambda r: (VARIANTS.index(r[1]) if r[1] in VARIANTS else 99, r[0])
    ):
        e2e_p50 = inf_ms + prepost_ms + net_ms
        print(f"{MODEL_LABELS.get(model, model):26} {variant:36} {fps:7.2f} {inf_ms:12.2f} {prepost_ms:11.2f} {net_ms:9.2f} {e2e_p50:9.2f} {e2e_lo:8.2f} {e2e_hi:8.2f}")
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

    rows = load_rows()
    print_summary(rows)
    plot(rows)


if __name__ == "__main__":
    main()