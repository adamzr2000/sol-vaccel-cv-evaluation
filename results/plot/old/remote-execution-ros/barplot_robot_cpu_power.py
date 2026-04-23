#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# ---------------- CONFIG ----------------
cfg = load_config()
CPU_FILE = str(get_path("system_cpu_summary"))

OUTPUT_FILE = "robot_cpu_power.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (18, 5.5)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

MODEL_TYPE_ORDER = get_model_type_order()

# --------------- CATEGORIES --------------
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]

CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# --------------- VARIANTS ----------------
# (Keeping your intent; fixed duplicates/typos in original list)
VARIANTS = [
    "Local (torch.compile)",
    "Local (SOL)",
    "Remote-CPU (torch.compile)",
    "Remote-CPU (SOL)",
    "Remote-GPU (torch.compile)",
    "Remote-GPU (SOL)",
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def compute_offsets(variants_present):
    n = len(variants_present)
    if n == 1:
        width = 0.55
        offsets = {variants_present[0]: 0.0}
        return width, offsets

    width = min(0.22, 0.8 / n)
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def hatch_for_variant_label(vlabel: str) -> str | None:
    s = vlabel.lower()
    # Offloaded variants are hatched; local robot variants not hatched
    if "remote-cpu" in s:
        return "///"
    if "remote-gpu" in s:
        return "++"
    return None


def classify_robot_variant(row: pd.Series) -> str | None:
    """
    Rows are robot-host CPU power measurements, including local and vAccel-client overhead for remote runs.
    Assumes:
      - local: backend == ptc/sol and device == cpu
      - remote: backend contains vaccel-remote-{torch,sol} and device contains target-cpu/target-gpu
    """
    backend = str(row.get("backend", "")).lower().strip()
    device = str(row.get("device", "")).lower().strip()

    # Local robot runs
    if backend == "ptc" and device == "cpu":
        return VARIANTS[0]
    if backend == "sol" and device == "cpu":
        return VARIANTS[1]

    # Remote (robot as client)
    if "vaccel-remote-torch" in backend:
        if "target-cpu" in device:
            return VARIANTS[2]
        if "target-gpu" in device:
            return VARIANTS[4]
    if "vaccel-remote-sol" in backend:
        if "target-cpu" in device:
            return VARIANTS[3]
        if "target-gpu" in device:
            return VARIANTS[5]

    return None


def load_robot_rows(cpu_df: pd.DataFrame):
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_variant(r)
        if v is None:
            continue
        rows.append(
            {
                "base_model": str(r["model"]).strip(),
                "variant": v,
                "mean": float(r["cpu_watts_mean"]),
                "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan,
            }
        )
    return rows


def apply_model_filter(rows, allowed_models):
    present = sorted({r["base_model"] for r in rows})
    dropped = sorted([m for m in present if m not in allowed_models])
    kept = [r for r in rows if r["base_model"] in allowed_models]
    return kept, dropped


def build_maps(rows, variants_present):
    base_models = ordered_models(sorted({r["base_model"] for r in rows}))
    mean_map = {(m, v): np.nan for m in base_models for v in variants_present}
    std_map = {(m, v): np.nan for m in base_models for v in variants_present}
    for r in rows:
        m, v = r["base_model"], r["variant"]
        if m in base_models and v in variants_present:
            mean_map[(m, v)] = r["mean"]
            std_map[(m, v)] = r["std"]
    return base_models, mean_map, std_map


def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.55))
    _, y_top = ax.get_ylim()
    pad_y = y_top * 0.02
    for x, y in zip(xs, ys):
        if not np.isfinite(y):
            continue
        ax.text(
            x,
            y + pad_y,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fs,
            color="black",
            fontweight="bold",
            clip_on=True,
            zorder=20,
        )


def plot_robot_power_categories(rows):
    # Variants to show (robot local + remote client overhead)
    variants_present = VARIANTS[:]  # all 6

    # Color map
    pal = sns.color_palette(cfg.get("palette"), n_colors=len(variants_present))
    color_map = {v: pal[i] for i, v in enumerate(variants_present)}

    # Group models into the three categories (only those present)
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    rows, dropped = apply_model_filter(rows, allowed_models)
    if dropped:
        print(f"\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  {dropped}\n")

    if not rows:
        raise SystemExit("ERROR: No robot rows remained after filtering.")

    # Determine which categories actually have data
    present_models = sorted({r["base_model"] for r in rows})
    cat_models = []
    cat_captions = []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in ordered_models(present_models) if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i])

    if not cat_models:
        raise SystemExit("ERROR: No category models found in robot rows.")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        1,
        len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,  # independent y-scale per category
        gridspec_kw={"width_ratios": widths},
    )
    if len(cat_models) == 1:
        axes = [axes]

    # Common bar geometry
    width, offsets = compute_offsets(variants_present)

    for ax_idx, (ax, current_models) in enumerate(zip(axes, cat_models)):
        # Build maps restricted to current category models
        cat_rows = [r for r in rows if r["base_model"] in current_models]
        base_models, mean_map, std_map = build_maps(cat_rows, variants_present)

        x = np.arange(len(base_models))

        # Per-category y limit (auto zoom)
        all_means = np.asarray([mean_map[(m, v)] for m in base_models for v in variants_present], dtype=float)
        all_stds = np.asarray([std_map[(m, v)] for m in base_models for v in variants_present], dtype=float)
        y_max = np.nanmax(all_means + np.nan_to_num(all_stds, nan=0.0))

        if np.isfinite(y_max) and y_max > 0:
            # smaller, more consistent headroom than 35%
            rel_pad = 0.05  # headroom
            abs_pad = 0.15  # +0.15 W absolute headroom (helps low-power cases)
            y_lim_top = y_max * (1.0 + rel_pad) + abs_pad

            # optional: round up to a "nice" step to avoid awkward top ticks
            step = 0.5 if y_lim_top >= 5 else 0.2
            y_lim_top = np.ceil(y_lim_top / step) * step
        else:
            y_lim_top = 1.0

        ax.set_ylim(0, y_lim_top)

        edgecolor = "black" if SHOW_ERROR_BARS else "none"
        linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

        for v in variants_present:
            xs = x + offsets[v]
            means = np.asarray([mean_map[(m, v)] for m in base_models], dtype=float)
            stds = np.asarray([std_map[(m, v)] for m in base_models], dtype=float)

            hatch_pattern = hatch_for_variant_label(v)

            ax.bar(
                xs,
                means,
                width=width,
                color=color_map[v],
                edgecolor=edgecolor,
                linewidth=linewidth,
                hatch=hatch_pattern,
                label=v if ax_idx == 0 else "",
                zorder=3,
            )

            if SHOW_ERROR_BARS:
                yerr = np.where(np.isfinite(stds), stds, 0.0)
                if np.any(yerr > 0):
                    ax.errorbar(
                        xs,
                        means,
                        yerr=yerr,
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.0,
                        capsize=2.5,
                        capthick=0.7,
                        zorder=10,
                    )

            if SHOW_VALUE_LABELS:
                add_value_labels(ax, xs, means)

        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in base_models], rotation=15, ha="right")
        ax.margins(x=0.015)
        style_axes(ax)

    # ---------- Layout + legend + aligned captions ----------
    # More space between panels + extra bottom for captions
    WSPACE = 0.10
    CAPTION_OFFSET = 0.135

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.22,
        top=0.78,
        wspace=WSPACE,
    )

    # Legend outside top
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Inference execution",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,  # 6 items -> 2 rows; set to 6 if you want one row
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    # Aligned (a)(b)(c) captions below each subplot
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes) - CAPTION_OFFSET
    for ax, cap in zip(axes, cat_captions):
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    # Global y-label (include in tight bbox)
    sy = fig.supylabel("Robot CPU\npower consumption (W)", x=0.04, ha="center", va="center", fontsize=plt.rcParams["font.size"] * 1.15)

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(leg, sy, *caption_artists),
    )
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    cpu_path = Path(CPU_FILE).resolve()
    if not cpu_path.exists():
        raise SystemExit(f"CPU CSV not found: {cpu_path}")

    cpu_df = pd.read_csv(cpu_path)

    # Normalize text cols used in classification
    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()

    # Only robot-host rows (local + vAccel client power during offload)
    robot_rows = load_robot_rows(cpu_df)

    if not robot_rows:
        raise SystemExit("ERROR: No robot rows found after filtering/variant classification.")

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "in", "ytick.direction": "in"},
        font_scale=FONT_SCALE,
    )

    plot_robot_power_categories(robot_rows)


if __name__ == "__main__":
    main()