#!/usr/bin/env python3
"""
barplot_energy_consumption.py

Plots Robot CPU, Edge CPU, and Edge GPU energy consumption (Joules) in a 3x3
grid grouped by model category.

Energy = mean_power_W × duration_sec (CPU panels)
       = gpu_energy_j column (GPU panel, pre-computed in summarize_system_stats.py)

Produces: energy-consumption.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
_HERE = Path(__file__).parent
cfg = load_config()
CPU_FILE     = str(get_path("system_cpu_summary"))   # e2e: edge-asus + robot remote rows
ISO_CPU_FILE = str(_HERE / "../../experiments/system-stats/vaccel/_summary/iso_overall_cpu_stats_wifi.csv")
GPU_FILE     = str(get_path("system_gpu_summary"))

OUTPUT_FILE = "energy-consumption.pdf"

FONT_SCALE   = 2.2
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.8
MAX_TICKS    = 5
TICK_FMT     = "%.1f"
FIG_SIZE     = (18, 11.0)

SHOW_ERROR_BARS = False

MODEL_TYPE_ORDER = get_model_type_order()

# --- MODEL CATEGORIES ---
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]

CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG   = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

VARIANTS = [
    "Local CPU (vAccel + Torch)",   # Index 0
    "Local CPU (vAccSOL)",          # Index 1
    "Remote CPU (vAccel + Torch)",  # Index 2
    "Remote CPU (vAccSOL)",         # Index 3
    "Remote GPU (vAccel + Torch)",  # Index 4
    "Remote GPU (vAccSOL)",         # Index 5
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def _safe_energy_cpu(row) -> float:
    """Compute CPU energy (kJ) = mean_power_W * duration_sec / 1000."""
    try:
        w = float(row["cpu_watts_mean"])
        d = float(row["duration_sec"])
        if np.isfinite(w) and np.isfinite(d) and d > 0:
            return w * d / 1000.0
    except (TypeError, ValueError, KeyError):
        pass
    return np.nan


def _safe_energy_gpu(row) -> float:
    """Use pre-computed gpu_energy_j, converted to kJ."""
    try:
        v = float(row["gpu_energy_j"])
        return v / 1000.0 if np.isfinite(v) else np.nan
    except (TypeError, ValueError, KeyError):
        return np.nan


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def classify_robot_cpu_variant(row) -> str | None:
    backend = str(row.get("backend", "")).lower().strip()
    device  = str(row.get("device",  "")).lower().strip()

    if backend == "vaccel-local-ptc" and device == "cpu":
        return VARIANTS[0]
    if backend == "vaccel-local-sol" and device == "cpu":
        return VARIANTS[1]
    if backend == "vaccel-remote-ptc" and "cpu" in device:
        return VARIANTS[2]
    if backend == "vaccel-remote-sol" and "cpu" in device:
        return VARIANTS[3]
    if backend == "vaccel-remote-ptc" and "gpu" in device:
        return VARIANTS[4]
    if backend == "vaccel-remote-sol" and "gpu" in device:
        return VARIANTS[5]
    return None


def load_robot_cpu_rows(cpu_df: pd.DataFrame):
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_cpu_variant(r)
        if v is None:
            continue
        energy = _safe_energy_cpu(r)
        rows.append({
            "base_model": str(r["model"]).strip(),
            "variant": v,
            "mean": energy,
            "std": np.nan,  # energy is a derived scalar per run; no std
        })
    return rows


def load_edge_cpu_remote_rows(cpu_df: pd.DataFrame):
    rows = []
    for _, r in cpu_df.iterrows():
        host    = str(r.get("host",    "")).lower()
        backend = str(r.get("backend", "")).lower()
        device  = str(r.get("device",  "")).lower()
        if "edge" not in host or "cpu" not in device:
            continue
        model  = str(r.get("model", "")).strip()
        energy = _safe_energy_cpu(r)

        if backend == "vaccel-remote-ptc":
            rows.append({"base_model": model, "variant": VARIANTS[2], "mean": energy, "std": np.nan})
        elif backend == "vaccel-remote-sol":
            rows.append({"base_model": model, "variant": VARIANTS[3], "mean": energy, "std": np.nan})
    return rows


def load_edge_gpu_remote_rows(gpu_df: pd.DataFrame):
    rows = []
    for _, r in gpu_df.iterrows():
        host    = str(r.get("host",    "")).lower()
        backend = str(r.get("backend", "")).lower()
        device  = str(r.get("device",  "")).lower()
        if "edge" not in host or "gpu" not in device:
            continue
        model  = str(r.get("model", "")).strip()
        energy = _safe_energy_gpu(r)

        if backend == "vaccel-remote-ptc":
            rows.append({"base_model": model, "variant": VARIANTS[4], "mean": energy, "std": np.nan})
        elif backend == "vaccel-remote-sol":
            rows.append({"base_model": model, "variant": VARIANTS[5], "mean": energy, "std": np.nan})
    return rows


def _apply_model_filter(rows, allowed_models):
    kept = [r for r in rows if r["base_model"] in allowed_models]
    return kept


def compute_offsets(variants_present, row_idx: int):
    n = len(variants_present)
    width = min(0.12, 0.8 / 6) if row_idx == 0 else min(0.25, 0.8 / n)
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


# ---------------------------------------------------------------------------
# Debug table
# ---------------------------------------------------------------------------

def print_debug_table(panels, base_models):
    print("\n" + "=" * 90)
    print(f"{'DEBUG: ENERGY CONSUMPTION (J)':^90}")
    print("=" * 90)
    for panel_key, ylabel, rows_data, vars_present in panels:
        print(f"\n--- {panel_key.upper()} ---")
        print(f"{'Model':<22} | {'Variant':<28} | {'Energy (J)':>12}")
        print("-" * 70)
        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
        for m in base_models:
            for v in vars_present:
                val = mean_map[(m, v)]
                if np.isfinite(val):
                    print(f"{m:<22} | {v:<28} | {val:12.2f}")
    print("\n" + "=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_combined_energy(robot_rows, edge_cpu_rows, edge_gpu_rows):
    base_models = ordered_models(
        sorted({r["base_model"] for r in robot_rows + edge_cpu_rows + edge_gpu_rows})
    )

    sns.set_theme(
        context="paper", style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=FONT_SCALE,
    )

    crest = sns.color_palette("crest", 10)
    flare = sns.color_palette("flare", 10)
    purp  = sns.color_palette("Purples", 10)

    color_map = {
        VARIANTS[0]: crest[6],
        VARIANTS[1]: crest[2],
        VARIANTS[2]: flare[6],
        VARIANTS[3]: flare[2],
        VARIANTS[4]: purp[3],
        VARIANTS[5]: purp[7],
    }

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        3, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.14, "hspace": 0.14},
    )

    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    panels = [
        ("robot",    "Robot CPU (kJ)", robot_rows,     VARIANTS),
        ("edge_cpu", "Edge CPU (kJ)",  edge_cpu_rows,  [VARIANTS[2], VARIANTS[3]]),
        ("edge_gpu", "Edge GPU (kJ)",  edge_gpu_rows,  [VARIANTS[4], VARIANTS[5]]),
    ]

    print_debug_table(panels, base_models)

    for row_idx, (panel_key, ylabel, rows_data, vars_present) in enumerate(panels):

        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]

        all_vals = np.asarray(
            [mean_map[(m, v)] for cm in cat_models for m in cm for v in vars_present],
            dtype=float,
        )
        y_max = np.nanmax(all_vals)
        if np.isfinite(y_max) and y_max > 0:
            y_lim_top = y_max * 1.05
            if y_lim_top >= 50:
                step = 10.0
            elif y_lim_top >= 20:
                step = 5.0
            elif y_lim_top >= 2:
                step = 1.0
            elif y_lim_top >= 0.5:
                step = 0.2
            else:
                step = 0.05
            row_y_lim_top = np.ceil(y_lim_top / step) * step
        else:
            row_y_lim_top = 1.0

        width, offsets = compute_offsets(vars_present, row_idx)

        for col_idx, current_models in enumerate(cat_models):
            ax = axes[row_idx, col_idx]
            x  = np.arange(len(current_models))

            ax.set_ylim(0, row_y_lim_top)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_TICKS))
            ax.yaxis.set_major_formatter(FormatStrFormatter(TICK_FMT))

            for v in vars_present:
                xs    = x + offsets[v]
                means = np.asarray([mean_map[(m, v)] for m in current_models], dtype=float)

                ax.bar(
                    xs, means, width=width,
                    color=color_map[v], edgecolor="black", linewidth=STROKE_WIDTH,
                    label=v if (row_idx == 0 and col_idx == 0) else "",
                    zorder=3,
                )

            # --- ENERGY SAVINGS INDICATOR ---
            for i, m in enumerate(current_models):
                if row_idx == 0:
                    local_base_vars = [VARIANTS[0], VARIANTS[1]]
                    compare_vars    = VARIANTS[2:]
                    x_center_compare = i + (offsets[VARIANTS[2]] + offsets[VARIANTS[-1]]) / 2.0
                elif row_idx == 1:
                    base_var         = VARIANTS[2]
                    compare_vars     = [VARIANTS[3]]
                    x_center_compare = i + offsets[VARIANTS[3]]
                else:
                    base_var         = VARIANTS[4]
                    compare_vars     = [VARIANTS[5]]
                    x_center_compare = i + offsets[VARIANTS[5]]

                if row_idx == 0:
                    local_vals = {
                        v: mean_map.get((m, v), np.nan)
                        for v in local_base_vars
                        if np.isfinite(mean_map.get((m, v), np.nan)) and mean_map.get((m, v), np.nan) > 0
                    }
                    if not local_vals:
                        continue
                    base_val = np.mean(list(local_vals.values()))
                    x_start  = i + min(offsets[v] for v in local_base_vars)
                else:
                    base_val = mean_map.get((m, base_var), np.nan)
                    x_start  = i + offsets[base_var]

                if not (np.isfinite(base_val) and base_val > 0):
                    continue

                comps = [mean_map[(m, v)] for v in compare_vars if np.isfinite(mean_map[(m, v)])]
                if not comps:
                    continue

                avg_comp    = np.mean(comps)
                savings_pct = (base_val - avg_comp) / base_val * 100
                print(f"Panel: {panel_key:<10} | Model: {m:<22} | Energy Savings: {savings_pct:>6.1f}%")

                if abs(savings_pct) <= 0.5:
                    continue

                is_reduction = savings_pct > 0
                badge_color  = "darkgreen" if is_reduction else "darkred"
                symbol       = "↓" if is_reduction else "↑"
                display_pct  = abs(savings_pct)
                x_end        = i + offsets[vars_present[-1]]

                ax.hlines(
                    y=base_val, xmin=x_start, xmax=x_end,
                    colors=badge_color, linestyles="-.", linewidth=STROKE_WIDTH,
                    alpha=0.75, zorder=4,
                    label="Savings reference" if (row_idx == 0 and col_idx == 0 and i == 0) else "",
                )

                arrow_offset = (base_val - avg_comp) * 0.15
                ax.annotate(
                    "", xy=(x_center_compare, avg_comp + arrow_offset), xycoords="data",
                    xytext=(x_center_compare, base_val), textcoords="data",
                    arrowprops=dict(arrowstyle="->", color=badge_color, lw=1.5),
                    zorder=4,
                )

                y_center = (base_val + avg_comp) / 2.0
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
                    x_center_compare, y_center, f"{symbol} {display_pct:.0f}%",
                    ha="center", va="center", fontsize=plt.rcParams["font.size"] * 0.7,
                    color=badge_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=badge_color, lw=1.2, alpha=0.95),
                    zorder=20,
                )

            ax.set_xticks(x)
            if row_idx == 2:
                ax.set_xticklabels(
                    [get_model_display_name(m) for m in current_models],
                    rotation=15, ha="right",
                )
            else:
                ax.set_xticklabels([])

            style_axes(ax)
            ax.margins(x=0.005)

            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)

    # --- Layout ---
    CAPTION_OFFSET = 0.08

    fig.subplots_adjust(
        left=0.07, right=0.995,
        bottom=0.17, top=0.88,
    )

    clean_handles = [
        mpatches.Patch(facecolor=color_map[v], edgecolor="none", linewidth=0.0)
        for v in VARIANTS
    ]
    from matplotlib.lines import Line2D
    clean_handles.append(
        Line2D([0], [0], color="darkgreen", linestyle="-.", linewidth=STROKE_WIDTH, alpha=0.75)
    )
    variant_labels = list(VARIANTS) + ["Energy savings"]

    leg = fig.legend(
        clean_handles, variant_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        edgecolor="black",
        fontsize=plt.rcParams["font.size"] * 0.85
    )
    leg.get_frame().set_linewidth(SPINES_WIDTH)
    leg.get_frame().set_boxstyle("square", pad=0.4)

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[2, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[2, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(
        OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.02,
        bbox_extra_artists=(leg, *caption_artists),
    )
    print(f"[OK] Saved energy plot to: {OUTPUT_FILE}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    cpu_path     = Path(CPU_FILE).resolve()
    iso_cpu_path = Path(ISO_CPU_FILE).resolve()
    gpu_path     = Path(GPU_FILE).resolve()

    if not cpu_path.exists():
        raise SystemExit(f"CPU CSV not found: {cpu_path}")
    if not gpu_path.exists():
        raise SystemExit(f"GPU CSV not found: {gpu_path}")

    cpu_df = pd.read_csv(cpu_path)
    if iso_cpu_path.exists():
        cpu_df = pd.concat([cpu_df, pd.read_csv(iso_cpu_path)], ignore_index=True)
    else:
        print(f"[WARNING] ISO CPU file not found, local robot rows will be missing: {iso_cpu_path}")
    gpu_df = pd.read_csv(gpu_path)

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    robot_rows    = load_robot_cpu_rows(cpu_df)
    edge_cpu_rows = load_edge_cpu_remote_rows(cpu_df)
    edge_gpu_rows = load_edge_gpu_remote_rows(gpu_df)

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    robot_rows    = _apply_model_filter(robot_rows,    allowed_models)
    edge_cpu_rows = _apply_model_filter(edge_cpu_rows, allowed_models)
    edge_gpu_rows = _apply_model_filter(edge_gpu_rows, allowed_models)

    if not robot_rows and not edge_cpu_rows and not edge_gpu_rows:
        raise SystemExit("ERROR: No rows remained after filtering!")

    plot_combined_energy(robot_rows, edge_cpu_rows, edge_gpu_rows)


if __name__ == "__main__":
    main()
