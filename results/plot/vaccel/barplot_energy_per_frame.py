#!/usr/bin/env python3
"""
barplot_energy_per_frame.py

Plots Robot CPU, Edge CPU, and Edge GPU energy per processed frame (J/frame)
in a 3×3 grid grouped by model category.

Energy per frame = (mean_power_W × duration_sec) / num_processed_frames
  - CPU: power from system-stats CSV; duration and num_frames from benchmark JSON
  - GPU: power from system-stats CSV; duration and num_frames from benchmark JSON

Local robot rows come from the iso run; remote rows from the e2e run.
Produces: energy-per-frame.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
_HERE = Path(__file__).parent
cfg = load_config()

CPU_FILE     = str(get_path("system_cpu_summary"))   # e2e CPU (edge-asus + robot remote)
ISO_CPU_FILE = str(_HERE / "../../experiments/system-stats/vaccel/_summary/iso_overall_cpu_stats_wifi.csv")
GPU_FILE     = str(get_path("system_gpu_summary"))   # e2e GPU

# Benchmark JSONs supplying num_processed_frames per run
E2E_PERF_FILE = str(get_path("model_summary"))       # e2e_benchmark_summary.json
ISO_PERF_FILE = str(_HERE / "../../experiments/model-stats/vaccel/_summary/iso_benchmark_summary.json")

OUTPUT_FILE = "energy-per-frame.pdf"

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
    "Local CPU (vAccel + Torch)",   # 0
    "Local CPU (vAccSOL)",          # 1
    "Remote CPU (vAccel + Torch)",  # 2
    "Remote CPU (vAccSOL)",         # 3
    "Remote GPU (vAccel + Torch)",  # 4
    "Remote GPU (vAccSOL)",         # 5
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m.strip(): i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def _norm_device(device: str) -> str:
    """Reduce device string to 'cpu' or 'gpu'."""
    d = device.lower()
    if "gpu" in d:
        return "gpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Build perf lookup: (host, model, backend, device, remote_host) → num_samples
# ---------------------------------------------------------------------------

def build_perf_map(json_path: str) -> dict:
    """
    Returns {(host, model, backend, device, remote_host): num_samples}.
    host/backend/device/remote_host are lowercased and stripped.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"[WARNING] Perf JSON not found: {path}")
        return {}
    with path.open() as f:
        data = json.load(f)
    runs = data if isinstance(data, list) else data.get("runs", [])
    perf = {}
    for r in runs:
        host        = str(r.get("host",        "")).lower().strip()
        model       = str(r.get("model",       "")).lower().strip()
        backend     = str(r.get("backend",     "")).lower().strip()
        device      = str(r.get("device",      "")).lower().strip()
        remote_host = str(r.get("remote_host", "")).lower().strip()
        n = int(r.get("num_samples", 0))
        if n > 0:
            perf[(host, model, backend, device, remote_host)] = n
    return perf


def lookup_frames(perf_map: dict, host: str, model: str,
                  backend: str, device: str, remote_host: str = "") -> int | None:
    key = (host.lower(), model.lower(), backend.lower(), device.lower(), remote_host.lower())
    return perf_map.get(key)


# ---------------------------------------------------------------------------
# Energy-per-frame calculation
# ---------------------------------------------------------------------------

def _epf_cpu(row, num_frames: int) -> tuple[float, float]:
    """(energy_per_frame_J, std_per_frame_J) from a CPU stats row."""
    w = float(row["cpu_watts_mean"])
    d = float(row["duration_sec"])
    if not (np.isfinite(w) and np.isfinite(d) and d > 0 and num_frames > 0):
        return np.nan, np.nan
    epf = w * d / num_frames
    try:
        ws = float(row["cpu_watts_std"])
        epf_std = ws * d / num_frames if np.isfinite(ws) else np.nan
    except (TypeError, ValueError, KeyError):
        epf_std = np.nan
    return epf, epf_std


def _epf_gpu(row, num_frames: int) -> tuple[float, float]:
    """(energy_per_frame_J, std_per_frame_J) from a GPU stats row."""
    w = float(row["power_draw_w_mean"])
    d = float(row["duration_sec"])
    if not (np.isfinite(w) and np.isfinite(d) and d > 0 and num_frames > 0):
        return np.nan, np.nan
    epf = w * d / num_frames
    try:
        ws = float(row["power_draw_w_std"])
        epf_std = ws * d / num_frames if np.isfinite(ws) else np.nan
    except (TypeError, ValueError, KeyError):
        epf_std = np.nan
    return epf, epf_std


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def classify_robot_cpu_variant(row) -> str | None:
    backend = str(row.get("backend", "")).lower().strip()
    device  = str(row.get("device",  "")).lower().strip()

    if backend == "vaccel-local-ptc" and "cpu" in device:
        return VARIANTS[0]
    if backend == "vaccel-local-sol" and "cpu" in device:
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


def load_robot_cpu_rows(cpu_df: pd.DataFrame,
                        iso_perf: dict, e2e_perf: dict) -> list:
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_cpu_variant(r)
        if v is None:
            continue
        model   = str(r["model"]).strip()
        backend = str(r["backend"]).lower().strip()
        device  = _norm_device(str(r["device"]))

        if v in (VARIANTS[0], VARIANTS[1]):
            # local iso run — no remote_host
            n = lookup_frames(iso_perf, "robot", model, backend, "cpu")
        else:
            # e2e remote run — remote_host = edge-asus
            n = lookup_frames(e2e_perf, "robot", model, backend, device,
                              remote_host="edge-asus")

        if not n:
            print(f"[WARN] No perf entry for robot/{model}/{backend}/{device} — skipped")
            continue

        epf, epf_std = _epf_cpu(r, n)
        rows.append({"base_model": model, "variant": v,
                     "mean": epf, "std": epf_std})
    return rows


def load_edge_cpu_remote_rows(cpu_df: pd.DataFrame, e2e_perf: dict) -> list:
    rows = []
    for _, r in cpu_df.iterrows():
        host    = str(r.get("host",    "")).lower()
        backend = str(r.get("backend", "")).lower()
        device  = str(r.get("device",  "")).lower()
        if "edge" not in host or "cpu" not in device:
            continue
        model = str(r.get("model", "")).strip()

        # num_frames from robot perspective (same benchmark run, device=cpu)
        n = lookup_frames(e2e_perf, "robot", model, backend, "cpu",
                          remote_host="edge-asus")
        if not n:
            print(f"[WARN] No perf entry for robot/{model}/{backend}/cpu — skipped")
            continue

        epf, epf_std = _epf_cpu(r, n)
        if backend == "vaccel-remote-ptc":
            rows.append({"base_model": model, "variant": VARIANTS[2],
                         "mean": epf, "std": epf_std})
        elif backend == "vaccel-remote-sol":
            rows.append({"base_model": model, "variant": VARIANTS[3],
                         "mean": epf, "std": epf_std})
    return rows


def load_edge_gpu_remote_rows(gpu_df: pd.DataFrame, e2e_perf: dict) -> list:
    rows = []
    for _, r in gpu_df.iterrows():
        host    = str(r.get("host",    "")).lower()
        backend = str(r.get("backend", "")).lower()
        device  = str(r.get("device",  "")).lower()
        if "edge" not in host or "gpu" not in device:
            continue
        model = str(r.get("model", "")).strip()

        # num_frames from robot perspective (same benchmark run, device=gpu)
        n = lookup_frames(e2e_perf, "robot", model, backend, "gpu",
                          remote_host="edge-asus")
        if not n:
            print(f"[WARN] No perf entry for robot/{model}/{backend}/gpu — skipped")
            continue

        epf, epf_std = _epf_gpu(r, n)
        if backend == "vaccel-remote-ptc":
            rows.append({"base_model": model, "variant": VARIANTS[4],
                         "mean": epf, "std": epf_std})
        elif backend == "vaccel-remote-sol":
            rows.append({"base_model": model, "variant": VARIANTS[5],
                         "mean": epf, "std": epf_std})
    return rows


def _apply_model_filter(rows, allowed_models):
    return [r for r in rows if r["base_model"] in allowed_models]


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
    print("\n" + "=" * 95)
    print(f"{'DEBUG: ENERGY PER FRAME (J/frame)':^95}")
    print("=" * 95)
    for panel_key, ylabel, rows_data, vars_present in panels:
        print(f"\n--- {panel_key.upper()} ---")
        print(f"{'Model':<22} | {'Variant':<28} | {'J/frame':>10} | {'Std':>10}")
        print("-" * 80)
        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        std_map  = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
            std_map [(r["base_model"], r["variant"])] = r["std"]
        for m in base_models:
            for v in vars_present:
                val = mean_map[(m, v)]
                if np.isfinite(val):
                    s = std_map[(m, v)]
                    std_str = f"{s:10.4f}" if np.isfinite(s) else f"{'N/A':>10}"
                    print(f"{m:<22} | {v:<28} | {val:10.4f} | {std_str}")
    print("\n" + "=" * 95 + "\n")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_combined_epf(robot_rows, edge_cpu_rows, edge_gpu_rows):
    base_models = ordered_models(
        sorted({r["base_model"] for r in robot_rows + edge_cpu_rows + edge_gpu_rows})
    )

    sns.set_theme(
        context="paper", style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out",
            "font.family": "serif"},
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
        gridspec_kw={"width_ratios": widths, "wspace": 0.32, "hspace": 0.14},
    )

    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    panels = [
        ("robot",    "Robot CPU \n(Joules/frame)", robot_rows,     VARIANTS),
        ("edge_cpu", "Edge CPU \n(Joules/frame)",  edge_cpu_rows,  [VARIANTS[2], VARIANTS[3]]),
        ("edge_gpu", "Edge GPU \n(Joules/frame)",  edge_gpu_rows,  [VARIANTS[4], VARIANTS[5]]),
    ]

    print_debug_table(panels, base_models)

    def _rounded_ylim(y_max: float) -> float:
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

    # Per-panel y-limit (row x col independent)
    panel_ylim = {}
    for row_idx, (panel_key, ylabel, rows_data, vars_present) in enumerate(panels):
        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
        for col_idx, current_models in enumerate(cat_models):
            vals = np.asarray(
                [mean_map[(m, v)] for m in current_models for v in vars_present],
                dtype=float,
            )
            y_max = np.nanmax(vals) if np.any(np.isfinite(vals)) else np.nan
            panel_ylim[(row_idx, col_idx)] = (
                _rounded_ylim(float(y_max)) if np.isfinite(y_max) and y_max > 0 else 1.0
            )

    for row_idx, (panel_key, ylabel, rows_data, vars_present) in enumerate(panels):

        mean_map = {(m, v): np.nan for m in base_models for v in vars_present}
        std_map  = {(m, v): np.nan for m in base_models for v in vars_present}
        for r in rows_data:
            mean_map[(r["base_model"], r["variant"])] = r["mean"]
            std_map [(r["base_model"], r["variant"])] = r["std"]

        width, offsets = compute_offsets(vars_present, row_idx)

        for col_idx, current_models in enumerate(cat_models):
            ax = axes[row_idx, col_idx]
            x  = np.arange(len(current_models))

            ax.set_ylim(0, panel_ylim[(row_idx, col_idx)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_TICKS))
            ax.yaxis.set_major_formatter(FormatStrFormatter(TICK_FMT))

            for v in vars_present:
                xs    = x + offsets[v]
                means = np.asarray([mean_map[(m, v)] for m in current_models], dtype=float)
                stds  = np.asarray([std_map [(m, v)] for m in current_models], dtype=float)

                ax.bar(
                    xs, means, width=width,
                    color=color_map[v], edgecolor="black", linewidth=STROKE_WIDTH,
                    label=v if (row_idx == 0 and col_idx == 0) else "",
                    zorder=3,
                )

                if SHOW_ERROR_BARS:
                    yerr = np.where(np.isfinite(stds), stds, 0.0)
                    if np.any(yerr > 0):
                        ax.errorbar(xs, means, yerr=yerr, fmt="none",
                                    ecolor="black", elinewidth=STROKE_WIDTH,
                                    capsize=4, capthick=STROKE_WIDTH, zorder=10)

            # --- SAVINGS INDICATOR ---
            for i, m in enumerate(current_models):
                if row_idx == 0:
                    local_base_vars  = [VARIANTS[0], VARIANTS[1]]
                    compare_vars     = VARIANTS[2:]
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
                        if np.isfinite(mean_map.get((m, v), np.nan))
                        and mean_map.get((m, v), np.nan) > 0
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

                comps = [mean_map[(m, v)] for v in compare_vars
                         if np.isfinite(mean_map[(m, v)])]
                if not comps:
                    continue

                avg_comp    = np.mean(comps)
                savings_pct = (base_val - avg_comp) / base_val * 100
                print(f"Panel: {panel_key:<10} | Model: {m:<22} | "
                      f"Energy/frame savings: {savings_pct:>6.1f}%")

                if abs(savings_pct) <= 0.5:
                    continue

                is_reduction = savings_pct > 0
                badge_color  = "darkgreen" if is_reduction else "darkred"
                symbol       = "↓" if is_reduction else "↑"
                x_end        = i + offsets[vars_present[-1]]

                ax.hlines(
                    y=base_val, xmin=x_start, xmax=x_end,
                    colors=badge_color, linestyles="-.", linewidth=STROKE_WIDTH,
                    alpha=0.75, zorder=4,
                    label="Savings reference"
                    if (row_idx == 0 and col_idx == 0 and i == 0) else "",
                )

                arrow_offset = (base_val - avg_comp) * 0.15
                ax.annotate(
                    "", xy=(x_center_compare, avg_comp + arrow_offset),
                    xytext=(x_center_compare, base_val),
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
                        y_center = min(max(y_center, y_min_ax + y_pad),
                                       y_max_ax - y_pad)

                ax.text(
                    x_center_compare, y_center,
                    f"{symbol} {abs(savings_pct):.0f}%",
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

    # --- Layout ---
    CAPTION_OFFSET = 0.08

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.86)

    clean_handles = [
        mpatches.Patch(facecolor=color_map[v], edgecolor="none", linewidth=0.0)
        for v in VARIANTS
    ]
    clean_handles.append(
        Line2D([0], [0], color="darkgreen", linestyle="-.",
               linewidth=STROKE_WIDTH, alpha=0.75)
    )
    variant_labels = list(VARIANTS) + ["Energy savings"]

    leg = fig.legend(
        clean_handles, variant_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fancybox=False,
        edgecolor="black",
        prop={"size": plt.rcParams["font.size"] * 0.8}
    )
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_boxstyle("square", pad=0.3)

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[2, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[2, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        caption_artists.append(
            fig.text(x_center, y_caption, cap, ha="center", va="top")
        )

    fig.savefig(
        OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.02,
        bbox_extra_artists=(leg, *caption_artists),
    )
    print(f"[OK] Saved energy-per-frame plot to: {OUTPUT_FILE}")
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
        print(f"[WARNING] ISO CPU file not found — local robot rows will be missing: {iso_cpu_path}")
    gpu_df = pd.read_csv(gpu_path)

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    iso_perf = build_perf_map(ISO_PERF_FILE)
    e2e_perf = build_perf_map(E2E_PERF_FILE)
    print(f"Perf map: iso={len(iso_perf)} entries, e2e={len(e2e_perf)} entries")

    robot_rows    = load_robot_cpu_rows(cpu_df, iso_perf, e2e_perf)
    edge_cpu_rows = load_edge_cpu_remote_rows(cpu_df, e2e_perf)
    edge_gpu_rows = load_edge_gpu_remote_rows(gpu_df, e2e_perf)

    allowed = [m.strip() for m in MODEL_TYPE_ORDER]
    robot_rows    = _apply_model_filter(robot_rows,    allowed)
    edge_cpu_rows = _apply_model_filter(edge_cpu_rows, allowed)
    edge_gpu_rows = _apply_model_filter(edge_gpu_rows, allowed)

    if not robot_rows and not edge_cpu_rows and not edge_gpu_rows:
        raise SystemExit("ERROR: No rows remained after filtering!")

    plot_combined_epf(robot_rows, edge_cpu_rows, edge_gpu_rows)


if __name__ == "__main__":
    main()
