#!/usr/bin/env python3
"""
barplot_local_latency_isolated.py

Local inference latency (p50 / median) from iso runs.
5 rows (deployment targets) × 3 columns (model categories).
Backends: ptc, sol, vaccel-remote-ptc, vaccel-remote-sol.
Missing data cells are left blank — no errors raised.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ── paths ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
INPUT_FILE_ISO   = _HERE / "../experiments/model-stats/_summary/iso_benchmark_summary.json"
INPUT_FILE_IROS2 = _HERE / "../experiments/model-stats/_summary/iros2_benchmark_summary.json"
OUTPUT_FILE = _HERE / "iso_local_latency.pdf"

# robot ptc/sol data comes from iros2 run; all other entries from iso
_ROBOT_IROS2_BACKENDS = {"ptc", "sol"}

# ── backend rename ───────────────────────────────────────────────────────────
BACKEND_MAP = {
    "ptc":               "Runtime JIT (torch.compile)",
    "sol":               "SOL",
    "vaccel-remote-ptc": "vAccel+Offline AOTI (.pt2)",
    "vaccel-remote-sol": "vAccel+SOL",
}
BACKENDS = list(BACKEND_MAP.values())   # display-order preserved

# ── rows: (host, device, row label) ─────────────────────────────────────────
ROWS = [
    ("robot",       "cpu", "Robot CPU"),
    ("edge-asus",   "cpu", "edge-asus CPU"),
    ("edge-asus",   "gpu", "edge-asus GPU"),
    # ("edge-xtreme", "cpu", "edge-xtreme CPU"),  # no data yet
    # ("edge-xtreme", "gpu", "edge-xtreme GPU"),  # no data yet
]

# ── columns: model categories ────────────────────────────────────────────────
CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "r3d_18", "r2plus1d_18"]
CAT_SEG   = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]

CATEGORIES = [
    ("(a) Image Classification",    CAT_IMAGE),
    ("(b) Video Action Recognition", CAT_VIDEO),
    ("(c) Semantic Segmentation",   CAT_SEG),
]

MODEL_DISPLAY = {
    "resnet50":            "ResNet-50",
    "swin_t":              "Swin-T",
    "swin_s":              "Swin-S",
    "swin_v2_b":           "SwinV2-B",
    "swin3d_s":            "Swin3D-S",
    "swin3d_b":            "Swin3D-B",
    "r3d_18":              "R3D-18",
    "r2plus1d_18":         "R(2+1)D-18",
    "deeplabv3_resnet50":  "DLv3-R50",
    "deeplabv3_resnet101": "DLv3-R101",
    "fcn_resnet50":        "FCN-R50",
    "fcn_resnet101":       "FCN-R101",
}

# ── style ────────────────────────────────────────────────────────────────────
BAR_WIDTH       = 0.18
SPINES_LW       = 0.8
FONT_SCALE      = 1.5
SHOW_BAR_VALUES  = True
SHOW_ERROR_BARS  = True   # std dev for mean, IQR for median
METRIC           = "mean"   # "median" (p50) | "mean"


# ── helpers ──────────────────────────────────────────────────────────────────
def _parse_runs(raw: dict, *, robot_only: bool = False, skip_robot: bool = False) -> dict:
    out = {}
    for run in raw.get("runs", []):
        host    = str(run.get("host",    "")).strip().lower()
        device  = str(run.get("device",  "")).strip().lower()
        backend = str(run.get("backend", "")).strip().lower()
        model   = str(run.get("model",   "")).strip()
        label   = BACKEND_MAP.get(backend)
        if label is None:
            continue
        if robot_only and not (host == "robot" and backend in _ROBOT_IROS2_BACKENDS):
            continue
        if skip_robot and (host == "robot" and backend in _ROBOT_IROS2_BACKENDS):
            continue
        inf = run.get("inference_ms") or {}
        stat_key = "p50" if METRIC == "median" else "mean"
        value = inf.get(stat_key)
        if value is None:
            continue
        value = float(value)
        if METRIC == "median":
            p25 = float(inf.get("p25", value))
            p75 = float(inf.get("p75", value))
            err_low, err_high = value - p25, p75 - value
        else:
            std = float(inf.get("std", 0.0))
            err_low = err_high = std
        out[(host, device, label, model)] = (value, err_low, err_high)
    return out


def load_data(iso_path: Path, iros2_path: Path) -> dict:
    """Returns {(host, device, backend_label, model): median_ms}.

    Robot ptc/sol entries come from iros2; everything else from iso.
    """
    with iso_path.open() as f:
        data = _parse_runs(json.load(f), skip_robot=True)

    if iros2_path.exists():
        with iros2_path.open() as f:
            data.update(_parse_runs(json.load(f), robot_only=True))
    else:
        print(f"[warn] iros2 summary not found: {iros2_path} — robot ptc/sol will be blank")

    return data


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="gray")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINES_LW)
    ax.spines["bottom"].set_linewidth(SPINES_LW)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    path = INPUT_FILE_ISO.resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    data = load_data(INPUT_FILE_ISO.resolve(), INPUT_FILE_IROS2.resolve())

    # ── debug print ───────────────────────────────────────────────────────
    stat_key = "p50" if METRIC == "median" else "mean"
    print(f"\n{'host':<14} {'device':<6} {'backend':<36} {'model':<24} {stat_key:>8}  err_low  err_hi")
    print("-" * 105)
    for (host, device, backend, model), (val, el, eh) in sorted(data.items()):
        print(f"{host:<14} {device:<6} {backend:<36} {model:<24} {val:>8.2f}  {el:>7.2f}  {eh:>6.2f}")
    print()

    # Paired colormap: light/dark pairs — ptc=lighter, sol=darker
    paired = plt.cm.get_cmap("Paired")
    color_map = {
        "Runtime JIT (torch.compile)": paired(0 / 11),   # light blue
        "SOL":                          paired(1 / 11),   # dark blue
        "vAccel+Offline AOTI (.pt2)":   paired(2 / 11),   # light green
        "vAccel+SOL":                   paired(3 / 11),   # dark green
    }

    n_rows = len(ROWS)
    n_cols = len(CATEGORIES)

    plt.rcParams.update({
        "font.family":      "serif",
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "font.size":        9 * FONT_SCALE,
        "axes.labelsize":   9 * FONT_SCALE,
        "xtick.labelsize":  8 * FONT_SCALE,
        "ytick.labelsize":  8 * FONT_SCALE,
        "legend.fontsize":  8.5 * FONT_SCALE,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, n_rows * 2.6),
        sharey=False,
        gridspec_kw={
            "wspace":        0.05,
            "hspace":        0.38,
            "width_ratios":  [len(c[1]) for c in CATEGORIES],
        },
    )

    _nan_entry = (np.nan, 0.0, 0.0)

    # per-row shared y-max (so categories in the same row are comparable)
    # include bar + upper error so whiskers never overflow the axes
    row_ymaxes = []
    for host, device, _ in ROWS:
        tops = [
            data[(host, device, b, m)][0] + (data[(host, device, b, m)][2] if SHOW_ERROR_BARS else 0.0)
            for _, models in CATEGORIES
            for m in models
            for b in BACKENDS
            if (host, device, b, m) in data
        ]
        row_ymaxes.append(max(tops) * 1.12 if tops else 1.0)

    # ── draw ──────────────────────────────────────────────────────────────
    n_b = len(BACKENDS)
    span = BAR_WIDTH * n_b
    offsets = np.linspace(-span / 2 + BAR_WIDTH / 2, span / 2 - BAR_WIDTH / 2, n_b)

    for row_idx, (host, device, row_label) in enumerate(ROWS):
        for col_idx, (cat_title, models) in enumerate(CATEGORIES):
            ax = axes[row_idx, col_idx]
            x  = np.arange(len(models))

            any_data = False
            for b_idx, backend in enumerate(BACKENDS):
                tuples = [data.get((host, device, backend, m), _nan_entry) for m in models]
                vals    = np.array([t[0] for t in tuples], dtype=float)
                err_low = np.array([t[1] for t in tuples], dtype=float)
                err_hi  = np.array([t[2] for t in tuples], dtype=float)
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                any_data = True
                xs = x[valid] + offsets[b_idx]
                bars = ax.bar(
                    xs, vals[valid],
                    width=BAR_WIDTH,
                    color=color_map[backend],
                    edgecolor="white", linewidth=0.3,
                    label=backend if (row_idx == 0 and col_idx == 0) else "",
                    zorder=3,
                )
                if SHOW_ERROR_BARS:
                    ax.errorbar(
                        xs, vals[valid],
                        yerr=[err_low[valid], err_hi[valid]],
                        fmt="none", color="black",
                        capsize=2, linewidth=0.8, zorder=5,
                    )
                if SHOW_BAR_VALUES:
                    for bar, v in zip(bars, vals[valid]):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() / 2,
                            f"{v:.0f}",
                            ha="center", va="center",
                            fontsize=7.5, color="black",
                            fontweight="bold", rotation=90,
                            zorder=4,
                        )

            ax.set_ylim(0, row_ymaxes[row_idx])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.set_xticks(x)
            ax.margins(x=0.04)
            style_ax(ax)

            # x-tick labels on bottom row only
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [MODEL_DISPLAY.get(m, m) for m in models],
                    rotation=20, ha="right",
                )
            else:
                ax.set_xticklabels([])

            # y-axis label on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(row_label)
            else:
                ax.tick_params(labelleft=False)

            # category caption below bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(cat_title, labelpad=10)

            # blank cell indicator
            if not any_data:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="gray", fontstyle="italic")

    metric_label = "Median" if METRIC == "median" else "Mean"
    if SHOW_ERROR_BARS:
        error_label = " ± IQR" if METRIC == "median" else " ± Std Dev"
    else:
        error_label = ""
    fig.suptitle(f"Local Inference Latency ({metric_label}{error_label}, ms)", fontsize=10 * FONT_SCALE, fontweight="bold", y=1.02)

    # ── legend ────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=color_map[b], label=b) for b in BACKENDS]
    fig.legend(
        handles, BACKENDS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=n_b,
        frameon=True, framealpha=0.92,
        borderpad=0.4, handlelength=1.3,
        columnspacing=1.0,
    )

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] Saved: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
