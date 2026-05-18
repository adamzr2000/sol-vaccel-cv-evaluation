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
INPUT_FILE = _HERE / "../experiments/model-stats/_summary/iso_benchmark_summary.json"
OUTPUT_FILE = _HERE / "iso_local_latency.pdf"

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
    ("edge-xtreme", "cpu", "edge-xtreme CPU"),
    ("edge-xtreme", "gpu", "edge-xtreme GPU"),
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
BAR_WIDTH   = 0.18
SPINES_LW   = 0.8
FONT_SCALE  = 1.1


# ── helpers ──────────────────────────────────────────────────────────────────
def load_data(path: Path) -> dict:
    """Returns {(host, device, backend_label, model): median_ms}."""
    with path.open() as f:
        raw = json.load(f)
    out = {}
    for run in raw.get("runs", []):
        host    = str(run.get("host",    "")).strip().lower()
        device  = str(run.get("device",  "")).strip().lower()
        backend = str(run.get("backend", "")).strip().lower()
        model   = str(run.get("model",   "")).strip()
        label   = BACKEND_MAP.get(backend)
        if label is None:
            continue
        median = (run.get("inference_ms") or {}).get("p50")
        if median is None:
            continue
        out[(host, device, label, model)] = float(median)
    return out


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
    path = INPUT_FILE.resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    data = load_data(path)

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

    # per-row shared y-max (so categories in the same row are comparable)
    row_ymaxes = []
    for host, device, _ in ROWS:
        vals = [
            data[(host, device, b, m)]
            for _, models in CATEGORIES
            for m in models
            for b in BACKENDS
            if (host, device, b, m) in data
        ]
        row_ymaxes.append(max(vals) * 1.12 if vals else 1.0)

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
                vals = np.array(
                    [data.get((host, device, backend, m), np.nan) for m in models],
                    dtype=float,
                )
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                any_data = True
                xs = x[valid] + offsets[b_idx]
                ax.bar(
                    xs, vals[valid],
                    width=BAR_WIDTH,
                    color=color_map[backend],
                    edgecolor="white", linewidth=0.3,
                    label=backend if (row_idx == 0 and col_idx == 0) else "",
                    zorder=3,
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
                ax.set_ylabel(f"{row_label} (ms)")
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

    # ── legend ────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=color_map[b], label=b) for b in BACKENDS]
    fig.legend(
        handles, BACKENDS,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=n_b,
        frameon=True, framealpha=0.92,
        borderpad=0.5, handlelength=1.3,
        columnspacing=1.0,
    )

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"[OK] Saved: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
