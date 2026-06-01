#!/usr/bin/env python3
"""
barplot_local_latency_isolated.py

Local inference latency (mean ± std dev) from iso runs.
3 rows (deployment targets) × 3 columns (model categories).

7 backends — ordering: PTC first (JIT baseline), then AOTI family, then SOL family:
  PTC  |  AOTI · vAccel Local+PTC · vAccel Remote+PTC†  |  SOL · vAccel Local+SOL · vAccel Remote+SOL†

† vAccel Remote: agent co-located on the same host (loopback).
Missing data cells are left blank — no errors raised.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ── paths ────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
INPUT_FILE  = _HERE / "../experiments/model-stats/_summary/iso_benchmark_summary.json"
OUTPUT_FILE = _HERE / "iso_local_latency.pdf"

# ── backend ordering and display names ───────────────────────────────────────
# PTC first (de-emphasised JIT baseline); then AOTI/vAccel-PTC family;
# then SOL/vAccel-SOL family — keeps natural "native vs. vAccel" comparisons adjacent.
BACKEND_MAP = {
    "ptc":               "JIT-Inductor",
    "aoti":              "AOT-Inductor",
    "vaccel-local-ptc":  "vAccel Local (AOT-Inductor)",
    "vaccel-remote-ptc": "vAccel RPC (AOT-Inductor)",
    "sol":               "SOL",
    "vaccel-local-sol":  "vAccel Local (SOL)",
    "vaccel-remote-sol": "vAccel RPC (SOL)",
}
BACKENDS = list(BACKEND_MAP.values())

# ── rows: (host, device, y-axis label) ───────────────────────────────────────
ROWS = [
    ("robot",     "cpu", "Robot CPU"),
    ("edge-asus", "cpu", "edge-asus CPU"),
    ("edge-asus", "gpu", "edge-asus GPU"),
    # ("edge-xtreme",    "cpu", "edge-xtreme CPU"),    # no data yet
    # ("edge-xtreme",    "gpu", "edge-xtreme GPU"),    # no data yet
    # ("edge-alienware", "cpu", "edge-alienware CPU"), # no data yet
    # ("edge-alienware", "gpu", "edge-alienware GPU"), # no data yet
]

# ── model categories ─────────────────────────────────────────────────────────
CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "r3d_18", "r2plus1d_18"]
CAT_SEG   = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]

CATEGORIES = [
    ("(a) Image Classification",     CAT_IMAGE),
    ("(b) Video Action Recognition",  CAT_VIDEO),
    ("(c) Semantic Segmentation",     CAT_SEG),
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

# ── style params ──────────────────────────────────────────────────────────────
BAR_WIDTH       = 0.11
SPINES_LW       = 0.8
FONT_SCALE      = 1.8
SHOW_BAR_VALUES = False
SHOW_ERROR_BARS = True
METRIC          = "median"   # "median" | "mean"

# ── color scheme ──────────────────────────────────────────────────────────────
# PTC:  neutral gray (de-emphasised baseline)
# AOTI family: blue gradient  dark → medium → light
# SOL  family: green gradient dark → medium → light
# Remote variants additionally carry /// hatching to signal indirect execution.
COLOR_MAP = {
    "JIT-Inductor":              "#9e9e9e",   # neutral gray
    "AOT-Inductor":              "#1a6dbf",   # dark blue
    "vAccel Local (AOT-Inductor)": "#5b9bd5", # medium blue
    "vAccel RPC (AOT-Inductor)": "#9fc5e8",   # light blue
    "SOL":                       "#2d7a3e",   # dark green
    "vAccel Local (SOL)":        "#5aab6e",   # medium green
    "vAccel RPC (SOL)":          "#9fd4a8",   # light green
}

HATCH_MAP = {
    "vAccel RPC (AOT-Inductor)": "///",
    "vAccel RPC (SOL)":          "///",
}
HATCH_EDGE_MAP = {
    "vAccel RPC (AOT-Inductor)": "#1a6dbf",   # dark blue hatch on light blue
    "vAccel RPC (SOL)":          "#2d7a3e",   # dark green hatch on light green
}


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(path: Path) -> dict:
    """Return {(host, device, backend_label, model): (value, err_low, err_high)}."""
    stat_key = "p50" if METRIC == "median" else "mean"
    out: dict = {}
    with path.open() as f:
        raw = json.load(f)
    for run in raw.get("runs", []):
        host    = str(run.get("host",    "")).strip().lower()
        device  = str(run.get("device",  "")).strip().lower()
        backend = str(run.get("backend", "")).strip().lower()
        model   = str(run.get("model",   "")).strip()
        label   = BACKEND_MAP.get(backend)
        if label is None:
            continue
        inf   = run.get("inference_ms") or {}
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


# ── axis styling ──────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="gray")
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINES_LW)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    path = INPUT_FILE.resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    data = load_data(path)

    # debug table
    stat_key = "p50" if METRIC == "median" else "mean"
    print(f"\n{'host':<14} {'device':<6} {'backend':<36} {'model':<24} {stat_key:>8}  err_low  err_hi")
    print("-" * 105)
    for (host, device, backend, model), (val, el, eh) in sorted(data.items()):
        print(f"{host:<14} {device:<6} {backend:<36} {model:<24} {val:>8.2f}  {el:>7.2f}  {eh:>6.2f}")
    print()

    n_rows = len(ROWS)
    n_cols = len(CATEGORIES)
    n_b    = len(BACKENDS)

    plt.rcParams.update({
        "font.family":     "serif",
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
        "font.size":       9  * FONT_SCALE,
        "axes.labelsize":  9  * FONT_SCALE,
        "xtick.labelsize": 8  * FONT_SCALE,
        "ytick.labelsize": 8  * FONT_SCALE,
        "legend.fontsize": 8  * FONT_SCALE,
    })

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(22, n_rows * 3.0),
        sharey=False,
        gridspec_kw={
            "wspace":       0.18,
            "hspace":       0.42,
            "width_ratios": [len(c[1]) for c in CATEGORIES],
        },
    )

    _nan_entry = (np.nan, 0.0, 0.0)
    span    = BAR_WIDTH * n_b
    offsets = np.linspace(-span / 2 + BAR_WIDTH / 2, span / 2 - BAR_WIDTH / 2, n_b)

    # per-panel y ceiling — each subplot scales independently to maximise bar visibility
    panel_ymaxes: dict = {}
    for host, device, _ in ROWS:
        for cat_title, models in CATEGORIES:
            tops = [
                data[(host, device, b, m)][0]
                + (data[(host, device, b, m)][2] if SHOW_ERROR_BARS else 0.0)
                for m in models
                for b in BACKENDS
                if (host, device, b, m) in data
            ]
            panel_ymaxes[(host, device, cat_title)] = max(tops) * 1.12 if tops else 1.0

    for row_idx, (host, device, row_label) in enumerate(ROWS):
        for col_idx, (cat_title, models) in enumerate(CATEGORIES):
            ax = axes[row_idx, col_idx]
            x  = np.arange(len(models))
            any_data = False

            # bar_pos[(backend, model_idx)] = (x_center, y_val) — used for RPC connectors
            bar_pos: dict = {}

            for b_idx, backend in enumerate(BACKENDS):
                tuples  = [data.get((host, device, backend, m), _nan_entry) for m in models]
                vals    = np.array([t[0] for t in tuples], dtype=float)
                err_low = np.array([t[1] for t in tuples], dtype=float)
                err_hi  = np.array([t[2] for t in tuples], dtype=float)
                valid   = ~np.isnan(vals)
                if not valid.any():
                    continue
                any_data = True

                hatch = HATCH_MAP.get(backend, "")
                hedge = HATCH_EDGE_MAP.get(backend, "white")
                xs    = x[valid] + offsets[b_idx]

                bars = ax.bar(
                    xs, vals[valid],
                    width=BAR_WIDTH,
                    color=COLOR_MAP[backend],
                    hatch=hatch,
                    edgecolor=hedge,
                    linewidth=0.3,
                    label=backend if (row_idx == 0 and col_idx == 0) else "",
                    zorder=3,
                )

                # track bar tops for connector lines
                for mi, xc, v in zip(np.where(valid)[0], xs, vals[valid]):
                    bar_pos[(backend, int(mi))] = (float(xc), float(v))

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
                            fontsize=6.5, color="black",
                            fontweight="bold", rotation=90,
                            zorder=4,
                        )

            # ── RPC trend connectors ───────────────────────────────────────
            # Thin lines linking each native backend bar top to its vAccel RPC
            # counterpart: solid = RPC faster, dashed = RPC adds overhead.
            for nat, rpc, col in [
                ("vAccel Local (AOT-Inductor)", "vAccel RPC (AOT-Inductor)", "#0d47a1"),
                ("vAccel Local (SOL)",          "vAccel RPC (SOL)",           "#1b5e20"),
            ]:
                for mi in range(len(models)):
                    p1 = bar_pos.get((nat, mi))
                    p2 = bar_pos.get((rpc, mi))
                    if p1 is None or p2 is None:
                        continue
                    x1, y1 = p1
                    x2, y2 = p2
                    ls = "--" if y2 > y1 else "-"
                    ax.plot([x1, x2], [y1, y2],
                            color=col, linewidth=1.1, linestyle=ls,
                            alpha=0.72, zorder=6)

            ax.set_ylim(0, panel_ymaxes[(host, device, cat_title)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.set_xticks(x)
            ax.margins(x=0.03)
            style_ax(ax)

            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [MODEL_DISPLAY.get(m, m) for m in models],
                    rotation=20, ha="right",
                )
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(row_label)

            if row_idx == n_rows - 1:
                ax.set_xlabel(cat_title, labelpad=10)

            if not any_data:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="gray", fontstyle="italic")

    metric_label = "Median" if METRIC == "median" else "Mean"
    error_label  = (" ± IQR" if METRIC == "median" else " ± Std Dev") if SHOW_ERROR_BARS else ""
    fig.suptitle(
        f"Local Inference Latency ({metric_label}{error_label}, ms)",
        fontsize=10 * FONT_SCALE, fontweight="bold", y=1.02,
    )

    # PGF/TikZ-style legend — backends + connector-line key
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(
            facecolor=COLOR_MAP[b],
            hatch=HATCH_MAP.get(b, ""),
            edgecolor=HATCH_EDGE_MAP.get(b, "black"),
            label=b,
        )
        for b in BACKENDS
    ]
    handles += [
        Line2D([], [], color="dimgray", linewidth=1.1, linestyle="-",
               label="Local→RPC: speedup"),
        Line2D([], [], color="dimgray", linewidth=1.1, linestyle="--",
               label="Local→RPC: overhead"),
    ]
    labels = BACKENDS + ["Local→RPC: speedup", "Local→RPC: overhead"]
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=len(handles),
        frameon=True, framealpha=0.9, fancybox=False, edgecolor="black",
        borderpad=0.5, handlelength=1.6, handletextpad=0.5,
        columnspacing=0.9,
    )

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"[OK] Saved: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
