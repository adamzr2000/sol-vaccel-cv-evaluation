#!/usr/bin/env python3
"""
plot_style.py

Single source of truth for the visual style shared by the e2e FPS/latency
barplots (barplot_e2e_fps_and_latency.py, barplot_e2e_fps.py,
barplot_e2e_latency.py). Change a color, the legend look, grid, or font
HERE and every script that imports this module picks it up — nothing to
replicate by hand across files.

Not a generic plotting library: the VARIANTS list and color assignment are
specific to this project's 6 deployment scenarios (Local/Remote CPU/GPU x
Torch/SOL). Add a plot that uses the same scenarios and it plugs in for
free; a genuinely different chart should define its own style module.
"""
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# --- Typography / line weights ---------------------------------------
FONT_SCALE = 1.8
SPINES_WIDTH = 1.0
STROKE_WIDTH = 0.7
LATENCY_MAX_TICKS = 5
LATENCY_TICK_FMT = "%.0f"

# --- Deployment scenario order (fixed — never re-derive/cycle this) ---
VARIANTS = [
    "Local CPU (vAccel + Torch)",
    "Local CPU (vAccSOL)",
    "Remote CPU (vAccel + Torch)",
    "Remote CPU (vAccSOL)",
    "Remote GPU (vAccel + Torch)",
    "Remote GPU (vAccSOL)",
]

# Composition breakdown, in stacking order (bottom -> top)
LATENCY_COMPONENTS = ["Inference", "Pre/Post-processing", "Network + vAccel remoting layer"]

LEGEND_STYLE = dict(
    frameon=True,
    framealpha=0.9,
    borderpad=0.4,
    handlelength=1.4,
    fancybox=False,
    edgecolor="black",
)


def apply_theme(font_scale: float | None = None):
    """Call once per script, before building any figure.

    font_scale defaults to the shared FONT_SCALE above; pass an explicit value
    to override for a single script (e.g. a standalone figure with more room
    to breathe than the dense combined one) without touching the shared default.
    """
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "out", "ytick.direction": "out", "font.family": "serif"},
        font_scale=font_scale if font_scale is not None else FONT_SCALE,
    )
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    plt.rcParams["hatch.linewidth"] = STROKE_WIDTH


def get_color_map():
    """Paired palette: light/dark pair per scenario (blue=Local, green=Remote CPU,
    red=Remote GPU; light=vAccel+Torch, dark=vAccSOL). Fixed order, not cycled."""
    paired_pal = sns.color_palette("Paired", 6)
    return {v: paired_pal[i] for i, v in enumerate(VARIANTS)}


def light(color, alpha=0.4):
    """Lighter, semi-transparent variant of a base color (used for stacked overhead segments)."""
    return mcolors.to_rgba(color, alpha=alpha)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def compute_offsets(variants_present, wide: bool):
    """Bar x-offsets for a group of variants sharing one model's tick.
    wide=False: many variants side by side (e.g. FPS row, all 6). wide=True:
    fewer, wider bars (e.g. one latency row, 2 variants: Torch vs SOL)."""
    n = len(variants_present)
    width = 0.35 if wide else 0.12
    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def hatch_handles():
    """Legend patches for the latency composition breakdown (neutral gray, not tied
    to any scenario color, so they read as 'component' rather than 'scenario')."""
    base = "lightgray"
    return {
        "Inference": mpatches.Patch(facecolor=base, edgecolor="black", hatch="", label="Inference"),
        "Pre/Post-processing": mpatches.Patch(
            facecolor=light(base), edgecolor="black", hatch="..", label="Pre/Post-processing"
        ),
        "Network + vAccel remoting layer": mpatches.Patch(
            facecolor=light(base), edgecolor="black", hatch="//", label="Network + vAccel remoting layer"
        ),
    }
