#!/usr/bin/env python3
"""
barplot_e2e_latency.py

Standalone E2E latency composition breakdown (inference / pre-post / network)
across the three remoting scenarios (Local CPU, Remote CPU, Remote GPU),
split out of barplot_e2e_fps_and_latency.py's bottom three rows — see
barplot_e2e_fps.py for the companion throughput figure.

Style (colors/legend/grid/font) lives in plot_style.py; data extraction
lives in plot_data_e2e.py. Both are shared with the other two plots in this
family — change either module and all three stay in sync.

Produces: e2e-latency.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from plot_config import get_model_display_name
from plot_style import (
    apply_theme, get_color_map, style_axes, compute_offsets, hatch_handles,
    VARIANTS, LEGEND_STYLE, STROKE_WIDTH, LATENCY_MAX_TICKS, LATENCY_TICK_FMT,
)
from plot_data_e2e import load_rows, filter_to_known_models, build_value_maps, print_debug_info

OUTPUT_FILE = "e2e-latency.pdf"
FIG_SIZE = (18, 11.5)  # 3-row layout


def plot_latency(rows):
    if not rows:
        print("No matching rows found.")
        return

    rows, base_models, cat_models, cat_captions = filter_to_known_models(rows)
    variants = VARIANTS

    fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map = build_value_maps(
        rows, base_models, variants
    )
    print_debug_info(base_models, variants, fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map)

    color_map = get_color_map()
    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        3, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.14, "hspace": 0.14},
    )
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    # --- PRE-CALCULATE PER-PANEL Y-LIMITS ---
    panel_lat_limits = {}  # (row_idx, col_idx) -> ylim
    for col_idx, current_models in enumerate(cat_models):
        for row_idx, v_present in [(0, variants[0:2]), (1, variants[2:4]), (2, variants[4:6])]:
            totals = np.asarray(
                [inf_map[(m, v)] + net_map[(m, v)] + pre_map[(m, v)] + upper_map[(m, v)]
                 for m in current_models for v in v_present],
                dtype=float,
            )
            ymax = np.nanmax(totals) if np.any(np.isfinite(totals)) else 1.0
            panel_lat_limits[(row_idx, col_idx)] = (ymax * 1.08) if ymax > 0 else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        latency_rows = [
            (0, axes[0, col_idx], variants[0:2], "Local CPU\nE2E Latency (ms)"),
            (1, axes[1, col_idx], variants[2:4], "Remote CPU\nE2E Latency (ms)"),
            (2, axes[2, col_idx], variants[4:6], "Remote GPU\nE2E Latency (ms)"),
        ]

        for row_idx, ax_lat, vars_present, ylabel in latency_rows:
            width_lat, offsets_lat = compute_offsets(vars_present, wide=True)

            ax_lat.set_ylim(0, panel_lat_limits[(row_idx, col_idx)])
            ax_lat.yaxis.set_major_locator(MaxNLocator(nbins=LATENCY_MAX_TICKS))
            ax_lat.yaxis.set_major_formatter(FormatStrFormatter(LATENCY_TICK_FMT))

            for v in vars_present:
                xs = x + offsets_lat[v]
                inf_vals = np.asarray([inf_map[(m, v)] for m in current_models], dtype=float)
                net_vals = np.asarray([net_map[(m, v)] for m in current_models], dtype=float)
                pre_vals = np.asarray([pre_map[(m, v)] for m in current_models], dtype=float)
                yerr = np.array([
                    [lower_map[(m, v)] for m in current_models],
                    [upper_map[(m, v)] for m in current_models],
                ], dtype=float)

                tot_vals = inf_vals + net_vals + pre_vals

                base_color = color_map[v]
                light_color = mcolors.to_rgba(base_color, alpha=0.4)

                ax_lat.bar(
                    xs, inf_vals, width=width_lat, facecolor=base_color, edgecolor="black",
                    linewidth=STROKE_WIDTH, zorder=3
                )
                ax_lat.bar(
                    xs, pre_vals, bottom=inf_vals, width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="..", zorder=3
                )
                ax_lat.bar(
                    xs, net_vals, bottom=(inf_vals + pre_vals), width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="//", zorder=3
                )

                ax_lat.errorbar(
                    xs, tot_vals, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10,
                )

            ax_lat.set_xticks(x)
            if row_idx == 2:
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])

            style_axes(ax_lat)
            ax_lat.margins(x=0.005)

            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)

    CAPTION_OFFSET = 0.09
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.80)

    # This figure has no FPS row (no bar sets label=), so build the 6 scenario
    # swatches directly from color_map rather than pulling handles off an axis.
    # Same "Torch triplet, then SOL triplet" grouping as the combined figure.
    import matplotlib.patches as mpatches
    torch_vars = [variants[0], variants[2], variants[4]]
    sol_vars = [variants[1], variants[3], variants[5]]
    color_handles = [mpatches.Patch(facecolor=color_map[v], edgecolor="black") for v in torch_vars + sol_vars]
    color_labels = torch_vars + sol_vars

    hatches = hatch_handles()
    combined_handles = color_handles + [
        hatches["Inference"], hatches["Pre/Post-processing"], hatches["Network + vAccel remoting layer"],
    ]
    combined_labels = color_labels + [
        "Inference", "Pre/Post-processing", "Network + vAccel remoting layer",
    ]

    leg = fig.legend(
        combined_handles, combined_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        **LEGEND_STYLE,
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[2, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[2, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(OUTPUT_FILE, bbox_extra_artists=(leg, *caption_artists))
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    rows = load_rows()
    apply_theme()
    plot_latency(rows)


if __name__ == "__main__":
    main()
