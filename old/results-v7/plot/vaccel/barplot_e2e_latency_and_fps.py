#!/usr/bin/env python3
"""
barplot_e2e_fps_and_latency.py

Combines FPS (top) and Latency Breakdown (middle & bottom) into a 4-row grid.
Row-normalized auto-zooming Y-axes across all rows, shared X-axes per column,
hidden inner Y-axis labels for cleanliness, unified legend.

Style (colors/legend/grid/font) lives in plot_style.py; data extraction lives
in plot_data_e2e.py. Both are shared with barplot_e2e_fps.py and
barplot_e2e_latency.py — change either module and all three plots stay in
sync. This script only owns the combined-figure layout.

Produces: e2e-fps-and-latency.pdf
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

OUTPUT_FILE = "e2e-latency-and-fps.pdf"
FIG_SIZE = (18, 15.0)  # 4-row layout
FONT_SCALE = 2.2 # tune this figure locally without changing the shared style default


def plot_combined(rows):
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
        4, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.20, "hspace": 0.12},
    )

    # Ensure axes is always 2D array [row, col]
    if len(cat_models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])

    # --- PRE-CALCULATE PER-PANEL Y-LIMITS (each subplot scales independently) ---
    panel_fps_limits = {}  # col_idx -> ylim
    panel_lat_limits = {}  # (r_idx, col_idx) -> ylim

    for col_idx, current_models in enumerate(cat_models):
        # Row 0: FPS — scale to the tallest bar in this column
        fps_vals = np.asarray(
            [fps_val_map[(m, v)] for m in current_models for v in variants], dtype=float
        )
        ymax = np.nanmax(fps_vals) if np.any(np.isfinite(fps_vals)) else 1.0
        panel_fps_limits[col_idx] = (ymax * 1.12) if ymax > 0 else 1.0

        # Rows 1-3: Latency — scale to tallest stacked bar + upper error bar
        for r_idx, v_present in [(1, variants[0:2]), (2, variants[2:4]), (3, variants[4:6])]:
            totals = np.asarray(
                [inf_map[(m, v)] + net_map[(m, v)] + pre_map[(m, v)] + upper_map[(m, v)]
                 for m in current_models for v in v_present],
                dtype=float,
            )
            ymax = np.nanmax(totals) if np.any(np.isfinite(totals)) else 1.0
            panel_lat_limits[(r_idx, col_idx)] = (ymax * 1.08) if ymax > 0 else 1.0

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))

        # ==========================================
        # --- ROW 0: FPS (All 6 Variants) ---
        # ==========================================
        ax_fps = axes[0, col_idx]
        row_0_vars = variants
        width_fps, offsets_fps = compute_offsets(row_0_vars, wide=False)

        ax_fps.set_ylim(0, panel_fps_limits[col_idx])

        for v in row_0_vars:
            xs = x + offsets_fps[v]
            vals = np.asarray([fps_val_map[(m, v)] for m in current_models], dtype=float)

            ax_fps.bar(
                xs, vals, width=width_fps, color=color_map[v],
                edgecolor="black", linewidth=STROKE_WIDTH,
                label=v if col_idx == 0 else "", zorder=3,
            )

        ax_fps.set_xticks(x)
        ax_fps.set_xticklabels([])
        style_axes(ax_fps)
        ax_fps.margins(x=0.005)

        if col_idx == 0:
            ax_fps.set_ylabel("Frame rate (FPS)")

        # ==========================================
        # --- ROWS 1-3: LATENCY BREAKDOWN ---
        # ==========================================
        latency_rows = [
            (1, axes[1, col_idx], variants[0:2], "Local CPU\nLatency (ms)"),
            (2, axes[2, col_idx], variants[2:4], "Remote CPU\nLatency (ms)"),
            (3, axes[3, col_idx], variants[4:6], "Remote GPU\nLatency (ms)"),
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

                # Bottom: Inference — solid color
                ax_lat.bar(
                    xs, inf_vals, width=width_lat, facecolor=base_color, edgecolor="black",
                    linewidth=STROKE_WIDTH, zorder=3
                )
                # Middle: Pre/Post-processing — light, dotted hatch
                ax_lat.bar(
                    xs, pre_vals, bottom=inf_vals, width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="..", zorder=3
                )
                # Top: Network + vAccel remoting layer — light, diagonal hatch
                ax_lat.bar(
                    xs, net_vals, bottom=(inf_vals + pre_vals), width=width_lat, facecolor=light_color,
                    edgecolor="black", linewidth=STROKE_WIDTH, hatch="//", zorder=3
                )

                ax_lat.errorbar(
                    xs, tot_vals, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10,
                )

            ax_lat.set_xticks(x)
            if row_idx == 3:
                ax_lat.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
            else:
                ax_lat.set_xticklabels([])

            style_axes(ax_lat)
            ax_lat.margins(x=0.005)

            if col_idx == 0:
                ax_lat.set_ylabel(ylabel)

    # ---- Manual Layout ----
    CAPTION_OFFSET = 0.075

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.10, top=0.84)

    color_handles, color_labels = axes[0, 0].get_legend_handles_labels()
    hatches = hatch_handles()

    # 3 rows x 3 cols (filled left-to-right, top-to-bottom):
    #   Row 1: Local CPU (Torch)   |  Remote CPU (Torch)  |  Remote GPU (Torch)
    #   Row 2: Local CPU (SOL)     |  Remote CPU (SOL)    |  Remote GPU (SOL)
    #   Row 3: Inference | Pre/Post-processing | Network + vAccel remoting layer
    combined_handles = [
        color_handles[0], color_handles[2], color_handles[4],
        color_handles[1], color_handles[3], color_handles[5],
        hatches["Inference"], hatches["Pre/Post-processing"], hatches["Network + vAccel remoting layer"],
    ]
    combined_labels = [
        color_labels[0], color_labels[2], color_labels[4],
        color_labels[1], color_labels[3], color_labels[5],
        "Inference", "Pre/Post-processing", "Network + vAccel remoting layer",
    ]

    leg = fig.legend(
        combined_handles, combined_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        **LEGEND_STYLE,
    )

    # Aligned (a)(b)(c) captions below the bottom row
    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes[3, :]) - CAPTION_OFFSET
    for ax, cap in zip(axes[3, :], cat_captions):
        if not cap:
            continue
        bbox = ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        t = fig.text(x_center, y_caption, cap, ha="center", va="top")
        caption_artists.append(t)

    fig.savefig(
        OUTPUT_FILE,
        bbox_extra_artists=(leg, *caption_artists),
    )
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    rows = load_rows()
    apply_theme(font_scale=FONT_SCALE)
    plot_combined(rows)


if __name__ == "__main__":
    main()
