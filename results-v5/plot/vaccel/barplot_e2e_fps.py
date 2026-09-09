#!/usr/bin/env python3
"""
barplot_e2e_fps.py

Standalone FPS comparison across all 6 deployment scenarios (split out of
barplot_e2e_fps_and_latency.py's top row for a cleaner, independently
sized/captioned figure — see barplot_e2e_latency.py for the companion
latency breakdown).

Style (colors/legend/grid/font) lives in plot_style.py; data extraction
lives in plot_data_e2e.py. Both are shared with the other two plots in this
family — change either module and all three stay in sync.

Produces: e2e-fps.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_config import get_model_display_name
from plot_style import apply_theme, get_color_map, style_axes, compute_offsets, VARIANTS, LEGEND_STYLE
from plot_data_e2e import load_rows, filter_to_known_models, build_value_maps, print_debug_info

OUTPUT_FILE = "e2e-fps.pdf"
FIG_SIZE = (18, 4.6)
FONT_SCALE = 2.2  # this standalone figure has more room than the dense combined one


def plot_fps(rows):
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
        1, len(cat_models),
        figsize=FIG_SIZE,
        sharey=False,
        gridspec_kw={"width_ratios": widths, "wspace": 0.14},
    )
    if len(cat_models) == 1:
        axes = np.array([axes])

    width, offsets = compute_offsets(variants, wide=False)

    for col_idx, current_models in enumerate(cat_models):
        x = np.arange(len(current_models))
        ax = axes[col_idx]

        fps_vals_all = np.asarray(
            [fps_val_map[(m, v)] for m in current_models for v in variants], dtype=float
        )
        ymax = np.nanmax(fps_vals_all) if np.any(np.isfinite(fps_vals_all)) else 1.0
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)

        for v in variants:
            xs = x + offsets[v]
            vals = np.asarray([fps_val_map[(m, v)] for m in current_models], dtype=float)
            ax.bar(
                xs, vals, width=width, color=color_map[v],
                edgecolor="none", linewidth=0.0,
                label=v if col_idx == 0 else "", zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in current_models], rotation=15, ha="right")
        style_axes(ax)
        ax.margins(x=0.005)

        if col_idx == 0:
            ax.set_ylabel("Frame rate (FPS)")

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.72)

    color_handles, color_labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        color_handles, color_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        **LEGEND_STYLE,
    )

    caption_artists = []
    y_caption = min(ax.get_position().y0 for ax in axes) - 0.18
    for ax, cap in zip(axes, cat_captions):
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
    apply_theme(font_scale=FONT_SCALE)
    plot_fps(rows)


if __name__ == "__main__":
    main()
