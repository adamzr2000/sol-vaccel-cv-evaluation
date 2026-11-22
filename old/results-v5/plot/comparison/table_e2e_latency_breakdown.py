#!/usr/bin/env python3
"""
table_e2e_latency_breakdown.py (comparison, paper-facing)

Same model x scenario x framework/backend layout as the E2E latency table in
table_e2e_fps_and_latency_simplified.py, but each cell additionally breaks
the total down into its three components -- Inference / Pre-Post-processing
/ Net+Framework Overhead (same three categories, same order, as the stacked
bars in barplot_e2e_latency.py) -- shown in ms as a small second line under
the bold total, e.g.:
    151.4
    72/3/77
Lowest total per model x scenario group is still highlighted with the same
light-green cell fill as table_e2e_fps_and_latency_simplified.py.

Table shell (headers, cell layout, resizebox, category grouping) lives in
table_style_e2e.py -- shared with the other tables built the same way.

Produces: e2e-latency-breakdown-table.tex
"""
from __future__ import annotations

import barplot_e2e_latency_and_fps as src
import table_style_e2e as tstyle

OUTPUT_FILE = "e2e-latency-breakdown-table.tex"


def main():
    merged = src.load_merged_rows()
    _base_models, cat_models, cat_captions, _fps_map, inf_map, pre_map, net_map, _lo_map, _hi_map = src.build_maps(merged)

    tstyle.print_breakdown_debug_info(
        "DEBUG: E2E LATENCY BREAKDOWN (ms) -- total (inf/pre/net)",
        cat_models, inf_map, pre_map, net_map, decimals=1,
    )

    table = tstyle.build_breakdown_table(
        cat_models, cat_captions, inf_map, pre_map, net_map,
        caption=(
            r"End-to-end (E2E) latency (ms) of the robot vision pipeline for a "
            r"single-camera input, measured from image capture to result delivery "
            r"(lower is better; robot as client, median values). The number below "
            r"each latency value breaks it down (ms) into inference, "
            r"pre-/post-processing, and network/framework overhead."
        ),
        label="tab:e2e-latency-breakdown",
        total_decimals=1, part_decimals=0,
    )

    content = tstyle.PREAMBLE_NOTE + table + "\n"
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
