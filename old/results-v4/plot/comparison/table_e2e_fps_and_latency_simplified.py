#!/usr/bin/env python3
"""
table_e2e_fps_and_latency_simplified.py (comparison, paper-facing)

Unified generator for the two compact e2e comparison tables (FPS, and total
E2E latency) -- replaces the separate table_e2e_fps.py / table_e2e_latency.py.
Both tables are written as continuous LaTeX into a single .tex file (two
back-to-back `table*` environments), since they share one data load and are
two views of the same underlying measurement: FPS = 1000 / total latency.
The FPS table's caption says so explicitly and cross-references the latency
table.

Table shell (headers, color, resizebox, category grouping) lives in
table_style_e2e.py -- shared with any other table built the same way.

Produces: e2e-latency-and-fps-simplified-table.tex
"""
from __future__ import annotations

import barplot_e2e_latency_and_fps as src
import table_style_e2e as tstyle

OUTPUT_FILE = "e2e-latency-and-fps-simplified-table.tex"


def main():
    merged = src.load_merged_rows()
    base_models, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map, _lo_map, _hi_map = src.build_maps(merged)

    total_map = {k: inf_map[k] + pre_map[k] + net_map[k] for k in inf_map}

    tstyle.print_debug_info("DEBUG: E2E FPS TABLE VALUES", cat_models, fps_map, decimals=2)
    tstyle.print_debug_info("DEBUG: E2E TOTAL LATENCY TABLE VALUES (ms)", cat_models, total_map, decimals=1)

    # Terminology kept consistent across both captions ("end-to-end (E2E)
    # latency" in both, not "total" in one and "end-to-end" in the other).
    # METRIC_LABEL is interpolated in (not auto-appended by the table
    # builder) so the caption text stays exactly as specified but the
    # metric name still can't silently go stale if METRIC ever changes.
    fps_table = tstyle.build_compact_table(
        cat_models, cat_captions, fps_map,
        caption=(
            r"Frame rate (FPS) of the robot vision pipeline for a single-camera "
            rf"input (higher is better; robot as client, {tstyle.METRIC_LABEL}). "
            r"FPS is calculated as the inverse of the end-to-end (E2E) latency "
            r"reported in Table~\ref{tab:e2e-latency}. Values in brackets denote "
            r"the speedup over the best Local CPU configuration, highlighting "
            r"the performance benefits of edge offloading."
        ),
        label="tab:e2e-fps",
        decimals=1, best="max",
        gain_vs_scenario="Local CPU",
    )
    latency_table = tstyle.build_compact_table(
        cat_models, cat_captions, total_map,
        caption=(
            r"End-to-end (E2E) latency (ms) of the robot vision pipeline for a "
            r"single-camera input, measured from image capture to result delivery "
            rf"(lower is better; robot as client, {tstyle.METRIC_LABEL})."
        ),
        label="tab:e2e-latency",
        decimals=1, best="min",
    )

    content = tstyle.PREAMBLE_NOTE + fps_table + "\n\n" + latency_table + "\n"

    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX tables to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
