#!/usr/bin/env python3
"""
table_energy_consumption.py (comparison, paper-facing)

Compact LaTeX energy tables -- same visual style as
table_e2e_fps_and_latency_simplified.py (bold headers/models, light-green
cell fill + bold for the best value, resizebox-fit, category grouping,
booktabs), but for energy instead of FPS/latency: total energy consumption
(kJ) and energy per processed frame (J/frame), as two independent tables
in one continuous .tex file, so you can pick whichever representation
reads better in the paper. Lower is always better for energy, so the
minimum per column group is bold + filled (same BEST_FILL as the e2e
tables).

Data reuses barplot_energy_consumption.py / barplot_energy_per_frame.py's
own loaders (load_merged_maps()) -- no data logic duplicated here. Table
shape (5 column groups: Robot CPU x 3 scenarios, Edge CPU, Edge GPU) lives
in table_style_energy.py, which itself reuses styling primitives from
table_style_e2e.py -- change either and both energy tables + all e2e
tables that share the primitives stay in sync where applicable.

Produces: energy-consumption-and-per-frame-table.tex
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# vaccel/, ros2/, and this comparison/ dir each have their own same-named
# barplot_energy_consumption.py / barplot_energy_per_frame.py (and
# barplot_e2e_latency_and_fps.py). Plain `import X` is ambiguous -- it
# resolves to whichever same-named file's directory got inserted into
# sys.path first, which depends on import order elsewhere in the process.
# Load every same-named module by explicit path so this script is correct
# regardless of what else runs first (same pattern as table_style_e2e.py).
_HERE = Path(__file__).parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import table_style_energy as tstyle
from table_style_e2e import PREAMBLE_NOTE

energy_kj = _load_module(_HERE / "barplot_energy_consumption.py", "_table_energy_kj_src")
energy_jpf = _load_module(_HERE / "barplot_energy_per_frame.py", "_table_energy_jpf_src")


def _cat_models_and_captions(robot_map, categories, cat_captions_src, ordered_models_fn):
    present = sorted({m for (m, *_rest) in robot_map})
    base_models = ordered_models_fn(present)
    cat_models, cat_captions = [], []
    for cat, cap in zip(categories, cat_captions_src):
        in_cat = [m for m in base_models if m in cat]
        if in_cat:
            cat_models.append(in_cat)
            cat_captions.append(cap)
    return cat_models, cat_captions


def main():
    kj_robot, kj_edge_cpu, kj_edge_gpu = energy_kj.load_merged_maps()
    (jpf_robot, jpf_edge_cpu, jpf_edge_gpu,
     jpf_robot_debug, jpf_edge_cpu_debug, jpf_edge_gpu_debug) = energy_jpf.load_merged_maps_with_debug()

    kj_cat_models, kj_cat_captions = _cat_models_and_captions(
        kj_robot, energy_kj.CATEGORIES, energy_kj.CAT_CAPTIONS, energy_kj.vaccel_data.ordered_models
    )
    jpf_cat_models, jpf_cat_captions = _cat_models_and_captions(
        jpf_robot, energy_jpf.CATEGORIES, energy_jpf.CAT_CAPTIONS, energy_jpf.vaccel_data.ordered_models
    )

    tstyle.print_debug_info(
        "DEBUG: TOTAL ENERGY CONSUMPTION TABLE VALUES (kJ)",
        kj_cat_models, kj_robot, kj_edge_cpu, kj_edge_gpu, energy_kj.GROUP_ORDER, decimals=3,
    )
    tstyle.print_debug_info(
        "DEBUG: ENERGY PER FRAME TABLE VALUES (J/frame)",
        jpf_cat_models, jpf_robot, jpf_edge_cpu, jpf_edge_gpu, energy_jpf.GROUP_ORDER, decimals=3,
    )

    # Energy values here are exact hardware-counter (RAPL/NVML) totals, idle-
    # baseline subtracted -- a run-level total, not the per-frame median
    # (p50) the e2e FPS/latency tables use -- so the caption doesn't borrow
    # that "median (p50)" phrasing; it would be inaccurate.
    kj_table = tstyle.build_energy_table(
        kj_cat_models, kj_cat_captions, kj_robot, kj_edge_cpu, kj_edge_gpu,
        energy_kj.GROUP_ORDER,
        caption=(
            r"Total energy consumed (kJ) on the robot and edge resources "
            r"(lower is better). Green cells highlight the best "
            r"configuration for each execution option. Energy is computed "
            r"from the difference between counter readings at the start "
            r"and end of each experiment."
        ),
        label="tab:energy-kj",
        decimals=2,
    )
    jpf_table = tstyle.build_energy_table(
        jpf_cat_models, jpf_cat_captions, jpf_robot, jpf_edge_cpu, jpf_edge_gpu,
        energy_jpf.GROUP_ORDER,
        caption=(
            r"Energy consumed per processed frame (J/frame) on the robot "
            r"and edge resources (lower is better). Green cells highlight "
            r"the best configuration for each execution option. Values are "
            r"computed by normalizing total energy consumption by the "
            r"number of processed frames."
        ),
        label="tab:energy-jpf",
        decimals=2,
        robot_debug=jpf_robot_debug, edge_cpu_debug=jpf_edge_cpu_debug, edge_gpu_debug=jpf_edge_gpu_debug,
    )

    # Same total-energy (kJ) values as the first table, but annotated with
    # how many frames were processed and how long that took -- lets a reader
    # check whether a lower total tracks a genuinely shorter run (speed) or
    # a genuinely lower power draw, rather than assuming the latter. Reuses
    # the J/frame table's (energy_j, n_frames, duration_sec) debug maps for
    # n_frames/duration_sec -- kj_* and jpf_* come from independent load
    # paths (kJ never needed a frame count), so a handful of cells the kJ
    # table fills in may show "--" here if that run's frame-count lookup
    # failed on the J/frame side; the main kJ value itself is unaffected.
    kj_speed_table = tstyle.build_energy_table(
        kj_cat_models, kj_cat_captions, kj_robot, kj_edge_cpu, kj_edge_gpu,
        energy_kj.GROUP_ORDER,
        caption=(
            r"Total energy consumed (kJ) on the robot and edge resources "
            r"(lower is better), same values as Table~\ref{tab:energy-kj} "
            r"-- repeated here with the number of processed frames and the "
            r"wall-clock time taken to process them shown below each value, "
            r"to make explicit how much of the difference between "
            r"configurations tracks how long the run took rather than the "
            r"power draw itself."
        ),
        label="tab:energy-kj-speed",
        decimals=2,
        robot_debug=jpf_robot_debug, edge_cpu_debug=jpf_edge_cpu_debug, edge_gpu_debug=jpf_edge_gpu_debug,
        sub_line="frames_duration",
    )

    content = PREAMBLE_NOTE + kj_table + "\n\n" + jpf_table + "\n\n" + kj_speed_table + "\n"

    OUTPUT_FILE = "energy-consumption-and-per-frame-table.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX tables to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
