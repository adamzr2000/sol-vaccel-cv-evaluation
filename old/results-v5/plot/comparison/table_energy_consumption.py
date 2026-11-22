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
            r"Energy consumption (kJ) of CPU and GPU resources during E2E "
            r"inference under local and edge execution settings (lower is "
            r"better; robot acts as client). Energy was measured using "
            r"Intel Running Average Power Limit (RAPL) counters, which "
            r"provide hardware-level estimates of energy usage for CPU "
            r"package and memory domains, and computed from the difference "
            r"between counter readings taken before and after execution."
        ),
        label="tab:energy-kj",
        decimals=2,
    )
    jpf_table = tstyle.build_energy_table(
        jpf_cat_models, jpf_cat_captions, jpf_robot, jpf_edge_cpu, jpf_edge_gpu,
        energy_jpf.GROUP_ORDER,
        caption=(
            r"Energy per processed frame (J/frame) of CPU and GPU resources "
            r"during E2E inference under local and edge execution settings "
            r"(lower is better; robot as client) --- normalizes "
            r"Table~\ref{tab:energy-kj} by frame count. The number below "
            r"each value shows the total energy (J) and frame count it was "
            r"computed from."
        ),
        label="tab:energy-jpf",
        decimals=2,
        robot_debug=jpf_robot_debug, edge_cpu_debug=jpf_edge_cpu_debug, edge_gpu_debug=jpf_edge_gpu_debug,
    )

    content = PREAMBLE_NOTE + kj_table + "\n\n" + jpf_table + "\n"

    OUTPUT_FILE = "energy-consumption-and-per-frame-table.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX tables to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
