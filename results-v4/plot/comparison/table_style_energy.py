#!/usr/bin/env python3
"""
table_style_energy.py

Shared compact-table builder for the energy comparison LaTeX tables
(table_energy_consumption.py: total energy in kJ, and energy per frame in
J/frame). Reuses the primitives from table_style_e2e.py (escape, fmt,
BEST_COLOR, CONFIG_HEADER) so a color/header-text change there applies here
too, but defines its own table shape: unlike the e2e FPS/latency tables'
clean 3-scenario grid, energy has 3 measurement DOMAINS that don't all
apply to every scenario --

  Robot CPU: measured in all 3 scenarios (Local CPU, Remote CPU, Remote GPU)
  Edge CPU:  Remote CPU scenario only (edge is idle for Local/Remote GPU)
  Edge GPU:  Remote GPU scenario only

-- so this flattens the source plots' 3-panel-row layout into 5 grouped
column blocks per row (Robot CPU x 3 scenarios, Edge CPU, Edge GPU) instead
of the e2e tables' 3.

Lower energy is always better, so the best (minimum) value per column
group is bold + colored, same convention as the e2e latency table.

Requires \\usepackage{booktabs}, \\usepackage{graphicx} (for \\resizebox),
and \\usepackage{xcolor} (for \\textcolor) in the preamble -- none loaded
by IEEEtran by default.
"""
from __future__ import annotations

import numpy as np

# Import order matters: table_style_e2e pulls in comparison/barplot_e2e_
# latency_and_fps.py, whose module-level sys.path.insert(0, .../vaccel) is
# what makes the plain `from plot_config import ...` below resolvable at
# all -- see the comment in table_energy_consumption.py for the fuller story
# on why this repo's same-named files across vaccel/ros2/comparison make
# import order load-bearing here.
from table_style_e2e import escape, fmt, BEST_COLOR, CONFIG_HEADER
from plot_config import get_model_display_name

# (group_label, domain_map_key, scenario) -- domain_map_key selects which of
# the three maps passed to build_energy_table() this column group reads
# from; scenario is only meaningful for "robot" (indexes robot_cpu_map's
# (model, scenario, fw, be) key), the edge maps are already scenario-fixed.
COLUMN_GROUPS = [
    ("Robot CPU -- Local", "robot", "Local CPU"),
    ("Robot CPU -- Remote CPU", "robot", "Remote CPU"),
    ("Robot CPU -- Remote GPU", "robot", "Remote GPU"),
    ("Edge CPU -- Remote CPU", "edge_cpu", None),
    ("Edge GPU -- Remote GPU", "edge_gpu", None),
]


def _lookup(group_key: str, scenario, model, fw, be, robot_map, edge_cpu_map, edge_gpu_map):
    if group_key == "robot":
        return robot_map.get((model, scenario, fw, be), np.nan)
    if group_key == "edge_cpu":
        return edge_cpu_map.get((model, fw, be), np.nan)
    return edge_gpu_map.get((model, fw, be), np.nan)


def build_energy_table(
    cat_models, cat_captions, robot_map, edge_cpu_map, edge_gpu_map,
    group_order, *, caption: str, label: str, decimals: int = 2,
) -> str:
    """robot_map: dict[(model, scenario, fw, be)] -> float.
    edge_cpu_map / edge_gpu_map: dict[(model, fw, be)] -> float.
    group_order: the GROUP_ORDER list ([(fw, be), ...]) from whichever
    energy source module is being rendered (kJ or J/frame script) --
    both use the identical 4-entry order, passed explicitly rather than
    imported so this module has no hard dependency on either.
    Lower is always better here (energy); best per column group is
    bold + colored.
    """
    n_groups = len(COLUMN_GROUPS)
    n_configs = len(group_order)
    n_cols = 1 + n_groups * n_configs
    col_spec = "l" + "r" * (n_groups * n_configs)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    # Same overflow story as table_style_e2e.py's build_compact_table --
    # 5 groups x 4 configs is even wider than the e2e tables' 3 x 4;
    # resizebox guarantees an exact fit regardless.
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: column-group headers
    header1 = [""]
    for group_label, _, _ in COLUMN_GROUPS:
        header1.append(rf"\multicolumn{{{n_configs}}}{{c}}{{\textbf{{{escape(group_label)}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = []
    start = 2
    for _ in COLUMN_GROUPS:
        end = start + n_configs - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmidrules))

    # Row 2: config sub-headers
    header2 = [r"\textbf{Model}"]
    for _ in COLUMN_GROUPS:
        for fw, be in group_order:
            header2.append(rf"\textbf{{{CONFIG_HEADER[(fw, be)]}}}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat_idx, (models, cap) in enumerate(zip(cat_models, cat_captions)):
        if cat_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{escape(cap)}}}}} \\")
        lines.append(r"\midrule")

        for model in models:
            cells = [rf"\textbf{{{escape(get_model_display_name(model))}}}"]
            for group_label, group_key, scenario in COLUMN_GROUPS:
                vals = {
                    (fw, be): _lookup(group_key, scenario, model, fw, be, robot_map, edge_cpu_map, edge_gpu_map)
                    for fw, be in group_order
                }
                finite_vals = [v for v in vals.values() if np.isfinite(v)]
                best_val = min(finite_vals) if finite_vals else None
                for fw, be in group_order:
                    v = vals[(fw, be)]
                    text = fmt(v, decimals)
                    if best_val is not None and np.isfinite(v) and v == best_val:
                        text = rf"\textcolor{{{BEST_COLOR}}}{{\textbf{{{text}}}}}"
                    cells.append(text)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def print_debug_info(title: str, cat_models, robot_map, edge_cpu_map, edge_gpu_map, group_order, decimals: int = 2):
    W = 130
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    header = f"{'Model':<22} | " + " | ".join(
        f"{gl} {CONFIG_HEADER[(fw, be)]:>10}" for gl, _, _ in COLUMN_GROUPS for fw, be in group_order
    )
    print(header)
    print("-" * W)
    for models in cat_models:
        for model in models:
            cells = [f"{model:<22}"]
            for group_label, group_key, scenario in COLUMN_GROUPS:
                for fw, be in group_order:
                    v = _lookup(group_key, scenario, model, fw, be, robot_map, edge_cpu_map, edge_gpu_map)
                    cells.append(f"{v:10.{decimals}f}" if np.isfinite(v) else f"{'-':>10}")
            print(" | ".join(cells))
    print("=" * W + "\n")
