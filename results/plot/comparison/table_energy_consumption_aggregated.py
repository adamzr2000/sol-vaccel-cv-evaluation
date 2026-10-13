#!/usr/bin/env python3
"""
table_energy_consumption_aggregated.py (comparison, paper-facing)

Collapsed variant of table_energy_consumption.py's total-energy (kJ) table:
instead of 5 column groups (Robot CPU x 3 scenarios, Edge CPU, Edge GPU)
this shows just 3 -- Local (robot-only), Remote CPU (robot+edge CPU
summed), Remote GPU (robot+edge GPU summed) -- so each offloading option
reads as one end-to-end energy number instead of splitting robot/edge
across separate groups. Lower is still better; the minimum per column
group is bold + light-green filled (same BEST_FILL convention as every
other table in this directory). The sub-value under each Remote cell
shows the robot+edge split that was summed, so the aggregate doesn't hide
where the energy actually went.

Reuses table_energy_consumption.py's data loaders (barplot_energy_consumption
.py's load_merged_maps()) unmodified -- this file only changes how the
table is laid out, not how the numbers are computed. Does not modify
table_energy_consumption.py or table_style_energy.py.

Produces: energy-consumption-aggregated-table.tex
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from table_style_e2e import escape, fmt, BEST_FILL, CONFIG_HEADER, get_model_display_name, PREAMBLE_NOTE

_HERE = Path(__file__).parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


energy_kj = _load_module(_HERE / "barplot_energy_consumption.py", "_table_energy_agg_kj_src")

# (group_label, scenario, edge_map_attr) -- edge_map_attr is None for the
# Local group (robot-only, edge is idle) and otherwise names which of
# edge_cpu_map/edge_gpu_map to sum into the robot value.
AGG_COLUMN_GROUPS = [
    ("Local (Robot CPU)", "Local CPU", None),
    ("Remote CPU (Robot+Edge)", "Remote CPU", "edge_cpu"),
    ("Remote GPU (Robot+Edge)", "Remote GPU", "edge_gpu"),
]


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


def _parts(scenario, edge_attr, model, fw, be, robot_map, edge_cpu_map, edge_gpu_map):
    robot_val = robot_map.get((model, scenario, fw, be), np.nan)
    if edge_attr is None:
        return robot_val, None
    edge_map = edge_cpu_map if edge_attr == "edge_cpu" else edge_gpu_map
    edge_val = edge_map.get((model, fw, be), np.nan)
    return robot_val, edge_val


def build_aggregated_table(
    cat_models, cat_captions, robot_map, edge_cpu_map, edge_gpu_map, group_order,
    *, caption: str, label: str, decimals: int = 2,
) -> str:
    n_groups = len(AGG_COLUMN_GROUPS)
    n_configs = len(group_order)
    n_cols = 1 + n_groups * n_configs
    col_spec = "l" + "r" * (n_groups * n_configs)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1 = [""]
    for group_label, _, _ in AGG_COLUMN_GROUPS:
        header1.append(rf"\multicolumn{{{n_configs}}}{{c}}{{\textbf{{{escape(group_label)}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = []
    start = 2
    for _ in AGG_COLUMN_GROUPS:
        end = start + n_configs - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmidrules))

    header2 = [r"\textbf{Model}"]
    for _ in AGG_COLUMN_GROUPS:
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
            for group_label, scenario, edge_attr in AGG_COLUMN_GROUPS:
                parts = {
                    (fw, be): _parts(scenario, edge_attr, model, fw, be, robot_map, edge_cpu_map, edge_gpu_map)
                    for fw, be in group_order
                }
                totals = {
                    key: (robot_v + edge_v if edge_v is not None else robot_v)
                    if np.isfinite(robot_v) and (edge_v is None or np.isfinite(edge_v)) else np.nan
                    for key, (robot_v, edge_v) in parts.items()
                }
                finite_totals = [v for v in totals.values() if np.isfinite(v)]
                best_val = min(finite_totals) if finite_totals else None

                for fw, be in group_order:
                    robot_v, edge_v = parts[(fw, be)]
                    total = totals[(fw, be)]
                    text = fmt(total, decimals)
                    is_best = best_val is not None and np.isfinite(total) and total == best_val
                    top = rf"\textbf{{{text}}}" if is_best else text

                    if edge_attr is not None:
                        sub = (
                            f"{robot_v:.1f}+{edge_v:.1f}"
                            if np.isfinite(robot_v) and np.isfinite(edge_v) else "--"
                        )
                        cell = (
                            rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\"
                            rf"{{\scriptsize {sub}}}\end{{tabular}}"
                        )
                    else:
                        cell = top

                    if is_best:
                        cell = rf"\cellcolor{{{BEST_FILL}}}{cell}"
                    cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    kj_robot, kj_edge_cpu, kj_edge_gpu = energy_kj.load_merged_maps()

    kj_cat_models, kj_cat_captions = _cat_models_and_captions(
        kj_robot, energy_kj.CATEGORIES, energy_kj.CAT_CAPTIONS, energy_kj.vaccel_data.ordered_models
    )

    kj_agg_table = build_aggregated_table(
        kj_cat_models, kj_cat_captions, kj_robot, kj_edge_cpu, kj_edge_gpu,
        energy_kj.GROUP_ORDER,
        caption=(
            r"Total energy consumed (kJ) per execution option, aggregated "
            r"across robot and edge resources (lower is better). The Local "
            r"column is robot-only; the Remote CPU and Remote GPU columns "
            r"sum robot and edge energy for that offloading scenario, with "
            r"the robot+edge split shown below each value. Green cells "
            r"highlight the best configuration per column."
        ),
        label="tab:energy-kj-aggregated",
        decimals=2,
    )

    content = PREAMBLE_NOTE + kj_agg_table + "\n"

    OUTPUT_FILE = "energy-consumption-aggregated-table.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
