#!/usr/bin/env python3
"""
table_e2e_latency_and_fps.py (comparison, paper-facing)

LaTeX table companion to barplot_e2e_latency_and_fps.py -- same merged
ROS2-vs-vAccel data (13 models x 3 scenarios x 4 framework/backend configs),
rendered as three IEEE `table*` environments (one per scenario: Local CPU,
Remote CPU, Remote GPU) instead of bars, for when the raw numbers read
clearer than the chart.

Columns per row: FPS, Inference (ms), Pre/Post (ms), Network (ms), Total (ms).
Models are grouped by category (Image Classification / Video Action
Recognition / Semantic Segmentation) with a spanning subheader row, each
model spans its 4 configs via \\multirow. Best FPS and best Total latency
per model are bolded.

Requires \\usepackage{booktabs} and \\usepackage{multirow} in the paper
preamble (not loaded by IEEEtran by default).

Produces: e2e-latency-and-fps-table.tex
"""
from __future__ import annotations

import numpy as np

import barplot_e2e_latency_and_fps as src
from plot_config import get_model_display_name

OUTPUT_FILE = "e2e-latency-and-fps-table.tex"

# Metric formatting: (label, unit, decimals)
METRIC_COLUMNS = [
    ("FPS", "", 2),
    ("Inference", "ms", 1),
    ("Pre/Post", "ms", 1),
    ("Network", "ms", 1),
    ("Total", "ms", 1),
]

SCENARIO_LABELS = {
    "Local CPU": "Local CPU",
    "Remote CPU": "Remote CPU",
    "Remote GPU": "Remote GPU",
}


def _escape(s: str) -> str:
    """Defensive LaTeX escaping for anything that ends up in table text."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _fmt(value: float, decimals: int) -> str:
    if value is None or not np.isfinite(value) or value == 0.0:
        return "--"
    return f"{value:.{decimals}f}"


def _row_metrics(fps_map, inf_map, pre_map, net_map, model, scenario, fw, be):
    fps = fps_map[(model, scenario, fw, be)]
    inf = inf_map[(model, scenario, fw, be)]
    pre = pre_map[(model, scenario, fw, be)]
    net = net_map[(model, scenario, fw, be)]
    total = inf + pre + net if np.isfinite(inf) else np.nan
    return [fps, inf, pre, net, total]


def build_scenario_table(scenario: str, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map) -> str:
    n_cols = 2 + len(METRIC_COLUMNS)  # Model, Config, + metrics
    col_spec = "ll" + "r" * len(METRIC_COLUMNS)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{End-to-end FPS and latency breakdown --- {SCENARIO_LABELS[scenario]}}}"
    )
    lines.append(rf"\label{{tab:e2e-{scenario.lower().replace(' ', '-')}}}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = ["Model", "Configuration"] + [
        (f"{label} ({unit})" if unit else label) for label, unit, _ in METRIC_COLUMNS
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for cat_idx, (models, caption) in enumerate(zip(cat_models, cat_captions)):
        if cat_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{_escape(caption)}}}}} \\")
        lines.append(r"\midrule")

        for m_idx, model in enumerate(models):
            display_name = _escape(get_model_display_name(model))

            # Gather all 4 configs' metrics first, to find the row-group's
            # best FPS (highest) and best Total latency (lowest) for bolding.
            rows = {}
            for fw, be in src.GROUP_ORDER:
                rows[(fw, be)] = _row_metrics(fps_map, inf_map, pre_map, net_map, model, scenario, fw, be)

            fps_vals = [r[0] for r in rows.values() if np.isfinite(r[0])]
            total_vals = [r[4] for r in rows.values() if np.isfinite(r[4])]
            best_fps = max(fps_vals) if fps_vals else None
            best_total = min(total_vals) if total_vals else None

            for g_idx, (fw, be) in enumerate(src.GROUP_ORDER):
                metrics = rows[(fw, be)]
                cells = []
                for col_idx, (value, (_, _, decimals)) in enumerate(zip(metrics, METRIC_COLUMNS)):
                    text = _fmt(value, decimals)
                    is_fps = col_idx == 0
                    is_total = col_idx == 4
                    if is_fps and best_fps is not None and np.isfinite(value) and value == best_fps:
                        text = rf"\textbf{{{text}}}"
                    if is_total and best_total is not None and np.isfinite(value) and value == best_total:
                        text = rf"\textbf{{{text}}}"
                    cells.append(text)

                config_label = _escape(src.GROUP_LABEL[(fw, be)])
                if g_idx == 0:
                    row = rf"\multirow{{{len(src.GROUP_ORDER)}}}{{*}}{{{display_name}}} & {config_label} & " + " & ".join(cells) + r" \\"
                else:
                    row = " & " + config_label + " & " + " & ".join(cells) + r" \\"
                lines.append(row)

            is_last_model_in_cat = m_idx == len(models) - 1
            is_last_cat = cat_idx == len(cat_models) - 1
            if not (is_last_model_in_cat and is_last_cat):
                lines.append(rf"\cmidrule(lr){{1-{n_cols}}}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def print_debug_info(cat_models, fps_map, inf_map, pre_map, net_map):
    """Console mirror of the table content, same convention as the other
    comparison scripts' print_debug_info."""
    W = 118
    print("\n" + "=" * W)
    print(f"{'DEBUG: E2E FPS + LATENCY TABLE VALUES':^{W}}")
    print("=" * W)
    header = (f"{'Model':<22} | {'Scenario':<11} | {'Config':<14} | "
              f"{'FPS':>7} | {'Inf(ms)':>8} | {'Pre/Post':>8} | {'Net(ms)':>8} | {'Total(ms)':>9}")
    print(header)
    print("-" * W)
    for models in cat_models:
        for model in models:
            for scenario, _ in src.SCENARIOS:
                for fw, be in src.GROUP_ORDER:
                    fps, inf, pre, net, total = _row_metrics(fps_map, inf_map, pre_map, net_map, model, scenario, fw, be)
                    if not np.isfinite(fps) and inf == 0.0:
                        continue
                    print(
                        f"{model:<22} | {scenario:<11} | {src.GROUP_LABEL[(fw, be)]:<14} | "
                        f"{fps:7.2f} | {inf:8.2f} | {pre:8.2f} | {net:8.2f} | {total:9.2f}"
                    )
        print("-" * W)
    print("=" * W + "\n")


def main():
    merged = src.load_merged_rows()
    base_models, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map, _lo_map, _hi_map = src.build_maps(merged)

    print_debug_info(cat_models, fps_map, inf_map, pre_map, net_map)

    tables = [
        build_scenario_table(scenario, cat_models, cat_captions, fps_map, inf_map, pre_map, net_map)
        for scenario, _ in src.SCENARIOS
    ]

    preamble_note = (
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow} in the paper preamble.\n"
        "% Best FPS (highest) and best Total latency (lowest) per model x scenario group are bolded.\n"
    )
    content = preamble_note + "\n\n".join(tables) + "\n"

    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
