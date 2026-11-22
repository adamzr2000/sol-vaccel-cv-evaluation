#!/usr/bin/env python3
"""
table_style_e2e.py

Shared compact-table builder for the e2e comparison LaTeX tables
(table_e2e_fps.py, table_e2e_latency.py). One row per model, scenarios
(Local CPU / Remote CPU / Remote GPU) as grouped column headers, 4
sub-columns each (ROS2+Torch, ROS2+SOL, vAccel+Torch, vAccSOL). Change the
table shell here (headers, highlight color, resizebox/sizing) and every
table using it stays in sync -- same "single source of truth" pattern as
plot_style.py for the figures.

These are internal working tables (not paper-final), so bold + color for
the best value per model x scenario group is fair game.

Requires \\usepackage{booktabs}, \\usepackage{graphicx} (for \\resizebox),
and \\usepackage{xcolor} (for \\textcolor) in the preamble -- none loaded
by IEEEtran by default.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# vaccel/, ros2/, and this comparison/ dir each have their own
# barplot_e2e_latency_and_fps.py (and several other same-named files, e.g.
# barplot_energy_consumption.py). A plain `import barplot_e2e_latency_and_fps`
# is ambiguous -- it silently resolves to whichever same-named file's
# directory got inserted into sys.path first, which depends on import
# order elsewhere in the process (a real bug this hit: importing an energy
# script before this one made `src` resolve to vaccel/'s copy instead of
# comparison/'s, and `src.vaccel_data` didn't exist there). Load by
# explicit path instead, so this is correct regardless of what else has
# been imported.
_HERE = Path(__file__).parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


src = _load_module(_HERE / "barplot_e2e_latency_and_fps.py", "_table_style_e2e_src")
from plot_config import get_model_display_name  # noqa: E402

# Column headers -- spelled out (Torch/SOL, not T/S -- abbreviations read
# as confusingly similar at a glance).
CONFIG_HEADER = {
    ("ROS2", "Torch"): "ROS2+Torch",
    ("ROS2", "SOL"): "ROS2+SOL",
    ("vAccel", "Torch"): "vAcc+Torch",
    ("vAccel", "SOL"): "vAccSOL",
}

BEST_COLOR = "blue!70!black"

# Both data sources measure from the robot's perspective as client (vaccel
# side filters explicitly on host=="robot"; ROS2 side's t_e2e_ms is the
# client-observed round trip) and both currently report the per-model
# median (p50) -- vaccel_data.METRIC is a real toggle, the ROS2 loader's
# median is hardcoded with no toggle, so if METRIC ever changes to "mean"
# the two sides would silently disagree. Exposed here (not appended to
# captions automatically -- callers own their exact caption wording and
# interpolate this where they want it) so it can't go stale unnoticed.
METRIC_LABEL = "median (p50)" if src.vaccel_data.METRIC == "median" else src.vaccel_data.METRIC

PREAMBLE_NOTE = (
    "% Requires \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{xcolor}.\n"
    "% Best value per model x scenario group is bold + colored (internal working table).\n"
    "% vAcc = vAccel.\n"
)


def escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value: float, decimals: int) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}f}"


def build_compact_table(
    cat_models, cat_captions, value_map, *, caption: str, label: str,
    decimals: int = 1, best: str = "max",
    gain_vs_scenario: str | None = None,
) -> str:
    """value_map: dict[(model, scenario, fw, be)] -> float.
    caption: full, final caption text (verbatim) -- interpolate METRIC_LABEL
    into it yourself if you want the metric name to stay dynamic.
    best: "max" (higher is better, e.g. FPS) or "min" (lower is better,
    e.g. latency) -- which direction gets the bold+color highlight.
    gain_vs_scenario: if set to a scenario name (e.g. "Local CPU"), every
    OTHER scenario's own best value (per model -- the same cell already
    bold+colored) is additionally annotated with its gain relative to the
    best value in this baseline scenario, e.g. "11.8 (x5.9)" -- so each of
    Remote CPU and Remote GPU gets its own gain-vs-baseline figure, styled
    the same bold+color as the number itself. Direction follows `best`
    (ratio for "max", inverse ratio for "min"). Opt-in per table, not a
    default.
    """
    best_fn = max if best == "max" else min

    n_scenarios = len(src.SCENARIOS)
    n_configs = len(src.GROUP_ORDER)
    n_cols = 1 + n_scenarios * n_configs
    col_spec = "l" + "r" * (n_scenarios * n_configs)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    # This many columns overflows the page at any fixed font size (verified
    # by an actual pdflatex compile, not just eyeballed); resizebox
    # guarantees an exact fit to \textwidth regardless of column count.
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: scenario group headers spanning n_configs columns each
    header1 = [""]
    for scenario, _ in src.SCENARIOS:
        header1.append(rf"\multicolumn{{{n_configs}}}{{c}}{{\textbf{{{escape(scenario)}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = []
    start = 2
    for _ in src.SCENARIOS:
        end = start + n_configs - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmidrules))

    # Row 2: config sub-headers
    header2 = [r"\textbf{Model}"]
    for _ in src.SCENARIOS:
        for fw, be in src.GROUP_ORDER:
            header2.append(rf"\textbf{{{CONFIG_HEADER[(fw, be)]}}}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat_idx, (models, cap) in enumerate(zip(cat_models, cat_captions)):
        if cat_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{escape(cap)}}}}} \\")
        lines.append(r"\midrule")

        for model in models:
            # Baseline (e.g. Local CPU) best value, per model -- used to
            # annotate every OTHER scenario's own best cell with its gain
            # over this baseline (see gain_vs_scenario docstring above).
            baseline_best = None
            if gain_vs_scenario is not None:
                baseline_vals = [
                    value_map[(model, gain_vs_scenario, fw, be)]
                    for fw, be in src.GROUP_ORDER
                ]
                baseline_vals = [v for v in baseline_vals if np.isfinite(v)]
                baseline_best = best_fn(baseline_vals) if baseline_vals else None

            cells = [rf"\textbf{{{escape(get_model_display_name(model))}}}"]
            for scenario, _ in src.SCENARIOS:
                vals = {
                    (fw, be): value_map[(model, scenario, fw, be)]
                    for fw, be in src.GROUP_ORDER
                }
                finite_vals = [v for v in vals.values() if np.isfinite(v)]
                best_val = best_fn(finite_vals) if finite_vals else None
                for fw, be in src.GROUP_ORDER:
                    v = vals[(fw, be)]
                    text = fmt(v, decimals)
                    is_best = best_val is not None and np.isfinite(v) and v == best_val
                    if is_best:
                        text = rf"\textcolor{{{BEST_COLOR}}}{{\textbf{{{text}}}}}"
                        if baseline_best and scenario != gain_vs_scenario:
                            ratio = v / baseline_best if best == "max" else baseline_best / v
                            text += rf" \textcolor{{{BEST_COLOR}}}{{\textbf{{(x{ratio:.1f})}}}}"
                    cells.append(text)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def print_debug_info(title: str, cat_models, value_map, decimals: int = 1):
    W = 118
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    header = f"{'Model':<22} | " + " | ".join(
        f"{scenario} {CONFIG_HEADER[(fw, be)]:>8}" for scenario, _ in src.SCENARIOS for fw, be in src.GROUP_ORDER
    )
    print(header)
    print("-" * W)
    for models in cat_models:
        for model in models:
            cells = [f"{model:<22}"]
            for scenario, _ in src.SCENARIOS:
                for fw, be in src.GROUP_ORDER:
                    v = value_map[(model, scenario, fw, be)]
                    cells.append(f"{v:8.{decimals}f}" if np.isfinite(v) else f"{'-':>8}")
            print(" | ".join(cells))
    print("=" * W + "\n")
