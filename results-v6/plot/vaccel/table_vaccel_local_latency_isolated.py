#!/usr/bin/env python3
"""
table_vaccel_local_latency_isolated.py

Compact LaTeX table version of barplot_local_latency_isolated_simplified.py --
same data (iso local-inference latency, 5 backends x 3 deployment targets),
same category grouping, but as a table instead of a figure: one row per
model, 3 grouped column blocks (Robot CPU / Edge CPU / Edge GPU), 5
sub-columns each (JIT-Inductor, AOT-Inductor, vAccel Local (AOT-Inductor),
SOL, vAccel Local (SOL)). No AOT-Inductor-vs-SOL trend/connector -- this is
values only. Fastest backend per model x deployment-target is bold +
light-green cell fill (same convention as the comparison/ tables).

Reuses barplot_local_latency_isolated_simplified.py's own data loader,
BACKEND_MAP, CATEGORIES, and MODEL_DISPLAY (plain import -- this script
lives in the same directory as its source, so there's no cross-directory
same-named-module ambiguity the comparison/ scripts have to work around).

Produces: vaccel-local-latency-isolated-table.tex
"""
from __future__ import annotations

import numpy as np

import barplot_local_latency_isolated_simplified as src

BEST_FILL = "green!15"

# Column groups: (host, device, group_label) -- same targets as the barplot's
# ROWS, just relabeled here as table column groups instead of figure rows.
TABLE_ROWS = [
    ("robot",     "cpu", "Robot CPU"),
    ("edge-asus", "cpu", "Edge CPU"),
    ("edge-asus", "gpu", "Edge GPU"),
]

BACKENDS = list(src.BACKEND_MAP.values())

SHORT_BACKEND_LABEL = {
    "JIT-Inductor":                "JIT",
    "AOT-Inductor":                "AOTI",
    "vAccel Local (AOT-Inductor)": "vAcc-AOTI",
    "SOL":                         "SOL",
    "vAccel Local (SOL)":          "vAcc-SOL",
}


def escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}f}"


def print_debug_info(data: dict, decimals: int = 2) -> None:
    W = 150
    print("\n" + "=" * W)
    print(f"{'DEBUG: LOCAL INFERENCE LATENCY TABLE VALUES (ms)':^{W}}")
    print("=" * W)
    header = f"{'Model':<22} | " + " | ".join(
        f"{gl} {SHORT_BACKEND_LABEL[be]:>9}" for _h, _d, gl in TABLE_ROWS for be in BACKENDS
    )
    print(header)
    print("-" * W)
    for _cat_title, models in src.CATEGORIES:
        for m in models:
            cells = [f"{m:<22}"]
            for host, device, _gl in TABLE_ROWS:
                for be in BACKENDS:
                    entry = data.get((host, device, be, m))
                    v = entry[0] if entry else float("nan")
                    cells.append(f"{v:9.{decimals}f}" if np.isfinite(v) else f"{'-':>9}")
            print(" | ".join(cells))
    print("=" * W + "\n")


def build_table(data: dict, *, caption: str, label: str, decimals: int = 2) -> str:
    """One row per model, grouped by category. Column groups are the 3
    deployment targets (TABLE_ROWS), sub-columns are the 5 backends
    (BACKENDS). Lower latency is better -- the minimum per model x
    deployment-target group gets bold + light-green cell fill. Missing
    (host, device, backend, model) combos render as "--", matching the
    barplot's own "leave blank, no errors" convention.
    """
    n_groups = len(TABLE_ROWS)
    n_configs = len(BACKENDS)
    n_cols = 1 + n_groups * n_configs
    col_spec = "l" + "r" * (n_groups * n_configs)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    # Same overflow story as the other comparison tables -- 3 groups x 5
    # configs is wide enough to need resizebox regardless of font size.
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: deployment-target group headers
    header1 = [""]
    for _h, _d, group_label in TABLE_ROWS:
        header1.append(rf"\multicolumn{{{n_configs}}}{{c}}{{\textbf{{{escape(group_label)}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = []
    start = 2
    for _ in TABLE_ROWS:
        end = start + n_configs - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmidrules))

    # Row 2: backend sub-headers
    header2 = [r"\textbf{Model}"]
    for _ in TABLE_ROWS:
        for be in BACKENDS:
            header2.append(rf"\textbf{{{escape(SHORT_BACKEND_LABEL.get(be, be))}}}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat_idx, (cat_title, models) in enumerate(src.CATEGORIES):
        if cat_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{escape(cat_title)}}}}} \\")
        lines.append(r"\midrule")

        for m in models:
            cells = [rf"\textbf{{{escape(src.MODEL_DISPLAY.get(m, m))}}}"]
            for host, device, _gl in TABLE_ROWS:
                vals = {}
                for be in BACKENDS:
                    entry = data.get((host, device, be, m))
                    vals[be] = entry[0] if entry else float("nan")
                finite_vals = [v for v in vals.values() if np.isfinite(v)]
                best_val = min(finite_vals) if finite_vals else None

                for be in BACKENDS:
                    v = vals[be]
                    text = fmt(v, decimals)
                    is_best = best_val is not None and np.isfinite(v) and v == best_val
                    if is_best:
                        text = rf"\cellcolor{{{BEST_FILL}}}\textbf{{{text}}}"
                    cells.append(text)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    path = src.INPUT_FILE.resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    data = src.load_data(path)
    print_debug_info(data, decimals=2)

    metric_label = "median (p50)" if src.METRIC == "median" else src.METRIC
    table = build_table(
        data,
        caption=(
            r"Local inference latency (ms) across deployment targets and "
            rf"backends ({metric_label}; lower is better). Green cells "
            r"indicate the fastest backend for each model and deployment "
            r"target."
        ),
        label="tab:vaccel-local-latency-isolated",
        decimals=2,
    )

    preamble_note = (
        "% Requires \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage[table]{xcolor}.\n"
        "% Fastest backend per model x deployment-target is bold + light-green cell fill.\n"
    )
    content = preamble_note + table + "\n"

    output_file = "vaccel-local-latency-isolated-table.tex"
    with open(output_file, "w") as f:
        f.write(content)
    print(f"[OK] Saved LaTeX table to: {output_file}")


if __name__ == "__main__":
    main()
