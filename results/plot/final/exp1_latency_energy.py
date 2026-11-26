#!/usr/bin/env python3
"""
exp1_latency_energy.py -- Experiment 1 (vAccSOL): one LaTeX table with latency and energy per model,
for local inference (robot CPU) and remote inference (edge CPU, edge GPU).

Reads the CSVs written by exp1_latency.py and exp1_energy.py (run those first) and prints the same
values as their own tables, without re-measuring or re-rounding anything:
  Latency (ms)  Total = median latency on the robot; Comm. = communication part of it (data transfer
                and vAccel processing), i.e. total - edge computation (exp1-latency.csv)
  Energy (J)    per inference; Robot = robot CPU (every target), Robot + Edge = robot + edge (edge CPU,
                + edge GPU for the edge GPU), from the printed 2-decimal values (exp1_energy.tex_energies)
Local inference has no communication and no edge energy: one latency and one energy (the robot's) column.
Shaded columns are the totals, in the execution target's colour (common.TEX_COLOUR).

Usage:    python3 exp1_latency_energy.py
Produces: exp1-latency-energy.tex
"""
from common import HERE, TEX_COLOUR, TEX_INDENT, TEX_MODEL, TEX_TASK, fmt_fps, read_rows
from exp1_energy import tex_energies

LAT_CSV, ENERGY_CSV = HERE / "exp1-latency.csv", HERE / "exp1-energy.csv"
TEX = HERE / "exp1-latency-energy.tex"
TARGETS = [("robot_cpu", "Local CPU"), ("edge_cpu", "Edge CPU"), ("edge_gpu", "Edge GPU")]
CAPTION = ("Exp.~1: Inference latency (ms), measured on the robot, and energy per inference (J) of models "
           "executed via vAccSOL. We compare local inference on the robot CPU (blue) with remote inference on "
           "the edge CPU (orange) and GPU (green). Comm.\\ is the communication overhead of remote inference. "
           "The equivalent frame rate is given in parentheses.")


def cells() -> tuple:
    """-> ({task: [models]}, {(model, target): (lat_total, lat_comm, e_robot, e_total)}) as printed strings
    (comm and robot energy None for local inference)."""
    tasks, lat = read_rows(LAT_CSV, 3)
    _, energy_raw = read_rows(ENERGY_CSV, 3)
    energy = tex_energies(energy_raw)
    out = {}
    for (m, t), d in lat.items():
        robot, _, total = energy[(m, t)]
        local = t == "robot_cpu"
        out[(m, t)] = (d["total_ms"], None if local else d["communication_ms"],
                       None if local else f"{robot:.2f}", f"{total:.2f}")
    return tasks, out


def write_tex() -> None:
    tasks, v = cells()
    # columns: (metric index into the cell tuple, target, header, shaded)
    cols = [(0, "robot_cpu", "Total", True), (1, "edge_cpu", "Comm.", False), (0, "edge_cpu", "Total", True),
            (1, "edge_gpu", "Comm.", False), (0, "edge_gpu", "Total", True),
            # energy: the robot's energy under "Robot" for every target (local: the robot is the only device,
            # so it is also the shaded total); remote totals named by what they add up
            (3, "robot_cpu", "Robot", True), (2, "edge_cpu", "Robot", False), (3, "edge_cpu", "\\shortstack{Robot\\\\+ Edge}", True),
            (2, "edge_gpu", "Robot", False), (3, "edge_gpu", "\\shortstack{Robot\\\\+ Edge}", True)]
    # each latency Total is followed by its equivalent frame rate, in small grey text inside the shaded block
    # ("(33 fps)" = 1000 / printed latency), as in exp1-latency.tex
    spec, header, n_cols = "", ["Model"], 1
    spans = []  # (metric group, target, number of tabular columns)
    for k, (i, t, h, shaded) in enumerate(cols):
        vals = [x[i] for (m, tt), x in v.items() if tt == t and x[i] is not None]
        fmt = f"{max(len(s.split('.')[0]) for s in vals)}.{len(vals[0].split('.')[1])}"  # siunitx width
        last = k == len(cols) - 1
        fps = i == 0  # latency total
        # shading stops at the table's end and at the latency/energy boundary (columns 4|5), so the
        # column spacing there stays white instead of the two halves' colours touching
        left = "0pt" if k == 5 else "\\tabcolsep"
        right = "0pt" if last else "\\tabcolsep"
        shade = f">{{\\columncolor{{{TEX_COLOUR[t][0]}}}[{left}][{right}]}}" if shaded else ""
        spec += f"{shade}S[table-format={fmt}]"
        header.append(f"{{{h}}}")
        width = 1
        if fps:  # fps cell: no left overhang (would paint over the latency's last digit)
            fps_right = "0pt" if k == 4 else "\\tabcolsep"
            spec += f"@{{\\,}}>{{\\columncolor{{{TEX_COLOUR[t][0]}}}[0pt][{fps_right}]\\scriptsize\\color{{black!65}}}}l"
            header.append("")
            width = 2
        group = "lat" if i < 2 else "energy"
        if spans and spans[-1][:2] == (group, t):
            spans[-1] = (group, t, spans[-1][2] + width)
        else:
            spans.append((group, t, width))
        n_cols += width
    names = dict(TARGETS)
    tgt, rules, col = [], [], 2
    for g, t, w in spans:
        tgt.append(f"\\multicolumn{{{w}}}{{c}}{{{names[t]}}}")
        rules.append(f"\\cmidrule({'l' if col + w - 1 == n_cols else 'lr'}){{{col}-{col + w - 1}}}")
        col += w
    n_lat = sum(w for g, _, w in spans if g == "lat")
    lines = ["% Requires: \\usepackage[table]{xcolor}, booktabs, array, siunitx",
             "% Generated by exp1_latency_energy.py from exp1-latency.csv and exp1-energy.csv -- do not edit by hand."]
    lines += [f"\\definecolor{{{n}}}{{HTML}}{{{rgb}}}" for n, rgb in TEX_COLOUR.values()]
    lines += ["\\begin{table*}[!t]", f"\\caption{{{CAPTION}}}", "\\label{tab:exp1-latency-energy}", "\\centering",
              "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}", f"\\begin{{tabular}}{{@{{}}l{spec}@{{}}}}",
              "\\toprule",
              f" & \\multicolumn{{{n_lat}}}{{c}}{{Inference latency (ms)}} & "
              f"\\multicolumn{{{n_cols - 1 - n_lat}}}{{c}}{{Energy per inference (J)}} \\\\",
              f"\\cmidrule(lr){{2-{1 + n_lat}}}\\cmidrule(lr){{{2 + n_lat}-{n_cols}}}",
              " & " + " & ".join(tgt) + " \\\\", "".join(rules),
              " & ".join(header) + " \\\\"]
    for task, models in tasks.items():
        lines += ["\\midrule", f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{TEX_TASK[task]}}}}} \\\\"]
        for m in models:
            row = [TEX_INDENT + TEX_MODEL[m]]
            for i, t, _, _ in cols:
                cell = v[(m, t)][i] if (m, t) in v else "{--}"
                row.append(cell)
                if i == 0:
                    row.append(f"({fmt_fps(1000 / float(cell))}~fps)" if (m, t) in v else "")
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"[OK] Saved LaTeX table to: {TEX.name}")


if __name__ == "__main__":
    write_tex()
