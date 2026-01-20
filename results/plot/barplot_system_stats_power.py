#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CPU_FILE = "../experiments/system-stats/_summary/run1_overall_cpu_stats.csv"
GPU_FILE = "../experiments/system-stats/_summary/run1_overall_gpu_stats.csv"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_BASENAME = "system_stats_power"  # combined -> <basename>.pdf, separate -> <basename>_<panel>.pdf

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE_SINGLE = (11.2, 5.6)
FIG_SIZE_COMBINED = (11.2, 13.6)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
    "swin_t",
]

VARIANTS = [
    "Local · PyTorch @ Robot CPU",
    "Local · SOL @ Robot CPU",
    "Remote · SOL + vAccel @ Edge CPU",
    "Remote · SOL + vAccel @ Edge GPU",
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def base_model_name(model: str) -> str:
    m = str(model).strip()
    return m[:-4] if m.endswith("_sol") else m


def add_value_labels(ax, xs, ys, yerrs, y_top):
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    for x, y, e in zip(xs, ys, yerrs):
        if not np.isfinite(y):
            continue
        err = float(e) if (SHOW_ERROR_BARS and np.isfinite(e)) else 0.0
        ax.text(
            x, y + err + 0.02 * y_top, f"{y:.1f}",
            ha="center", va="bottom",
            fontsize=fs, color="black",
            clip_on=False, zorder=20,
        )


def classify_robot_cpu_variant(row) -> str | None:
    backend = str(row.get("backend", "")).lower().strip()
    device = str(row.get("device", "")).lower().strip()
    model = str(row.get("model", "")).strip()
    is_sol = model.endswith("_sol")

    if backend == "stock" and device == "cpu":
        return VARIANTS[0] if not is_sol else VARIANTS[1]

    if backend == "vaccel-remote" and is_sol:
        if device == "cpu_target-cpu":
            return VARIANTS[2]
        if device == "cpu_target-gpu":
            return VARIANTS[3]

    return None


def load_robot_cpu_rows(cpu_df: pd.DataFrame):
    sub = cpu_df[cpu_df["host"] == "robot"].copy()
    rows = []
    for _, r in sub.iterrows():
        v = classify_robot_cpu_variant(r)
        if v is None:
            continue
        rows.append({
            "base_model": base_model_name(r["model"]),
            "variant": v,
            "mean": float(r["cpu_watts_mean"]),
            "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan,
        })
    return rows


def load_edge_cpu_remote_rows(cpu_df: pd.DataFrame):
    sub = cpu_df[
        (cpu_df["host"] == "edge")
        & (cpu_df["backend"] == "vaccel-remote")
        & (cpu_df["device"] == "cpu")
    ].copy()

    rows = []
    for _, r in sub.iterrows():
        model = str(r.get("model", "")).strip()
        if not model.endswith("_sol"):
            continue
        rows.append({
            "base_model": base_model_name(model),
            "variant": VARIANTS[2],
            "mean": float(r["cpu_watts_mean"]),
            "std": float(r["cpu_watts_std"]) if pd.notna(r["cpu_watts_std"]) else np.nan,
        })
    return rows


def load_edge_gpu_remote_rows(gpu_df: pd.DataFrame):
    sub = gpu_df[
        (gpu_df["host"] == "edge")
        & (gpu_df["backend"] == "vaccel-remote")
        & (gpu_df["device"] == "gpu")
    ].copy()

    rows = []
    for _, r in sub.iterrows():
        model = str(r.get("model", "")).strip()
        if not model.endswith("_sol"):
            continue
        rows.append({
            "base_model": base_model_name(model),
            "variant": VARIANTS[3],
            "mean": float(r["power_draw_w_mean"]),
            "std": float(r["power_draw_w_std"]) if pd.notna(r["power_draw_w_std"]) else np.nan,
        })
    return rows


def compute_offsets(variants_present):
    n = len(variants_present)
    if n == 1:
        width = 0.55
        offsets = {variants_present[0]: 0.0}
        return width, offsets

    width = min(0.22, 0.8 / n)
    if n == 2:
        offsets = {variants_present[0]: -width / 2, variants_present[1]: +width / 2}
        return width, offsets

    center = (n - 1) / 2.0
    offsets = {v: (i - center) * width for i, v in enumerate(variants_present)}
    return width, offsets


def plot_panel(ax, rows, title, variants_present, color_map):
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    base_models = ordered_models(sorted({r["base_model"] for r in rows}))
    mean_map = {(m, v): np.nan for m in base_models for v in variants_present}
    std_map = {(m, v): np.nan for m in base_models for v in variants_present}

    for r in rows:
        m, v = r["base_model"], r["variant"]
        if m in base_models and v in variants_present:
            mean_map[(m, v)] = r["mean"]
            std_map[(m, v)] = r["std"]

    all_means = np.asarray([mean_map[(m, v)] for m in base_models for v in variants_present], dtype=float)
    all_stds = np.asarray([std_map[(m, v)] for m in base_models for v in variants_present], dtype=float)
    y_max = np.nanmax(all_means + np.nan_to_num(all_stds, nan=0.0))
    y_lim_top = (y_max * 1.95) if np.isfinite(y_max) and y_max > 0 else 1.0

    x = np.arange(len(base_models))
    width, offsets = compute_offsets(variants_present)

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    for v in variants_present:
        xs = x + offsets[v]
        means = np.asarray([mean_map[(m, v)] for m in base_models], dtype=float)
        stds = np.asarray([std_map[(m, v)] for m in base_models], dtype=float)

        ax.bar(
            xs, means, width=width,
            color=color_map[v],
            edgecolor=edgecolor, linewidth=linewidth,
            label=v, zorder=3,
        )

        if SHOW_ERROR_BARS:
            yerr = np.where(np.isfinite(stds), stds, 0.0)
            if np.any(yerr > 0):
                ax.errorbar(
                    xs, means, yerr=yerr,
                    fmt="none", ecolor="black",
                    elinewidth=1.5, capsize=4, capthick=1.5, zorder=10
                )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, means, stds, y_lim_top)

    ax.set_title(title)
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Power (W)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(
        title="Execution stack @ execution hardware",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )


def main():
    cpu_path = Path(CPU_FILE).resolve()
    gpu_path = Path(GPU_FILE).resolve()
    if not cpu_path.exists():
        raise SystemExit(f"CPU CSV not found: {cpu_path}")
    if not gpu_path.exists():
        raise SystemExit(f"GPU CSV not found: {gpu_path}")

    cpu_df = pd.read_csv(cpu_path)
    gpu_df = pd.read_csv(gpu_path)

    for c in ("host", "model", "backend", "device"):
        cpu_df[c] = cpu_df[c].astype(str).str.lower().str.strip()
        gpu_df[c] = gpu_df[c].astype(str).str.lower().str.strip()

    robot_rows = load_robot_cpu_rows(cpu_df)
    edge_cpu_rows = load_edge_cpu_remote_rows(cpu_df)
    edge_gpu_rows = load_edge_gpu_remote_rows(gpu_df)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANTS))
    color_map = {v: pal[i] for i, v in enumerate(VARIANTS)}

    panels = [
        ("robot_cpu", "Robot CPU power", robot_rows, VARIANTS),
        ("edge_cpu", "Edge CPU power", edge_cpu_rows, [VARIANTS[2]]),
        ("edge_gpu", "Edge GPU power", edge_gpu_rows, [VARIANTS[3]]),
    ]

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "combined":
        fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_COMBINED)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, (_key, title, rows, vars_present) in zip(axes, panels):
            plot_panel(ax, rows, title, vars_present, color_map)

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for key, title, rows, vars_present in panels:
            fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_SINGLE)
            plot_panel(ax, rows, title, vars_present, color_map)

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{key}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
