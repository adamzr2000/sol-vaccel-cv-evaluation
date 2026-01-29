#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../../experiments/system-stats/_summary/run1_overall_gpu_stats.csv"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "system_stats_gpu_local_exec.pdf"
FIG_SIZE_COMBINED = (10.5, 11.5)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

FONT_SCALE = 1.2
SPINES_WIDTH = 1.5
FIG_SIZE_SINGLE = (9.0, 5.4)

INCLUDE_VACCEL_LOCAL = False
VARIANT_ORDER = ["PyTorch", "SOL", "SOL + vAccel"] if INCLUDE_VACCEL_LOCAL else ["PyTorch", "SOL"]
LEGEND_LOC = "upper left"

MODEL_TYPE_ORDER = [
    "mobilenet_v3_large", "resnet50", "swin_t", "swin_s", "swin_v2_b",
    "swin3d_t", "swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "fcn_resnet50", "fcn_resnet101",
]

PLOTS = [
    dict(
        out="system_stats_gpu_vram_local_exec.pdf",
        y="mem_used_mb_mean",
        yerr="mem_used_mb_std",
        ylabel="Edge GPU\nVRAM utilization (MB)",
    ),
    dict(
        out="system_stats_gpu_utilization_local_exec.pdf",
        y="util_gpu_percent_mean",
        yerr="util_gpu_percent_std",
        ylabel="Edge GPU utilization\n(%)",
    ),
    dict(
        out="system_stats_gpu_power_local_exec.pdf",
        y="power_draw_w_mean",
        yerr="power_draw_w_std",
        ylabel="Edge GPU power\n(W)",
    ),
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def split_variant(model: str, backend: str):
    backend = str(backend).lower().strip()
    model = str(model).strip()

    is_sol = model.endswith("_sol")
    base = model[:-4] if is_sol else model

    if backend == "stock" and not is_sol:
        return base, "PyTorch"
    if backend == "stock" and is_sol:
        return base, "SOL"
    if INCLUDE_VACCEL_LOCAL and backend == "vaccel-local" and is_sol:
        return base, "SOL + vAccel"

    return base, None


def add_value_labels(ax, xs, ys, yerrs, y_top, show_errors: bool):
    fs = max(8, int(plt.rcParams["font.size"] * 0.6))
    for x, y, e in zip(xs, ys, yerrs):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        err = 0.0
        if show_errors and e is not None and not (isinstance(e, float) and np.isnan(e)):
            err = float(e)
        y_text = y + err + 0.02 * y_top
        ax.text(
            x, y_text, f"{y:.1f}",
            ha="center", va="bottom",
            color="black", fontsize=fs,
            clip_on=False, zorder=20
        )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def plot_metric(df, y_col, yerr_col, ylabel, color_map, ax=None, out_file=None):
    base_models = ordered_models(sorted(df["base_model"].unique().tolist()))
    d = df.copy()
    d["base_model"] = pd.Categorical(d["base_model"], categories=base_models, ordered=True)
    d["variant"] = pd.Categorical(d["variant"], categories=VARIANT_ORDER, ordered=True)

    mean_map = {(m, v): np.nan for m in base_models for v in VARIANT_ORDER}
    std_map = {(m, v): np.nan for m in base_models for v in VARIANT_ORDER}
    for _, r in d.iterrows():
        m = r["base_model"]
        v = r["variant"]
        mean_map[(m, v)] = float(r[y_col]) if pd.notna(r[y_col]) else np.nan
        std_map[(m, v)] = float(r[yerr_col]) if pd.notna(r[yerr_col]) else np.nan

    all_means = np.asarray([mean_map[(m, v)] for m in base_models for v in VARIANT_ORDER], dtype=float)
    all_stds = np.asarray([std_map[(m, v)] for m in base_models for v in VARIANT_ORDER], dtype=float)
    y_max = np.nanmax(all_means + np.nan_to_num(all_stds, nan=0.0))
    y_lim_top = (y_max * 1.50) if np.isfinite(y_max) and y_max > 0 else 1.0

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        created_fig = True

    x = np.arange(len(base_models))
    if len(VARIANT_ORDER) == 3:
        width = 0.24
        offsets = {"PyTorch": -width, "SOL": 0.0, "SOL + vAccel": +width}
    else:
        width = 0.34
        offsets = {"PyTorch": -width / 2, "SOL": +width / 2}

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    for v in VARIANT_ORDER:
        xs = x + offsets[v]

        means = [mean_map[(m, v)] for m in base_models]
        stds = [std_map[(m, v)] for m in base_models]

        means_np = np.array(means, dtype=float)
        stds_np = np.array(stds, dtype=float)

        ax.bar(
            xs, means, width=width,
            color=color_map[v],
            edgecolor=edgecolor, linewidth=linewidth,
            label=v, zorder=3
        )

        if SHOW_ERROR_BARS:
            mask = ~np.isnan(means_np)
            if np.any(mask):
                ax.errorbar(
                    xs[mask],
                    means_np[mask],
                    yerr=stds_np[mask],
                    fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=4, capthick=1.0, zorder=10
                )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, means, stds, y_lim_top, SHOW_ERROR_BARS)

    # no title (per request)
    ax.set_xlabel("ML Model")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(
        loc=LEGEND_LOC,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )

    if created_fig:
        if not out_file:
            raise ValueError("out_file must be provided when ax is None.")
        plt.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved plot to: {out_file}")


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path)

    needed = {
        "host", "model", "backend", "device",
        "power_draw_w_mean", "power_draw_w_std",
        "util_gpu_percent_mean", "util_gpu_percent_std",
        "mem_used_mb_mean", "mem_used_mb_std",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    df["host"] = df["host"].astype(str).str.lower().str.strip()
    df["device"] = df["device"].astype(str).str.lower().str.strip()
    df["backend"] = df["backend"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    allowed_backends = {"stock"}
    if INCLUDE_VACCEL_LOCAL:
        allowed_backends.add("vaccel-local")

    df = df[
        (df["host"] == "edge")
        & (df["device"] == "gpu")
        & (df["backend"].isin(allowed_backends))
    ].copy()
    if df.empty:
        raise SystemExit("No rows after filtering host='edge', device='gpu', backend in {stock,vaccel-local}.")

    base_var = df.apply(lambda r: split_variant(r["model"], r["backend"]), axis=1)
    df["base_model"] = base_var.apply(lambda t: t[0])
    df["variant"] = base_var.apply(lambda t: t[1])
    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    if df.empty:
        raise SystemExit("No rows after parsing variants.")

    # --- STRICT MODEL FILTER (match your other plots) ---
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    dropped = sorted({m for m in df["base_model"].unique() if m not in allowed_models})
    if dropped:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {dropped}\n")
    df = df[df["base_model"].isin(allowed_models)].copy()
    if df.empty:
        raise SystemExit("ERROR: No rows remained after filtering! Check the [WARNING] above.")
    # ---------------------------------------------------

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "separate":
        for cfg in PLOTS:
            plot_metric(df, cfg["y"], cfg["yerr"], cfg["ylabel"], color_map, ax=None, out_file=cfg["out"])
        return

    fig, axes = plt.subplots(len(PLOTS), 1, figsize=FIG_SIZE_COMBINED)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, cfg in zip(axes, PLOTS):
        plot_metric(df, cfg["y"], cfg["yerr"], cfg["ylabel"], color_map, ax=ax, out_file=None)

    plt.tight_layout()
    fig.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved combined plot to: {OUTPUT_COMBINED}")


if __name__ == "__main__":
    main()
