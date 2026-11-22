#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order

cfg = load_config()
INPUT_FILE = str(get_path("docker_summary"))

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_COMBINED = "docker_stats_ram_local_exec.pdf"

FONT_SCALE = 1.2
SPINES_WIDTH = 1.0
FIG_SIZE_WIDTH = 10.5
FIG_HEIGHT_PER_SUBPLOT = 4.0

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

# --- VARIANT CONFIGURATION ---
INCLUDE_VACCEL_LOCAL = False

VARIANT_ORDER = ["Torchcompile", "SOL"]
if INCLUDE_VACCEL_LOCAL:
    VARIANT_ORDER.append("SOL + vAccel")

SMOOTH = False
SMOOTH_WINDOW = 3

MODEL_TYPE_ORDER = get_model_type_order()

# --- HARDCODED TARGETS ---
# [UPDATED] Removed ("edge-asus", "gpu") to only show CPU RAM usage
TARGETS = [
    ("robot", "cpu", "docker_stats_ram_robot_cpu_barplot.pdf", "upper left"),
    ("edge-asus", "cpu", "docker_stats_ram_edge_asus_cpu_barplot.pdf", "upper left"),
]


def split_model_variant(model: str, backend: str):
    """
    Decides the legend label (Variant) based on backend.
    """
    backend = str(backend).lower().strip()
    model = str(model).strip()

    base = model

    if backend == "stock":
        return base, "PyTorch"
    
    if backend == "ptc":
        return base, "Torchcompile"

    if backend == "sol":
        return base, "SOL"

    if INCLUDE_VACCEL_LOCAL and backend == "vaccel-local-sol":
        return base, "SOL + vAccel"

    return base, None


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def add_value_labels(ax, xs, ys, yerrs, y_top, show_errors: bool):
    fs = max(8, int(plt.rcParams["font.size"] * 0.8))
    for x, y, e in zip(xs, ys, yerrs):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        err = 0.0
        if show_errors and e is not None and not (isinstance(e, float) and np.isnan(e)):
            err = float(e)
        y_text = y + err + 0.02 * y_top
        ax.text(
            x, y_text, f"{y:.0f}",
            ha="center", va="bottom",
            color="black", fontsize=fs,
            clip_on=False, zorder=20,
        )


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def moving_average(arr, window: int):
    a = np.asarray(arr, dtype=float)
    if window <= 1:
        return a
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(a, kernel, mode="same")


def plot_target(ax, sub, host, device, base_models, color_map, leg_loc):
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {host}-{device}", ha="center", va="center", transform=ax.transAxes)
        return

    # Dynamic Y-limit based on max value in this subplot
    y_max = (sub["mem_mb_mean"].astype(float) + sub["mem_mb_std"].fillna(0).astype(float)).max()
    y_lim_top = (y_max * 1.25) if (not pd.isna(y_max) and y_max > 0) else 1.0

    x = np.arange(len(base_models))

    # --- DYNAMIC BAR OFFSET CALCULATION ---
    n_vars = len(VARIANT_ORDER)
    group_width = 0.8
    bar_width = group_width / n_vars
    
    offsets_arr = np.linspace(
        -group_width/2 + bar_width/2, 
        group_width/2 - bar_width/2, 
        n_vars
    )
    offsets = {v: off for v, off in zip(VARIANT_ORDER, offsets_arr)}

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    stats = {v: {"mean": [], "std": []} for v in VARIANT_ORDER}
    for m in base_models:
        for v in VARIANT_ORDER:
            r = sub[(sub["base_model"] == m) & (sub["variant"] == v)]
            stats[v]["mean"].append(float(r.iloc[0]["mem_mb_mean"]) if not r.empty else np.nan)
            stats[v]["std"].append(float(r.iloc[0]["mem_mb_std"]) if not r.empty else np.nan)

    for v in VARIANT_ORDER:
        xs = x + offsets[v]
        means = np.asarray(stats[v]["mean"], dtype=float)
        stds = np.asarray(stats[v]["std"], dtype=float)

        ax.bar(
            xs, means, width=bar_width,
            color=color_map[v],
            edgecolor=edgecolor, linewidth=linewidth,
            label=v, zorder=3,
        )

        if SHOW_ERROR_BARS:
            valid_mask = ~np.isnan(means)
            if np.any(valid_mask):
                ax.errorbar(
                    xs[valid_mask], means[valid_mask], yerr=stds[valid_mask],
                    fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=4, capthick=1.0, zorder=10,
                )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, means, stds, y_lim_top, SHOW_ERROR_BARS)

        if SMOOTH:
            y = means.copy()
            if np.any(~np.isnan(y)):
                y[np.isnan(y)] = np.interp(
                    np.flatnonzero(np.isnan(y)),
                    np.flatnonzero(~np.isnan(y)),
                    y[~np.isnan(y)],
                )
            else:
                y[:] = 0.0
            y_s = moving_average(y, SMOOTH_WINDOW)
            ax.plot(x, y_s, linewidth=1.8, color="black", alpha=0.35, zorder=6)

    # Labeling
    host_u = str(host).strip()

    #ax.set_xlabel("ML Model")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    ax.set_ylabel(f"{host_u}\nRAM utilization (MB)")

    style_axes(ax)
    ax.legend(
        loc=leg_loc,
        frameon=True, framealpha=0.9, borderpad=0.4,
        handlelength=1.4, fontsize="small", title_fontsize="small",
    )


def main():
    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize columns
    df["host"] = df["host"].astype(str).str.lower().str.strip()
    df["device"] = df["device"].astype(str).str.lower().str.strip()
    df["backend"] = df["backend"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["container"] = df["container"].astype(str).str.strip()

    # UPDATED: Backends to include in this plot
    backends = ["stock", "ptc", "sol"] + (["vaccel-local-sol"] if INCLUDE_VACCEL_LOCAL else [])

    # Filter for container and backend only (don't filter device yet!)
    df = df[
        (df["container"] == "torchvision-app")
        & (df["backend"].isin(backends))
    ].copy()

    if df.empty:
        raise SystemExit(f"No rows matched container='torchvision-app' and backend in {backends}.")

    # Split variants
    base_variant = df.apply(lambda r: split_model_variant(r["model"], r["backend"]), axis=1)
    df["base_model"] = base_variant.apply(lambda t: t[0])
    df["variant"] = base_variant.apply(lambda t: t[1])
    df = df[df["variant"].notna()].copy()

    # Model Filter
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    df = df[df["base_model"].isin(allowed_models)].copy()
    if df.empty:
        raise SystemExit("No rows remained after filtering models.")

    # Setup Ordering
    base_models = ordered_models(sorted(df["base_model"].unique().tolist()))
    df["base_model"] = pd.Categorical(df["base_model"], categories=base_models, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    # --- SEPARATE ---
    if PLOT_MODE == "separate":
        for host, device, out_file, leg_loc in TARGETS:
            sub = df[(df["host"] == host) & (df["device"] == device)].copy()
            if sub.empty:
                print(f"[SKIP] No data for {host}-{device}")
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            plot_target(ax, sub, host, device, base_models, color_map, leg_loc)
            plt.tight_layout()
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved plot to: {out_file}")
            plt.close(fig)

    # --- COMBINED ---
    else:
        num_plots = len(TARGETS)
        if num_plots == 0:
            print("No targets configured.")
            return

        total_height = num_plots * FIG_HEIGHT_PER_SUBPLOT
        fig, axes = plt.subplots(num_plots, 1, figsize=(FIG_SIZE_WIDTH, total_height))
        if num_plots == 1: axes = [axes]

        for ax, (host, device, _, leg_loc) in zip(axes, TARGETS):
            sub = df[(df["host"] == host) & (df["device"] == device)].copy()
            plot_target(ax, sub, host, device, base_models, color_map, leg_loc)

        plt.tight_layout()
        fig.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {OUTPUT_COMBINED}")
        plt.close(fig)


if __name__ == "__main__":
    main()