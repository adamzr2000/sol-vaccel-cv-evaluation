#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../../experiments/docker-stats/_summary/run1_overall_resource_usage_per_container.csv"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_BASENAME = "./docker_stats_cpu"

FONT_SCALE = 1.2
SPINES_WIDTH = 1.0
FIG_SIZE = (8, 5)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

HOST_ORDER = ["robot", "edge"]

# Toggle: 2 bars (PyTorch, SOL) vs 3 bars (PyTorch, SOL, SOL + vAccel)
INCLUDE_VACCEL_LOCAL = False
VARIANT_ORDER = ["PyTorch", "SOL", "SOL + vAccel"] if INCLUDE_VACCEL_LOCAL else ["PyTorch", "SOL"]

# Optional: show a light "trend" line per variant across models (simple smoothing)
SMOOTH = False
SMOOTH_WINDOW = 3  # moving average window (odd works best)

LEGEND_LOC = {
    "robot": "upper left",
    "edge": "upper left",
}

# --- FILTER CONFIGURATION (match your other plots) ---
MODEL_TYPE_ORDER = [
    "mobilenet_v3_large", "resnet50", "swin_t", "swin_s", "swin_v2_b",
    "swin3d_t", "swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "fcn_resnet50", "fcn_resnet101",
]
def split_model_variant(model: str, backend: str):
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
            x,
            y_text,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=fs,
            clip_on=False,
            zorder=20,
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


def plot_host(ax, sub, host, base_models, color_map, y_lim_top):
    x = np.arange(len(base_models))

    if len(VARIANT_ORDER) == 3:
        width = 0.24
        offsets = {"PyTorch": -width, "SOL": 0.0, "SOL + vAccel": +width}
    else:
        width = 0.34
        offsets = {"PyTorch": -width / 2, "SOL": +width / 2}

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    stats = {v: {"mean": [], "std": []} for v in VARIANT_ORDER}
    for m in base_models:
        for v in VARIANT_ORDER:
            r = sub[(sub["base_model"] == m) & (sub["variant"] == v)]
            stats[v]["mean"].append(float(r.iloc[0]["cpu_percent_mean"]) if not r.empty else np.nan)
            stats[v]["std"].append(float(r.iloc[0]["cpu_percent_std"]) if not r.empty else np.nan)

    for v in VARIANT_ORDER:
        xs = x + offsets[v]
        means_pct = np.asarray(stats[v]["mean"], dtype=float)
        stds_pct = np.asarray(stats[v]["std"], dtype=float)

        # Convert percent -> vCPUs (100% ~= 1 vCPU)
        means = means_pct / 100.0
        stds = stds_pct / 100.0

        ax.bar(
            xs,
            means,
            width=width,
            color=color_map[v],
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=v,
            zorder=3,
        )

        if SHOW_ERROR_BARS:
            valid_mask = ~np.isnan(means)
            ax.errorbar(
                xs[valid_mask],
                means[valid_mask],
                yerr=stds[valid_mask],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                capthick=1.0,
                zorder=10,
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

    # no title (per request)
    ax.set_xlabel("ML Model")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    ax.set_ylabel(f"{host.capitalize()} CPU utilization (vCPUs)")

    style_axes(ax)
    ax.legend(
        loc=LEGEND_LOC.get(host, "upper right"),
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )


def main():
    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    needed = {
        "container", "host", "device", "model", "backend",
        "cpu_percent_mean", "cpu_percent_std",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    df["host"] = df["host"].astype(str).str.lower().str.strip()
    df["device"] = df["device"].astype(str).str.lower().str.strip()
    df["backend"] = df["backend"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["container"] = df["container"].astype(str).str.strip()

    backends = ["stock"] + (["vaccel-local"] if INCLUDE_VACCEL_LOCAL else [])
    df = df[
        (df["container"] == "torchvision-app")
        & (df["backend"].isin(backends))
        & (df["device"] == "cpu")
    ].copy()
    if df.empty:
        raise SystemExit("No rows after filtering container='torchvision-app', backends, device='cpu'.")

    base_variant = df.apply(lambda r: split_model_variant(r["model"], r["backend"]), axis=1)
    df["base_model"] = base_variant.apply(lambda t: t[0])
    df["variant"] = base_variant.apply(lambda t: t[1])
    df = df[df["variant"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows after parsing variants (check backends + *_sol availability).")

    # --- STRICT MODEL FILTER (match your other plots) ---
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    dropped = sorted({m for m in df["base_model"].unique() if m not in allowed_models})
    if dropped:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {dropped}\n")
    df = df[df["base_model"].isin(allowed_models)].copy()
    if df.empty:
        raise SystemExit("ERROR: No rows remained after filtering! Check the [WARNING] above.")
    # ---------------------------------------------------

    present_hosts = [h for h in HOST_ORDER if h in set(df["host"])]
    if not present_hosts:
        present_hosts = sorted(df["host"].unique().tolist())

    base_models = ordered_models(sorted(df["base_model"].unique().tolist()))
    df["host"] = pd.Categorical(df["host"], categories=present_hosts, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["base_model"] = pd.Categorical(df["base_model"], categories=base_models, ordered=True)

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(VARIANT_ORDER))
    color_map = {v: pal[i] for i, v in enumerate(VARIANT_ORDER)}

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    if PLOT_MODE == "combined":
        y_lim_top_by_host = {}
        for host in present_hosts:
            subh = df[df["host"] == host].copy()
            y_max_pct = (subh["cpu_percent_mean"].astype(float) + subh["cpu_percent_std"].fillna(0).astype(float)).max()
            y_max = (y_max_pct / 100.0) if (not pd.isna(y_max_pct) and y_max_pct > 0) else 0.0
            y_lim_top_by_host[host] = (y_max * 1.25) if y_max > 0 else 1.0

        n = len(present_hosts)
        fig, axes = plt.subplots(
            nrows=n, ncols=1,
            figsize=(FIG_SIZE[0], FIG_SIZE[1] * n),
            sharex=False,
            sharey=False,
        )
        if n == 1:
            axes = [axes]

        for ax, host in zip(axes, present_hosts):
            sub = df[df["host"] == host].copy()
            if sub.empty:
                ax.axis("off")
                continue
            plot_host(ax, sub, host, base_models, color_map, y_lim_top_by_host[host])

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}_local_exec.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for host in present_hosts:
            sub = df[df["host"] == host].copy()
            if sub.empty:
                continue

            y_max_pct = (sub["cpu_percent_mean"].astype(float) + sub["cpu_percent_std"].fillna(0).astype(float)).max()
            y_max = (y_max_pct / 100.0) if (not pd.isna(y_max_pct) and y_max_pct > 0) else 0.0
            y_lim_top = (y_max * 1.25) if y_max > 0 else 1.0

            fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
            plot_host(ax, sub, host, base_models, color_map, y_lim_top)

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{host}_local_exec.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {host} plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
