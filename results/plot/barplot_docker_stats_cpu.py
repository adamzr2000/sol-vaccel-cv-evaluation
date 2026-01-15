#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/docker-stats/_summary/run1_overall_resource_usage_per_container.csv"
OUTPUT_FILE = "./docker_stats_cpu.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (8, 5)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = False

HOST_ORDER = ["robot", "edge"]
VARIANT_ORDER = ["PyTorch", "SOL"]

LEGEND_LOC = {
    "robot": "upper right",
    "edge": "upper left",
}

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]


def split_model_variant(model: str):
    if isinstance(model, str) and model.endswith("_sol"):
        return model[:-4], "SOL"
    return model, "PyTorch"


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
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
            f"{y:.0f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=fs,
            clip_on=False,
            zorder=20,
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

    df = df[
        (df["container"] == "torchvision-app")
        & (df["backend"] == "stock")
        & (df["device"] == "cpu")
    ].copy()
    if df.empty:
        raise SystemExit("No rows after filtering container='torchvision-app', backend='stock', device='cpu'.")

    base_variant = df["model"].apply(split_model_variant)
    df["base_model"] = base_variant.apply(lambda t: t[0])
    df["variant"] = base_variant.apply(lambda t: t[1])

    df = df[df["variant"].isin(VARIANT_ORDER)].copy()
    if df.empty:
        raise SystemExit("No rows after parsing variants (PyTorch/SOL).")

    present_hosts = [h for h in HOST_ORDER if h in set(df["host"])]
    if not present_hosts:
        present_hosts = sorted(df["host"].unique().tolist())

    base_models = ordered_models(sorted(df["base_model"].unique().tolist()))
    df["host"] = pd.Categorical(df["host"], categories=present_hosts, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["base_model"] = pd.Categorical(df["base_model"], categories=base_models, ordered=True)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=2)
    color_map = {"PyTorch": pal[0], "SOL": pal[1]}

    y_max = (df["cpu_percent_mean"].astype(float) + df["cpu_percent_std"].fillna(0).astype(float)).max()
    y_lim_top = (y_max * 1.25) if (not pd.isna(y_max) and y_max > 0) else 1.0

    n = len(present_hosts)
    fig, axes = plt.subplots(1, n, figsize=(FIG_SIZE[0] * n, FIG_SIZE[1]), sharey=True)
    if n == 1:
        axes = [axes]

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    for ax, host in zip(axes, present_hosts):
        sub = df[df["host"] == host].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(base_models))
        width = 0.34

        means_pt, std_pt, means_sol, std_sol = [], [], [], []
        for m in base_models:
            r_pt = sub[(sub["base_model"] == m) & (sub["variant"] == "PyTorch")]
            r_sol = sub[(sub["base_model"] == m) & (sub["variant"] == "SOL")]

            means_pt.append(float(r_pt.iloc[0]["cpu_percent_mean"]) if not r_pt.empty else np.nan)
            std_pt.append(float(r_pt.iloc[0]["cpu_percent_std"]) if not r_pt.empty else np.nan)

            means_sol.append(float(r_sol.iloc[0]["cpu_percent_mean"]) if not r_sol.empty else np.nan)
            std_sol.append(float(r_sol.iloc[0]["cpu_percent_std"]) if not r_sol.empty else np.nan)

        xs_pt = x - width / 2
        xs_sol = x + width / 2

        ax.bar(
            xs_pt, means_pt, width=width,
            color=color_map["PyTorch"],
            edgecolor=edgecolor, linewidth=linewidth,
            label="PyTorch", zorder=3,
        )
        ax.bar(
            xs_sol, means_sol, width=width,
            color=color_map["SOL"],
            edgecolor=edgecolor, linewidth=linewidth,
            label="SOL", zorder=3,
        )

        if SHOW_ERROR_BARS:
            ax.errorbar(xs_pt, means_pt, yerr=std_pt, fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)
            ax.errorbar(xs_sol, means_sol, yerr=std_sol, fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=10)

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs_pt, means_pt, std_pt, y_lim_top, SHOW_ERROR_BARS)
            add_value_labels(ax, xs_sol, means_sol, std_sol, y_lim_top, SHOW_ERROR_BARS)

        host_title = host.capitalize()

        ax.set_title(f"{host_title} CPU utilization")
        ax.set_xlabel("ML Model")
        ax.set_xticks(x)
        ax.set_xticklabels(base_models, rotation=20, ha="right")
        ax.set_ylim(0, y_lim_top)

        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)

        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(SPINES_WIDTH)

        ax.set_ylabel(f"CPU (%)\n(100% ≈ one logical core)")

        ax.legend(
            loc=LEGEND_LOC.get(host, "upper right"),
            frameon=True,
            framealpha=0.9,
            borderpad=0.4,
            handlelength=1.4,
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
