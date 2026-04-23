#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
cfg = load_config()
REMOTE_HOST = cfg.get("remote_host", "edge-asus")
INPUT_FILE = str(get_path("docker_summary"))

PLOT_MODE = "separate"  # "combined" or "separate"
OUTPUT_BASENAME = "./docker_stats_cpu"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (10.2, 5.4)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

# Define variants and where they should appear
VARIANT_DEFINITIONS = [
    {
        "id": "local_torchcompile_cpu",
        "label": "Robot CPU (vaccel-local-torch.compile)",
        "show_on_robot": True,
        "show_on_edge": False
    },
    {
        "id": "local_sol_cpu",
        "label": "Robot CPU (vaccel-local-sol)",
        "show_on_robot": True,
        "show_on_edge": False
    },
    {
        "id": "remote_ptc_edge_cpu",
        "label": "Edge CPU (vaccel-remote-torch.compile)",
        "show_on_robot": True,
        "show_on_edge": True
    },
    {
        "id": "remote_sol_edge_cpu",
        "label": "Edge CPU (vaccel-remote-sol)",
        "show_on_robot": True,
        "show_on_edge": True
    },
    {
        "id": "remote_ptc_edge_gpu",
        "label": "Edge GPU (vaccel-remote-torch.compile)",
        "show_on_robot": True,
        "show_on_edge": True
    },
    {
        "id": "remote_sol_edge_gpu",
        "label": "Edge GPU (vaccel-remote-sol)",
        "show_on_robot": True,
        "show_on_edge": True
    },
]

# Generate simple list for ordering/colors
VARIANTS_ALL = [v["label"] for v in VARIANT_DEFINITIONS]

MODEL_TYPE_ORDER = get_model_type_order()

LEGEND_LOC = {
    "robot": "upper right",
    "edge": "upper right", # Fallback default
}



def pretty_host_name(host: str) -> str:
    h = str(host).lower()
    if "robot" in h:
        return "Robot"
    if "edge" in h:
        return "Edge"
    return host  # fallback for unexpected hostnames

def variants_for_host(host: str):
    """
    Returns list of variant labels that should be plotted for the given host.
    """
    host = str(host).lower().strip()

    # 1. Robot Host
    if "robot" in host:
        return [v["label"] for v in VARIANT_DEFINITIONS if v["show_on_robot"]]

    # 2. Edge Host (matches "edge", "edge-asus", "edge-xtreme", etc.)
    if "edge" in host:
        return [v["label"] for v in VARIANT_DEFINITIONS if v["show_on_edge"]]

    return []


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def classify_variant(row: dict):
    """
    Maps a CSV row to a Variant Label defined in VARIANT_DEFINITIONS based
    on the new explicit backend names (ptc, sol, vaccel-remote-ptc, vaccel-remote-sol).
    """
    container = str(row.get("container", "")).lower().strip()
    host = str(row.get("host", "")).lower().strip()
    backend = str(row.get("backend", "")).lower().strip()
    device = str(row.get("device", "")).lower().strip()
    base = str(row.get("model", "")).strip()

    def get_label(vid):
        for v in VARIANT_DEFINITIONS:
            if v["id"] == vid: return v["label"]
        return None

    # --- CASE 1: ROBOT HOST ---
    if "robot" in host:
        if container == "torchvision-app":
            # Local Execution
            if backend == "ptc" and device == "cpu":
                return base, get_label("local_torchcompile_cpu")
            if backend == "sol" and device == "cpu":
                return base, get_label("local_sol_cpu")

            # Remote Execution (Robot Perspective)
            if "vaccel-remote-ptc" in backend:
                if "target-cpu" in device:
                    return base, get_label("remote_ptc_edge_cpu")
                if "target-gpu" in device:
                    return base, get_label("remote_ptc_edge_gpu")
            if "vaccel-remote-sol" in backend:
                if "target-cpu" in device:
                    return base, get_label("remote_sol_edge_cpu")
                if "target-gpu" in device:
                    return base, get_label("remote_sol_edge_gpu")

    # --- CASE 2: EDGE HOST ---
    if "edge" in host:
        if container == "torchvision-app-agent":
            if "vaccel-remote-ptc" in backend:
                if device == "cpu":
                    return base, get_label("remote_ptc_edge_cpu")
                if device == "gpu":
                    return base, get_label("remote_ptc_edge_gpu")
            if "vaccel-remote-sol" in backend:
                if device == "cpu":
                    return base, get_label("remote_sol_edge_cpu")
                if device == "gpu":
                    return base, get_label("remote_sol_edge_gpu")

    return None, None

def hatch_for_variant_label(vlabel: str) -> str | None:
    # Only hatch edge variants (remote ones), and split CPU vs GPU
    if "Edge CPU" in vlabel:
        return "////"
    if "Edge GPU" in vlabel:
        return "oooo"
    return None

def plot_host(ax, dfh: pd.DataFrame, host: str, base_models, color_map, y_lim_top):
    variants = variants_for_host(host)
    if not variants:
        ax.text(0.5, 0.5, f"No configured variants for {host}", ha="center")
        return

    x = np.arange(len(base_models))
    n = len(variants)

    # Auto-adjust bar width and offsets
    group_width = 0.8  # total width for all bars in a group
    width = min(0.18, group_width / n)
    start = -((n - 1) * width) / 2
    offsets = {v: start + i * width for i, v in enumerate(variants)}

    mean_map = {(m, vv): np.nan for m in base_models for vv in variants}
    std_map = {(m, vv): np.nan for m in base_models for vv in variants}

    for _, r in dfh.iterrows():
        m = r["base_model"]
        vv = r["variant"]
        if (m in base_models) and (vv in variants):
            mean_map[(m, vv)] = float(r["cpu_percent_mean"])
            std_map[(m, vv)] = float(r["cpu_percent_std"]) if pd.notna(r["cpu_percent_std"]) else np.nan

    edgecolor = "black" if SHOW_ERROR_BARS else "none"
    linewidth = 1.0 if SHOW_ERROR_BARS else 0.0

    # Define which variants should be hatched (remote variants)
    hatched_variants = [
        v["label"] for v in VARIANT_DEFINITIONS
        if v["id"].startswith("remote_")
    ]

    for vv in variants:
        xs = x + offsets[vv]
        means = np.asarray([mean_map[(m, vv)] for m in base_models], dtype=float)
        stds = np.asarray([std_map[(m, vv)] for m in base_models], dtype=float)

        # Convert percent to vCPUs
        means = means / 100.0
        stds = stds / 100.0

        hatch_pattern = hatch_for_variant_label(vv)

        ax.bar(
            xs, means, width=width,
            color=color_map[vv],
            edgecolor=edgecolor, linewidth=linewidth,
            label=vv, zorder=3,
            hatch=hatch_pattern,
        )

        if SHOW_ERROR_BARS:
            yerr = np.where(np.isfinite(stds), stds, 0.0)
            if np.any(yerr > 0):
                ax.errorbar(
                    xs, means, yerr=yerr,
                    fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10
                )

    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.set_ylim(0, y_lim_top)
    ax.margins(x=0.015)
    style_axes(ax)

    # Pick legend loc based on host string (default to upper right)
    loc = "upper right"
    if "robot" in host.lower():
        loc = LEGEND_LOC.get("robot", "upper right")
    elif "edge" in host.lower():
        loc = LEGEND_LOC.get("edge", "upper right")

    leg = ax.legend(
        title=None,
        loc=loc,
        ncol=1,
        fontsize="small",
        frameon=True, framealpha=0.9,
        borderpad=0.4, handlelength=1.4,
    )
    if leg:
        leg.set_zorder(30)


def main():
    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Cleanup strings
    for c in ["container", "host", "device", "backend"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    # Model Filter
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]

    rows = []
    dropped_models = set()

    for _, r in df.iterrows():
        base, variant = classify_variant(r.to_dict())

        if variant is None:
            continue

        if base not in allowed_models:
            dropped_models.add(base)
            continue

        rows.append({
            "host": r["host"],
            "base_model": base,
            "variant": variant,
            "cpu_percent_mean": r["cpu_percent_mean"],
            "cpu_percent_std": r["cpu_percent_std"],
        })

    if dropped_models:
        print(f"\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  {sorted(list(dropped_models))}\n")

    if not rows:
        raise SystemExit("ERROR: No rows remained after filtering.")

    df2 = pd.DataFrame(rows)
    print(f"Plotting {len(df2)} data points...")

    # Auto-discover available hosts in the data
    present_hosts = sorted(df2["host"].unique().tolist())
    # Sort to put robot first
    present_hosts.sort(key=lambda h: (0 if "robot" in h else 1, h))

    base_models = ordered_models(sorted(df2["base_model"].unique().tolist()))

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette(cfg.get("palette"), n_colors=len(VARIANTS_ALL))
    color_map = {v: pal[i] for i, v in enumerate(VARIANTS_ALL)}

    # Plot
    if PLOT_MODE == "combined":
        y_lim_top_by_host = {}
        for host in present_hosts:
            subh = df2[df2["host"] == host].copy()
            # Calculate max in vCPUs (divide by 100)
            y_max = (subh["cpu_percent_mean"].astype(float) + subh["cpu_percent_std"].fillna(0).astype(float)).max() / 100.0
            y_lim_top_by_host[host] = (y_max * 1.55) if (pd.notna(y_max) and y_max > 0) else 1.0

        n = len(present_hosts)
        fig, axes = plt.subplots(
            nrows=n, ncols=1,
            figsize=(FIG_SIZE[0], FIG_SIZE[1] * n),
            sharex=False, sharey=False,
        )
        if n == 1: axes = [axes]

        for ax, host in zip(axes, present_hosts):
            sub = df2[df2["host"] == host].copy()
            plot_host(ax, sub, host, base_models, color_map, y_lim_top_by_host[host])
            ax.set_ylabel(f"{pretty_host_name(host)} CPU utilization (vCPUs)")

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for host in present_hosts:
            sub = df2[df2["host"] == host].copy()
            if sub.empty: continue

            y_max = (sub["cpu_percent_mean"].astype(float) + sub["cpu_percent_std"].fillna(0).astype(float)).max() / 100.0
            y_lim_top = (y_max * 1.25) if (pd.notna(y_max) and y_max > 0) else 1.0

            fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
            plot_host(ax, sub, host, base_models, color_map, y_lim_top)
            ax.set_ylabel(f"{pretty_host_name(host)} CPU utilization (vCPUs)")

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{host.replace('-', '_')}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {host} plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()