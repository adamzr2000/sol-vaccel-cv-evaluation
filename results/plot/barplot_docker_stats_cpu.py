#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/docker-stats/_summary/run1_overall_resource_usage_per_container.csv"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_BASENAME = "./docker_stats_cpu"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (10.2, 5.4)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = True

HOST_ORDER = ["robot", "edge"]

LEGEND_LOC = {
    "robot": "upper right",
    "edge": "upper right",
}

# --- FILTER CONFIGURATION ---
MODEL_TYPE_ORDER = [
    "swin_t",
    "resnet50",
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50"
]

VARIANTS_ALL = [
    "Local · PyTorch @ Robot CPU",
    "Local · SOL @ Robot CPU",
    "Remote · SOL + vAccel @ Edge CPU",
    "Remote · SOL + vAccel @ Edge GPU",
]


def variants_for_host(host: str):
    host = str(host).lower().strip()
    if host == "robot":
        return VARIANTS_ALL
    return [
        "Remote · SOL + vAccel @ Edge CPU",
        "Remote · SOL + vAccel @ Edge GPU",
    ]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    # We strip whitespace from the order keys just to be safe
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    
    # Sort by rank. If a model somehow isn't in rank (shouldn't happen with filter), push to end.
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def split_model_base(model: str):
    m = str(model).strip()
    is_sol = m.endswith("_sol")
    base = m[:-4] if is_sol else m
    return base, is_sol


def classify_variant(row: dict):
    container = str(row.get("container", "")).lower().strip()
    host = str(row.get("host", "")).lower().strip()
    backend = str(row.get("backend", "")).lower().strip()
    device = str(row.get("device", "")).lower().strip()
    model = str(row.get("model", "")).strip()

    base, is_sol = split_model_base(model)

    # ROBOT (torchvision-app): local + remote
    if host == "robot":
        if container == "torchvision-app" and backend == "stock" and device == "cpu":
            return base, (VARIANTS_ALL[0] if not is_sol else VARIANTS_ALL[1])

        if container == "torchvision-app" and backend == "vaccel-remote" and is_sol:
            if "cpu_target-cpu" in device:
                return base, VARIANTS_ALL[2]
            if "cpu_target-gpu" in device:
                return base, VARIANTS_ALL[3]

        return None, None

    # EDGE (torchvision-app-agent): ONLY remote
    if host == "edge":
        if container == "torchvision-app-agent" and backend == "vaccel-remote" and is_sol:
            if device == "cpu":
                return base, VARIANTS_ALL[2]
            if device == "gpu":
                return base, VARIANTS_ALL[3]
        return None, None

    return None, None


def plot_host(ax, dfh: pd.DataFrame, host: str, base_models, color_map, y_lim_top):
    variants = variants_for_host(host)
    x = np.arange(len(base_models))

    n = len(variants)
    width = 0.24 if n == 2 else 0.18
    if n == 2:
        offsets = {variants[0]: -width / 2, variants[1]: +width / 2}
    else:
        offsets = {
            variants[0]: -1.5 * width,
            variants[1]: -0.5 * width,
            variants[2]: +0.5 * width,
            variants[3]: +1.5 * width,
        }

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

    for vv in variants:
        xs = x + offsets[vv]
        means = np.asarray([mean_map[(m, vv)] for m in base_models], dtype=float)
        stds = np.asarray([std_map[(m, vv)] for m in base_models], dtype=float)

        ax.bar(
            xs, means, width=width,
            color=color_map[vv],
            edgecolor=edgecolor, linewidth=linewidth,
            label=vv, zorder=3,
        )

        if SHOW_ERROR_BARS:
            yerr = np.where(np.isfinite(stds), stds, 0.0)
            if np.any(yerr > 0):
                ax.errorbar(
                    xs, means, yerr=yerr,
                    fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=4, capthick=1.0, zorder=10
                )

    # if host == "robot":
    #     ax.set_title("Robot (application container)")
    # else:
    #     ax.set_title("Edge (vAccel agent container)")

    ax.set_xlabel("ML Model")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(
        title="Execution mode · Backend @ Hardware",
        loc=LEGEND_LOC.get(host, "upper right"),
        frameon=True, framealpha=0.9,
        borderpad=0.4, handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )


def main():
    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic cleanup
    for c in ["container", "host", "device", "backend"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    # --- DIAGNOSTICS: Check what is in the CSV ---
    unique_models_in_csv = sorted(df["model"].unique())
    unique_bases_in_csv = sorted({split_model_base(m)[0] for m in unique_models_in_csv})
    print(f"Found {len(unique_models_in_csv)} unique raw models in CSV.")
    print(f"Found {len(unique_bases_in_csv)} unique BASE models in CSV: {unique_bases_in_csv}")
    
    # Clean the User's List
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    print(f"Filtering for only these {len(allowed_models)} models: {allowed_models}")

    rows = []
    dropped_models = set()

    for _, r in df.iterrows():
        base, variant = classify_variant(r.to_dict())
        
        # 1. Skip if it's not a relevant container/variant
        if variant is None:
            continue
        
        # 2. STRICT FILTER CHECK
        if base not in allowed_models:
            dropped_models.add(base)
            continue

        # 3. CONVERT: Divide by 100 to get vCPUs/Cores
        try:
            mean_val = float(r["cpu_percent_mean"]) / 100.0
            std_val = float(r["cpu_percent_std"]) / 100.0 if pd.notna(r["cpu_percent_std"]) else np.nan
        except (ValueError, TypeError):
            continue

        rows.append({
            "host": r["host"],
            "base_model": base,
            "variant": variant,
            "cpu_percent_mean": mean_val,
            "cpu_percent_std": std_val,
        })

    if dropped_models:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {sorted(list(dropped_models))}\n")

    if not rows:
        raise SystemExit("ERROR: No rows remained after filtering! Check the [WARNING] above to see what was dropped.")

    df2 = pd.DataFrame(rows)
    print(f"Plotting {len(df2)} data points...")

    present_hosts = [h for h in HOST_ORDER if h in set(df2["host"])]
    if not present_hosts:
        present_hosts = sorted(df2["host"].unique().tolist())

    # Ensure we use the exact order from the allowed list
    base_models = ordered_models(sorted(df2["base_model"].unique().tolist()))

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)

    pal = sns.color_palette("colorblind", n_colors=len(VARIANTS_ALL))
    color_map = {v: pal[i] for i, v in enumerate(VARIANTS_ALL)}

    y_label = "CPU utilization (vCPUs)"

    if PLOT_MODE == "combined":
        y_lim_top_by_host = {}
        for host in present_hosts:
            subh = df2[df2["host"] == host].copy()
            y_max = (subh["cpu_percent_mean"] + subh["cpu_percent_std"].fillna(0)).max()
            y_lim_top_by_host[host] = (y_max * 1.55) if (pd.notna(y_max) and y_max > 0) else 1.0

        n = len(present_hosts)
        fig, axes = plt.subplots(
            nrows=n, ncols=1,
            figsize=(FIG_SIZE[0], FIG_SIZE[1] * n),
            sharex=False, sharey=False,
        )
        if n == 1:
            axes = [axes]

        for ax, host in zip(axes, present_hosts):
            sub = df2[df2["host"] == host].copy()
            plot_host(ax, sub, host, base_models, color_map, y_lim_top_by_host[host])
            ax.set_ylabel(f"{host.capitalize()} CPU utilization (vCPUs)")

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for host in present_hosts:
            sub = df2[df2["host"] == host].copy()
            if sub.empty:
                continue
            y_max = (sub["cpu_percent_mean"] + sub["cpu_percent_std"].fillna(0)).max()
            y_lim_top = (y_max * 1.25) if (pd.notna(y_max) and y_max > 0) else 1.0

            fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
            plot_host(ax, sub, host, base_models, color_map, y_lim_top)
            ax.set_ylabel(f"{host.capitalize()} CPU utilization (vCPUs)")

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{host}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {host} plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()