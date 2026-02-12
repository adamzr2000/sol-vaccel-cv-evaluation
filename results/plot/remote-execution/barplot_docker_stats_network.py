#!/usr/bin/env python3
# docker_stats_network.py
#
# Network traffic bar plot from docker-stats summary CSV.
# - Strict model filtering (MODEL_TYPE_ORDER) consistent with your other plots
# - No titles
# - Y-axis label: "Robot Traffic (Mbps)" / "Edge Traffic (Mbps)"
# - When TRAFFIC_MODE="both": color encodes Execution Mode, hatch encodes TX/RX,
#   and bars are paired per model: CPU(TX,RX) then GPU(TX,RX)
# - Two legends: Execution Mode (color) + traffic direction (hatch)

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from plot_config import get_path, load_config, get_model_type_order

# --- CONFIGURATION ---
cfg = load_config()
REMOTE_HOST = cfg.get("remote_host", "edge-asus")
INPUT_FILE = str(get_path("docker_summary"))

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_BASENAME = "./docker_stats_network"  # combined -> <basename>.pdf, separate -> <basename>_<host>.pdf

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (9.2, 5.2)

SHOW_VALUE_LABELS = True

MODEL_TYPE_ORDER = get_model_type_order()

# Defines the plot order and labels for the legend
BASE_SERIES = [
    f"Remote · SOL + vAccel @ {REMOTE_HOST} CPU",
    f"Remote · SOL + vAccel @ {REMOTE_HOST} GPU",
]

LEGEND_LOC = {"robot": "upper left", "edge": "upper left"}

# What to plot: "tx", "rx", or "both"
TRAFFIC_MODE = "both"

# Hatch encodes traffic direction (used when TRAFFIC_MODE="both")
HATCH_MAP = {"TX": "/////", "RX": "xx"}


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def base_model_name(model: str) -> str:
    m = str(model).strip()
    return m[:-4] if m.endswith("_sol") else m


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def add_value_labels(ax, xs, ys, y_top):
    fs = max(7, int(plt.rcParams["font.size"] * 0.4))
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        ax.text(
            x, y + 0.02 * y_top, f"{y:.1f}",
            ha="center", va="bottom",
            fontsize=fs, color="black",
            clip_on=False, zorder=20,
        )


def _norm_traffic_mode(m: str) -> str:
    m = str(m).lower().strip()
    if m in {"tx", "rx", "both"}:
        return m
    return "tx"


def _traffic_fields(mode: str):
    mode = _norm_traffic_mode(mode)
    if mode == "tx":
        return [("TX", "net_tx_mbps")]
    if mode == "rx":
        return [("RX", "net_rx_mbps")]
    return [("RX", "net_rx_mbps"), ("TX", "net_tx_mbps")]


def classify_base_series(host: str, device: str) -> str | None:
    host = str(host).lower().strip()
    device = str(device).lower().strip()

    # Robot Side (Client)
    if "robot" in host:
        if "cpu_target-cpu" in device:
            return BASE_SERIES[0]
        if "cpu_target-gpu" in device:
            return BASE_SERIES[1]
        return None

    # Edge Side (Server)
    if "edge" in host:
        if device == "cpu":
            return BASE_SERIES[0]
        if device == "gpu":
            return BASE_SERIES[1]
        return None

    return None


def current_series_list(traffic_mode: str):
    mode = _norm_traffic_mode(traffic_mode)
    if mode in {"tx", "rx"}:
        tag = mode.upper()
        return [f"{s} · {tag}" for s in BASE_SERIES]

    # both: pair by exec mode -> TX then RX
    return [
        f"{BASE_SERIES[0]} · TX",
        f"{BASE_SERIES[0]} · RX",
        f"{BASE_SERIES[1]} · TX",
        f"{BASE_SERIES[1]} · RX",
    ]


def extract_rows(df: pd.DataFrame, traffic_mode: str) -> pd.DataFrame:
    tf = _traffic_fields(traffic_mode)

    rows = []
    for _, r in df.iterrows():
        backend = str(r.get("backend", "")).lower().strip()
        host = str(r.get("host", "")).lower().strip()
        container = str(r.get("container", "")).strip()
        device = str(r.get("device", "")).lower().strip()
        model = r.get("model", "")

        # Only remote runs for network plots
        if backend != "vaccel-remote":
            continue
        
        # Valid hosts are 'robot' or any 'edge...'
        if not ("robot" in host or "edge" in host):
            continue

        # Container identity (robot vs edge agent)
        if "robot" in host:
            if container != "torchvision-app":
                continue
        else:  # edge
            if container != "torchvision-app-agent":
                continue

        base_series = classify_base_series(host, device)
        if base_series is None:
            continue

        base_model = base_model_name(model)

        for direction, col in tf:
            v = r.get(col, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            rows.append({
                "host": host,
                "base_model": base_model,
                "series": f"{base_series} · {direction}",
                "value": v,
            })

    return pd.DataFrame(rows)


def plot_host(
    ax,
    sub: pd.DataFrame,
    host: str,
    base_models: list[str],
    series_list: list[str],
    color_map_exec: dict[str, tuple],
    y_lim_top: float,
):
    x = np.arange(len(base_models))

    if len(series_list) == 2:
        width = 0.34
        offsets = {series_list[0]: -width / 2, series_list[1]: +width / 2}
    else:
        width = 0.18
        offsets = {
            series_list[0]: -1.5 * width,  # CPU TX
            series_list[1]: -0.5 * width,  # CPU RX
            series_list[2]: +0.5 * width,  # GPU TX
            series_list[3]: +1.5 * width,  # GPU RX
        }

    value_map = {(m, s): np.nan for m in base_models for s in series_list}
    for _, r in sub.iterrows():
        m, s, v = r["base_model"], r["series"], r["value"]
        if m in base_models and s in series_list:
            value_map[(m, s)] = float(v) if pd.notna(v) else np.nan

    # Bars
    for s in series_list:
        xs = x + offsets[s]
        ys = np.asarray([value_map[(m, s)] for m in base_models], dtype=float)

        # parse label: "{exec} · {dir}"
        exec_mode, direction = [p.strip() for p in s.rsplit("·", 1)]
        direction = direction.upper()

        ax.bar(
            xs, ys, width=width,
            color=color_map_exec.get(exec_mode),
            hatch=HATCH_MAP.get(direction, ""),
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, ys, y_lim_top)

    ax.set_ylabel(f"{host}\nTraffic (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)

    # Legend 1: Execution Mode (colors)
    exec_handles = [
        mpatches.Patch(
            facecolor=color_map_exec[BASE_SERIES[0]],
            edgecolor="black",
            label=BASE_SERIES[0],
        ),
        mpatches.Patch(
            facecolor=color_map_exec[BASE_SERIES[1]],
            edgecolor="black",
            label=BASE_SERIES[1],
        ),
    ]
    leg_loc = LEGEND_LOC.get("robot" if "robot" in host else "edge", "upper left")
    
    leg1 = ax.legend(
        handles=exec_handles,
        title="Execution Mode",
        loc=leg_loc,
        frameon=True, framealpha=0.9,
        borderpad=0.4, handlelength=1.4,
        fontsize="small", title_fontsize="small",
    )
    ax.add_artist(leg1)

    # Legend 2: direction (hatches)
    dir_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_MAP["TX"], label="TX"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_MAP["RX"], label="RX"),
    ]
    ax.legend(
        handles=dir_handles,
        title="Traffic",
        loc="upper right",
        frameon=True, framealpha=0.9,
        borderpad=0.4, handlelength=1.4,
        fontsize="small", title_fontsize="small",
    )


def main():
    traffic_mode = _norm_traffic_mode(TRAFFIC_MODE)
    series_list = current_series_list(traffic_mode)

    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    needed = {
        "container", "host", "device", "model", "backend",
        "net_rx_mbps", "net_tx_mbps",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    # Basic cleanup
    for c in ["container", "host", "device", "backend"]:
        df[c] = df[c].astype(str).str.lower().str.strip()
    df["container"] = df["container"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    print(f"Filtering for only these {len(allowed_models)} models: {allowed_models}")

    long_df = extract_rows(df, traffic_mode)
    if long_df.empty:
        raise SystemExit("No rows matched (vaccel-remote + host/container filters + net_rx/net_tx).")

    # Strict filter
    dropped_models = sorted({m for m in long_df["base_model"].unique() if m not in allowed_models})
    if dropped_models:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {dropped_models}\n")
    long_df = long_df[long_df["base_model"].isin(allowed_models)].copy()
    if long_df.empty:
        raise SystemExit("ERROR: No rows remained after filtering! Check the [WARNING] above.")

    base_models = ordered_models(sorted(long_df["base_model"].unique().tolist()))

    # Categoricals for stable ordering
    # Dynamically find hosts in data
    present_hosts = sorted(long_df["host"].unique().tolist())
    present_hosts.sort(key=lambda h: (0 if "robot" in h else 1, h))

    long_df["host"] = pd.Categorical(long_df["host"], categories=present_hosts, ordered=True)
    long_df["series"] = pd.Categorical(long_df["series"], categories=series_list, ordered=True)
    long_df["base_model"] = pd.Categorical(long_df["base_model"], categories=base_models, ordered=True)

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)

    # Color map is per Execution Mode (2 colors), direction uses hatches
    pal = sns.color_palette("colorblind", n_colors=len(BASE_SERIES))
    color_map_exec = {BASE_SERIES[i]: pal[i] for i in range(len(BASE_SERIES))}

    if PLOT_MODE not in {"combined", "separate"}:
        raise SystemExit("PLOT_MODE must be 'combined' or 'separate'.")

    def y_top_for(host: str) -> float:
        subh = long_df[long_df["host"] == host].copy()
        if subh.empty:
            return 1.0
        y_max = subh["value"].astype(float).max()
        return (y_max * 1.25) if (pd.notna(y_max) and y_max > 0) else 1.0

    if PLOT_MODE == "combined":
        fig, axes = plt.subplots(
            nrows=len(present_hosts), ncols=1,
            figsize=(FIG_SIZE[0], FIG_SIZE[1] * len(present_hosts)),
            sharex=False, sharey=False,
        )
        if len(present_hosts) == 1: axes = [axes]

        for ax, host in zip(axes, present_hosts):
            sub = long_df[long_df["host"] == host].copy()
            if sub.empty:
                ax.axis("off")
                continue
            plot_host(ax, sub, host, base_models, series_list, color_map_exec, y_top_for(host))

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for host in present_hosts:
            sub = long_df[long_df["host"] == host].copy()
            if sub.empty:
                print(f"[SKIP] No data for host={host}")
                continue

            fig, ax = plt.subplots(figsize=FIG_SIZE)
            plot_host(ax, sub, host, base_models, series_list, color_map_exec, y_top_for(host))

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{host.replace('-', '_')}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {host} plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()