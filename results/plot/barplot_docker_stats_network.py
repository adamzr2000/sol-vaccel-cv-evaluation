#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/docker-stats/_summary/run1_overall_resource_usage_per_container.csv"

PLOT_MODE = "combined"  # "combined" or "separate"
OUTPUT_BASENAME = "./docker_stats_network"  # combined -> <basename>.pdf, separate -> <basename>_<host>.pdf

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (9.2, 5.2)

SHOW_VALUE_LABELS = True

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
    "swin_t",
]

HOSTS = ["robot", "edge"]

# What to plot: "tx", "rx", or "both"
TRAFFIC_MODE = "tx"

# Base series (execution stack + hw) kept stable for color consistency
BASE_SERIES = [
    "Remote · SOL + vAccel @ Edge CPU",
    "Remote · SOL + vAccel @ Edge GPU",
]

LEGEND_TITLE = "Execution stack @ execution hardware · Traffic"
LEGEND_LOC = {"robot": "upper right", "edge": "upper right"}


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def base_model_name(model: str) -> str:
    m = str(model).strip()
    return m[:-4] if m.endswith("_sol") else m


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def add_value_labels(ax, xs, ys, y_top):
    fs = max(8, int(plt.rcParams["font.size"] * 0.75))
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

    if host == "robot":
        if device == "cpu_target-cpu":
            return "Remote · SOL + vAccel @ Edge CPU"
        if device == "cpu_target-gpu":
            return "Remote · SOL + vAccel @ Edge GPU"
        return None

    if host == "edge":
        if device == "cpu":
            return "Remote · SOL + vAccel @ Edge CPU"
        if device == "gpu":
            return "Remote · SOL + vAccel @ Edge GPU"
        return None

    return None


def current_series_list(traffic_mode: str):
    mode = _norm_traffic_mode(traffic_mode)
    if mode in {"tx", "rx"}:
        tag = mode.upper()
        return [f"{s} · {tag}" for s in BASE_SERIES]
    return [f"{s} · RX" for s in BASE_SERIES] + [f"{s} · TX" for s in BASE_SERIES]


def extract_rows(df: pd.DataFrame, traffic_mode: str) -> pd.DataFrame:
    tf = _traffic_fields(traffic_mode)

    rows = []
    for _, r in df.iterrows():
        backend = str(r.get("backend", "")).lower().strip()
        host = str(r.get("host", "")).lower().strip()
        container = str(r.get("container", "")).strip()
        device = str(r.get("device", "")).lower().strip()
        model = r.get("model", "")

        if backend != "vaccel-remote":
            continue
        if host not in {"robot", "edge"}:
            continue

        if host == "robot":
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


def plot_host(ax, sub: pd.DataFrame, host: str, base_models: list[str], series_list: list[str],
              color_map: dict[str, tuple], y_lim_top: float):
    x = np.arange(len(base_models))

    if len(series_list) == 2:
        width = 0.34
        offsets = {series_list[0]: -width / 2, series_list[1]: +width / 2}
    else:
        width = 0.18
        offsets = {
            series_list[0]: -1.5 * width,
            series_list[1]: -0.5 * width,
            series_list[2]: +0.5 * width,
            series_list[3]: +1.5 * width,
        }

    value_map = {(m, s): np.nan for m in base_models for s in series_list}
    for _, r in sub.iterrows():
        m, s, v = r["base_model"], r["series"], r["value"]
        if m in base_models and s in series_list:
            value_map[(m, s)] = float(v) if pd.notna(v) else np.nan

    for s in series_list:
        xs = x + offsets[s]
        ys = np.asarray([value_map[(m, s)] for m in base_models], dtype=float)

        ax.bar(xs, ys, width=width, color=color_map[s], edgecolor="none", label=s, zorder=3)

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, ys, y_lim_top)

    ax.set_title("Robot network traffic (torchvision-app container)" if host == "robot"
                 else "Edge network traffic (vaccel-agent container)")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Traffic (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(
        title=LEGEND_TITLE,
        loc=LEGEND_LOC.get(host, "upper right"),
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )


def main():
    traffic_mode = _norm_traffic_mode(TRAFFIC_MODE)
    series_list = current_series_list(traffic_mode)

    csv_path = Path(INPUT_FILE).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    needed = {
        "container", "host", "device", "model", "backend",
        "net_rx_mbps", "net_tx_mbps",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    df["host"] = df["host"].astype(str).str.lower().str.strip()
    df["device"] = df["device"].astype(str).str.lower().str.strip()
    df["backend"] = df["backend"].astype(str).str.lower().str.strip()
    df["container"] = df["container"].astype(str).str.strip()

    long_df = extract_rows(df, traffic_mode)
    if long_df.empty:
        raise SystemExit("No rows matched (vaccel-remote + host/container filters + net_rx/net_tx).")

    base_models = ordered_models(sorted(long_df["base_model"].unique().tolist()))
    long_df["host"] = pd.Categorical(long_df["host"], categories=HOSTS, ordered=True)
    long_df["series"] = pd.Categorical(long_df["series"], categories=series_list, ordered=True)
    long_df["base_model"] = pd.Categorical(long_df["base_model"], categories=base_models, ordered=True)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(series_list))
    color_map = {s: pal[i] for i, s in enumerate(series_list)}

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
            nrows=2, ncols=1,
            figsize=(FIG_SIZE[0], FIG_SIZE[1] * 2),
            sharex=False, sharey=False,
        )
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, host in zip(axes, HOSTS):
            sub = long_df[long_df["host"] == host].copy()
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No data for host={host}", ha="center", va="center", transform=ax.transAxes)
                continue
            plot_host(ax, sub, host, base_models, series_list, color_map, y_top_for(host))

        plt.tight_layout()
        out = f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved combined plot to: {out}")
        plt.close(fig)

    else:
        for host in HOSTS:
            sub = long_df[long_df["host"] == host].copy()
            if sub.empty:
                print(f"[SKIP] No data for host={host}")
                continue

            fig, ax = plt.subplots(figsize=FIG_SIZE)
            plot_host(ax, sub, host, base_models, series_list, color_map, y_top_for(host))

            plt.tight_layout()
            out = f"{OUTPUT_BASENAME}_{host}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {host} plot to: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
