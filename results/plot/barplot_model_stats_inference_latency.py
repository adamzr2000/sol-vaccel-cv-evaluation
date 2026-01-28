#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/model-stats/_summary/run1_benchmark_summary.json"
OUTPUT_FILE = "model_stats_inference_latency.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (11.2, 5.6)

SHOW_VALUE_LABELS = True
SHOW_ERROR_BARS = True

MODEL_TYPE_ORDER = [
    "swin_t","swin_s", "swin_v2_b",
    "swin3d_t","swin3d_s","mc3_18", "r3d_18","r2plus1d_18",
    "deeplabv3_resnet50","deeplabv3_resnet101",
    "fcn_resnet50","fcn_resnet101", 
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
    ax.grid(axis="both", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def moving_average(arr, window: int):
    a = np.asarray(arr, dtype=float)
    if window <= 1:
        return a
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(a, kernel, mode="same")


def base_model_name(model: str) -> str:
    m = str(model).strip()
    return m[:-4] if m.endswith("_sol") else m


def classify_variant(run: dict):
    run_id = str(run.get("run_id", "")).strip()
    backend = str(run.get("backend", "")).lower().strip()
    host = str(run.get("host", "")).lower().strip()
    model = str(run.get("model", "")).strip()
    device = str(run.get("device", "")).lower().strip()

    # we plot robot-side observations only
    if host != "robot":
        return None

    is_sol = model.endswith("_sol")

    # Local on robot CPU
    if backend == "stock" and device == "cpu":
        return VARIANTS[0] if not is_sol else VARIANTS[1]

    # Remote offloading (robot measures end-to-end)
    if backend == "vaccel-remote" and is_sol:
        if "cpu_target-cpu" in run_id:
            return VARIANTS[2]
        if "cpu_target-gpu" in run_id:
            return VARIANTS[3]

    return None


def extract_rows(runs):
    """
    Returns rows as tuples:
      (base_model, variant_label, mean_ms, std_ms_or_nan)
    """
    rows = []
    for r in runs:
        # 1. Get the cleaned model name first
        b_model = base_model_name(r.get("model", ""))

        # 2. Check if it is in your approved list. If not, skip it.
        if b_model not in MODEL_TYPE_ORDER:
            continue
        variant = classify_variant(r)
        if variant is None:
            continue

        inf = r.get("inference_latency_ms", {}) or {}
        mean = inf.get("mean", None)
        std = inf.get("std", None)

        if mean is None:
            continue

        try:
            mean_f = float(mean)
        except Exception:
            continue

        try:
            std_f = float(std) if std is not None else np.nan
        except Exception:
            std_f = np.nan

        rows.append((
            b_model,
            variant,
            mean_f,
            std_f,
        ))
    return rows

def add_value_labels(ax, xs, ys, yerrs, y_top, show_errors: bool):
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    pad = 0.02 * y_top
    for x, y, e in zip(xs, ys, yerrs):
        if not np.isfinite(y):
            continue

        err = 0.0
        if show_errors and e is not None and np.isfinite(e):
            err = float(e)

        ax.text(
            x,
            y + err + pad,          # <- key fix: mean + std + padding
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=fs,
            color="black",
            clip_on=False,
            zorder=20,
        )


def plot_latency(rows):
    if not rows:
        raise SystemExit("No matching rows found (robot stock + robot vaccel-remote).")

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}

    for m, v, mu, sd in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = float(mu)
            std_map[(m, v)] = float(sd) if sd is not None else np.nan

    all_vals = np.asarray([val_map[(m, v)] for m in base_models for v in variants], dtype=float)
    all_std = np.asarray([std_map[(m, v)] for m in base_models for v in variants], dtype=float)

    y_max = np.nanmax(all_vals + (np.nan_to_num(all_std, nan=0.0) if SHOW_ERROR_BARS else 0.0))
    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x = np.arange(len(base_models))
    width = 0.18
    offsets = {
        variants[0]: -1.5 * width,
        variants[1]: -0.5 * width,
        variants[2]: +0.5 * width,
        variants[3]: +1.5 * width,
    }

    for v in variants:
        xs = x + offsets[v]
        vals = np.asarray([val_map[(m, v)] for m in base_models], dtype=float)
        yerr = np.asarray([std_map[(m, v)] for m in base_models], dtype=float)

        ax.bar(
            xs, vals, width=width,
            color=color_map[v],
            edgecolor=("black" if SHOW_ERROR_BARS else "none"),
            linewidth=(1.0 if SHOW_ERROR_BARS else 0.0),
            label=v, zorder=3,
        )

        if SHOW_ERROR_BARS:
            ax.errorbar(
                xs, vals, yerr=yerr, fmt="none",
                ecolor="black", elinewidth=1.0, capsize=4, capthick=1.0, zorder=10
            )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, vals, yerr, y_lim_top, SHOW_ERROR_BARS)


    # ax.set_title("Robot-side inference latency under local execution and edge offloading")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=20, ha="right")
    ax.set_ylim(0, y_lim_top)

    style_axes(ax)
    ax.legend(
        title="Execution mode · Backend @ Hardware",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
        fontsize="small",
        title_fontsize="small",
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot to: {OUTPUT_FILE}")
    plt.close(fig)


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    rows = extract_rows(runs)
    plot_latency(rows)


if __name__ == "__main__":
    main()
