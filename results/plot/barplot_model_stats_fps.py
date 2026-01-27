#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Figure caption (example)
# Inference FPS measured at the robot for multiple vision models. Local execution uses PyTorch and SOL-optimized libraries on the robot CPU, while remote execution offloads inference to edge CPU/GPU resources via vAccel.

INPUT_FILE = "../experiments/model-stats/_summary/run1_benchmark_summary.json"
OUTPUT_FILE = "model_stats_inference_fps.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (11.2, 5.6)

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = False  # no fps std available in summary json

SMOOTH = False
SMOOTH_WINDOW = 3

MODEL_TYPE_ORDER = [
    "swin_t",
    "resnet50",
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50"
]

VARIANTS = [
    "Local · PyTorch @ Robot CPU",
    "Local · SOL @ Robot CPU",
    "Remote · SOL + vAccel @ Edge CPU",
    "Remote · SOL + vAccel @ Edge GPU",
]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    # strip whitespace in order keys to be safe
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
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
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        fps = (r.get("fps", {}) or {}).get("inference", None)
        if fps is None:
            continue

        try:
            fps_f = float(fps)
        except Exception:
            continue

        rows.append((
            base_model_name(r.get("model", "")),
            variant,
            fps_f,
        ))
    return rows


def add_value_labels(ax, xs, ys, y_top):
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    for x, y in zip(xs, ys):
        if not np.isfinite(y):
            continue
        ax.text(
            x, y + 0.02 * y_top, f"{y:.2f}",
            ha="center", va="bottom",
            fontsize=fs, color="black",
            clip_on=False, zorder=20,
        )


def plot_fps(rows):
    if not rows:
        raise SystemExit("No matching rows found (robot local stock + robot vaccel-remote).")

    # --- MODEL_TYPE_ORDER strict filter (same behavior as your other plots) ---
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {dropped}\n")

    rows = [(m, v, fps) for (m, v, fps) in rows if m in allowed_models]
    if not rows:
        raise SystemExit("ERROR: No rows remained after filtering! Check the [WARNING] above.")
    # -----------------------------------------------------------------------

    base_models = ordered_models(sorted({m for m, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    for m, v, fps in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = fps

    all_vals = np.asarray([val_map[(m, v)] for m in base_models for v in variants], dtype=float)
    y_max = np.nanmax(all_vals)
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

        ax.bar(
            xs, vals, width=width,
            color=color_map[v],
            edgecolor="none", linewidth=0.0,
            label=v, zorder=3,
        )

        if SHOW_VALUE_LABELS:
            add_value_labels(ax, xs, vals, y_lim_top)

        if SMOOTH:
            y = vals.copy()
            if np.any(np.isfinite(y)):
                y2 = y.copy()
                if np.any(~np.isfinite(y2)):
                    idx_ok = np.flatnonzero(np.isfinite(y2))
                    if idx_ok.size >= 2:
                        idx_bad = np.flatnonzero(~np.isfinite(y2))
                        y2[idx_bad] = np.interp(idx_bad, idx_ok, y2[idx_ok])
                ax.plot(
                    x, moving_average(y2, SMOOTH_WINDOW),
                    linewidth=1.8, color="black", alpha=0.35, zorder=6
                )

    # ax.set_title("Robot-side inference FPS under local execution and edge offloading")
    ax.set_xlabel("ML Model")
    ax.set_ylabel("FPS (inference)")
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
    plot_fps(rows)


if __name__ == "__main__":
    main()
