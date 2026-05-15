#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order, get_model_display_name

# --- CONFIGURATION ---
cfg = load_config()
REMOTE_HOST = cfg.get("remote_host", "edge-asus")
INPUT_FILE = str(get_path("model_summary"))
OUTPUT_FILE = "model_stats_accuracy.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (16, 5.5)

SHOW_VALUE_LABELS = True

MODEL_TYPE_ORDER = get_model_type_order()

# --- MODEL CATEGORIES ---
CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# Per-subplot Y labels (replaces generic "Accuracy / Score (%)" and removes titles)
CATEGORY_YLABELS = ["Top-1 Accuracy (%)", "Top-1 Accuracy (%)", "mIoU (%)"]

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {
        "label": "Robot-CPU (Torch.compile)",
        "match": {"host": "robot", "backend": "stock", "device": "cpu"},
    },
    {
        "label": "Robot-CPU (SOL)",
        "match": {"host": "robot", "backend": "sol", "device": "cpu"},
    },
    {
        "label": "Edge-CPU (Torch.compile)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-torch",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "Edge-CPU (SOL)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-sol",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "Edge-GPU (Torch.compile)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-torch",
        "run_id_contains": ["target-gpu"],
    },
    {
        "label": "Edge-GPU (SOL)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-sol",
        "run_id_contains": ["target-gpu"],
    },
]
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


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


def base_model_name(model: str) -> str:
    return str(model).strip()


def classify_variant(run: dict):
    run_id = str(run.get("run_id", "")).strip()
    backend = str(run.get("backend", "")).lower().strip()
    host = str(run.get("host", "")).lower().strip()
    device = str(run.get("device", "")).lower().strip()

    for v_def in VARIANT_DEFINITIONS:
        match_criteria = v_def.get("match", {})
        matches = True
        for key, val in match_criteria.items():
            if locals().get(key) != val:
                matches = False
                break
        if not matches:
            continue

        backend_sub = v_def.get("backend_contains")
        if backend_sub and backend_sub not in backend:
            continue

        substrings = v_def.get("run_id_contains", [])
        if substrings and not all(sub in run_id for sub in substrings):
            continue

        return v_def["label"]

    return None


def extract_rows(runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        acc_data = r.get("accuracy", {})
        if not acc_data:
            continue

        score = acc_data.get("score", None)
        if score is None:
            continue

        try:
            # Convert decimal score (0.7188) to percentage (71.88)
            score_pct = float(score) * 100.0
        except Exception:
            continue

        rows.append((base_model_name(r.get("model", "")), variant, score_pct))
    return rows


def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.8))
    _, y_top = ax.get_ylim()
    pad_y = y_top * 0.015

    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue
        ax.text(
            x,
            y + pad_y,
            f"{y:.1f}",  # 1 decimal place for percentages (e.g., 71.9)
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fs,
            color="black",
            fontweight="bold",
            clip_on=True,
            zorder=20,
        )


def plot_accuracy(rows):
    if not rows:
        print("No matching rows found for accuracy.")
        return

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print(
            "\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n"
            f"  {dropped}\n"
        )

    rows = [(m, v, acc) for (m, v, acc) in rows if m in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering! Check the [WARNING] above.")
        return

    base_models = ordered_models(sorted({m for m, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    for m, v, acc in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = acc

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={"xtick.direction": "in", "ytick.direction": "in"},
        font_scale=FONT_SCALE,
    )
    pal = sns.color_palette(cfg.get("palette"), n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    # Group valid base models into the three categories
    cat_models = []
    cat_ylabels_used = []
    for idx, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_ylabels_used.append(CATEGORY_YLABELS[idx])

    widths = [len(cm) for cm in cat_models]

    fig, axes = plt.subplots(
        1,
        len(cat_models),
        figsize=FIG_SIZE,
        sharey=True,
        gridspec_kw={"width_ratios": widths},
    )
    if len(cat_models) == 1:
        axes = [axes]

    # Lock Y-Axis from 0 to 115 to fit percentages (100 max + room for vertical text labels)
    y_lim_top = 115.0

    n_vars = len(variants)
    group_width = 0.8
    bar_width = min(0.2, group_width / n_vars)

    start = -((n_vars - 1) * bar_width) / 2
    offsets = {v: start + i * bar_width for i, v in enumerate(variants)}

    for ax_idx, (ax, current_models) in enumerate(zip(axes, cat_models)):
        ax.set_ylim(0, y_lim_top)

        # No subplot titles; y-axis label carries the metric name
        x = np.arange(len(current_models))

        for v in variants:
            xs = x + offsets[v]
            vals = np.asarray([val_map[(m, v)] for m in current_models], dtype=float)

            ax.bar(
                xs,
                vals,
                width=bar_width,
                color=color_map[v],
                edgecolor="none",
                zorder=3,
                label=v if ax_idx == 0 else "",
            )

            if SHOW_VALUE_LABELS:
                add_value_labels(ax, xs, vals)

        ax.set_xticks(x)
        display_labels = [get_model_display_name(m) for m in current_models]
        ax.set_xticklabels(display_labels, rotation=30, ha="right")

        ax.margins(x=0.005)
        style_axes(ax)

        # Per-subplot metric label
        ax.set_ylabel(cat_ylabels_used[ax_idx])

    fig.subplots_adjust(
        left=0.05,
        right=0.995,
        bottom=0.22,
        top=0.78,   # reclaimed space since titles are removed
        wspace=0.08,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Execution via vAccel",
        fontsize="small",
        title_fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    fig.savefig(OUTPUT_FILE, dpi=300)
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
    plot_accuracy(rows)


if __name__ == "__main__":
    main()