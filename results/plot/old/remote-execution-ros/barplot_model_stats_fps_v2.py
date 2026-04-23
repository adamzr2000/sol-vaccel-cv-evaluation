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
OUTPUT_FILE = "model_stats_inference_fps_v2.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (16, 5.5)

SHOW_VALUE_LABELS = True
SHOW_ERROR_BARS = False

MODEL_TYPE_ORDER = get_model_type_order()

# --- MODEL CATEGORIES ---
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]

CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {
        "label": "Robot-CPU (torch.compile)",
        "match": {"host": "robot", "backend": "ptc", "device": "cpu"},
    },
    {
        "label": "Robot-CPU (SOL)",
        "match": {"host": "robot", "backend": "sol", "device": "cpu"},
    },
    {
        "label": "Edge-CPU (torch.compile)",
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
        "label": "Edge-GPU (torch.compile)",
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

        fps_data = r.get("fps", {}) or {}
        fps = fps_data.get("system", None)
        fps_err = fps_data.get("system_std", np.nan)
        if fps is None:
            continue

        try:
            fps_f = float(fps)
            fps_err_f = float(fps_err) if fps_err is not None else np.nan
        except Exception:
            continue

        rows.append((base_model_name(r.get("model", "")), variant, fps_f, fps_err_f))
    return rows


def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.9))
    _, y_top = ax.get_ylim()
    pad_y = y_top * 0.015

    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue
        ax.text(
            x,
            y + pad_y,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fs,
            color="black",
            fontweight="bold",
            clip_on=True,
            zorder=20,
        )


def plot_fps(rows):
    if not rows:
        print("No matching rows found.")
        return

    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print(
            "\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n"
            f"  {dropped}\n"
        )

    rows = [(m, v, fps, err) for (m, v, fps, err) in rows if m in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering! Check the [WARNING] above.")
        return

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    err_map = {(m, v): np.nan for m in base_models for v in variants}
    for m, v, fps, err in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = fps
            err_map[(m, v)] = err

    all_vals = np.asarray(
        [val_map[(m, v)] for m in base_models for v in variants], dtype=float
    )
    all_errs = np.asarray(
        [err_map[(m, v)] for m in base_models for v in variants], dtype=float
    )
    y_max = np.nanmax(
        all_vals + (np.nan_to_num(all_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0)
    )

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
    cat_captions = []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            # Keep captions aligned with only-present categories
            cap = CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else ""
            cat_captions.append(cap)

    widths = [len(cm) for cm in cat_models]

    # Create horizontal subplots with shared Y axis
    # (We set spacing with subplots_adjust later to avoid tight_layout warnings)
    fig, axes = plt.subplots(
        1,
        len(cat_models),
        figsize=FIG_SIZE,
        sharey=True,
        gridspec_kw={"width_ratios": widths},
    )
    if len(cat_models) == 1:
        axes = [axes]

    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0

    n_vars = len(variants)
    group_width = 0.8
    bar_width = min(0.2, group_width / n_vars)

    start = -((n_vars - 1) * bar_width) / 2
    offsets = {v: start + i * bar_width for i, v in enumerate(variants)}

    for ax_idx, (ax, current_models) in enumerate(zip(axes, cat_models)):
        ax.set_ylim(0, y_lim_top)
        x = np.arange(len(current_models))

        for v in variants:
            xs = x + offsets[v]
            vals = np.asarray([val_map[(m, v)] for m in current_models], dtype=float)
            yerr = np.asarray([err_map[(m, v)] for m in current_models], dtype=float)

            ax.bar(
                xs,
                vals,
                width=bar_width,
                color=color_map[v],
                edgecolor=("black" if SHOW_ERROR_BARS else "none"),
                linewidth=(1.0 if SHOW_ERROR_BARS else 0.0),
                label=v if ax_idx == 0 else "",
                zorder=3,
            )

            if SHOW_ERROR_BARS:
                ax.errorbar(
                    xs,
                    vals,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                    capthick=1.0,
                    zorder=10,
                )

            if SHOW_VALUE_LABELS:
                add_value_labels(ax, xs, vals)

        ax.set_xticks(x)
        display_labels = [get_model_display_name(m) for m in current_models]
        ax.set_xticklabels(display_labels, rotation=30, ha="right")

        # Caption under each subplot (category label)
        caption = cat_captions[ax_idx] if ax_idx < len(cat_captions) else ""
        if caption:
            ax.text(
                0.5,
                -0.32,  # tune together with fig.subplots_adjust(bottom=...)
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
            )

        # Keep categories snug; don't add extra whitespace
        ax.margins(x=0.005)

        style_axes(ax)

        if ax_idx == 0:
            ax.set_ylabel("Frames per second (fps)")
        else:
            ax.set_ylabel("")

    # ---- Manual layout control (no tight_layout warning) ----
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.30,  # room for rotated xticks + captions
        top=0.78,     # room for legend
        wspace=0.04,  # horizontal gap between subplots
    )

    # Single legend for the full figure
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        title="Execution via vAccel",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
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