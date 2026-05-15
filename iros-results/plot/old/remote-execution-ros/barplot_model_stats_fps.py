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
OUTPUT_FILE = "model_stats_inference_fps.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (11.2, 4.5)  # Decreased height from 5.6 to 4.5

SHOW_VALUE_LABELS = False
SHOW_ERROR_BARS = False
VALUE_LABEL_OFFSET = 0.005  # Closer to axis: 0.01-0.005, higher up: 0.03-0.05

MODEL_TYPE_ORDER = get_model_type_order()

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {
        "label": "Robot–CPU (Torch.compile)",
        "match": {"host": "robot", "backend": "ptc", "device": "cpu"},
    },
    {
        "label": "Robot–CPU (SOL)",
        "match": {"host": "robot", "backend": "sol", "device": "cpu"},
    },
    {
        "label": "Edge–CPU (Torch.compile)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-torch",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "Edge–CPU (SOL)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-sol",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "Edge–GPU (Torch.compile)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-torch",
        "run_id_contains": ["target-gpu"],
    },
    {
        "label": "Edge–GPU (SOL)",
        "match": {"host": "robot"},
        "backend_contains": "vaccel-remote-sol",
        "run_id_contains": ["target-gpu"],
    },
]

# Extract just the labels for ordering/colors
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


def print_fps_table(base_models, variants, val_map, err_map):
    # Header
    print("\n=== Inference FPS (system) ===")
    print("model\tvariant\tfps\tstd")

    for m in base_models:
        for v in variants:
            fps = float(val_map[(m, v)])
            err = float(err_map[(m, v)])
            if not np.isfinite(fps):
                continue
            std_s = f"{err:.2f}" if np.isfinite(err) else "-"
            print(f"{m}\t{v}\t{fps:.2f}\t{std_s}")
    print()

def ordered_models(models):
    models = list(dict.fromkeys(models))
    # strip whitespace in order keys to be safe
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
    # Model names are already clean in the new pipeline
    return str(model).strip()


def classify_variant(run: dict):
    """
    Generic classifier that checks the run against VARIANT_DEFINITIONS.
    """
    run_id = str(run.get("run_id", "")).strip()
    backend = str(run.get("backend", "")).lower().strip()
    host = str(run.get("host", "")).lower().strip()
    device = str(run.get("device", "")).lower().strip()

    for v_def in VARIANT_DEFINITIONS:
        # 1. Check exact matches (host, device, backend)
        match_criteria = v_def.get("match", {})
        matches = True
        for key, val in match_criteria.items():
            if locals().get(key) != val:
                matches = False
                break
        
        if not matches:
            continue

        # 2. Check backend substring (useful for vaccel-remote-sol)
        backend_sub = v_def.get("backend_contains")
        if backend_sub and backend_sub not in backend:
            continue

        # 3. Check Run ID substrings (e.g. "target-gpu")
        substrings = v_def.get("run_id_contains", [])
        if substrings:
            if not all(sub in run_id for sub in substrings):
                continue

        return v_def["label"]

    return None


def extract_rows(runs):
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        # Get exact system FPS and STD directly from your new JSON structure
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

        rows.append((
            base_model_name(r.get("model", "")),
            variant,
            fps_f,
            fps_err_f
        ))
    return rows


def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.6))
    
    # Calculate a small padding so the text floats just above the bar
    _, y_top = ax.get_ylim()
    pad_y = y_top * 0.015  
    
    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue

        ax.text(
            x,
            y + pad_y,  # Position exactly above the bar + a little padding
            f"{y:.0f}",
            ha="center",
            va="bottom",  # Aligns the bottom of the text with the y+pad_y coordinate
            rotation=90,  # Keep rotated so labels on thin bars don't overlap
            fontsize=fs,
            color="black",  # Always black since it's on the white background now
            fontweight="bold",
            clip_on=True,
            zorder=20,
        )

# def add_value_labels(ax, xs, ys):
#     fs = max(6, int(plt.rcParams["font.size"] * 0.55))
    
#     # Calculate a nice bottom padding so text doesn't touch the x-axis
#     y_bottom, y_top = ax.get_ylim()
#     pad_y = y_top * VALUE_LABEL_OFFSET
    
#     for x, y in zip(xs, ys):
#         if not np.isfinite(y) or y <= 0:
#             continue

#         # Fallback: if the bar is tiny, put text in the middle of it instead of fixed bottom
#         text_y = pad_y if y > (pad_y * 1.5) else (y / 2.0)

#         ax.text(
#             x,
#             text_y,
#             f"{y:.0f}",
#             ha="center",
#             va="bottom",
#             rotation=90,  # Reads bottom-to-top inside the bar
#             fontsize=fs,
#             color="white" if y > (pad_y * 1.5) else "black", 
#             fontweight="bold",
#             clip_on=True,
#             zorder=20,
#         )


def plot_fps(rows):
    if not rows:
        print("No matching rows found.")
        return

    # --- MODEL_TYPE_ORDER strict filter ---
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print(f"\n[WARNING] Dropped the following models because they are not in MODEL_TYPE_ORDER:\n  {dropped}\n")

    rows = [(m, v, fps, err) for (m, v, fps, err) in rows if m in allowed_models]
    if not rows:
        print("ERROR: No rows remained after filtering! Check the [WARNING] above.")
        return
    # --------------------------------------

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    err_map = {(m, v): np.nan for m in base_models for v in variants}
    
    for m, v, fps, err in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = fps
            err_map[(m, v)] = err
    print_fps_table(base_models, variants, val_map, err_map)

    all_vals = np.asarray([val_map[(m, v)] for m in base_models for v in variants], dtype=float)
    all_errs = np.asarray([err_map[(m, v)] for m in base_models for v in variants], dtype=float)
    
    y_max = np.nanmax(all_vals + (np.nan_to_num(all_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0))

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette(cfg.get("palette"), n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --- SET LIMITS BEFORE DRAWING SO TEXT CAN FIND THE BOTTOM ---
    y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0
    ax.set_ylim(0, y_lim_top)

    x = np.arange(len(base_models))
    
    # Calculate offsets dynamically based on number of variants
    n_vars = len(variants)
    group_width = 0.8
    bar_width = min(0.2, group_width / n_vars)
    
    start = -((n_vars - 1) * bar_width) / 2
    offsets = {v: start + i * bar_width for i, v in enumerate(variants)}

    for v in variants:
        xs = x + offsets[v]
        vals = np.asarray([val_map[(m, v)] for m in base_models], dtype=float)
        yerr = np.asarray([err_map[(m, v)] for m in base_models], dtype=float)

        ax.bar(
            xs, vals, width=bar_width,
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
            add_value_labels(ax, xs, vals)

    ax.set_ylabel("Frames per second (fps)")
    ax.set_xticks(x)
    display_labels = [get_model_display_name(m) for m in base_models]
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.margins(x=0.015)

    style_axes(ax)
    ax.legend(
        title="Execution via vAccel",
        loc="upper right",
        fontsize="small",
        title_fontsize="small",
        ncol=1,
        
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
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
    
    # Generate linear scale plot
    plot_fps(rows)


if __name__ == "__main__":
    main()