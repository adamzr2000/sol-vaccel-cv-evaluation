#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import get_path, load_config, get_model_type_order

# --- CONFIGURATION ---
cfg = load_config()
REMOTE_HOST = cfg.get("remote_host", "edge-asus")
INPUT_FILE = str(get_path("model_summary"))
OUTPUT_FILE = "model_stats_inference_fps.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (11.2, 5.6)

SHOW_VALUE_LABELS = True
SHOW_ERROR_BARS = True

MODEL_TYPE_ORDER = get_model_type_order()

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {
        "label": "Robot CPU (vaccel-local-torch.compile)",
        "match": {"host": "robot", "backend": "ptc", "device": "cpu"},
    },
    {
        "label": "Robot CPU (vaccel-local-sol)",
        "match": {"host": "robot", "backend": "sol", "device": "cpu"},
    },
    {
        "label": "Edge CPU (vaccel-remote-sol)",
        "match": {"host": "robot"},
        "backend_contains": "remote",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "Edge GPU (vaccel-remote-sol)",
        "match": {"host": "robot"},
        "backend_contains": "remote",
        "run_id_contains": ["target-gpu"],
    },
]

# Extract just the labels for ordering/colors
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


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
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    is_log = ax.get_yscale() == "log"
    
    # Calculate a nice bottom padding so text doesn't touch the x-axis
    y_bottom, y_top = ax.get_ylim()
    # Offset to place labels just above the bar for better contrast
    if is_log:
        # small multiplicative offset for log scale
        def _above(y):
            return y * 1.10
    else:
        span = max(1e-6, (y_top - y_bottom))
        offset = span * 0.02
        def _above(y):
            return y + offset

    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue

        text_y = _above(y)

        ax.text(
            x,
            text_y,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=fs,
            color="black",
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )


def plot_fps(rows, log_scale=False):
    if not rows:
        print("No matching rows found.")
        return

    # --- MODEL_TYPE_ORDER strict filter ---
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, _, _, _ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped and not log_scale: # Only print warning once
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

    all_vals = np.asarray([val_map[(m, v)] for m in base_models for v in variants], dtype=float)
    all_errs = np.asarray([err_map[(m, v)] for m in base_models for v in variants], dtype=float)
    
    y_max = np.nanmax(all_vals + (np.nan_to_num(all_errs, nan=0.0) if SHOW_ERROR_BARS else 0.0))

    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --- SET LIMITS BEFORE DRAWING SO TEXT CAN FIND THE BOTTOM ---
    if log_scale:
        ax.set_yscale("log")
        valid_vals = all_vals[all_vals > 0]
        # using 0.1 as a safe bottom fallback for < 1.0 FPS bounds
        y_min = np.nanmin(valid_vals) if len(valid_vals) > 0 else 0.1
        ax.set_ylim(bottom=max(0.1, y_min * 0.5), top=(y_max * 2.0))
    else:
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
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.margins(x=0.015)

    style_axes(ax)
    ax.legend(
        title=None,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.4,
    )

    plt.tight_layout()
    out_file = OUTPUT_FILE.replace(".pdf", "_log.pdf") if log_scale else OUTPUT_FILE
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot to: {out_file}")
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
    
    # Generate both linear and log scale plots
    plot_fps(rows, log_scale=False)
    plot_fps(rows, log_scale=True)


if __name__ == "__main__":
    main()