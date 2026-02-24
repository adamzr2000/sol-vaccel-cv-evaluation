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
OUTPUT_FILE = "model_stats_inference_latency.pdf"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.0
FIG_SIZE = (11.2, 5.6)

SHOW_VALUE_LABELS = True  # Enabled this for you
SHOW_ERROR_BARS = True

HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH = False

MODEL_TYPE_ORDER = get_model_type_order()

# --- VARIANT CONFIGURATION ---
VARIANT_DEFINITIONS = [
    {
        "label": "Robot CPU (torch.compile)",
        "match": {"host": "robot", "backend": "ptc", "device": "cpu"},
    },
    {
        "label": "Robot CPU (SOL)",
        "match": {"host": "robot", "backend": "sol", "device": "cpu"},
    },
    {
        "label": "5G Edge CPU (vAccel + SOL)",
        "match": {"host": "robot"},
        "backend_contains": "remote",
        "run_id_contains": ["target-cpu"],
    },
    {
        "label": "5G Edge GPU (vAccel + SOL)",
        "match": {"host": "robot"},
        "backend_contains": "remote",
        "run_id_contains": ["target-gpu"],
    },
]


# Extract just the labels for ordering/colors
VARIANTS = [v["label"] for v in VARIANT_DEFINITIONS]


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(SPINES_WIDTH)


def base_model_name(model: str) -> str:
    # Model names are already clean in the new pipeline (e.g., 'resnet50')
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
        b_model = base_model_name(r.get("model", ""))

        if b_model not in MODEL_TYPE_ORDER:
            continue

        variant = classify_variant(r)
        if variant is None:
            continue

        inf = r.get("system_ms", {}) or {}
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

        rows.append((b_model, variant, mean_f, std_f))
    return rows


# --- NEW SIMPLIFIED VALUE LABELS ---
def add_value_labels(ax, xs, ys):
    fs = max(6, int(plt.rcParams["font.size"] * 0.45))
    is_log = ax.get_yscale() == "log"
    
    # Calculate a nice bottom padding so text doesn't touch the x-axis
    y_bottom, y_top = ax.get_ylim()
    pad_y = y_bottom * 1.3 if is_log else y_top * 0.02
    
    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue

        # Fallback: if the bar is tiny, put text in the middle of it instead of fixed bottom
        text_y = pad_y if y > (pad_y * 1.5) else (y / 2.0)

        ax.text(
            x,
            text_y,
            f"{y:.0f}",
            ha="center",
            va="bottom",
            rotation=90,  # 90 degrees reads bottom-to-top inside the bar
            fontsize=fs,
            color="white",
            fontweight="bold",
            clip_on=True,
            zorder=20,
        )


def print_debug_table(val_map, base_models, variants):
    """Prints a formatted ASCII table of the plot data to the console."""
    print("\n" + "="*115)
    print(f"{'DEBUG: EXTRACTED TOTAL SYSTEM LATENCY VALUES (ms)':^115}")
    print("="*115)
    
    header = f"{'Model':<20}"
    for v in variants:
        label = (v[:19] + '...') if len(v) > 22 else v
        header += f" | {label:<20}"
    print(header)
    print("-" * 115)
    
    for m in base_models:
        row_str = f"{m:<20}"
        for v in variants:
            val = val_map.get((m, v), np.nan)
            if np.isnan(val):
                row_str += f" | {'N/A':<20}"
            else:
                row_str += f" | {val:<20.2f}"
        print(row_str)
    print("="*115 + "\n")


def plot_latency(rows, log_scale=False):
    if not rows:
        print("No matching rows found.")
        return

    base_models = ordered_models(sorted({m for m, _, _, _ in rows}))
    variants = VARIANTS

    val_map = {(m, v): np.nan for m in base_models for v in variants}
    std_map = {(m, v): np.nan for m in base_models for v in variants}

    for m, v, mu, sd in rows:
        if m in base_models and v in variants:
            val_map[(m, v)] = float(mu)
            std_map[(m, v)] = float(sd) if sd is not None else np.nan

    if not log_scale:
        print_debug_table(val_map, base_models, variants)

    all_vals = np.asarray([val_map[(m, v)] for m in base_models for v in variants], dtype=float)
    all_std = np.asarray([std_map[(m, v)] for m in base_models for v in variants], dtype=float)

    y_max = np.nanmax(all_vals + (np.nan_to_num(all_std, nan=0.0) if SHOW_ERROR_BARS else 0.0))
    
    sns.set_theme(context="paper", style="ticks", rc={"xtick.direction": "in", "ytick.direction": "in"}, font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=len(variants))
    color_map = {v: pal[i] for i, v in enumerate(variants)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --- SET LIMITS BEFORE DRAWING SO TEXT CAN FIND THE BOTTOM ---
    if log_scale:
        ax.set_yscale("log")
        valid_vals = all_vals[all_vals > 0]
        y_min = np.nanmin(valid_vals) if len(valid_vals) > 0 else 1.0
        ax.set_ylim(bottom=max(1.0, y_min * 0.5), top=(y_max * 2.0))
    else:
        y_lim_top = (y_max * 1.25) if np.isfinite(y_max) and y_max > 0 else 1.0
        ax.set_ylim(0, y_lim_top)

    x = np.arange(len(base_models))
    
    n_vars = len(variants)
    group_width = 0.8
    bar_width = min(0.2, group_width / n_vars)
    
    start = -((n_vars - 1) * bar_width) / 2
    offsets = {v: start + i * bar_width for i, v in enumerate(variants)}

    for v in variants:
        xs = x + offsets[v]
        vals = np.asarray([val_map[(m, v)] for m in base_models], dtype=float)
        yerr = np.asarray([std_map[(m, v)] for m in base_models], dtype=float)

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

    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, rotation=30, ha="right")
    ax.margins(x=0.015)

    if HIGHLIGHT_SOL_SLOWER_THAN_PYTORCH:
        if len(VARIANTS) >= 2:
            v0, v1 = VARIANTS[0], VARIANTS[1]
            for tick, m in zip(ax.get_xticklabels(), base_models):
                mu_tc = val_map.get((m, v0), np.nan)
                mu_sol = val_map.get((m, v1), np.nan)
                if np.isfinite(mu_tc) and np.isfinite(mu_sol) and (mu_sol > mu_tc):
                    tick.set_color("red")

    style_axes(ax)
    
    ax.legend(
        title=None,
        loc="upper right",
        frameon=True,
        ncol=2,
        framealpha=0.9,
        borderpad=0.4
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
    
    plot_latency(rows, log_scale=False)
    plot_latency(rows, log_scale=True)


if __name__ == "__main__":
    main()