#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "../experiments/model-stats/_summary/run1_benchmark_summary.json"

FONT_SCALE = 1.5
SPINES_WIDTH = 1.5
FIG_SIZE = (8.5, 5.2)

MODEL_TYPE_ORDER = [
    "mc3_18", "r3d_18",
    "deeplabv3_resnet50", "fcn_resnet50",
    "resnet50", "mobilenet_v3_large",
]

VARIANT_ORDER = ["PyTorch", "SOL"]
BACKEND_FILTER = "stock"

TARGETS = [
    ("robot", "cpu", "model_stats_inference_fps_robot_cpu_barplot.pdf", "upper right"),
    ("edge", "cpu", "model_stats_inference_fps_edge_cpu_barplot.pdf", "upper left"),
    ("edge", "gpu", "model_stats_inference_fps_edge_gpu_barplot.pdf", "upper left"),
]


def split_model_variant(model: str):
    if isinstance(model, str) and model.endswith("_sol"):
        return model[:-4], "SOL"
    return model, "PyTorch"


def ordered_models(models):
    models = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_TYPE_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def main():
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    rows = []
    for r in runs:
        backend = str(r.get("backend", "")).lower()
        if backend != BACKEND_FILTER:
            continue

        host = str(r.get("host", "")).lower()
        device = str(r.get("device", "")).lower()
        model = r.get("model", "")
        base_model, variant = split_model_variant(model)

        fps = (r.get("fps", {}) or {}).get("inference", None)
        if fps is None:
            continue

        try:
            fps = float(fps)
        except Exception:
            continue

        rows.append({
            "host": host,
            "device": device,
            "base_model": base_model,
            "variant": variant,
            "fps_inference": fps,
        })

    if not rows:
        raise SystemExit("No rows matched backend='stock' with fps.inference present.")

    df = pd.DataFrame(rows)

    sns.set_theme(context="paper", style="ticks", font_scale=FONT_SCALE)
    pal = sns.color_palette("colorblind", n_colors=2)
    color_map = {"PyTorch": pal[0], "SOL": pal[1]}

    for host, device, out_file, leg_loc in TARGETS:
        sub = df[(df["host"] == host) & (df["device"] == device)].copy()
        if sub.empty:
            print(f"[SKIP] No runs for host={host}, device={device}, backend={BACKEND_FILTER}")
            continue

        base_models = ordered_models(sorted(sub["base_model"].unique().tolist()))
        sub["base_model"] = pd.Categorical(sub["base_model"], categories=base_models, ordered=True)
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        sns.barplot(
            data=sub,
            x="base_model",
            y="fps_inference",
            hue="variant",
            palette=color_map,
            errorbar=None,
            ax=ax,
        )

        ax.set_title(f"Inference FPS - {BACKEND_FILTER} @ {host}-{device}")
        ax.set_xlabel("ML Model")
        ax.set_ylabel("FPS (inference)")

        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.8)

        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(SPINES_WIDTH)

        ax.legend(loc=leg_loc, frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.4, title=None)

        plt.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved plot to: {out_file}")
        plt.close(fig)


if __name__ == "__main__":
    main()
