from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).with_name("plot_config.json")

def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_tag = cfg.get("run_tag", "").strip()
    link = cfg.get("link", "").strip()
    if not run_tag or not link:
        raise SystemExit("plot_config.json must define non-empty 'run_tag' and 'link'")

    # Expand templated paths
    paths = cfg.get("paths", {})
    expanded = {}
    for k, v in paths.items():
        expanded[k] = str(v).format(run_tag=run_tag, link=link)

    cfg["paths_expanded"] = expanded
    return cfg

def get_path(key: str) -> Path:
    cfg = load_config()
    p = cfg["paths_expanded"].get(key)
    if not p:
        raise SystemExit(f"Missing path key in config: {key}")
    return Path(p).resolve()

def get_model_type_order() -> list[str]:
    cfg = load_config()
    order = cfg.get("model_type_order")

    if not isinstance(order, list) or not order:
        raise SystemExit(
            "plot_config.json must define non-empty 'model_type_order' list"
        )

    return [str(x).strip() for x in order]


def get_seg_remote_run_tag() -> str:
    """
    Run tag that remote (vaccel-remote-*) rows for segmentation models
    (get_seg_models()) are sourced from instead of the base run_tag - e.g.
    "e2efps12" for a 12fps-capped rerun. "e2e" disables the override (use
    the base run_tag for everything).
    """
    cfg = load_config()
    return str(cfg.get("seg_remote_run_tag", cfg.get("run_tag", "e2e"))).strip()


def get_seg_models() -> set[str]:
    cfg = load_config()
    return {str(m).strip() for m in cfg.get("seg_models", [])}


def seg_override_enabled() -> bool:
    """True if get_seg_remote_run_tag() differs from the base run_tag (i.e.
    there's actually an override to apply)."""
    cfg = load_config()
    return get_seg_remote_run_tag() != str(cfg.get("run_tag", "")).strip()


def is_seg_remote(model: str, backend: str) -> bool:
    return str(model).strip().lower() in get_seg_models() and str(backend).strip().lower().startswith("vaccel-remote")


def get_seg_remote_model_summary_path() -> Path:
    """Path to get_seg_remote_run_tag()'s model_summary JSON - used to source
    num_processed_frames for segmentation-model remote rows consistently with
    apply_seg_remote_override()'s system-stats data (same run, same frame
    count)."""
    return (Path(__file__).parent
            / f"../../experiments/model-stats/vaccel/_summary/{get_seg_remote_run_tag()}_benchmark_summary.json").resolve()


def apply_seg_remote_override(df, stats_kind: str, link: str | None = None):
    """
    For segmentation models' remote (vaccel-remote-*) rows only, swap in data
    from get_seg_remote_run_tag()'s run instead of the base run already in df
    - everything else (non-seg models, seg models' local rows) is untouched.
    No-op if the configured tag matches the base run_tag.

    df must already have lowercased/stripped host/model/backend/device
    columns (same convention the CSV-based plot scripts already apply).
    stats_kind: "cpu" | "gpu" - selects {tag}_overall_{kind}_stats_{link}.csv.
    """
    import pandas as pd

    if not seg_override_enabled():
        return df

    seg_tag = get_seg_remote_run_tag()
    seg_models = get_seg_models()
    is_seg_remote_mask = df["model"].isin(seg_models) & df["backend"].str.startswith("vaccel-remote")
    df = df[~is_seg_remote_mask].copy()

    cfg = load_config()
    link = link or str(cfg.get("link", "wifi")).strip()
    alt_path = (Path(__file__).parent
                / f"../../experiments/system-stats/vaccel/_summary/{seg_tag}_overall_{stats_kind}_stats_{link}.csv").resolve()
    if not alt_path.exists():
        print(f"[WARNING] seg_remote_run_tag={seg_tag!r} file not found: {alt_path}")
        return df

    alt_df = pd.read_csv(alt_path)
    for c in ("host", "model", "backend", "device"):
        alt_df[c] = alt_df[c].astype(str).str.lower().str.strip()
    alt_is_seg_remote = alt_df["model"].isin(seg_models) & alt_df["backend"].str.startswith("vaccel-remote")
    return pd.concat([df, alt_df[alt_is_seg_remote]], ignore_index=True)


# --- Append to plot_config.py ---

MODEL_DISPLAY_NAMES = {
    "resnet50":            "ResNet-50",
    "swin_t":              "Swin-T",
    "swin_s":              "Swin-S",
    "swin_v2_b":           "SwinV2-B",
    "swin3d_t":            "Swin3D-T",
    "swin3d_s":            "Swin3D-S",
    "swin3d_b":            "Swin3D-B",
    "mc3_18":              "MC3-18",
    "r3d_18":              "R3D-18",
    "r2plus1d_18":         "R(2+1)D-18",
    "deeplabv3_resnet50":  "DLv3-R50",
    "deeplabv3_resnet101": "DLv3-R101",
    "fcn_resnet50":        "FCN-R50",
    "fcn_resnet101":       "FCN-R101",
}

def get_model_display_name(internal_name: str) -> str:
    """Returns a clean, human-readable model name for plot labels."""
    return MODEL_DISPLAY_NAMES.get(internal_name, internal_name)