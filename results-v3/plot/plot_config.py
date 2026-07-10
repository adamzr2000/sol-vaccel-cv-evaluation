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