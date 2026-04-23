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