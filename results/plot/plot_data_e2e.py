#!/usr/bin/env python3
"""
plot_data_e2e.py

Shared data loading + extraction for the e2e FPS/latency barplot family
(barplot_e2e_fps_and_latency.py, barplot_e2e_fps.py, barplot_e2e_latency.py).
Variant matching, model categories, and the inference/pre-post/network
split live here once — change how a variant is matched or how network
time is derived and all three plots stay consistent.

See plot_style.py for the visual side (colors/legend/grid/font).
"""
from pathlib import Path
import json

import numpy as np

from plot_config import get_path, load_config, get_model_type_order

_HERE = Path(__file__).parent
INPUT_FILE = str(get_path("model_summary"))
# iso run: edge-asus vaccel-remote-{ptc,sol} inference_ms used as pure-inference baseline for remote breakdown
REMOTE_INFERENCE_FILE = _HERE / "../experiments/model-stats/vaccel/_summary/iso_benchmark_summary.json"

METRIC = "median"  # "median" (p50) | "mean" — controls FPS derivation and latency bars
LOCAL_ROBOT_MODE = "vaccel-rpc"  # "legacy"     -> ptc/sol backends, iros2 run tag
                                 # "vaccel-rpc" -> vaccel-remote-*/remote_host=robot (loopback), iso run tag

MODEL_TYPE_ORDER = get_model_type_order()

# --- Model categories ---
CAT_CAPTIONS = [
    "(a) Image Classification",
    "(b) Video Action Recognition",
    "(c) Semantic Segmentation",
]
CAT_IMAGE = ["resnet50", "swin_t", "swin_s", "swin_v2_b"]
CAT_VIDEO = ["swin3d_s", "swin3d_b", "mc3_18", "r3d_18", "r2plus1d_18"]
CAT_SEG = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
CATEGORIES = [CAT_IMAGE, CAT_VIDEO, CAT_SEG]

# --- Variant definitions (which run matches which deployment scenario) ---
if LOCAL_ROBOT_MODE == "legacy":
    _local_defs = [
        {"label": "Local CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "ptc", "device": "cpu"}},
        {"label": "Local CPU (vAccSOL)", "match": {"host": "robot", "backend": "sol", "device": "cpu"}},
    ]
else:  # vaccel-rpc: robot local via vaccel-local-* (iso run tag, no remote_host)
    _local_defs = [
        {"label": "Local CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-local-ptc", "device": "cpu"}},
        {"label": "Local CPU (vAccSOL)", "match": {"host": "robot", "backend": "vaccel-local-sol", "device": "cpu"}},
    ]

VARIANT_DEFINITIONS = _local_defs + [
    {"label": "Remote CPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-remote-ptc", "device": "cpu", "remote_host": "edge-asus"}},
    {"label": "Remote CPU (vAccSOL)", "match": {"host": "robot", "backend": "vaccel-remote-sol", "device": "cpu", "remote_host": "edge-asus"}},
    {"label": "Remote GPU (vAccel + Torch)", "match": {"host": "robot", "backend": "vaccel-remote-ptc", "device": "gpu", "remote_host": "edge-asus"}},
    {"label": "Remote GPU (vAccSOL)", "match": {"host": "robot", "backend": "vaccel-remote-sol", "device": "gpu", "remote_host": "edge-asus"}},
]


def base_model_name(model: str) -> str:
    return str(model).strip()


def ordered_models(models):
    models = list(dict.fromkeys(models))
    clean_order = [m.strip() for m in MODEL_TYPE_ORDER]
    rank = {m: i for i, m in enumerate(clean_order)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def classify_variant(run: dict):
    run_id = str(run.get("run_id", "")).strip()
    backend = str(run.get("backend", "")).lower().strip()
    host = str(run.get("host", "")).lower().strip()
    device = str(run.get("device", "")).lower().strip()
    remote_host = str(run.get("remote_host", "")).lower().strip()

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


def get_remote_pure_inference(remote_data, model_name, variant_label):
    """Return inference_ms[METRIC] for pure edge inference (no network).

    Aligned with LOCAL_ROBOT_MODE:
      legacy     -> ptc / sol backends (direct, no vAccel RPC layer)
      vaccel-rpc -> vaccel-remote-ptc / vaccel-remote-sol edge-asus loopback
    """
    is_gpu = "gpu" in variant_label.lower()
    is_sol = "sol" in variant_label.lower()

    target_host = "edge-asus"
    target_device = "gpu" if is_gpu else "cpu"
    stat_key = "p50" if METRIC == "median" else "mean"

    if LOCAL_ROBOT_MODE == "legacy":
        target_backend = "sol" if is_sol else "ptc"
        match_remote_host = None  # legacy runs have no remote_host field
    else:  # vaccel-rpc
        target_backend = "vaccel-local-sol" if is_sol else "vaccel-local-ptc"
        match_remote_host = None  # vaccel-local-* runs have no remote_host field

    for r in remote_data:
        if base_model_name(r.get("model", "")) != model_name:
            continue
        if r.get("host", "").lower() != target_host:
            continue
        if r.get("device", "").lower() != target_device:
            continue
        if r.get("backend", "").lower() != target_backend:
            continue
        if match_remote_host is not None:
            if r.get("remote_host", "").lower() != match_remote_host:
                continue
        inf = r.get("inference_ms", {}) or {}
        return float(inf.get(stat_key, 0.0))

    return 0.0


def extract_rows(runs, remote_runs):
    """Returns a list of tuples:
    (model, variant, fps, inference_ms, network_ms, pre_post_ms, err_lower, err_upper)
    """
    stat_key = "p50" if METRIC == "median" else "mean"
    rows = []
    for r in runs:
        variant = classify_variant(r)
        if variant is None:
            continue

        b_model = base_model_name(r.get("model", ""))

        # Latency data (robot perspective = system_ms e2e)
        sys_data = r.get("system_ms", {}) or {}
        sys_metric = sys_data.get(stat_key, None)
        if sys_metric is None:
            continue

        p25 = float(sys_data.get("p25", sys_metric))
        p75 = float(sys_data.get("p75", sys_metric))
        if METRIC == "median":
            err_lower = float(sys_metric) - p25  # asymmetric lower
            err_upper = p75 - float(sys_metric)  # asymmetric upper
        else:
            std = float(sys_data.get("std", 0.0))
            err_lower = std
            err_upper = std

        inf_data = r.get("inference_ms", {}) or {}
        inf_metric = inf_data.get(stat_key, 0.0)

        # FPS derived from system latency (robot e2e) using the chosen metric
        fps = 1000.0 / float(sys_metric) if float(sys_metric) > 0 else None
        if fps is None:
            continue

        pre_data = r.get("preprocessing_ms", {}) or {}
        post_data = r.get("postprocessing_ms", {}) or {}

        try:
            fps_f = float(fps)
            total_f = float(sys_metric)
            inf_f = float(inf_metric) if inf_metric else 0.0
            pre_post_f = float(pre_data.get(stat_key, 0.0)) + float(post_data.get(stat_key, 0.0))
        except Exception:
            continue

        is_remote = "remote" in variant.lower()

        if is_remote:
            pure_inf_f = get_remote_pure_inference(remote_runs, b_model, variant)
            inference_final = pure_inf_f if pure_inf_f > 0 and pure_inf_f <= inf_f else inf_f
            # Network = residual so stack sums exactly to system_ms.p50
            network_f = max(0.0, total_f - inference_final - pre_post_f)
        else:
            network_f = 0.0
            inference_final = inf_f

        rows.append((b_model, variant, fps_f, inference_final, network_f, pre_post_f, err_lower, err_upper))

    return rows


def load_rows():
    """Reads the configured summary JSONs and returns extract_rows(...) output."""
    path = Path(INPUT_FILE).resolve()
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")

    remote_path = REMOTE_INFERENCE_FILE.resolve()
    remote_data = []
    if remote_path.exists():
        with remote_path.open("r") as f:
            remote_data = json.load(f).get("runs", [])
    else:
        print(f"[WARNING] Remote inference file missing: {REMOTE_INFERENCE_FILE}")

    with path.open("r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("Input JSON does not contain a non-empty 'runs' list.")

    # In vaccel-rpc local mode, robot loopback data lives in iso (not iros2)
    if LOCAL_ROBOT_MODE == "vaccel-rpc":
        iso_path = REMOTE_INFERENCE_FILE.resolve()
        if iso_path.exists():
            with iso_path.open() as f:
                runs = runs + json.load(f).get("runs", [])
        else:
            print(f"[WARNING] iso file not found for vaccel-rpc local mode: {iso_path}")

    return extract_rows(runs, remote_data)


def filter_to_known_models(rows):
    """Drops rows whose model isn't in MODEL_TYPE_ORDER (warns once) and returns
    (rows, base_models, cat_models, cat_captions) ready for plotting."""
    allowed_models = [m.strip() for m in MODEL_TYPE_ORDER]
    present_models = sorted({m for m, *_ in rows})
    dropped = sorted([m for m in present_models if m not in allowed_models])
    if dropped:
        print("\n[WARNING] Dropped models not in MODEL_TYPE_ORDER:\n  " + str(dropped) + "\n")

    rows = [row for row in rows if row[0] in allowed_models]
    if not rows:
        raise SystemExit("ERROR: No rows remained after filtering!")

    base_models = ordered_models(sorted({r[0] for r in rows}))

    cat_models, cat_captions = [], []
    for i, cat in enumerate(CATEGORIES):
        models_in_cat = [m for m in base_models if m in cat]
        if models_in_cat:
            cat_models.append(models_in_cat)
            cat_captions.append(CAT_CAPTIONS[i] if i < len(CAT_CAPTIONS) else "")

    return rows, base_models, cat_models, cat_captions


def print_debug_info(base_models, variants, fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map):
    """Prints a clean summary of the plotted data to the console."""
    W = 118
    print("\n" + "=" * W)
    print(f"{'DEBUG: PLOTTED FPS AND LATENCY (ms)':^{W}}")
    print("=" * W)
    header = (f"{'Model':<22} | {'Variant':<28} | {'FPS':>8} | "
              f"{'Inf(ms)':>8} | {'Pre/Post':>8} | {'Net(ms)':>8} | {'Total(ms)':>10} | {'ErrLo':>7} | {'ErrHi':>7}")
    print(header)
    print("-" * W)
    for m in base_models:
        for v in variants:
            fps = fps_val_map.get((m, v), np.nan)
            inf = inf_map.get((m, v), 0.0)
            net = net_map.get((m, v), 0.0)
            pre = pre_map.get((m, v), 0.0)
            lo = lower_map.get((m, v), np.nan)
            hi = upper_map.get((m, v), np.nan)
            total = inf + pre + net

            if np.isfinite(fps) or inf > 0:
                fps_str = f"{fps:8.2f}" if np.isfinite(fps) else "     N/A"
                inf_str = f"{inf:8.2f}" if inf > 0 else "       -"
                pre_str = f"{pre:8.2f}" if pre > 0 else "       -"
                net_str = f"{net:8.2f}" if net > 0 else "       -"
                tot_str = f"{total:10.2f}"
                lo_str = f"{lo:7.2f}" if np.isfinite(lo) else "      -"
                hi_str = f"{hi:7.2f}" if np.isfinite(hi) else "      -"
                print(f"{m:<22} | {v:<28} | {fps_str} | {inf_str} | {pre_str} | {net_str} | {tot_str} | {lo_str} | {hi_str}")
    print("\n" + "=" * W + "\n")


def build_value_maps(rows, base_models, variants):
    """Turns extract_rows()'s row tuples into the per-(model,variant) dicts the
    plotting scripts index into."""
    fps_val_map = {(m, v): np.nan for m in base_models for v in variants}
    inf_map = {(m, v): 0.0 for m in base_models for v in variants}
    net_map = {(m, v): 0.0 for m in base_models for v in variants}
    pre_map = {(m, v): 0.0 for m in base_models for v in variants}
    lower_map = {(m, v): np.nan for m in base_models for v in variants}
    upper_map = {(m, v): np.nan for m in base_models for v in variants}

    for m, v, fps, inf, net, pre, e_lower, e_upper in rows:
        if m in base_models and v in variants:
            fps_val_map[(m, v)] = fps
            inf_map[(m, v)] = inf
            net_map[(m, v)] = net
            pre_map[(m, v)] = pre
            lower_map[(m, v)] = e_lower
            upper_map[(m, v)] = e_upper

    return fps_val_map, inf_map, net_map, pre_map, lower_map, upper_map
