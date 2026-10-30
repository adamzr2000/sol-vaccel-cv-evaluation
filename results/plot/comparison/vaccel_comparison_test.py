#!/usr/bin/env python3
"""
Per-input latency breakdown (inference / pre+post / network+framework overhead)
for a single remote-GPU Fast-DDS-TCP condition, comparing ros-vision-service
(plain ROS2/DDS, sol backend) against vaccel-remote-sol.

Condition: --dds fast (TCP) + --backend sol + --samples 1000, --run-tag remote-gpu-tcp.
Both sides now do uplink/downlink payload-size parity + pre/postprocessing on the
client (robot) -- see each project's own recent history for why. Network variation
and other backends are deliberately out of scope here (see network_perturbation_test.py
in this same directory for the WiFi-load sweep across backends/conditions).

HOW "inference" IS SOURCED, AND WHY -- this is the part most likely to be misread:

- ros2: inference_node.py's edge-side t_inference_ms (pipeline_*.json) is PURE GPU
  forward-pass time, nothing else -- it's a local in-process call, no RPC involved at
  any point in this architecture.

- vaccel: the REMOTE run's own inference_ms (benchmark_summary.json, robot-side) wraps
  adapter.infer() for the WHOLE RPC call -- network transport and remote GPU compute
  are inseparably bundled in that one field by construction (ttrpc request/response).
  Using it directly would silently fold real network time into "inference" and make
  net/fw look artificially tiny. So instead we use the matching ISOLATED LOCAL run
  (iso_{model}_sol_gpu/benchmark_summary.json, edge-asus, host-local SOL -- no vaccel
  plugin layer, no RPC at all) for vaccel's "inference" column here, exactly like
  network_perturbation_test.py already does for its 3 models. That isolates pure
  edge-asus GPU time, stripped of both RPC and vaccel-plugin dispatch overhead.

With that substitution, "net/fw" now means the same kind of thing on both rows:
total - inference(isolated/local) - [the part of pre/post actually on the critical
path -- see VIDEO MODELS note below] = whatever full round-trip cost isn't accounted
for by pure local GPU compute or by pre/post. For ros2 that's dominated by real DDS
network transport (upload+download) + edge buffer-wait. For vaccel that's dominated
by the real RPC round-trip (network + ttrpc/vaccel framework overhead) that the
remote run's own system_ms/inference_ms don't separate out.

VIDEO MODELS -- no longer need special-casing here. go2_camerapublisher now ships
one full 16-frame clip per publish, redoing the real torchvision transform over all
16 frames fresh on every single publish (no caching of previously-transformed
frames) -- deliberately matching vaccel's own preprocess_frames(), which also redoes
its transform over the whole clip, synchronously, every sample. So on BOTH sides,
the recorded preprocessing cost genuinely IS the full clip's cost, and it's genuinely
all on the critical path -- no x16 scaling, and no "undo the scaling for total/net_fw
but not for pre/post" split, needed anywhere. pre/post and total/net_fw now use the
exact same pre value, same as classification/segmentation.

Run from results/plot/comparison/:
  python3 vaccel_comparison_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "model-stats"
VACCEL_ROBOT_DIR = RESULTS_DIR / "robot"
VACCEL_EDGE_DIR = RESULTS_DIR / "edge-asus"

ROS_VISION_SERVICE_DIR = Path(__file__).resolve().parents[4] / "ros-vision-service"
ROS_MONITOR_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "robot"
ROS_PIPELINE_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "edge-asus"

# energy_totals_*.json, same sources/convention as network_perturbation_test.py.
SV_SYSTEM_ROBOT = RESULTS_DIR.parent / "system-stats" / "robot"
SV_SYSTEM_EDGE = RESULTS_DIR.parent / "system-stats" / "edge-asus"
ROS_SYSTEM_ROBOT = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "system-stats" / "robot"
ROS_SYSTEM_EDGE = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "system-stats" / "edge-asus"

RUN_TAG = "remote-gpu-tcp"
VACCEL_RUN_SUFFIX = "vaccel-remote-sol_edge-asus-gpu"
# vaccel run tag per category. Segmentation uses MAX_FPS=7 so vaccel runs at
# the same achieved rate as ros2's --publish-max-fps 12 (7.1/s: native ~14.2fps
# halved by the skip-based throttle) -- same link load and same window length,
# so energy totals compare directly. Unthrottled vaccel seg runs ~9.5/s.
VACCEL_RUN_TAG = {"semantic_segmentation": "e2eGSt-7fps"}
VACCEL_RUN_TAG_DEFAULT = "e2eGSt"

# model -> category, same classification inference_node.py itself uses. No longer
# changes how pre/post or total/net_fw are computed (see module docstring) -- kept
# only for the per-category console section headers.
MODELS = [
    ("resnet50", "image_classification"),
    ("swin_t", "image_classification"),
    ("swin_s", "image_classification"),
    ("swin_v2_b", "image_classification"),
    ("swin3d_s", "video_classification"),
    ("swin3d_b", "video_classification"),
    ("r3d_18", "video_classification"),
    ("r2plus1d_18", "video_classification"),
    ("deeplabv3_resnet50", "semantic_segmentation"),
    ("deeplabv3_resnet101", "semantic_segmentation"),
    ("fcn_resnet50", "semantic_segmentation"),
    ("fcn_resnet101", "semantic_segmentation"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_one(base_dir: Path, pattern: str) -> Path:
    """Glob for the file matching model+tag, ignoring the timestamp suffix --
    picks the most recent if more than one matches (e.g. a rerun)."""
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {base_dir}")
    if len(matches) > 1:
        print(f"[warn] {len(matches)} files match {pattern!r} in {base_dir}, using the most recent: {matches[-1].name}")
    return matches[-1]


def load_energy_totals(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def energy_row(robot: dict | None, edge: dict | None, samples: int) -> tuple:
    """(samples, duration_s, robot_cpu_J, edge_cpu_J, edge_gpu_J) -- totals over
    the run's window, same fields network_perturbation_test.py reads. duration_s
    is that same window's duration_sec (robot's, edge's as fallback), so it
    always matches the energy totals it sits next to."""
    window = robot or edge
    return (
        samples,
        window.get("duration_sec") if window else None,
        robot.get("cpu_energy_j_total") if robot else None,
        edge.get("cpu_energy_j_total") if edge else None,
        (edge.get("gpu") or [{}])[0].get("gpu_energy_j_total") if edge else None,
    )


def ros_contrib(model: str, category: str) -> dict:
    monitor_path = find_one(ROS_MONITOR_DIR, f"monitor_{model}_sol_{RUN_TAG}_*.json")
    monitor = load_json(monitor_path)
    pipeline = load_json(find_one(ROS_PIPELINE_DIR, f"pipeline_{model}_sol_{RUN_TAG}_*.json"))

    m_sum = monitor["summary"]
    p_sum = pipeline["summary"]

    # inference: edge-side, pure GPU forward pass (pipeline_*.json) -- never the
    # robot-side e2e, which also includes network + pre/post.
    inference_p50 = p_sum["t_inference_ms"]["p50"]
    # pre/post: robot-side (monitor_*.json), since both run client-side here.
    # pipeline_*.json's own t_preprocess_ms/t_postprocess_ms are NOT used -- the
    # former is near-zero by design (PREPROCESSED_INPUT skips real work on edge),
    # the latter doesn't exist at all (SKIP_POSTPROCESS). For video models this is
    # now a genuine, un-scaled full-clip cost (go2_camerapublisher redoes the whole
    # clip's transform fresh on every publish, matching vaccel) -- no scaling
    # needed, same as classification/segmentation.
    pre = m_sum["t_preprocess_ms"]
    post = m_sum["t_postprocess_ms"]
    prepost_p50 = pre["p50"] + post["p50"]

    e2e = m_sum["t_e2e_ms"]
    # t_e2e_ms's clock starts when go2_camerapublisher sets header.stamp, which is
    # AFTER its own preprocessing already ran (see _preprocess_for_category/
    # _preprocess_video_clip call order), and stops when monitoring_node.py
    # captures t_result_received, which is AFTER on_classification/on_seg_result's
    # own postprocess block already ran (see _on_result call order). So t_e2e_ms
    # already includes postprocess but is MISSING preprocess entirely -- add it
    # back so "total" means the same thing vaccel's system_ms does (that one
    # starts timing before its own preprocessing too).
    total_p50 = e2e["p50"] + pre["p50"]
    p25 = e2e["p25"] + pre["p25"]
    p75 = e2e["p75"] + pre["p75"]

    # Reduces algebraically to e2e - inference - post (the pre terms cancel) --
    # the honest "unaccounted-for" residual either way.
    net_fw_p50 = max(0.0, total_p50 - inference_p50 - pre["p50"] - post["p50"])

    return {
        "inference_p50": inference_p50,
        "prepost_p50": prepost_p50,
        "net_fw_p50": net_fw_p50,
        "total_p50": total_p50,
        "p25": p25,
        "p75": p75,
        "n_matched": monitor["metadata"]["n_matched"],
        "camera_published": monitor["metadata"].get("camera_frames_published"),
        "edge_received": pipeline["metadata"]["n_frames_received"],
        "edge_processed": pipeline["metadata"]["n_frames"],
        # None = run predates the field (not recorded), not "default".
        "omp_wait_policy": monitor["metadata"].get("camera_omp_wait_policy"),
        "glibc_tunables": monitor["metadata"].get("camera_glibc_tunables"),
        # Same timestamp suffix as the monitor file for this run, since
        # energy_totals_*.json is written by the same bracketed window.
        # monitor's n_frames (matched-result count), as network_perturbation_test.py.
        "energy": energy_row(
            load_energy_totals(ROS_SYSTEM_ROBOT / f"energy_totals_{monitor_path.stem[len('monitor_'):]}.json"),
            load_energy_totals(ROS_SYSTEM_EDGE / f"energy_totals_{monitor_path.stem[len('monitor_'):]}.json"),
            monitor["metadata"]["n_frames"],
        ),
    }


def vaccel_contrib(model: str, category: str) -> dict:
    tag = VACCEL_RUN_TAG.get(category, VACCEL_RUN_TAG_DEFAULT)
    run_id = f"{tag}_{model}_{VACCEL_RUN_SUFFIX}"
    d = load_json(VACCEL_ROBOT_DIR / run_id / "benchmark_summary.json")

    # inference: NOT this file's own inference_ms (that's RPC-inclusive, see module
    # docstring) -- the matching isolated local-GPU run instead, same as
    # network_perturbation_test.py's iso_{model}_sol_gpu convention. Plain "sol"
    # backend (not "vaccel-local-sol"), matching that script's own precedent, since
    # the intent is pure GPU time stripped of both RPC AND the vaccel plugin layer.
    iso = load_json(VACCEL_EDGE_DIR / f"iso_{model}_sol_gpu" / "benchmark_summary.json")
    inference_p50 = iso["inference_ms"]["p50"]

    # pre/post and total still come from the REMOTE run (robot-side) -- these were
    # never RPC-ambiguous, they're genuinely this run's own client-side cost/total,
    # and (unlike ros2) ALL of pre/post is genuinely on vaccel's critical path every
    # time (see module docstring, VIDEO MODELS section) -- no scaling needed here.
    prepost_p50 = d["preprocessing_ms"]["p50"] + d["postprocessing_ms"]["p50"]
    total = d["system_ms"]
    net_fw_p50 = max(0.0, total["p50"] - inference_p50 - prepost_p50)

    return {
        "inference_p50": inference_p50,
        "prepost_p50": prepost_p50,
        "net_fw_p50": net_fw_p50,
        "total_p50": total["p50"],
        "p25": total["p25"],
        "p75": total["p75"],
        "num_samples": d["num_samples"],
        "received_frames": d["stream"]["received_frames"],
        "processed_frames": d["stream"]["processed_frames"],
        "drop_ratio": d["stream"]["drop_ratio"],
        "omp_wait_policy": d["stream"].get("omp_wait_policy"),
        "glibc_tunables": d["stream"].get("glibc_tunables"),
        "fresh_frame": d["stream"].get("fresh_frame"),
        "low_priority_decode": d["stream"].get("low_priority_decode"),
        "energy": energy_row(
            load_energy_totals(SV_SYSTEM_ROBOT / f"energy_totals_{run_id}.json"),
            load_energy_totals(SV_SYSTEM_EDGE / f"energy_totals_{run_id}.json"),
            d["num_samples"],
        ),
    }


def _tun(v) -> str:
    """GLIBC_TUNABLES as recorded -> short label (None = run predates the field)."""
    return "unrecorded" if v is None else ("default" if v == "default" else "on")


def print_header() -> None:
    header = f"{'framework':<12} {'inference':>10} {'pre/post':>10} {'net/fw':>10} {'total':>10} {'IQR':>10}"
    print(header)
    print("-" * len(header))


def print_row(label: str, c: dict) -> None:
    iqr = c["p75"] - c["p25"]
    print(
        f"{label:<12} {c['inference_p50']:>10.3f} {c['prepost_p50']:>10.3f} "
        f"{c['net_fw_p50']:>10.3f} {c['total_p50']:>10.3f} {iqr:>10.3f}"
    )


def print_energy_header() -> None:
    header = (f"{'framework':<12} {'samples':>8} {'duration_s':>11} {'robot_cpu_J':>12} "
              f"{'edge_cpu_J':>11} {'edge_gpu_J':>11} {'total_J':>10}")
    print(header)
    print("-" * len(header))


def print_energy_row(label: str, samples: int, duration_s, robot_cpu_j, edge_cpu_j, edge_gpu_j) -> None:
    def fmt(v):
        return "n/a" if v is None else f"{v:.1f}"
    parts = (robot_cpu_j, edge_cpu_j, edge_gpu_j)
    total = None if None in parts else sum(parts)
    print(f"{label:<12} {samples:>8} {fmt(duration_s):>11} {fmt(robot_cpu_j):>12} "
          f"{fmt(edge_cpu_j):>11} {fmt(edge_gpu_j):>11} {fmt(total):>10}")


def main() -> None:
    print(f"Condition: --dds fast (TCP) --backend sol --run-tag {RUN_TAG}, --samples 1000")
    print("All values in ms, p50 unless noted. See module docstring for what 'net/fw' means per framework.\n")

    for model, category in MODELS:
        print(f"\n=== {model} ({category}) ===")
        print_header()

        try:
            ros = ros_contrib(model, category)
            print_row("ros2", ros)
        except FileNotFoundError as e:
            print(f"ros2         [missing] {e}")
            ros = None

        try:
            vaccel = vaccel_contrib(model, category)
            print_row("vaccel-rpc", vaccel)
        except FileNotFoundError as e:
            print(f"vaccel-rpc   [missing] {e}")
            vaccel = None

        if ros is not None:
            print(
                f"  ros2:   n_matched={ros['n_matched']}  camera_published={ros['camera_published']}  "
                f"edge_received={ros['edge_received']}  edge_processed={ros['edge_processed']}  "
                f"omp_wait_policy={ros['omp_wait_policy'] or 'unrecorded'}  "
                f"malloc_tunables={_tun(ros['glibc_tunables'])}"
            )
        if vaccel is not None:
            print(
                f"  vaccel: num_samples={vaccel['num_samples']}  received={vaccel['received_frames']}  "
                f"processed={vaccel['processed_frames']}  drop_ratio={vaccel['drop_ratio']:.4f}  "
                f"omp_wait_policy={vaccel['omp_wait_policy'] or 'unrecorded'}  "
                f"malloc_tunables={_tun(vaccel['glibc_tunables'])}  "
                f"fresh_frame={'unrecorded' if vaccel['fresh_frame'] is None else vaccel['fresh_frame']}  "
                f"low_priority_decode={'unrecorded' if vaccel['low_priority_decode'] is None else vaccel['low_priority_decode']}"
            )
        # Total energy over each run's window (not per-frame -- samples shown
        # alongside; every run is --samples 1000, so totals compare directly).
        print(f"--- {model} energy ---")
        print_energy_header()
        if ros is not None:
            print_energy_row("ros2", *ros["energy"])
        if vaccel is not None:
            print_energy_row("vaccel-rpc", *vaccel["energy"])

        # pre/post is only comparable when both robot-side torch processes ran
        # with the same OpenMP wait policy (see evaluate_models_ros_v2.sh /
        # start_go2.sh --omp-wait-policy).
        if ros is not None and vaccel is not None:
            policies = {ros["omp_wait_policy"], vaccel["omp_wait_policy"]}
            if None in policies:
                print("  [warn] omp_wait_policy unrecorded on at least one side -- pre/post match not verifiable")
            elif len(policies) > 1:
                print(f"  [warn] omp_wait_policy MISMATCH (ros2={ros['omp_wait_policy']}, "
                      f"vaccel={vaccel['omp_wait_policy']}) -- pre/post not comparable")
            tunables = {ros["glibc_tunables"], vaccel["glibc_tunables"]}
            if None not in tunables and len(tunables) > 1:
                print(f"  [warn] malloc_tunables MISMATCH (ros2={_tun(ros['glibc_tunables'])}, "
                      f"vaccel={_tun(vaccel['glibc_tunables'])}) -- pre not comparable")


if __name__ == "__main__":
    main()
