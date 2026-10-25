#!/usr/bin/env python3
"""
Breakdown of p50 total latency into inference / pre+post-processing /
network+framework overhead, across WiFi background-load levels, per model.

sol-vaccel (vaccel-remote-sol):
- inference p50: edge-side isolated run (iso_resnet50_sol_gpu), since the
  robot-side inference_ms also includes RPC/vaccel overhead, not pure GPU time.
- pre/post p50: sum of preprocessing_ms.p50 + postprocessing_ms.p50 from the
  robot-side run (preprocessing happens on the robot in this architecture).
- network/framework overhead p50: residual = max(0, system_ms.p50 - inference - pre/post).

ros-vision-service (plain ROS2/DDS, sol backend):
- inference p50 / pre+post p50: t_inference_ms / t_preprocess_ms + t_postprocess_ms
  from the edge-side pipeline_*.json. Unlike sol-vaccel, the ENTIRE pipeline
  (preprocess+inference+postprocess) runs on edge-asus here -- the robot only
  ships the raw camera frame over DDS. So "pre/post" is not measuring the same
  thing architecturally as sol-vaccel's robot-side preprocessing, even though
  "total" and "net/fw overhead" are still fair end-to-end comparisons.
- total p50: t_e2e_ms.p50 from the robot-side monitor_*.json.
- network/framework overhead p50: residual = max(0, total - inference - pre/post).

Overhead clamping (max(0, ...)) matches the convention already established in
results/plot/vaccel/plot_data_e2e.py and results/plot/ros2/barplot_e2e_latency_and_fps.py.
IQR is reported as a single p75-p25 width rather than those scripts' asymmetric
(p50-p25, p75-p50) error-bar split.

Assumes sol-vaccel-cv-evaluation and ros-vision-service are sibling checkouts
(same parent directory).

Run from results/plot/comparison/:
  python3 network_perturbation_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "model-stats"
ROBOT_DIR = RESULTS_DIR / "robot"
EDGE_DIR = RESULTS_DIR / "edge-asus"
SV_SYSTEM_ROBOT = RESULTS_DIR.parent / "system-stats" / "robot"
SV_SYSTEM_EDGE = RESULTS_DIR.parent / "system-stats" / "edge-asus"
SV_WIFI_ROBOT = RESULTS_DIR.parent / "wifi-stats" / "robot"

ROS_VISION_SERVICE_DIR = Path(__file__).resolve().parents[4] / "ros-vision-service"
ROS_MONITOR_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "robot"
ROS_PIPELINE_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "edge-asus"
ROS_SYSTEM_ROBOT = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "system-stats" / "robot"
ROS_SYSTEM_EDGE = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "system-stats" / "edge-asus"
ROS_WIFI_ROBOT = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "wifi-stats" / "robot"

MODELS = [
    ("resnet50", "iso_resnet50_sol_gpu"),
    ("deeplabv3_resnet50", "iso_deeplabv3_resnet50_sol_gpu"),
    ("r3d_18", "iso_r3d_18_sol_gpu"),
]

# vaccel-side run_id tag per model/level -- NOT a single shared list, since the
# run-tag convention changed between when resnet50 was run ("net-load-100",
# no suffix) and when deeplabv3_resnet50/r3d_18 were run ("net-load-100M",
# "M" suffix added for clarity). Levels also differ by availability: only
# non-"Resized" runs are used here (Resized is the --publish-width/height
# experiment, a different condition); deeplabv3_resnet50 has no non-Resized
# run past 300 Mbps, r3d_18 has one more at 400 Mbps.
VACCEL_RUNS = {
    "resnet50": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
        ("500 Mbps", "net-load-500M"),
        ("600 Mbps", "net-load-600M"),
    ],
    "deeplabv3_resnet50": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
        ("500 Mbps", "net-load-500M"),
        ("600 Mbps", "net-load-600M"),
    ],
    "r3d_18": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
        ("500 Mbps", "net-load-500M"),
        ("600 Mbps", "net-load-600M"),
    ],
}

# Monitor (robot) / pipeline (edge-asus) tag per model/level -- the file's
# timestamp isn't matched, just globbed for (see find_ros_file), since it's
# never meaningful and differs slightly between monitor/pipeline anyway.
# Levels differ by availability, same reasoning as VACCEL_RUNS.
#
# RAW: robot ships the camera's native 1280x720 frame unresized -- the default,
# main-comparison condition (payload size difference vs vaccel IS what's being
# measured, per the earlier discussion on why this shouldn't be the default).
ROS_RUNS_RAW = {
    "resnet50": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
    ],
    "deeplabv3_resnet50": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
    ],
    "r3d_18": [
        ("baseline", "net-baseline"),
        ("100 Mbps", "net-load-100M"),
        ("200 Mbps", "net-load-200M"),
        ("300 Mbps", "net-load-300M"),
    ],
}

# RESIZED: robot resizes before publishing (--publish-width/height, this
# session's feature), matched to vaccel's tensor resolution -- the deliberate,
# separate transport-only/payload-held-equal condition, kept distinct from RAW.
# deeplabv3_resnet50 has no 400 Mbps run on this side (goes 300 -> 500); the
# other two models have the full 100-600 Mbps sweep.
ROS_RUNS_RESIZED = {
    "resnet50": [
        ("baseline", "net-baselineResized"),
        ("100 Mbps", "net-loadResized-100M"),
        ("200 Mbps", "net-loadResized-200M"),
        ("300 Mbps", "net-loadResized-300M"),
        ("400 Mbps", "net-loadResized-400M"),
        ("500 Mbps", "net-loadResized-500M"),
        ("600 Mbps", "net-loadResized-600M"),
    ],
    "deeplabv3_resnet50": [
        ("baseline", "net-baselineResized"),
        ("100 Mbps", "net-loadResized-100M"),
        ("200 Mbps", "net-loadResized-200M"),
        ("300 Mbps", "net-loadResized-300M"),
        ("500 Mbps", "net-loadResized-500M"),
        ("600 Mbps", "net-loadResized-600M"),
    ],
    "r3d_18": [
        ("baseline", "net-baselineResized"),
        ("100 Mbps", "net-loadResized-100M"),
        ("200 Mbps", "net-loadResized-200M"),
        ("300 Mbps", "net-loadResized-300M"),
        ("400 Mbps", "net-loadResized-400M"),
        ("500 Mbps", "net-loadResized-500M"),
        ("600 Mbps", "net-loadResized-600M"),
    ],
}


def load_summary(base_dir: Path, run_id: str) -> dict:
    return json.loads((base_dir / run_id / "benchmark_summary.json").read_text())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_ros_file(base_dir: Path, prefix: str, model: str, tag: str) -> Path:
    """Glob for the file matching model+tag, ignoring the timestamp suffix.
    Excludes "Resized" runs automatically -- e.g. tag "net-load-100M" does not
    match "net-loadResized-100M", since there's no wildcard between the two.
    """
    pattern = f"{prefix}_{model}_sol_remote-gpu-{tag}_*.json"
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {base_dir}")
    if len(matches) > 1:
        print(f"[warn] {len(matches)} files match {pattern!r} in {base_dir}, using the earliest: {matches[0].name}")
    return matches[0]


def print_header() -> None:
    header = f"{'condition':<10} {'inference':>10} {'pre/post':>10} {'net/fw':>10} {'total':>10} {'IQR':>10}"
    print(header)
    print("-" * len(header))


def print_row(label: str, inference_p50: float, prepost_p50: float, total_p50: float, p25: float, p75: float) -> None:
    net_overhead_p50 = max(0.0, total_p50 - inference_p50 - prepost_p50)
    iqr = p75 - p25
    print(
        f"{label:<10} {inference_p50:>10.2f} {prepost_p50:>10.2f} "
        f"{net_overhead_p50:>10.2f} {total_p50:>10.2f} {iqr:>10.2f}"
    )


def print_loss_header() -> None:
    header = f"{'condition':<10} {'upload_loss':>12} {'edge_skip':>10} {'never_pub':>10} {'download_loss':>14}"
    print(header)
    print("-" * len(header))


def print_loss_row(label: str, camera_published, n_frames_received: int, n_frames: int,
                    n_publish_failed: int, n_matched: int) -> None:
    # upload_loss: real network loss, robot's camera -> edge (n/a if the camera
    # stats endpoint wasn't reachable for this run).
    upload_loss = "n/a" if camera_published is None else str(camera_published - n_frames_received)
    # edge_skip: NOT loss -- frames that arrived at the edge but were deliberately
    # skipped (AVOID_DUPLICATES/MAX_FPS), informational only.
    edge_skip = n_frames_received - n_frames
    # download_loss: real network loss, edge -> robot, of the results the edge
    # actually handed to DDS (excludes never_pub, which never left the edge at all).
    download_loss = (n_frames - n_publish_failed) - n_matched
    print(f"{label:<10} {upload_loss:>12} {edge_skip:>10} {n_publish_failed:>10} {download_loss:>14}")


def load_energy_totals(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def print_energy_header() -> None:
    header = f"{'condition':<10} {'samples':>8} {'robot_cpu_J':>12} {'edge_cpu_J':>11} {'edge_gpu_J':>11}"
    print(header)
    print("-" * len(header))


def print_energy_row(label: str, samples: int, robot_cpu_j, edge_cpu_j, edge_gpu_j) -> None:
    def fmt(v):
        return "n/a" if v is None else f"{v:.1f}"
    print(f"{label:<10} {samples:>8} {fmt(robot_cpu_j):>12} {fmt(edge_cpu_j):>11} {fmt(edge_gpu_j):>11}")


def load_wifi_totals(path: Path) -> dict | None:
    """wifi-stats-collector's /stop response wraps the actual fields one
    level down under "totals" -- unlike energy_totals.json, which is flat."""
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("totals")


def print_wifi_header() -> None:
    header = f"{'condition':<10} {'retry_rate':>11} {'rssi_start':>11} {'rssi_end':>10}"
    print(header)
    print("-" * len(header))


def print_wifi_row(label: str, wifi: dict | None) -> None:
    def fmt(v, spec=""):
        return "n/a" if v is None else f"{v:{spec}}"
    if wifi is None:
        print(f"{label:<10} {'n/a':>11} {'n/a':>11} {'n/a':>10}")
        return
    print(
        f"{label:<10} {fmt(wifi.get('retry_rate'), '.6f'):>11} "
        f"{fmt(wifi.get('rssi_dbm_start')):>11} {fmt(wifi.get('rssi_dbm_end')):>10}"
    )
    if wifi.get("counter_reset_suspected"):
        print(f"{'':<10} [warn] counter_reset_suspected=true for {label} -- retry_rate may be unreliable")


def print_ros_section(model: str, run_set: dict, section_label: str) -> None:
    print(f"\n=== {model} — ROS2/DDS, sol backend, {section_label} ===")
    print_header()
    loss_rows = []
    energy_rows = []
    wifi_rows = []
    for label, tag in run_set[model]:
        monitor_path = find_ros_file(ROS_MONITOR_DIR, "monitor", model, tag)
        monitor = load_json(monitor_path)
        pipeline = load_json(find_ros_file(ROS_PIPELINE_DIR, "pipeline", model, tag))

        inference_p50 = pipeline["summary"]["t_inference_ms"]["p50"]
        prepost_p50 = (
            pipeline["summary"]["t_preprocess_ms"]["p50"]
            + pipeline["summary"]["t_postprocess_ms"]["p50"]
        )
        t_e2e = monitor["summary"]["t_e2e_ms"]
        print_row(label, inference_p50, prepost_p50, t_e2e["p50"], t_e2e["p25"], t_e2e["p75"])

        loss_rows.append((
            label,
            monitor["metadata"].get("camera_frames_published"),
            pipeline["metadata"]["n_frames_received"],
            pipeline["metadata"]["n_frames"],
            pipeline["summary"].get("n_publish_failed", 0),
            monitor["metadata"]["n_matched"],
        ))

        # Same timestamp suffix as the monitor/pipeline files for this run,
        # since energy_totals_*.json is written by the same bracketed window.
        ts = monitor_path.stem.split("_")[-1]
        run_id = f"{model}_sol_remote-gpu-{tag}_{ts}"
        robot = load_energy_totals(ROS_SYSTEM_ROBOT / f"energy_totals_{run_id}.json")
        edge = load_energy_totals(ROS_SYSTEM_EDGE / f"energy_totals_{run_id}.json")
        energy_rows.append((
            label,
            # monitor's n_frames (matched-result count, ~1000), NOT pipeline's --
            # the energy monitoring window is bracketed by the robot's
            # COLLECT_SAMPLES stop condition, same as monitor.n_frames/n_matched,
            # not by the edge's internal processing count (which inflates under
            # congestion -- that's what the loss breakdown above already shows).
            monitor["metadata"]["n_frames"],
            robot.get("cpu_energy_j_total") if robot else None,
            edge.get("cpu_energy_j_total") if edge else None,
            (edge.get("gpu") or [{}])[0].get("gpu_energy_j_total") if edge else None,
        ))

        # Tag convention here differs from monitor/pipeline/energy_totals --
        # the actual --run-tag these were collected under was "remote-gpu-{tag}",
        # not "{tag}" alone (see run_vision_client.sh's _WIFI_TAG construction).
        wifi_rows.append((label, load_wifi_totals(ROS_WIFI_ROBOT / f"remote-gpu-{tag}_{model}_sol_totals.json")))

    # Loss breakdown, informational only -- see print_loss_row for what each
    # column means. Small +/-1-2 sample noise is expected (monitor/pipeline
    # bracket the same window via two independent start/stop reads, not one
    # atomic clock), so near-zero or occasionally negative values are normal,
    # not a broken measurement.
    print(f"--- {model} loss breakdown (informational) ---")
    print_loss_header()
    for row in loss_rows:
        print_loss_row(*row)

    # Total energy over each run's window (not per-frame -- samples shown
    # alongside so the reader can relate the two directly). robot_cpu_J is
    # RAPL-based (real energy, Intel-only); edge_gpu_J is the GPU's own
    # onboard energy counter (pynvml).
    print(f"--- {model} energy (informational) ---")
    print_energy_header()
    for row in energy_rows:
        print_energy_row(*row)

    # WiFi link stats (miniPC's wifi-stats-collector), informational. retry_rate
    # doesn't track load uniformly across models -- see the earlier finding that
    # it scales cleanly with load for large RAW payloads (e.g. deeplabv3_resnet50)
    # but not for small ones, so don't read this as a blanket congestion proxy.
    print(f"--- {model} wifi link stats (informational) ---")
    print_wifi_header()
    for row in wifi_rows:
        print_wifi_row(*row)


def main() -> None:
    print("Backend: vaccel-remote-sol | client: robot (CPU preprocessing) | inference: edge-asus GPU (remote)")

    for model, iso_run_id in MODELS:
        inference_p50 = load_summary(EDGE_DIR, iso_run_id)["inference_ms"]["p50"]

        print(f"\n=== {model} — vaccel-remote-sol, remote GPU (edge-asus) ===")
        print_header()
        energy_rows = []
        wifi_rows = []
        for label, tag in VACCEL_RUNS[model]:
            run_id = f"{tag}_{model}_vaccel-remote-sol_edge-asus-gpu"
            d = load_summary(ROBOT_DIR, run_id)
            prepost_p50 = d["preprocessing_ms"]["p50"] + d["postprocessing_ms"]["p50"]
            sys_ms = d["system_ms"]
            print_row(label, inference_p50, prepost_p50, sys_ms["p50"], sys_ms["p25"], sys_ms["p75"])

            wifi_rows.append((label, load_wifi_totals(SV_WIFI_ROBOT / f"{run_id}_totals.json")))

            robot = load_energy_totals(SV_SYSTEM_ROBOT / f"energy_totals_{run_id}.json")
            edge = load_energy_totals(SV_SYSTEM_EDGE / f"energy_totals_{run_id}.json")
            energy_rows.append((
                label,
                d["num_samples"],
                robot.get("cpu_energy_j_total") if robot else None,
                # vaccel's edge-asus runs were only monitored mode=[gpu,net] --
                # edge CPU energy was never collected for this framework, unlike
                # ROS2's edge runs (mode=[cpu,gpu,net]). Stays None/n/a here,
                # not a bug -- see module-level note if adding this for vaccel.
                edge.get("cpu_energy_j_total") if edge else None,
                (edge.get("gpu") or [{}])[0].get("gpu_energy_j_total") if edge else None,
            ))

        # Total energy over each run's window (not per-frame -- samples shown
        # alongside so the reader can relate the two directly). edge_cpu_J is
        # "n/a" for every vaccel row: that host's CPU was never monitored for
        # this framework (see comment above), not a missing-data error.
        print(f"--- {model} energy (informational) ---")
        print_energy_header()
        for row in energy_rows:
            print_energy_row(*row)

        # WiFi link stats, informational -- see print_ros_section's version
        # of this comment: retry_rate doesn't track load uniformly across
        # models/frameworks, don't read it as a blanket congestion proxy.
        print(f"--- {model} wifi link stats (informational) ---")
        print_wifi_header()
        for row in wifi_rows:
            print_wifi_row(*row)

    print("\nBackend: ros-vision-service (DDS, no vaccel) | sol backend | "
          "full pipeline runs on edge-asus GPU (remote); robot ships raw frame")
    print(">>> RAW upload: robot ships the camera's native 1280x720 frame, unresized <<<")
    for model, _ in MODELS:
        print_ros_section(model, ROS_RUNS_RAW, "RAW upload (1280x720)")

    print("\n" + "=" * 70)
    print(">>> RESIZED upload: robot resizes to ~vaccel's tensor resolution before")
    print(">>> publishing (--publish-width/height) -- transport-only comparison,")
    print(">>> payload size held roughly equal to vaccel's RPC tensor. NOT the main")
    print(">>> comparison condition -- see RAW above for that. <<<")
    print("=" * 70)
    for model, _ in MODELS:
        print_ros_section(model, ROS_RUNS_RESIZED, "RESIZED upload (~vaccel tensor size)")


if __name__ == "__main__":
    main()
