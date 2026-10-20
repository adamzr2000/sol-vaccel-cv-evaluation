#!/usr/bin/env python3
"""
Breakdown of p50 total latency into inference / pre+post-processing /
network+framework overhead, across WiFi background-load levels, for resnet50.

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

ROS_VISION_SERVICE_DIR = Path(__file__).resolve().parents[4] / "ros-vision-service"
ROS_MONITOR_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "robot"
ROS_PIPELINE_DIR = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "model-stats" / "edge-asus"

MODELS = [
    ("resnet50", "iso_resnet50_sol_gpu"),
    ("fcn_resnet50", "iso_fcn_resnet50_sol_gpu"),
]

LOAD_LEVELS = [
    ("baseline", "net-baseline"),
    ("100 Mbps", "net-load-100"),
    ("300 Mbps", "net-load-300"),
    ("500 Mbps", "net-load-500"),
]

# Monitor (robot) / pipeline (edge-asus) filename pairs per model -- timestamps
# aren't on a clean shared pattern (often off by 1s between the two), so these
# are listed explicitly rather than derived.
ROS_RUNS = {
    "resnet50": [
        ("baseline", "monitor_resnet50_sol_net-remote-gpu-baseline_20261013T153221.json",
                     "pipeline_resnet50_sol_net-remote-gpu-baseline_20261013T153221.json"),
        ("100 Mbps", "monitor_resnet50_sol_net-remote-gpu-load-100_20261013T154007.json",
                     "pipeline_resnet50_sol_net-remote-gpu-load-100_20261013T154007.json"),
        ("300 Mbps", "monitor_resnet50_sol_net-remote-gpu-load-300_20261013T154512.json",
                     "pipeline_resnet50_sol_net-remote-gpu-load-300_20261013T154512.json"),
        ("500 Mbps", "monitor_resnet50_sol_net-remote-gpu-load-500_20261013T154944.json",
                     "pipeline_resnet50_sol_net-remote-gpu-load-500_20261013T154945.json"),
    ],
    "fcn_resnet50": [
        ("baseline", "monitor_fcn_resnet50_sol_net-remote-gpu-baseline_20261013T153550.json",
                     "pipeline_fcn_resnet50_sol_net-remote-gpu-baseline_20261013T153550.json"),
        ("100 Mbps", "monitor_fcn_resnet50_sol_net-remote-gpu-load-100_20261013T155548.json",
                     "pipeline_fcn_resnet50_sol_net-remote-gpu-load-100_20261013T155549.json"),
        ("300 Mbps", "monitor_fcn_resnet50_sol_net-remote-gpu-load-300_20261013T160001.json",
                     "pipeline_fcn_resnet50_sol_net-remote-gpu-load-300_20261013T160001.json"),
        ("500 Mbps", "monitor_fcn_resnet50_sol_net-remote-gpu-load-500_20261013T160528.json",
                     "pipeline_fcn_resnet50_sol_net-remote-gpu-load-500_20261013T160529.json"),
    ],
}


def load_summary(base_dir: Path, run_id: str) -> dict:
    return json.loads((base_dir / run_id / "benchmark_summary.json").read_text())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def main() -> None:
    print("Backend: vaccel-remote-sol | client: robot (CPU preprocessing) | inference: edge-asus GPU (remote)")

    for model, iso_run_id in MODELS:
        inference_p50 = load_summary(EDGE_DIR, iso_run_id)["inference_ms"]["p50"]

        print(f"\n=== {model} — vaccel-remote-sol, remote GPU (edge-asus) ===")
        print_header()
        for label, tag in LOAD_LEVELS:
            run_id = f"{tag}_{model}_vaccel-remote-sol_edge-asus-gpu"
            d = load_summary(ROBOT_DIR, run_id)
            prepost_p50 = d["preprocessing_ms"]["p50"] + d["postprocessing_ms"]["p50"]
            sys_ms = d["system_ms"]
            print_row(label, inference_p50, prepost_p50, sys_ms["p50"], sys_ms["p25"], sys_ms["p75"])

    print("\nBackend: ros-vision-service (DDS, no vaccel) | sol backend | "
          "full pipeline runs on edge-asus GPU (remote); robot ships raw frame")
    for model, _ in MODELS:
        print(f"\n=== {model} — ROS2/DDS, sol backend, remote GPU (edge-asus) ===")
        print_header()
        for label, monitor_file, pipeline_file in ROS_RUNS[model]:
            monitor = load_json(ROS_MONITOR_DIR / monitor_file)
            pipeline = load_json(ROS_PIPELINE_DIR / pipeline_file)

            inference_p50 = pipeline["summary"]["t_inference_ms"]["p50"]
            prepost_p50 = (
                pipeline["summary"]["t_preprocess_ms"]["p50"]
                + pipeline["summary"]["t_postprocess_ms"]["p50"]
            )
            t_e2e = monitor["summary"]["t_e2e_ms"]
            print_row(label, inference_p50, prepost_p50, t_e2e["p50"], t_e2e["p25"], t_e2e["p75"])


if __name__ == "__main__":
    main()
