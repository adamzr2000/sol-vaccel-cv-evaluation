#!/usr/bin/env python3
"""
Network-load sweep: per-input latency breakdown (inference / pre+post /
network+framework overhead) and energy, per model, across WiFi background-load
levels -- ros2 over both DDS transports vs vaccel-remote-sol, remote GPU
(edge-asus), sol backend.

Runs (LOAD_LEVELS x MODELS):
  vaccel:              {vaccel_tag}_{model}_vaccel-remote-sol_edge-asus-gpu
                       vaccel_tag = net-baseline / net-load-100M / ...
  ros2-fastdds-tcp:    monitor_/pipeline_{model}_sol_remote-gpu-tcp-{level}_*.json
  ros2-cyclonedds-udp: monitor_/pipeline_{model}_sol_remote-gpu-udp-{level}_*.json

Every number comes from vaccel_ros_comparison.py's ros_contrib() /
vaccel_contrib(), so the definitions are exactly the main comparison's (see
that module's docstring): ros2 pre/post on the robot (monitor), total = e2e +
pre, inference = edge t_inference_ms; vaccel inference = the isolated
edge-asus SOL run; net/fw = total - inference - pre/post on both.

A run that doesn't exist (e.g. vaccel deeplabv3_resnet50 at 500 Mbps) prints
[missing] instead of stopping the script.
"""
from __future__ import annotations

import json
from pathlib import Path

from vaccel_ros_comparison import (
    LABEL_W, RESULTS_DIR, ROS_RUN_TAGS, ROS_VISION_SERVICE_DIR,
    print_energy_header, print_energy_row, print_header, print_row,
    ros_contrib, vaccel_contrib,
)

SV_WIFI_ROBOT = RESULTS_DIR.parent / "wifi-stats" / "robot"
ROS_WIFI_ROBOT = ROS_VISION_SERVICE_DIR / "results" / "experiments" / "wifi-stats" / "robot"

# (model, category) -- category only selects nothing here but is passed through
# to the shared loaders, which take it for their own console labels.
MODELS = [
    ("resnet50", "image_classification"),
    ("r3d_18", "video_classification"),
    ("deeplabv3_resnet50", "semantic_segmentation"),
]

# (console label, level suffix): vaccel run tag is "net-{suffix}", ros2's is
# "remote-gpu-{tcp|udp}-net-{suffix}" (ROS_RUN_TAGS' tag + "-net-" + suffix).
LOAD_LEVELS = [
    ("baseline", "baseline"),
    ("100 Mbps", "load-100M"),
    ("300 Mbps", "load-300M"),
    ("500 Mbps", "load-500M"),
    ("600 Mbps", "load-600M"),
]


def load_wifi_totals(path: Path) -> dict | None:
    """wifi-stats-collector's /stop response wraps the fields one level down
    under "totals" -- unlike energy_totals.json, which is flat."""
    return json.loads(path.read_text()).get("totals") if path.exists() else None


def collect(model: str, category: str, level: str) -> list[tuple[str, dict | None, str]]:
    """[(row label, contrib | None, missing-reason)] for every framework at one level."""
    rows = []
    for label, base_tag in ROS_RUN_TAGS:
        try:
            c = ros_contrib(model, category, f"{base_tag}-net-{level}")
            c["wifi"] = load_wifi_totals(ROS_WIFI_ROBOT / f"{base_tag}-net-{level}_{model}_sol_totals.json")
            rows.append((label, c, ""))
        except FileNotFoundError as e:
            rows.append((label, None, str(e)))
    try:
        c = vaccel_contrib(model, category, run_tag=f"net-{level}")
        c["wifi"] = load_wifi_totals(SV_WIFI_ROBOT / f"{c['run_id']}_totals.json")
        rows.append(("vaccel-rpc", c, ""))
    except FileNotFoundError as e:
        rows.append(("vaccel-rpc", None, str(e)))
    return rows


def print_loss_and_wifi(results: list[tuple[str, list]]) -> None:
    """Per level and framework: ros2 loss breakdown (upload/download loss,
    edge skips, never-published results) or vaccel frame drops, plus the
    WiFi retry rate / RSSI the miniPC recorded during the run."""
    header = (f"{'level':<9} {'framework':<{LABEL_W}} {'upload_loss':>11} {'edge_skip':>9} "
              f"{'never_pub':>9} {'down_loss':>9} {'retry_rate':>11} {'rssi_start':>10} {'rssi_end':>8}")
    print(header)
    print("-" * len(header))
    for level_label, rows in results:
        for label, c, _ in rows:
            if c is None:
                continue
            w = c.get("wifi") or {}
            wifi = (f"{w.get('retry_rate', float('nan')):>11.6f} {str(w.get('rssi_dbm_start', 'n/a')):>10} "
                    f"{str(w.get('rssi_dbm_end', 'n/a')):>8}" if w else f"{'n/a':>11} {'n/a':>10} {'n/a':>8}")
            if label == "vaccel-rpc":
                # vaccel reads the camera in-process: no upload/download legs to
                # lose a frame on; "drop" = camera frames it didn't process.
                loss = f"{'-':>11} {'-':>9} {'-':>9} {'-':>9}"
            else:
                up = "n/a" if c["camera_published"] is None else str(c["camera_published"] - c["edge_received"])
                skip = c["edge_received"] - c["edge_processed"]
                down = (c["edge_processed"] - c["n_publish_failed"]) - c["n_matched"]
                loss = f"{up:>11} {skip:>9} {c['n_publish_failed']:>9} {down:>9}"
            print(f"{level_label:<9} {label:<{LABEL_W}} {loss} {wifi}")
            if w.get("counter_reset_suspected"):
                print(f"{'':<9} [warn] counter_reset_suspected for {label} -- retry_rate may be unreliable")


def main() -> None:
    print("Network-load sweep | remote GPU (edge-asus), sol backend | "
          + ", ".join(f"{label}=remote-gpu-…-net-*" for label, _ in ROS_RUN_TAGS) + ", vaccel-rpc=net-*")
    print("All latency values in ms, p50 unless noted; energy in J (total per run).")

    for model, category in MODELS:
        results = [(level_label, collect(model, category, level)) for level_label, level in LOAD_LEVELS]

        print(f"\n{'=' * 30} {model} ({category}) {'=' * 30}")
        print(f"--- {model} latency ---")
        print(f"{'level':<9} ", end="")
        print_header()
        for level_label, rows in results:
            for label, c, missing in rows:
                print(f"{level_label:<9} ", end="")
                if c is None:
                    print(f"{label:<{LABEL_W}} [missing] {missing}")
                else:
                    print_row(label, c)

        print(f"--- {model} energy ---")
        print(f"{'level':<9} ", end="")
        print_energy_header()
        for level_label, rows in results:
            for label, c, _ in rows:
                if c is not None:
                    print(f"{level_label:<9} ", end="")
                    print_energy_row(label, *c["energy"])

        print(f"--- {model} loss + wifi link (informational) ---")
        print_loss_and_wifi(results)


if __name__ == "__main__":
    main()
