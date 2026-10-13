# Network Perturbation Tests

WiFi-link perturbation testing for **Phase 3 (robot remote)** only — robot acts as
WiFi6 STA via the miniPC. Not applicable to Phase 1 (edge isolation) or Phase 2
(robot local): neither has a WiFi hop.

Scope is deliberately minimal: **app performance (model-stats) + WiFi stats only**
— docker/system resource monitoring is skipped (`--monitoring false`), so
`run_monitoring.sh` is not needed for these runs. All commands below use a single
example model (`resnet50`) — keep it the same across baseline and load levels
within one comparison set.

## Prerequisites

- Benchmark container running on the robot (`./run_ros.sh cpu --iface enp2s0 --remote <edge-ip>:9125`)
  and the vAccel agent running on edge-asus (`./run_agent_sol.sh gpu`, matching the
  `vaccel-remote-sol` backend used below) — see main
  [README.md](README.md#phase-3--robot-remote-run-on-robot).
- `wifi-stats-collector` running on the miniPC:
  ```bash
  ./start_wifi_monitoring.sh   # on the miniPC, once
  ```
- Traffic generator: a **separate PC, wirelessly connected to the same WiFi6 AP**
  — not the miniPC, not the robot. Copy `network-perturbation/iperf3_congestion.sh`
  onto it and onto the congestion receiver (`10.5.99.12`, the edge server behind the
  WiFi6 AP). Needs `iperf3` installed on both.

## 1. Start the congestion receiver on 10.5.99.12

```bash
./network-perturbation/iperf3_congestion.sh server
```

## 2. Baseline (no perturbation)

On the robot (inside the benchmark container, from `/src`):
```bash
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol \
  --model resnet50 --duration 120 --run-tag net-baseline \
  --monitoring false --wifi-monitoring true --remote-host edge-asus
```

## 3. Increasing system load via background traffic

Levels stay below the 600 Mbps uplink cap, leaving headroom for the actual app
traffic. On the wireless test PC, start the congestion client *before* the
benchmark run, targeting `10.5.99.12` — it self-stops after `--duration`, no
manual cleanup needed. Kept simple on purpose: plain UDP, single stream (the
script's defaults — no `--proto`/`--parallel` needed).

```bash
# wireless test PC — light load
./network-perturbation/iperf3_congestion.sh client --target 10.5.99.12 --rate 150M --duration 150

# robot — while the above runs
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol \
  --model resnet50 --duration 120 --run-tag net-load-150 \
  --monitoring false --wifi-monitoring true --remote-host edge-asus
```

```bash
# wireless test PC — moderate load
./network-perturbation/iperf3_congestion.sh client --target 10.5.99.12 --rate 300M --duration 150
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol \
  --model resnet50 --duration 120 --run-tag net-load-300 \
  --monitoring false --wifi-monitoring true --remote-host edge-asus
```

```bash
# wireless test PC — heavy load
./network-perturbation/iperf3_congestion.sh client --target 10.5.99.12 --rate 450M --duration 150
./evaluate_models_ros.sh --host robot --device gpu --backend vaccel-remote-sol \
  --model resnet50 --duration 120 --run-tag net-load-450 \
  --monitoring false --wifi-monitoring true --remote-host edge-asus
```

Check `./network-perturbation/iperf3_congestion.sh status` on the wireless test PC
if a run needs to be confirmed still active; `stop client` to cut it early.

## 4. Stop monitoring

```bash
./stop_wifi_monitoring.sh                                # on the miniPC
./network-perturbation/iperf3_congestion.sh stop server   # on 10.5.99.12
```

## Results

```
results/experiments/model-stats/robot/{run-tag}_resnet50_vaccel-remote-sol_edge-asus-gpu/benchmark_summary.json
results/experiments/wifi-stats/robot/{run-tag}_resnet50_vaccel-remote-sol_totals.json
```
`retry_rate`, `rssi_dbm_start`/`rssi_dbm_end`, `counter_reset_suspected` per run-tag
in the wifi totals; latency/FPS per run-tag in `benchmark_summary.json`.
