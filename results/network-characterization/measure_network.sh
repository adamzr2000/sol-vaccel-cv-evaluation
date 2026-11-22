#!/usr/bin/env bash
# Robot -> edge network characterization. Run on the ROBOT; needs `iperf3 -s` on the edge.
# Usage: ./measure_network.sh [EDGE_IP] [OUT_DIR]
set -euo pipefail
EDGE="${1:-10.5.1.20}"
OUT="${2:-raw}"
RUNS=3          # iperf3 runs per direction (alternating up/down)
DUR=30          # seconds per iperf3 run
mkdir -p "$OUT"

echo "[net] latency: 300 pings, 0.2 s apart"
ping -c 300 -i 0.2 "$EDGE" > "$OUT/ping.txt"

for i in $(seq "$RUNS"); do
  echo "[net] run $i/$RUNS: uplink (robot -> edge), then downlink (edge -> robot)"
  iperf3 -c "$EDGE" -t "$DUR" -O 2 -J > "$OUT/iperf_up_$i.json";   sleep 2
  iperf3 -c "$EDGE" -t "$DUR" -O 2 -J -R > "$OUT/iperf_down_$i.json"; sleep 2
done
echo "[net] done: $OUT"
