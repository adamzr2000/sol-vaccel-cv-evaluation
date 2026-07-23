#!/usr/bin/env bash
set -euo pipefail

# collect_idle_stats.sh
#
# Collects an idle (no workload) system-stats baseline and exports it under
# results/experiments/system-stats/idle/, using the {host}-{cpu|gpu}-idle.csv
# naming convention already expected by idle/summarize_stats.py.
#
# cpu and gpu are always monitored in separate, sequential sessions (never
# together in one /monitor/start call) - mirrors how src/evaluate_models_ros.sh
# always monitors a single domain per run, and keeps each session's
# energy_totals_*.json 1:1 aligned with its own CSV.
#
# Requires the system-stats-collector container to already be running on
# this host (./run_monitoring.sh edge|robot) and reachable at --endpoint
# (default: http://localhost:6001).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_OUT_DIR="${SCRIPT_DIR}/results/experiments/system-stats/idle"
# The collector runs in its own container with ./results/experiments bind-mounted
# to /results/experiments (see docker-compose-monitoring-{edge,robot}.yml) - csv_dir
# in the API request must be that container-internal path, not the host path.
OUT_DIR="/results/experiments/system-stats/idle"

ENDPOINT="http://localhost:6001"
DURATION_SEC=60
MODE=""
HOST_NAME=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <name> --mode cpu|gpu|cpu,gpu [options]

Required:
  --host        identifier for this host (used in output filenames, e.g. edge-asus, robot)
  --mode        cpu|gpu|cpu,gpu (cpu,gpu runs as two separate, sequential
                sessions of --duration seconds each - never simultaneously)

Optional:
  --duration    seconds to sample idle stats, per mode (default: ${DURATION_SEC})
  --endpoint    system-stats-collector base URL (default: ${ENDPOINT})

Requires the system-stats-collector container to already be running on this
host (./run_monitoring.sh edge|robot) and reachable at --endpoint.

Output (one CSV + one energy_totals JSON per requested mode):
  results/experiments/system-stats/idle/<host>-cpu-idle.csv
  results/experiments/system-stats/idle/energy_totals_<host>-cpu-idle.json
  results/experiments/system-stats/idle/<host>-gpu-idle.csv
  results/experiments/system-stats/idle/energy_totals_<host>-gpu-idle.json

Examples:
  $(basename "$0") --host edge-asus --mode cpu,gpu --duration 120
  $(basename "$0") --host robot --mode cpu
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)     HOST_NAME="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --duration) DURATION_SEC="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "[err] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${HOST_NAME}" || -z "${MODE}" ]]; then
  echo "[err] Missing required arguments." >&2
  usage; exit 2
fi

IFS=',' read -ra MODE_PARTS <<< "${MODE}"
for raw_m in "${MODE_PARTS[@]}"; do
  m="${raw_m// /}"
  case "$m" in
    cpu|gpu) ;;
    *) echo "[err] invalid mode: '${raw_m}' (expected cpu, gpu, or cpu,gpu)" >&2; exit 2 ;;
  esac
done

mkdir -p "${HOST_OUT_DIR}"

if ! curl -sf "${ENDPOINT}/health" > /dev/null; then
  echo "[err] system-stats-collector not reachable at ${ENDPOINT} (is it running? see run_monitoring.sh)" >&2
  exit 1
fi

MONITOR_STARTED=0
cleanup() {
  if [[ "${MONITOR_STARTED}" -eq 1 ]]; then
    echo -e "\n[idle] Interrupted - stopping monitor..."
    curl -s -X POST "${ENDPOINT}/monitor/stop" > /dev/null || true
  fi
}
trap cleanup SIGINT SIGTERM

run_one_mode() {
  local m="$1"
  local csv_name="${HOST_NAME}-${m}-idle"

  echo "[idle] host=${HOST_NAME} mode=${m} duration=${DURATION_SEC}s endpoint=${ENDPOINT}"

  local start_payload
  start_payload=$(cat <<EOF
{
  "interval": 1.0,
  "csv_dir": "${OUT_DIR}",
  "tag": "idle-${HOST_NAME}-${m}",
  "mode": ["${m}"],
  "csv_names": {"${m}": "${csv_name}"},
  "stdout": false
}
EOF
)

  local start_resp start_code start_body
  start_resp=$(curl -s -w '\n%{http_code}' -X POST "${ENDPOINT}/monitor/start" \
    -H 'Content-Type: application/json' -d "${start_payload}")
  start_code="${start_resp##*$'\n'}"
  start_body="${start_resp%$'\n'*}"

  if [[ "${start_code}" != "200" ]]; then
    echo "[err] Failed to start monitor (HTTP ${start_code}): ${start_body}" >&2
    exit 1
  fi
  MONITOR_STARTED=1
  echo "[idle] monitor started: ${start_body}"

  echo "[idle] sampling idle ${m} stats for ${DURATION_SEC}s..."
  sleep "${DURATION_SEC}"

  local stop_resp stop_code stop_body
  stop_resp=$(curl -s -w '\n%{http_code}' -X POST "${ENDPOINT}/monitor/stop")
  stop_code="${stop_resp##*$'\n'}"
  stop_body="${stop_resp%$'\n'*}"
  MONITOR_STARTED=0

  if [[ "${stop_code}" != "200" ]]; then
    echo "[err] Failed to stop monitor (HTTP ${stop_code}): ${stop_body}" >&2
    exit 1
  fi
  echo "[idle] monitor stopped: ${stop_body}"
  echo "[idle] done: ${HOST_OUT_DIR}/${csv_name}.csv"
}

for raw_m in "${MODE_PARTS[@]}"; do
  m="${raw_m// /}"
  run_one_mode "$m"
done

echo "[idle] all done. Files written:"
for raw_m in "${MODE_PARTS[@]}"; do
  m="${raw_m// /}"
  echo "  ${HOST_OUT_DIR}/${HOST_NAME}-${m}-idle.csv"
  echo "  ${HOST_OUT_DIR}/energy_totals_${HOST_NAME}-${m}-idle.json"
done
