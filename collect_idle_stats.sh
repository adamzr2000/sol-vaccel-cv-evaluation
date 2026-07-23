#!/usr/bin/env bash
set -euo pipefail

# collect_idle_stats.sh
#
# Collects an idle (no workload) system-stats baseline and exports it under
# results/experiments/system-stats/idle/, using the {host}-{cpu|gpu}-idle.csv
# naming convention already expected by idle/summarize_stats.py.
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
  --mode        cpu|gpu|cpu,gpu

Optional:
  --duration    seconds to sample idle stats (default: ${DURATION_SEC})
  --endpoint    system-stats-collector base URL (default: ${ENDPOINT})

Requires the system-stats-collector container to already be running on this
host (./run_monitoring.sh edge|robot) and reachable at --endpoint.

Output:
  results/experiments/system-stats/idle/<host>-cpu-idle.csv
  results/experiments/system-stats/idle/<host>-gpu-idle.csv
  results/experiments/system-stats/idle/energy_totals_<host>-idle.json
  (CSVs only for the requested --mode; the energy totals sidecar always
  covers every monitored domain in one file, named mode-neutrally when
  both cpu and gpu are requested together)

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
MODE_JSON="["
CSV_NAMES_JSON="{"
first=1
for raw_m in "${MODE_PARTS[@]}"; do
  m="${raw_m// /}"
  case "$m" in
    cpu|gpu) ;;
    *) echo "[err] invalid mode: '${raw_m}' (expected cpu, gpu, or cpu,gpu)" >&2; exit 2 ;;
  esac
  if [[ $first -eq 0 ]]; then
    MODE_JSON+=","
    CSV_NAMES_JSON+=","
  fi
  MODE_JSON+="\"${m}\""
  CSV_NAMES_JSON+="\"${m}\":\"${HOST_NAME}-${m}-idle\""
  first=0
done
MODE_JSON+="]"
CSV_NAMES_JSON+="}"

mkdir -p "${HOST_OUT_DIR}"

echo "[idle] host=${HOST_NAME} mode=${MODE} duration=${DURATION_SEC}s endpoint=${ENDPOINT}"
echo "[idle] output dir: ${HOST_OUT_DIR}"

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

START_PAYLOAD=$(cat <<EOF
{
  "interval": 1.0,
  "csv_dir": "${OUT_DIR}",
  "tag": "idle-${HOST_NAME}",
  "mode": ${MODE_JSON},
  "csv_names": ${CSV_NAMES_JSON},
  "stdout": false
}
EOF
)

START_RESP=$(curl -s -w '\n%{http_code}' -X POST "${ENDPOINT}/monitor/start" \
  -H 'Content-Type: application/json' -d "${START_PAYLOAD}")
START_CODE="${START_RESP##*$'\n'}"
START_BODY="${START_RESP%$'\n'*}"

if [[ "${START_CODE}" != "200" ]]; then
  echo "[err] Failed to start monitor (HTTP ${START_CODE}): ${START_BODY}" >&2
  exit 1
fi
MONITOR_STARTED=1
echo "[idle] monitor started: ${START_BODY}"

echo "[idle] sampling idle stats for ${DURATION_SEC}s..."
sleep "${DURATION_SEC}"

STOP_RESP=$(curl -s -w '\n%{http_code}' -X POST "${ENDPOINT}/monitor/stop")
STOP_CODE="${STOP_RESP##*$'\n'}"
STOP_BODY="${STOP_RESP%$'\n'*}"
MONITOR_STARTED=0

if [[ "${STOP_CODE}" != "200" ]]; then
  echo "[err] Failed to stop monitor (HTTP ${STOP_CODE}): ${STOP_BODY}" >&2
  exit 1
fi
echo "[idle] monitor stopped: ${STOP_BODY}"

echo "[idle] done. Files written:"
for raw_m in "${MODE_PARTS[@]}"; do
  m="${raw_m// /}"
  echo "  ${HOST_OUT_DIR}/${HOST_NAME}-${m}-idle.csv"
done
echo "  ${HOST_OUT_DIR}/energy_totals_*.json  (see [idle] monitor stopped output above for the exact name)"
