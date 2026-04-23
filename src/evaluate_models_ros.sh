#!/usr/bin/env bash
set -euo pipefail

SCRIPT="model_benchmark_ros.py"
SLEEP_SEC=20
BACKEND="stock"
SELECTED_MODEL="" # Default empty means run all

# Cleanup on exit (Ctrl+C or error)
trap 'echo -e "\n[bench] Interrupted. Exiting..."; exit 1' SIGINT SIGTERM

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[err] ${SCRIPT} not found in $(pwd)"
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <edge-asus|edge-xtreme|robot> --device <cpu|gpu> --run-tag <tag> [options]

Required:
  --host      edge-asus|edge-xtreme|robot
  --device    cpu|gpu
  --run-tag   run identifier (e.g., run1)

Optional:
  --backend   stock|ptc|sol|vaccel-local-torch|vaccel-remote-torch\
              |vaccel-local-ptc|vaccel-remote-ptc\
              |vaccel-local-sol|vaccel-remote-sol (default: ${BACKEND})
  --sleep     seconds to wait between runs (default: ${SLEEP_SEC})
  --model     specific model to run (default: all)

Examples:
  $(basename "$0") --host edge-asus --device gpu --run-tag run1 --model resnet50
EOF
}

HOST=""
DEVICE=""
RUN_TAG=""
EXPORT_RESULTS=1
RESOURCE_MONITORING=0
EXPERIMENT_DURATION_SEC=60

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --sleep) SLEEP_SEC="$2"; shift 2 ;;
    --model) SELECTED_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[err] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---- Validate required args ----
if [[ -z "${HOST}" || -z "${DEVICE}" || -z "${RUN_TAG}" ]]; then
  echo "[err] Missing required arguments."
  usage; exit 2
fi

# ---- Validate backend ----
case "${BACKEND}" in
  stock|ptc|sol|\
  vaccel-local-torch|vaccel-remote-torch|\
  vaccel-local-ptc|vaccel-remote-ptc|\
  vaccel-local-sol|vaccel-remote-sol) ;;
  *) echo "[err] invalid backend: ${BACKEND}"; exit 2 ;;
esac

# ---- Host defaults ----
case "${HOST}" in
  robot) DOCKER_STATS_ENDPOINT="http://192.168.2.2:6000"; SYSTEM_STATS_ENDPOINT="http://192.168.2.2:6001" ;;
  edge-asus) DOCKER_STATS_ENDPOINT="http://10.5.1.20:6000"; SYSTEM_STATS_ENDPOINT="http://10.5.1.20:6001" ;;
  edge-xtreme) DOCKER_STATS_ENDPOINT="http://10.5.1.21:6000"; SYSTEM_STATS_ENDPOINT="http://10.5.1.21:6001" ;;
  *) echo "[err] Invalid host: ${HOST}"; exit 2 ;;
esac

# ---- Runner ----
run_one () {
  local model="$1"
  
  # If a specific model was requested, skip all others
  if [[ -n "${SELECTED_MODEL}" && "${model}" != "${SELECTED_MODEL}" ]]; then
    return
  fi

  echo "[bench] run: ${model}"

  export EXPORT_RESULTS="${EXPORT_RESULTS}" \
         EXPERIMENT_DURATION_SEC="${EXPERIMENT_DURATION_SEC}" \
         RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
         HOST="${HOST}" \
         DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
         SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
         DEVICE="${DEVICE}" \
         BACKEND="${BACKEND}" \
         MODEL="${model}" \
         RUN_TAG="${RUN_TAG}"

  if [[ "${HOST}" == edge* && "${BACKEND}" != vaccel-remote-* ]]; then
    export OMP_NUM_THREADS=12
  elif [[ "${HOST}" == "robot" && "${BACKEND}" == vaccel-remote-* ]]; then
    export DOCKER_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6000"
    export SYSTEM_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6001"
  fi

  python3 -u "${SCRIPT}"

  unset OMP_NUM_THREADS DOCKER_STATS_REMOTE_ENDPOINT SYSTEM_STATS_REMOTE_ENDPOINT
  echo "[bench] done: ${model} (sleep ${SLEEP_SEC}s)"
  sleep "${SLEEP_SEC}"
}

# ---- Model Lists ----
#MODELS=(
#  "resnet50" "swin_t" "swin_s" "swin_v2_b"
#  "swin3d_s" "swin3d_b" "mc3_18" "r3d_18" "r2plus1d_18"
#  "deeplabv3_resnet50" "fcn_resnet50" "deeplabv3_resnet101" "fcn_resnet101"
#)

MODELS=(
  "swin_v2_b"
  "swin3d_b" "r2plus1d_18"
)

# Execute
for m in "${MODELS[@]}"; do
  run_one "$m"
done

echo "[bench] all done ✅"
