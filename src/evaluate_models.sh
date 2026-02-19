#!/usr/bin/env bash
set -euo pipefail

SCRIPT="model_benchmark.py"
SLEEP_SEC=10
BACKEND="stock"

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <edge-asus|edge-xtreme|robot> --device <cpu|gpu> --run-tag <tag> [--backend <stock|ptc|sol|vaccel-local-sol|vaccel-remote-sol>] [--sleep <seconds>]

Required:
  --host     edge-asus|edge-xtreme|robot
  --device   cpu|gpu
  --run-tag  run identifier (e.g., run1)

Optional:
  --backend  stock|ptc|sol|vaccel-local-sol|vaccel-remote-sol (default: ${BACKEND})
  --sleep    seconds to wait between runs (default: ${SLEEP_SEC})

Examples:
  $(basename "$0") --host edge-asus   --device gpu --run-tag run1
  $(basename "$0") --host edge-xtreme --device gpu --run-tag run1 --backend vaccel-local-sol
  $(basename "$0") --host robot       --device cpu --run-tag testA --sleep 2
EOF
}

HOST=""
DEVICE=""
RUN_TAG=""

EXPORT_RESULTS=1
RESOURCE_MONITORING=1
NUM_IMAGES=512
NUM_VIDEOS=64

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || { echo "[err] --host requires a value"; usage; exit 2; }
      HOST="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || { echo "[err] --device requires a value"; usage; exit 2; }
      DEVICE="$2"
      shift 2
      ;;
    --run-tag)
      [[ $# -ge 2 ]] || { echo "[err] --run-tag requires a value"; usage; exit 2; }
      RUN_TAG="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || { echo "[err] --backend requires a value"; usage; exit 2; }
      BACKEND="$2"
      shift 2
      ;;
    --sleep)
      [[ $# -ge 2 ]] || { echo "[err] --sleep requires a value"; usage; exit 2; }
      SLEEP_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[err] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

# ---- Validate required args ----
if [[ -z "${HOST}" || -z "${DEVICE}" || -z "${RUN_TAG}" ]]; then
  echo "[err] Missing required arguments."
  usage
  exit 2
fi

if [[ "${HOST}" != "edge-asus" && "${HOST}" != "edge-xtreme" && "${HOST}" != "robot" ]]; then
  echo "[err] --host must be one of: edge-asus, edge-xtreme, robot (got: ${HOST})"
  usage
  exit 2
fi

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "gpu" ]]; then
  echo "[err] --device must be one of: cpu, gpu (got: ${DEVICE})"
  usage
  exit 2
fi

# ---- Validate backend ----
case "${BACKEND}" in
  stock|ptc|sol|vaccel-local-sol|vaccel-remote-sol) ;;
  *)
    echo "[err] --backend must be one of: stock, ptc, sol, vaccel-local-sol, vaccel-remote-sol (got: ${BACKEND})"
    usage
    exit 2
    ;;
esac

# ---- Host defaults (IP assignment) ----
case "${HOST}" in
  robot)
    DOCKER_STATS_ENDPOINT="http://192.168.2.2:6000"
    SYSTEM_STATS_ENDPOINT="http://192.168.2.2:6001"
    ;;
  edge-asus)
    DOCKER_STATS_ENDPOINT="http://10.5.1.20:6000"
    SYSTEM_STATS_ENDPOINT="http://10.5.1.20:6001"
    ;;
  edge-xtreme)
    DOCKER_STATS_ENDPOINT="http://10.5.1.21:6000"
    SYSTEM_STATS_ENDPOINT="http://10.5.1.21:6001"
    ;;
esac

echo "[bench] host=${HOST}"
echo "[bench] docker_stats=${DOCKER_STATS_ENDPOINT}"
echo "[bench] system_stats=${SYSTEM_STATS_ENDPOINT}"
echo "[bench] device=${DEVICE} run_tag=${RUN_TAG} sleep=${SLEEP_SEC}s"
echo "[bench] backend=${BACKEND}"

# ---- Runner ----
run_one () {
  local model="$1"

  echo "[bench] run: ${model}"

  # MATCH any edge host (asus/xtreme) for the CPU optimization block
  if [[ "${HOST}" == edge* && "${DEVICE}" == "cpu" && "${BACKEND}" != "vaccel-remote-sol" ]]; then
    local threads=10
    
    OMP_NUM_THREADS="${threads}" \
    EXPORT_RESULTS="${EXPORT_RESULTS}" \
    RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    NUM_VIDEOS="${NUM_VIDEOS}" \
    HOST="${HOST}" \
    DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
    SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
    DEVICE="${DEVICE}" \
    BACKEND="${BACKEND}" \
    MODEL="${model}" \
    RUN_TAG="${RUN_TAG}" \
    python3 "${SCRIPT}"

  elif [[ "${HOST}" == "robot" && "${DEVICE}" == "cpu" && "${BACKEND}" != "vaccel-remote-sol" ]]; then
    EXPORT_RESULTS="${EXPORT_RESULTS}" \
    RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    NUM_VIDEOS="${NUM_VIDEOS}" \
    HOST="${HOST}" \
    DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
    SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
    DEVICE="${DEVICE}" \
    BACKEND="${BACKEND}" \
    MODEL="${model}" \
    RUN_TAG="${RUN_TAG}" \
    python3 "${SCRIPT}"
    
  elif [[ "${HOST}" == "robot" && "${BACKEND}" == "vaccel-remote-sol" ]]; then
    EXPORT_RESULTS="${EXPORT_RESULTS}" \
    RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    NUM_VIDEOS="${NUM_VIDEOS}" \
    HOST="${HOST}" \
    DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
    SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
    DOCKER_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6000" \
    SYSTEM_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6001" \
    DEVICE="${DEVICE}" \
    BACKEND="${BACKEND}" \
    MODEL="${model}" \
    RUN_TAG="${RUN_TAG}" \
    python3 "${SCRIPT}"
  else
    EXPORT_RESULTS="${EXPORT_RESULTS}" \
    RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    NUM_VIDEOS="${NUM_VIDEOS}" \
    HOST="${HOST}" \
    DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
    SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
    DEVICE="${DEVICE}" \
    BACKEND="${BACKEND}" \
    MODEL="${model}" \
    RUN_TAG="${RUN_TAG}" \
    python3 "${SCRIPT}"
  fi

  echo "[bench] done: ${model} (sleep ${SLEEP_SEC}s)"
  sleep "${SLEEP_SEC}"
}

# ---- Image Classification ----
#run_one "mobilenet_v3_large"
run_one "resnet50"
run_one "swin_t"
run_one "swin_s"
run_one "swin_v2_b"

# ---- Video Classification ----
run_one "swin3d_t"
run_one "swin3d_s"
run_one "swin3d_b"
run_one "mc3_18"
run_one "r3d_18"
run_one "r2plus1d_18"

# ---- Semantic Segmentation ----
#run_one "deeplabv3_mobilenet_v3_large"
run_one "deeplabv3_resnet50"
run_one "fcn_resnet50"
run_one "deeplabv3_resnet101"
run_one "fcn_resnet101"

echo "[bench] all done ✅"
