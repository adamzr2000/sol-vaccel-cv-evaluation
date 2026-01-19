#!/usr/bin/env bash
set -euo pipefail

SCRIPT="model_benchmark_resources.py"
SLEEP_SEC=5
BACKEND="stock"   # NEW default

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <edge|robot> --device <cpu|gpu> --run-tag <tag> [--backend <stock|vaccel-local|vaccel-remote>] [--sleep <seconds>]

Required:
  --host     edge|robot
  --device   cpu|gpu
  --run-tag  run identifier (e.g., run1)

Optional:
  --backend  stock|vaccel-local|vaccel-remote (default: ${BACKEND})
  --sleep    seconds to wait between runs (default: ${SLEEP_SEC})

Examples:
  $(basename "$0") --host edge  --device gpu --run-tag run1
  $(basename "$0") --host edge  --device gpu --run-tag run1 --backend vaccel-local
  $(basename "$0") --host robot --device cpu --run-tag testA --sleep 2
EOF
}

HOST=""
DEVICE=""
RUN_TAG=""

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
    --backend)   # NEW
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

if [[ "${HOST}" != "edge" && "${HOST}" != "robot" ]]; then
  echo "[err] --host must be one of: edge, robot (got: ${HOST})"
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
  stock|vaccel-local|vaccel-remote) ;;
  *)
    echo "[err] --backend must be one of: stock, vaccel-local, vaccel-remote (got: ${BACKEND})"
    usage
    exit 2
    ;;
esac

USE_VACCEL=0
if [[ "${BACKEND}" == vaccel-* ]]; then
  USE_VACCEL=1
fi

# ---- Host defaults (mandatory envs derived from --host) ----
case "${HOST}" in
  robot)
    DOCKER_STATS_ENDPOINT="http://10.5.1.18:6000"
    SYSTEM_STATS_ENDPOINT="http://10.5.1.18:6001"
    ;;
  edge)
    DOCKER_STATS_ENDPOINT="http://10.5.1.20:6000"
    SYSTEM_STATS_ENDPOINT="http://10.5.1.20:6001"
    ;;
esac

echo "[bench] host=${HOST}"
echo "[bench] docker_stats=${DOCKER_STATS_ENDPOINT}"
echo "[bench] system_stats=${SYSTEM_STATS_ENDPOINT}"
echo "[bench] device=${DEVICE} run_tag=${RUN_TAG} sleep=${SLEEP_SEC}s"
echo "[bench] backend=${BACKEND} (vaccel=${USE_VACCEL})"

# ---- Runner ----
run_one () {
  local model="$1"

  # If backend is vaccel-*, only run *_sol models
  if [[ "${USE_VACCEL}" -eq 1 && "${model}" != *_sol ]]; then
    echo "[bench] skip (vaccel backend requires *_sol): ${model}"
    return 0
  fi

  echo "[bench] run: ${model}"

  if [[ "${HOST}" == "edge" && "${DEVICE}" == "cpu" && ( "${BACKEND}" == "stock" || "${BACKEND}" == "vaccel-local" ) ]]; then
    OMP_NUM_THREADS=10 \
    HOST="${HOST}" \
    DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
    SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
    DEVICE="${DEVICE}" \
    BACKEND="${BACKEND}" \
    MODEL="${model}" \
    RUN_TAG="${RUN_TAG}" \
    python3 "${SCRIPT}"
  else
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

# ---- Batch ----
run_one "deeplabv3_resnet50"
run_one "deeplabv3_resnet50_sol"

run_one "fcn_resnet50"
run_one "fcn_resnet50_sol"

run_one "resnet50"
run_one "resnet50_sol"

run_one "mobilenet_v3_large"
run_one "mobilenet_v3_large_sol"

run_one "swin_t"
run_one "swin_t_sol"

run_one "mc3_18"
run_one "mc3_18_sol"

run_one "r3d_18"
run_one "r3d_18_sol"

echo "[bench] all done ✅"
