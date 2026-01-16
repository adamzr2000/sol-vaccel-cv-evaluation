#!/usr/bin/env bash
set -euo pipefail

SCRIPT="model_benchmark_resources.py"
SLEEP_SEC=5

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <edge|robot> --device <cpu|gpu> --run-tag <tag> [--sleep <seconds>]

Required:
  --host     edge|robot
  --device   cpu|gpu
  --run-tag  run identifier (e.g., run1)

Optional:
  --sleep    seconds to wait between runs (default: ${SLEEP_SEC})

Examples:
  $(basename "$0") --host edge  --device gpu --run-tag run1
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

# ---- Host defaults (mandatory envs derived from --host) ----
case "${HOST}" in
  robot)
    DOCKER_STATS_ENDPOINT="http://10.5.1.19:6000"
    SYSTEM_STATS_ENDPOINT="http://10.5.1.19:6001"
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

# ---- Runner ----
run_one () {
  local model="$1"
  echo "[bench] run: ${model}"
  HOST="${HOST}" \
  DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
  SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
  DEVICE="${DEVICE}" \
  MODEL="${model}" \
  RUN_TAG="${RUN_TAG}" \
  python3 "${SCRIPT}"

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

run_one "mc3_18"
run_one "mc3_18_sol"

run_one "r3d_18"
run_one "r3d_18_sol"

# Only do the cuDNN reinstall on GPU runs (GPU-specific)
if [[ "${DEVICE}" == "gpu" ]]; then
  echo "[bench] pip: reinstall nvidia-cudnn-cu12==9.1.1.17 (no-deps)"
  python3 -m pip install --no-cache-dir --force-reinstall "nvidia-cudnn-cu12==9.1.1.17" --no-deps
  echo "[bench] sleep ${SLEEP_SEC}s"
  sleep "${SLEEP_SEC}"
else
  echo "[bench] skip cudnn reinstall (DEVICE=cpu)"
fi

run_one "mobilenet_v3_large"
run_one "mobilenet_v3_large_sol"

echo "[bench] all done ✅"
