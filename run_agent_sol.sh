#!/usr/bin/env bash
set -euo pipefail

IMAGE="torchvision-app:agent"
CONTAINER_NAME="torchvision-app-agent"
OMP_THREADS=12
SET_OMP=1
PORT=9125
PROFILING=0

usage() {
  echo "Usage: $0 <cpu|gpu> [--port PORT] [--no-omp] [--debug]"
  exit 1
}

MODE="${1:-}"
case "$MODE" in
  gpu) CUDA_VISIBLE_DEVICES_VAL="0" ;;
  cpu) CUDA_VISIBLE_DEVICES_VAL="" ;;
  *) usage ;;
esac
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --no-omp) SET_OMP=0; shift ;;
    --debug)  PROFILING=1; shift ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

OMP_ARGS=()
if [[ $SET_OMP -eq 1 ]]; then
  OMP_ARGS=(-e "OMP_NUM_THREADS=${OMP_THREADS}")
fi

echo "Starting ${CONTAINER_NAME} (${MODE}) on port ${PORT}"
if [[ $SET_OMP -eq 1 ]]; then echo "OMP_NUM_THREADS=${OMP_THREADS}"; else echo "OMP_NUM_THREADS=unset"; fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}"

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  "${OMP_ARGS[@]}" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}" \
  -e "VACCEL_LOG_LEVEL=3" \
  -e "VACCEL_PROFILING_ENABLED=${PROFILING}" \
  -p "${PORT}:9125" \
  "${IMAGE}"
