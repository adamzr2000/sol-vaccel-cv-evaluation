#!/bin/bash
set -euo pipefail

IMAGE="torchvision-app:agent"

PORT="${1:-9125}"   # usage: ./run.sh [port]
CONT_PORT=9125

docker run -it --rm \
  --name torchvision-app-agent \
  --gpus all \
  -e OMP_NUM_THREADS=10 \
  -e VACCEL_LOG_LEVEL=3 \
  -e VACCEL_PROFILING_ENABLED=0 \
  -p ${PORT}:${CONT_PORT} \
  "$IMAGE"
