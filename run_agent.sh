#!/bin/bash
set -euo pipefail

IMAGE="torchvision-app:agent"
PORT="${1:-9125}"
CONT_PORT=9125

# Default profiling to off
PROFILING=0

# Check if --debug flag is present in any position
for arg in "$@"; do
  if [ "$arg" == "--debug" ]; then
    PROFILING=1
    break
  fi
done

docker run -it --rm \
  --name torchvision-app-agent \
  --gpus all \
  --env OMP_NUM_THREADS=10 \
  --env VACCEL_LOG_LEVEL=3 \
  --env VACCEL_PROFILING_ENABLED=$PROFILING \
  --publish ${PORT}:${CONT_PORT} \
  "$IMAGE"