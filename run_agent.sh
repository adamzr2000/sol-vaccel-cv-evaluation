#!/bin/bash
set -euo pipefail

IMAGE="torchvision-app:agent"

PORT="${1:-9125}"   # usage: ./run.sh [port]
CONT_PORT=9125

docker run -it --rm \
  --name torchvision-app-agent \
  --gpus all \
  -p ${PORT}:${CONT_PORT} \
  "$IMAGE"
