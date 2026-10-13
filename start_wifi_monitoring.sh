#!/usr/bin/env bash
set -euo pipefail

IMAGE="wifi-stats-collector:latest"
CONTAINER="wifi-stats-collector"

if [[ -n "$(docker ps -q -f "name=^${CONTAINER}$")" ]]; then
  echo "[wifi-monitoring] already running."
  docker ps -f "name=^${CONTAINER}$"
  exit 0
fi

echo "[wifi-monitoring] building image (${IMAGE})..."
docker build -t "$IMAGE" ./monitoring/wifi-stats-collector

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[wifi-monitoring] starting container (${CONTAINER})..."
docker run -d \
  --name "$CONTAINER" \
  --network host \
  --privileged \
  --restart unless-stopped \
  -v "$(pwd)/results/experiments:/results/experiments" \
  "$IMAGE"

echo "[wifi-monitoring] started."
docker ps -f "name=^${CONTAINER}$"
