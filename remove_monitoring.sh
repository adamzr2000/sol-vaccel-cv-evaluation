#!/usr/bin/env bash
set -euo pipefail

echo "[monitoring] stopping/removing (docker compose down)..."
docker compose -f docker-compose-monitoring.yml down
echo "[monitoring] removed."
