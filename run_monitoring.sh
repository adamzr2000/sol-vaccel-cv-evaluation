#!/usr/bin/env bash
set -euo pipefail

echo "[monitoring] starting (docker compose up -d)..."
docker compose -f docker-compose-monitoring.yml up -d
echo "[monitoring] started. current status:"
docker compose -f docker-compose-monitoring.yml ps
