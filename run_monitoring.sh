#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-edge}"   # default to edge

case "$TARGET" in
  edge|robot) ;;
  *)
    echo "Usage: $0 [edge|robot]" >&2
    exit 2
    ;;
esac

COMPOSE_FILE="docker-compose-monitoring-${TARGET}.yml"

echo "[monitoring] starting ($TARGET) (docker compose up -d)..."
docker compose -f "$COMPOSE_FILE" up -d
echo "[monitoring] started. current status:"
docker compose -f "$COMPOSE_FILE" ps
