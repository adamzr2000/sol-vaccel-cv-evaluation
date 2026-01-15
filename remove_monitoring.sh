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

echo "[monitoring] stopping/removing ($TARGET) (docker compose down)..."
docker compose -f "$COMPOSE_FILE" down
echo "[monitoring] removed."
