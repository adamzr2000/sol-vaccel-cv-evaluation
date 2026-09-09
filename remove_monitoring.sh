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

# Both collectors run as root/privileged (unchanged — RAPL/thermal sysfs reads
# and docker.sock access depend on it, so we don't touch that). They write
# CSVs into results/experiments/ as root, which blocks non-root host users
# (e.g. `git pull` on another host) from later reading/replacing them.
# Reclaim host-side ownership now that both collectors have stopped writing —
# this only touches file ownership on the host, never the containers, so it
# does not affect data collection.
RESULTS_DIR="$(pwd)/results/experiments"
if [[ -d "$RESULTS_DIR" ]]; then
  sudo chown -R "$(id -u):$(id -g)" "$RESULTS_DIR" || \
    echo "[warn] Could not reclaim ownership of ${RESULTS_DIR}." \
         "Run manually: sudo chown -R \$(id -u):\$(id -g) ${RESULTS_DIR}" >&2
fi
