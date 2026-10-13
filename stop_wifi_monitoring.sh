#!/usr/bin/env bash
set -euo pipefail

CONTAINER="wifi-stats-collector"

if [[ -z "$(docker ps -aq -f "name=^${CONTAINER}$")" ]]; then
  echo "[wifi-monitoring] (was not running)"
else
  echo "[wifi-monitoring] stopping/removing (${CONTAINER})..."
  docker rm -f "$CONTAINER" >/dev/null
  echo "[wifi-monitoring] removed."
fi

# Runs privileged, so any files it writes under results/experiments/wifi-stats
# belong to root. Reclaim host-side ownership now that it has stopped writing
# (mirrors remove_monitoring.sh's handling of the other collectors).
RESULTS_DIR="$(pwd)/results/experiments/wifi-stats"
if [[ -d "$RESULTS_DIR" ]]; then
  sudo chown -R "$(id -u):$(id -g)" "$RESULTS_DIR" || \
    echo "[warn] Could not reclaim ownership of ${RESULTS_DIR}." \
         "Run manually: sudo chown -R \$(id -u):\$(id -g) ${RESULTS_DIR}" >&2
fi
