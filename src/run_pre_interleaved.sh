#!/bin/bash
# Interleaved, repeated subset of run_pre_contention.sh with cool-downs, so
# thermal/power drift over time can't masquerade as a config difference.
# Same stand-in camera and env as run_pre_contention.sh.
set -euo pipefail
cd "$(dirname "$0")"
VACCEL_IMG=torchvision-ros-app:gpu
SAMPLES="${SAMPLES:-300}"
COOL="${COOL:-25}"
docker rm -f pre-sender >/dev/null 2>&1 || true
docker run -d --rm --name pre-sender --net host -v "$PWD":/w --entrypoint bash go2-driver:latest -c '
  while true; do
    gst-launch-1.0 -q filesrc location=/w/clip.mkv ! matroskademux ! h264parse config-interval=-1 \
      ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600 sync=true
  done' >/dev/null
trap 'docker rm -f pre-sender >/dev/null 2>&1 || true' EXIT
sleep 3
probe() {
  echo "--- $*  (pkg temp before: $(( $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)/1000 ))C)"
  local extra=(); for kv in "$@"; do extra+=(-e "$kv"); done
  docker run --rm --net host -v "$PWD":/w -e OMP_WAIT_POLICY=ACTIVE -e OMP_NUM_THREADS=4 -e OMP_THREAD_LIMIT=4 \
    -e CUDA_VISIBLE_DEVICES= -e SAMPLES="$SAMPLES" "${extra[@]}" \
    --entrypoint python3 "$VACCEL_IMG" /w/pre_contention_probe.py 2>&1 | grep -E "pre p50|overlap_ms" || true
  sleep "$COOL"
}
for rep in 1 2; do
  echo "##### repetition $rep"
  probe MODE=R CATEGORY=image
  probe MODE=V CATEGORY=image GAP_MS=15
  probe MODE=V CATEGORY=image GAP_MS=15 FREEZE_AFTER=40
  probe MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 DEC_THREADS=1
  probe MODE=R CATEGORY=video
  probe MODE=V CATEGORY=video GAP_MS=41
  probe MODE=V CATEGORY=video GAP_MS=41 SYNC=1
done
