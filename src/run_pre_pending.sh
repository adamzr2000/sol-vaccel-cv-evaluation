#!/bin/bash
# Pending tests after run_pre_interleaved.sh, same stand-in camera/env:
#  C) per-launch "slow mode" on image preprocessing: 5 separate launches each
#     of ros2-like (R) and vaccel-like (V), with and without OpenMP thread
#     pinning (OMP_PROC_BIND=true OMP_PLACES=cores), interleaved. The probe's
#     colocated_frac shows whether a slow launch had threads sharing a core.
#  B) decode-overlap fixes on video / segmentation, 2 repetitions:
#     baseline vs low-priority single-threaded decode (NICE_DECODE) vs
#     fresh-frame sync (SYNC), plus R as the reference.
set -euo pipefail
cd "$(dirname "$0")"
IMG=torchvision-ros-app:gpu
COOL="${COOL:-10}"
docker rm -f pre-sender >/dev/null 2>&1 || true
docker run -d --rm --name pre-sender --net host -v "$PWD":/w --entrypoint bash go2-driver:latest -c '
  while true; do
    gst-launch-1.0 -q filesrc location=/w/clip.mkv ! matroskademux ! h264parse config-interval=-1 \
      ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600 sync=true
  done' >/dev/null
trap 'docker rm -f pre-sender >/dev/null 2>&1 || true' EXIT
sleep 3
PIN=(OMP_PROC_BIND=true OMP_PLACES=cores)
probe() {  # <label> KEY=VAL...
  local label="$1"; shift
  local extra=(); for kv in "$@"; do extra+=(-e "$kv"); done
  local out
  out=$(docker run --rm --net host -v "$PWD":/w -e OMP_WAIT_POLICY=ACTIVE -e OMP_NUM_THREADS=4 -e OMP_THREAD_LIMIT=4 \
    -e CUDA_VISIBLE_DEVICES= "${extra[@]}" --entrypoint python3 "$IMG" /w/pre_contention_probe.py 2>&1 \
    | grep -E "pre p50|overlap_ms" | tr -s ' ' | tr '\n' ' ')
  echo "$label | $out"
  sleep "$COOL"
}
echo "##### C: per-launch slow mode (image transform, 200 samples)"
for i in 1 2 3 4 5; do
  probe "C$i R  unpinned" SAMPLES=200 MODE=R CATEGORY=image
  probe "C$i V  unpinned" SAMPLES=200 MODE=V CATEGORY=image GAP_MS=15
  probe "C$i R  pinned  " SAMPLES=200 MODE=R CATEGORY=image "${PIN[@]}"
  probe "C$i V  pinned  " SAMPLES=200 MODE=V CATEGORY=image GAP_MS=15 "${PIN[@]}"
done
echo "##### B: decode-overlap fixes (300 samples)"
for r in 1 2; do
  probe "B$r video R (reference)     " SAMPLES=300 MODE=R CATEGORY=video
  probe "B$r video V baseline        " SAMPLES=300 MODE=V CATEGORY=video GAP_MS=41
  probe "B$r video V NICE_DECODE     " SAMPLES=300 MODE=V CATEGORY=video GAP_MS=41 DEC_THREADS=1 NICE_DECODE=1
  probe "B$r video V SYNC            " SAMPLES=300 MODE=V CATEGORY=video GAP_MS=41 SYNC=1
  probe "B$r seg   V baseline (1-thr)" SAMPLES=300 MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 DEC_THREADS=1
  probe "B$r seg   V NICE_DECODE     " SAMPLES=300 MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 DEC_THREADS=1 NICE_DECODE=1
done
