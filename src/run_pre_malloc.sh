#!/bin/bash
# D) Is the per-launch image "slow mode" page faults from the allocator
# handing memory back to the OS and re-faulting it during preprocessing?
# 5 launches each of R / V, default allocator vs GLIBC_TUNABLES that keep
# freed memory in the heap (mmap threshold at its 32 MiB max, trim threshold
# 256 MiB). Same stand-in camera/env as the other runners. Plus video SYNC
# with the tunables, to check the two fixes combine.
set -euo pipefail
cd "$(dirname "$0")"
IMG=torchvision-ros-app:gpu
COOL="${COOL:-8}"
TUN="glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=268435456"
docker rm -f pre-sender >/dev/null 2>&1 || true
docker run -d --rm --name pre-sender --net host -v "$PWD":/w --entrypoint bash go2-driver:latest -c '
  while true; do
    gst-launch-1.0 -q filesrc location=/w/clip.mkv ! matroskademux ! h264parse config-interval=-1 \
      ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600 sync=true
  done' >/dev/null
trap 'docker rm -f pre-sender >/dev/null 2>&1 || true' EXIT
sleep 3
probe() {
  local label="$1"; shift
  local extra=(); for kv in "$@"; do extra+=(-e "$kv"); done
  local out
  out=$(docker run --rm --net host -v "$PWD":/w -e OMP_WAIT_POLICY=ACTIVE -e OMP_NUM_THREADS=4 -e OMP_THREAD_LIMIT=4 \
    -e CUDA_VISIBLE_DEVICES= "${extra[@]}" --entrypoint python3 "$IMG" /w/pre_contention_probe.py 2>&1 \
    | grep -E "pre p50|minflt" | tr -s ' ' | sed -E 's/\| decode\+convert per frame p50=[0-9.]+ms cpu_mhz p50= ?[0-9]+ colocated_frac=[0-9.]+ \(n=[0-9]+\)//' | tr '\n' ' ')
  echo "$label | $out"
  sleep "$COOL"
}
echo "##### D: image, page faults, default vs tunables (200 samples)"
for i in 1 2 3 4 5; do
  probe "D$i R default " SAMPLES=200 MODE=R CATEGORY=image
  probe "D$i V default " SAMPLES=200 MODE=V CATEGORY=image GAP_MS=15
  probe "D$i R tunables" SAMPLES=200 MODE=R CATEGORY=image GLIBC_TUNABLES=$TUN
  probe "D$i V tunables" SAMPLES=200 MODE=V CATEGORY=image GAP_MS=15 GLIBC_TUNABLES=$TUN
done
echo "##### D-seg / D-video (300 samples)"
for r in 1 2; do
  probe "D$r seg   V default  " SAMPLES=300 MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143
  probe "D$r seg   V tunables " SAMPLES=300 MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 GLIBC_TUNABLES=$TUN
  probe "D$r video R reference" SAMPLES=300 MODE=R CATEGORY=video
  probe "D$r video V SYNC+tun " SAMPLES=300 MODE=V CATEGORY=video GAP_MS=41 SYNC=1 GLIBC_TUNABLES=$TUN
done
