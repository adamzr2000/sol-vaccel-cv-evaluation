#!/bin/bash
# Runs pre_contention_probe.py against a stand-in camera: a pre-encoded
# 1280x720 H264 clip (clip.mkv, see below) streamed over RTP to
# udp://127.0.0.1:5600 at its own framerate (14.2 fps, like the GO2 camera).
# Encoding is done once beforehand, so the sender only packetizes (~no CPU).
#
# One-time clip creation (robot, ~1 min):
#   docker run --rm -v "$PWD":/w --entrypoint gst-launch-1.0 go2-driver:latest -q -e \
#     videotestsrc num-buffers=4260 pattern=snow ! video/x-raw,width=1280,height=720,framerate=71/5 \
#     ! x264enc bitrate=4000 speed-preset=ultrafast tune=zerolatency key-int-max=30 \
#     ! h264parse ! matroskamux ! filesink location=/w/clip.mkv
#
# Usage (in the directory holding clip.mkv + pre_contention_probe.py):
#   ./run_pre_contention.sh            # full matrix
# Nothing else should be running on the robot (go2-driver, benchmarks).
set -euo pipefail
cd "$(dirname "$0")"

VACCEL_IMG=torchvision-ros-app:gpu
ROS_IMG=go2-driver:latest
SAMPLES="${SAMPLES:-600}"

docker rm -f pre-sender >/dev/null 2>&1 || true
docker run -d --rm --name pre-sender --net host -v "$PWD":/w --entrypoint bash "$ROS_IMG" -c '
  while true; do
    gst-launch-1.0 -q filesrc location=/w/clip.mkv ! matroskademux ! h264parse config-interval=-1 \
      ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600 sync=true
  done' >/dev/null
trap 'docker rm -f pre-sender >/dev/null 2>&1 || true' EXIT
sleep 3

# vaccel-like OpenMP setup (evaluate_models_ros_v2.sh): ACTIVE, OMP_NUM_THREADS=OMP_THREAD_LIMIT=nproc
VACCEL_ENV=(-e OMP_WAIT_POLICY=ACTIVE -e OMP_NUM_THREADS=4 -e OMP_THREAD_LIMIT=4 -e CUDA_VISIBLE_DEVICES=)
# ros-like (go2_camerapublisher): ACTIVE, torch default threads
ROS_ENV=(-e OMP_WAIT_POLICY=ACTIVE)

probe() {  # <image> <env-array-name> KEY=VAL...
  local img="$1" envname="$2"; shift 2
  local -n envs="$envname"
  local extra=()
  for kv in "$@"; do extra+=(-e "$kv"); done
  docker run --rm --net host -v "$PWD":/w "${envs[@]}" "${extra[@]}" -e SAMPLES="$SAMPLES" \
    --entrypoint python3 "$img" /w/pre_contention_probe.py 2>&1 | grep -v "^$" | grep -E "MODE=|pre p50|overlap_ms" || true
}

echo "### image category (classification/segmentation transform)"
echo "--- R, ros2 image (go2-driver)";               probe "$ROS_IMG"    ROS_ENV    MODE=R CATEGORY=image
echo "--- R, vaccel image";                           probe "$VACCEL_IMG" VACCEL_ENV MODE=R CATEGORY=image
echo "--- V, vaccel image, RPC 15ms (classification)"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=image GAP_MS=15
echo "--- V, vaccel image, RPC 92ms @7/s (segmentation)"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143

echo "--- V, vaccel image, RPC 92ms @7/s, single-threaded decode (exact overlap)"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 DEC_THREADS=1
echo "--- V, vaccel image, RPC 92ms @7/s, NO decode after warmup (idle pattern only)"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=image GAP_MS=92 PERIOD_MS=143 FREEZE_AFTER=40

echo "### video category (16-frame clip)"
echo "--- R, ros2 image (go2-driver)";               probe "$ROS_IMG"    ROS_ENV    MODE=R CATEGORY=video
echo "--- R, vaccel image";                           probe "$VACCEL_IMG" VACCEL_ENV MODE=R CATEGORY=video
echo "--- V, vaccel image, RPC 41ms";                 probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=video GAP_MS=41
echo "--- V, vaccel image, RPC 41ms, single-threaded decode"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=video GAP_MS=41 DEC_THREADS=1
echo "--- V, vaccel image, RPC 41ms, NO decode after warmup"; probe "$VACCEL_IMG" VACCEL_ENV MODE=V CATEGORY=video GAP_MS=41 FREEZE_AFTER=40
