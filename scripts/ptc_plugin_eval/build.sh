#!/usr/bin/env bash
# Rebuilds libvaccel-torch.so.0.2.1 with torch_profiling_sync.diff, inside the PyTorch agent
# image (same libtorch/vAccel as the agent). Works on a temporary copy; nothing in the image
# or in vaccel-plugin-torch is modified. Output: ./libvaccel-torch.so.0.2.1 (next to this script).
#
# The patch only changes profiling: torch_run now waits for the GPU (cuda synchronize; the
# following .cpu() waited anyway, so the call time is unchanged), and a new region
# "torch_predict" covers H2D + run + sync + D2H. Use it with check_agent_profile.py
# --region torch_predict+torch_out, which has the same scope as SOL's sol_predict.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="harbor.nbfc.io/desire6g/torchvision-vaccel:x86_64"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint bash \
  -v "${HERE}:/out" "${IMAGE}" -c '
    set -e
    mkdir /tmp/p && cd /workspace/yolo-torch && tar --exclude="./build*" -cf - . | tar -xf - -C /tmp/p
    cd /tmp/p && patch -p0 < /out/torch_profiling_sync.diff
    mkdir b && cd b && meson setup --cmake-prefix-path=/workspace/libtorch --prefix=/usr .. >/dev/null
    meson compile -C . >/dev/null && cp -L src/libvaccel-torch.so.0.2.1 /out/ && echo "built /out/libvaccel-torch.so.0.2.1"'
