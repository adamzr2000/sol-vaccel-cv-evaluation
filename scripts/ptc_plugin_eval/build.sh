#!/usr/bin/env bash
# Rebuilds libvaccel-torch.so.0.2.1 with torch_profiling_sync.diff (GPU sync in torch_run +
# torch_predict region) inside the PyTorch agent image. Output next to this script.
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
