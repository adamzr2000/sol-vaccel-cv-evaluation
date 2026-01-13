#!/bin/bash
set -euo pipefail

MODE="${1:-cpu}"   # usage: ./run.sh [cpu|gpu]

case "$MODE" in
  cpu)
    IMAGE="sol-deploy:cuda12-8"
    LIB_TYPE="lib_cpu"
    GPU_ARGS=()
    ;;
  gpu)
    IMAGE="sol-deploy:cuda12-8"
    LIB_TYPE="lib_gpu"
    GPU_ARGS=(--gpus all)
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]"
    exit 1
    ;;
esac

SOL_LIBS="/src/models/deeplabv3_resnet50_sol/${LIB_TYPE}:\
/src/models/fcn_resnet50_sol/${LIB_TYPE}:\
/src/models/mc3_18_sol/${LIB_TYPE}:\
/src/models/r3d_18_sol/${LIB_TYPE}:\
/src/models/resnet50_sol/${LIB_TYPE}:\
/src/models/mobilenet_v3_large_sol/${LIB_TYPE}"

# Add cuDNN wheel libs only for GPU runs
if [[ "$MODE" == "gpu" ]]; then
  CUDNN_LIBS="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
else
  CUDNN_LIBS=""
fi

# Build LD_LIBRARY_PATH (avoid trailing/duplicate colons)
LD_PARTS=("$SOL_LIBS")
if [[ -n "$CUDNN_LIBS" ]]; then
  LD_PARTS+=("$CUDNN_LIBS")
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_PARTS+=("$LD_LIBRARY_PATH")
fi

export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}")"

docker run -it --rm \
  --name torchvision-app \
  --privileged \
  -p 8000:8000 \
  -w /src \
  -v "$(pwd)"/src:/src \
  -v "$(pwd)"/results:/results \
  -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "${GPU_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE"
