#!/bin/bash
set -euo pipefail

MODE="${1:-cpu}"   # usage: ./run.sh [cpu|gpu] [remote_address]
REMOTE_ADDRESS="${2:-127.0.0.1:9125}"

case "$MODE" in
  cpu)
    IMAGE="torchvision-app:cpu"
    LIB_TYPE="lib_cpu"
    GPU_ARGS=()
    ;;
  gpu)
    IMAGE="torchvision-app:gpu"
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
/src/models/swin_t_sol/${LIB_TYPE}:\
/src/models/mobilenet_v3_large_sol/${LIB_TYPE}"

# Add cuDNN wheel libs only for GPU runs
if [[ "$MODE" == "gpu" ]]; then
  CUDNN_LIBS="/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
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
LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}"):/src/models"

docker run -it --rm \
  --name torchvision-app \
  -p 8000:8000 \
  -v "$(pwd)"/scripts:/scripts \
  -v "$(pwd)"/src:/src \
  -v "$(pwd)"/results/experiments:/results/experiments \
  -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  -e VACCEL_RPC_ADDRESS="tcp://${REMOTE_ADDRESS}" \
  "${GPU_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE"
