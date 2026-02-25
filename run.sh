#!/bin/bash
set -euo pipefail

MODE="${1:-cpu}"   # usage: ./run.sh [cpu|gpu] [remote_address]
REMOTE_ADDRESS="${2:-10.5.1.20:9125}"

case "$MODE" in
  cpu)
    IMAGE="torchvision-app:cpu"
    LIB_TYPE="lib_cpu"
    GPU_ARGS=()
    ;;
  gpu)
    IMAGE="torchvision-app:gpu"
    LIB_TYPE="lib_gpu"
    # Added --gpus all and the OMP_NUM_THREADS env flag
    GPU_ARGS=(--gpus all -e OMP_NUM_THREADS=12)
    ;;
  *)
    echo "Usage: $0 [cpu|gpu] [remote_address]"
    exit 1
    ;;
esac

SOL_LIBS="/src/models/deeplabv3_resnet50_sol/${LIB_TYPE}:\
/src/models/deeplabv3_resnet101_sol/${LIB_TYPE}:\
/src/models/deeplabv3_mobilenet_v3_large_sol/${LIB_TYPE}:\
/src/models/fcn_resnet50_sol/${LIB_TYPE}:\
/src/models/fcn_resnet101_sol/${LIB_TYPE}:\
/src/models/lraspp_mobilenet_v3_large_sol/${LIB_TYPE}:\
/src/models/resnet50_sol/${LIB_TYPE}:\
/src/models/mobilenet_v3_large_sol/${LIB_TYPE}:\
/src/models/swin_t_sol/${LIB_TYPE}:\
/src/models/swin_s_sol/${LIB_TYPE}:\
/src/models/swin_v2_b_sol/${LIB_TYPE}:\
/src/models/mc3_18_sol/${LIB_TYPE}:\
/src/models/r3d_18_sol/${LIB_TYPE}:\
/src/models/r2plus1d_18_sol/${LIB_TYPE}:\
/src/models/swin3d_t_sol/${LIB_TYPE}:\
/src/models/swin3d_s_sol/${LIB_TYPE}:\
/src/models/swin3d_b_sol/${LIB_TYPE}"

# For GPU runs
CUDA_HOME="/usr/local/cuda"
CUDA_LIBS=""
if [[ "$MODE" == "gpu" ]]; then
  CUDA_LIBS="${CUDA_HOME}/lib64:/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
  # Optional if you ever see cublas missing:
  # CUDA_LIBS="${CUDA_LIBS}:/.venv/lib/python3.10/site-packages/nvidia/cublas/lib"
fi

# Build LD_LIBRARY_PATH (cleanly)
LD_PARTS=("$SOL_LIBS")
[[ -n "$CUDA_LIBS" ]] && LD_PARTS+=("$CUDA_LIBS")
[[ -n "${LD_LIBRARY_PATH:-}" ]] && LD_PARTS+=("$LD_LIBRARY_PATH")

FINAL_LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}"):/src/models"

docker run -it --rm \
  --name torchvision-app \
  -p 8000:8000 \
  -v "$(pwd)"/scripts:/scripts \
  -v "$(pwd)"/src:/src \
  -v "$(pwd)"/results/experiments:/results/experiments \
  -e LD_LIBRARY_PATH="$FINAL_LD_LIBRARY_PATH" \
  -e CUDA_HOME="$CUDA_HOME" \
  -e VACCEL_RPC_ADDRESS="tcp://${REMOTE_ADDRESS}" \
  "${GPU_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE"
