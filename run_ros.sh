#!/bin/bash
set -euo pipefail

MODE="cpu"
REMOTE_ADDRESS="10.5.1.20:9125"
DDS_INTERFACE="enp130s0"
RMW="${RMW:-rmw_cyclonedds_cpp}"

usage() {
  echo "Usage: $0 [cpu|gpu] [--remote HOST:PORT] [--iface IFNAME]"
  exit 1
}

# Optional first positional: cpu|gpu
if [[ "${1:-}" == "cpu" || "${1:-}" == "gpu" ]]; then
  MODE="$1"
  shift
fi

# Long-option parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      [[ $# -ge 2 ]] || usage
      REMOTE_ADDRESS="$2"
      shift 2
      ;;
    --iface|--if|--interface)
      [[ $# -ge 2 ]] || usage
      DDS_INTERFACE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

case "$MODE" in
  cpu)
    IMAGE="torchvision-ros-app:cpu"
    LIB_TYPE="lib_cpu"
    GPU_ARGS=()
    ;;
  gpu)
    IMAGE="torchvision-ros-app:gpu"
    LIB_TYPE="lib_gpu"
    GPU_ARGS=(--gpus all)
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]"
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

# For GPU runs: prefer system CUDA, but add venv NVIDIA runtime libs (cudart + cudnn)
if [[ "$MODE" == "gpu" ]]; then
  CUDA_HOME="/usr/local/cuda"
  CUDA_LIBS="${CUDA_HOME}/lib64:\
/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:\
/.venv/lib/python3.10/site-packages/nvidia/cublas/lib"
else
  CUDA_LIBS=""
fi

# Build LD_LIBRARY_PATH (avoid trailing/duplicate colons)
LD_PARTS=("$SOL_LIBS")
if [[ -n "$CUDA_LIBS" ]]; then
  LD_PARTS+=("$CUDA_LIBS")
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_PARTS+=("$LD_LIBRARY_PATH")
fi

export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}"):/src/models"


HOST_DDS_CFG_DIR="./dds"
CONT_DDS_CFG_DIR="/etc/dds"

mkdir -p "$HOST_DDS_CFG_DIR"

cat > "$HOST_DDS_CFG_DIR/cyclonedds.xml" <<EOF
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="false" name="${DDS_INTERFACE}" priority="default" multicast="default"/>
      </Interfaces>
      <AllowMulticast>default</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Internal>
      <SocketReceiveBufferSize min="10MB"/>
      <Watermarks>
        <WhcHigh>500kB</WhcHigh>
      </Watermarks>
    </Internal>
  </Domain>
</CycloneDDS>
EOF

EXTRA_VOL_ARGS=(
  --volume="${HOST_DDS_CFG_DIR}:${CONT_DDS_CFG_DIR}:ro"
)
EXTRA_ENV_ARGS=(
  --env="RMW_IMPLEMENTATION=${RMW}"
  --env="CYCLONEDDS_URI=file://${CONT_DDS_CFG_DIR}/cyclonedds.xml"
  --env="VACCEL_PLUGINS=libvaccel-exec.so:libvaccel-rpc.so"
)

if [[ "$MODE" == "gpu" ]]; then
  # vaccel-torch plugin: required for vaccel-*-ptc backends (CUDA AOTI only)
  VACCEL_TORCH_SO="$(pwd)/vaccel-plugin-torch/libvaccel-torch.so"
  EXTRA_VOL_ARGS+=( --volume="${VACCEL_TORCH_SO}:/usr/local/lib/libvaccel-torch.so:ro" )
  EXTRA_ENV_ARGS+=( --env="VACCEL_PLUGINS=libvaccel-exec.so:libvaccel-rpc.so:libvaccel-torch.so" )
  EXTRA_ENV_ARGS+=( --env="CUDA_HOME=${CUDA_HOME}" )
  # /usr/local/lib is not in the container's LD_LIBRARY_PATH by default;
  # add it so dlopen can find libvaccel-torch.so at runtime.
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
fi

docker run -it --rm \
  --name torchvision-app \
  --net host \
  -v "$(pwd)"/scripts:/scripts \
  -v "$(pwd)"/src:/src \
  -v "$(pwd)"/results/experiments:/results/experiments \
  -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  -e VACCEL_RPC_ADDRESS="tcp://${REMOTE_ADDRESS}" \
  "${EXTRA_VOL_ARGS[@]}" \
  "${EXTRA_ENV_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE"
