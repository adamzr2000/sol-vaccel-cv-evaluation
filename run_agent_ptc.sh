#!/usr/bin/env bash
set -euo pipefail

IMAGE="harbor.nbfc.io/desire6g/torchvision-vaccel:x86_64"
CONTAINER_NAME="torchvision-app-agent"
OMP_THREADS=12
SET_OMP=1
PORT=9125
PROFILING=0
PROFILE_LOG=""   # --profile: per-call vAccel profiling, agent output appended to this file
# edge-asus: pin to the 8 P-cores (0-7) with 8 OpenMP threads (--no-pin: 12 threads, any core)
PIN_CPUS=""
if [[ "$(hostname)" == "asus-g815" ]]; then
  PIN_CPUS="0-7"
  OMP_THREADS=8
fi

# Profiling-patched plugin (GPU sync + torch_predict region, see scripts/ptc_plugin_eval); --stock-plugin: image's own
PATCHED_PLUGIN="$(cd "$(dirname "$0")" && pwd)/scripts/ptc_plugin_eval/libvaccel-torch.so.0.2.1"
USE_PATCHED_PLUGIN=1

usage() {
  echo "Usage: $0 <cpu|gpu> [--port PORT] [--no-omp] [--robot] [--no-pin] [--stock-plugin] [--profile LOGFILE]"
  exit 1
}

MODE="${1:-}"
case "$MODE" in
  gpu) CUDA_VISIBLE_DEVICES_VAL="0" ;;
  cpu) CUDA_VISIBLE_DEVICES_VAL="" ;;
  *) usage ;;
esac
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --no-omp) SET_OMP=0; shift ;;
    --robot)  OMP_THREADS=4; PIN_CPUS=""; shift ;;
    --no-pin) OMP_THREADS=12; PIN_CPUS=""; shift ;;
    --stock-plugin) USE_PATCHED_PLUGIN=0; shift ;;
    --profile) PROFILE_LOG="$2"; PROFILING=1; shift 2 ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

PIN_ARGS=()
if [[ -n "${PIN_CPUS}" ]]; then
  PIN_ARGS=(--cpuset-cpus "${PIN_CPUS}")
fi
PLUGIN_ARGS=()
PLUGIN_INFO="stock (image)"
if [[ $USE_PATCHED_PLUGIN -eq 1 ]]; then
  if [[ -f "${PATCHED_PLUGIN}" ]]; then
    PLUGIN_ARGS=(-v "${PATCHED_PLUGIN}:/usr/lib/x86_64-linux-gnu/libvaccel-torch.so.0.2.1:ro")
    PLUGIN_INFO="patched (${PATCHED_PLUGIN})"
  else
    echo "[warn] ${PATCHED_PLUGIN} not found -- using the image's stock plugin (no GPU sync in torch_run)"
  fi
fi
# -t always (whole log lines); -i only from a terminal (agent_server.py has none)
TTY_ARGS=(-t)
[[ -t 0 ]] && TTY_ARGS=(-it)

OMP_ARGS=()
if [[ $SET_OMP -eq 1 ]]; then
  OMP_ARGS=(
    -e "OMP_NUM_THREADS=${OMP_THREADS}"
    -e "OMP_THREAD_LIMIT=${OMP_THREADS}"
  )
fi

GPU_ARGS=()
CUDA_LD_ARGS=()
if [[ "${MODE}" == "gpu" ]]; then
  GPU_ARGS=(--gpus all)
  # venv cudnn/cublas before the system apt 9.7.0.66 (not on the image's LD_LIBRARY_PATH)
  # The harbor image has the pip package installed but /.venv/ is not in its LD_LIBRARY_PATH.
  VENV_NVIDIA="/.venv/lib/python3.10/site-packages/nvidia"
  # cuDNN 9.28 (src/models/runtime/cudnn-9.28, from scripts/sol_v09/install.sh) first on the library path (the image
  # alone would give libtorch 9.8). The SOL agent uses 9.20, the only version its GPU libraries run with.
  CUDNN_DIR="$(cd "$(dirname "$0")" && pwd)/src/models/runtime/cudnn-9.28"
  [[ -f "${CUDNN_DIR}/libcudnn.so.9" ]] || { echo "[err] ${CUDNN_DIR} missing (scripts/sol_v09/install.sh gpu ... copies it)"; exit 1; }
  CUDA_LD_ARGS=(-v "${CUDNN_DIR}:/cudnn:ro"
                -e "LD_LIBRARY_PATH=/cudnn:${VENV_NVIDIA}/cudnn/lib:${VENV_NVIDIA}/cublas/lib:/workspace/libtorch/lib:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64")
  echo "cuDNN: ${CUDNN_DIR}"
fi

echo "Starting ${CONTAINER_NAME} (${MODE}) on port ${PORT}"
if [[ $SET_OMP -eq 1 ]]; then echo "OMP_NUM_THREADS=${OMP_THREADS}"; else echo "OMP_NUM_THREADS=unset"; fi
echo "CPU pinning: ${PIN_CPUS:-none}"
echo "vaccel-torch plugin: ${PLUGIN_INFO}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}"

# --profile: per-call vAccel profiling ([prof] lines need log level 3)
LOG_LEVEL=1
if [[ -n "${PROFILE_LOG}" ]]; then
  LOG_LEVEL=3
  mkdir -p "$(dirname "${PROFILE_LOG}")"
  echo "Profiling ON: agent output appended to ${PROFILE_LOG}"
  echo "[host] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${MODE} agent, CPU governor(s): $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')cpuset: ${PIN_CPUS:-none} OMP threads: $([[ $SET_OMP -eq 1 ]] && echo "${OMP_THREADS}" || echo unset) plugin: ${PLUGIN_INFO}" \
    | tee -a "${PROFILE_LOG}"
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  "${PIN_ARGS[@]}" \
  "${PLUGIN_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "${OMP_ARGS[@]}" \
  "${CUDA_LD_ARGS[@]}" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}" \
  -e "VACCEL_LOG_LEVEL=${LOG_LEVEL}" \
  -e "TZ=UTC" \
  -e "VACCEL_PROFILING_ENABLED=${PROFILING}" \
  -e "TTRPC_TCP_NODELAY_ENABLED=1" \
  --network host \
  "${IMAGE}" \
  bash -lc "vaccel-rpc-agent -a tcp://0.0.0.0:${PORT}" 2>&1 | if [[ -n "${PROFILE_LOG}" ]]; then tee -a "${PROFILE_LOG}"; else cat; fi
