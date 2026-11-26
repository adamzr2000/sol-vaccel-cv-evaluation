#!/usr/bin/env bash
set -euo pipefail

IMAGE="torchvision-app:agent"
CONTAINER_NAME="torchvision-app-agent"
OMP_THREADS=12
SET_OMP=1
PORT=9125
PROFILE_LOG=""   # --profile: per-call vAccel profiling, agent output appended to this file
RPC_AGENT="zero-copy"   # zero-copy vAccel RPC agent (scripts/vaccel_rpc/build.sh); --stock-rpc: the image's agent
PROFILING=0
# edge-asus: pin to the 8 P-cores (0-7) with 8 OpenMP threads (--no-pin: 12 threads, any core)
PIN_CPUS=""
if [[ "$(hostname)" == "asus-g815" ]]; then
  PIN_CPUS="0-7"
  OMP_THREADS=8
fi

usage() {
  echo "Usage: $0 <cpu|gpu> [--port PORT] [--no-omp] [--debug] [--robot] [--no-pin] [--profile LOGFILE] [--stock-rpc]"
  echo "  --stock-rpc  run the image's vaccel-rpc-agent instead of the zero-copy agent (scripts/vaccel_rpc)"
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
    --debug)  PROFILING=1; shift ;;
    --robot)  OMP_THREADS=4; PIN_CPUS=""; shift ;;
    --no-pin) OMP_THREADS=12; PIN_CPUS=""; shift ;;
    --profile) PROFILE_LOG="$2"; PROFILING=1; shift 2 ;;
    --stock-rpc) RPC_AGENT="stock"; shift ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

PIN_ARGS=()
if [[ -n "${PIN_CPUS}" ]]; then
  PIN_ARGS=(--cpuset-cpus "${PIN_CPUS}")
fi
# -t always (whole log lines); -i only from a terminal (agent_server.py has none)
TTY_ARGS=(-t)
[[ -t 0 ]] && TTY_ARGS=(-it)

# CPU: SOL runs every inference on one worker thread of the wrapper (one warm OpenMP team instead of one per agent
# RPC thread; scripts/sol_v09/wrapper.c)
WORKER_ARGS=()
[[ "${MODE}" == "cpu" ]] && WORKER_ARGS=(-e "SOL_WORKER_THREAD=1")
# CPU and GPU: glibc keeps freed memory in the process (large mmap/trim thresholds). SOL (CPU) allocates its output
# and intermediate buffers per inference, and the vAccel RPC layer allocates the multi-MB request/response buffers
# per call; with glibc's defaults they go back to the kernel every call and are page-faulted in again.
# CPU: ResNet-50 14.0 -> 12.0 ms, FCN-R50 68.6 -> 56.7, R(2+1)D-18 110.9 -> 92.1 (8 P-cores);
# GPU: a 4.2 MB segmentation result from the robot's view 63.4 -> 58.1 ms per call.
WORKER_ARGS+=(-e "GLIBC_TUNABLES=glibc.malloc.mmap_threshold=268435456:glibc.malloc.trim_threshold=1073741824:glibc.malloc.top_pad=268435456")

OMP_ARGS=()
if [[ $SET_OMP -eq 1 ]]; then
  OMP_ARGS=(
    -e "OMP_NUM_THREADS=${OMP_THREADS}"
    -e "OMP_THREAD_LIMIT=${OMP_THREADS}"
  )
fi

GPU_ARGS=()
if [[ "${MODE}" == "gpu" ]]; then
  GPU_ARGS=(--gpus all)
  # cuDNN 9.20 (the SOL 0.9 GPU libraries run with this version only; the image ships 9.10), copied from
  # sol-deploy:rc2 by scripts/sol_v09/install.sh, first on the image's library path (rest unchanged).
  CUDNN_DIR="$(cd "$(dirname "$0")" && pwd)/src/models/runtime/cudnn-9.20"
  if [[ -f "${CUDNN_DIR}/libcudnn.so.9" ]]; then
    VENV_NV=/.venv/lib/python3.10/site-packages/nvidia
    GPU_ARGS+=(-v "${CUDNN_DIR}:/cudnn:ro"
               -e "LD_LIBRARY_PATH=/cudnn:/usr/local/cuda/lib64:${VENV_NV}/cuda_runtime/lib:${VENV_NV}/cudnn/lib:${VENV_NV}/cublas/lib")
    echo "cuDNN: ${CUDNN_DIR}"
  else
    echo "[err] ${CUDNN_DIR} missing (scripts/sol_v09/install.sh gpu ... copies it)"; exit 1
  fi
fi

# Agent binary: the zero-copy build (same protocol, without the redundant copies of request/response buffers) is
# mounted over the image's; --stock-rpc keeps the image's /usr/bin/vaccel-rpc-agent.
AGENT_BIN=/usr/bin/vaccel-rpc-agent
RPC_ARGS=()
if [[ "${RPC_AGENT}" == "zero-copy" ]]; then
  ZC_AGENT="$(cd "$(dirname "$0")" && pwd)/src/models/runtime/vaccel-rpc/vaccel-rpc-agent"
  [[ -x "${ZC_AGENT}" ]] || { echo "[err] ${ZC_AGENT} missing (scripts/vaccel_rpc/build.sh builds it; --stock-rpc for the image's agent)"; exit 1; }
  AGENT_BIN=/opt/vaccel-rpc/vaccel-rpc-agent
  RPC_ARGS=(-v "${ZC_AGENT}:${AGENT_BIN}:ro")
fi

echo "Starting ${CONTAINER_NAME} (${MODE}) on port ${PORT}"
echo "vAccel RPC agent: ${RPC_AGENT}"
if [[ $SET_OMP -eq 1 ]]; then echo "OMP_NUM_THREADS=${OMP_THREADS}"; else echo "OMP_NUM_THREADS=unset"; fi
echo "CPU pinning: ${PIN_CPUS:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}"

# --profile: per-call vAccel profiling ([prof] lines need log level 3)
LOG_LEVEL=1
if [[ -n "${PROFILE_LOG}" ]]; then
  LOG_LEVEL=3
  mkdir -p "$(dirname "${PROFILE_LOG}")"
  echo "Profiling ON: agent output appended to ${PROFILE_LOG}"
  echo "[host] $(date -u +%Y-%m-%dT%H:%M:%SZ) ${MODE} agent, CPU governor(s): $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')cpuset: ${PIN_CPUS:-none} rpc agent: ${RPC_AGENT} OMP threads: $([[ $SET_OMP -eq 1 ]] && echo "${OMP_THREADS}" || echo unset)" \
    | tee -a "${PROFILE_LOG}"
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  "${PIN_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "${OMP_ARGS[@]}" \
  "${WORKER_ARGS[@]}" \
  "${RPC_ARGS[@]}" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL}" \
  -e "VACCEL_LOG_LEVEL=${LOG_LEVEL}" \
  -e "TZ=UTC" \
  -e "VACCEL_PROFILING_ENABLED=${PROFILING}" \
  -e "TTRPC_TCP_NODELAY_ENABLED=1" \
  --network host \
  --entrypoint "${AGENT_BIN}" \
  "${IMAGE}" -a "tcp://0.0.0.0:${PORT}" 2>&1 | if [[ -n "${PROFILE_LOG}" ]]; then tee -a "${PROFILE_LOG}"; else cat; fi
