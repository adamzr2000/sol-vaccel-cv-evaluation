#!/usr/bin/env bash
set -euo pipefail

SCRIPT="model_benchmark_ros.py"
SLEEP_SEC=30
BACKEND="stock"
REMOTE_HOST="edge-asus"   # remote inference target for robot + vaccel-remote-* backends
VACCEL_RPC_PORT=9125      # vaccel agent RPC port (override with env var)
SELECTED_MODELS=""        # Comma-separated; empty means run all
CPU_AFFINITY=""           # CPU core mask for taskset (e.g. "0-7" for P-cores only); empty = no pinning

# Cleanup on exit (Ctrl+C or error)
trap 'echo -e "\n[bench] Interrupted. Exiting..."; exit 1' SIGINT SIGTERM

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[err] ${SCRIPT} not found in $(pwd)"
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") --host <edge-asus|edge-xtreme|edge-alienware|robot> --device <cpu|gpu> --run-tag <tag> [options]

Required:
  --host        edge-asus|edge-xtreme|edge-alienware|robot
  --device      cpu|gpu
  --run-tag     run identifier (e.g., run1)

Optional:
  --backend       stock|ptc|aoti|sol|vaccel-local-torch|vaccel-remote-torch\
                  |vaccel-local-ptc|vaccel-remote-ptc\
                  |vaccel-local-sol|vaccel-remote-sol (default: ${BACKEND})
  --sleep         seconds to wait between runs (default: ${SLEEP_SEC})
  --model         comma-separated model(s) to run (default: all)
                  e.g. --model resnet50 or --model resnet50,swin_t,fcn_resnet50
  --duration      experiment duration in seconds per model (default: ${EXPERIMENT_DURATION_SEC})
  --no-monitoring  disable resource monitoring only (export still runs)
  --no-export      disable both export and resource monitoring
  --cpu-affinity   CPU core list for taskset (e.g. "0-7" pins to P-cores only for fair cross-backend comparison)
  --remote-host    robot|edge-asus|edge-xtreme|edge-alienware (default: ${REMOTE_HOST})
                  remote inference target for vaccel-remote-* backends (use --host value for loopback)
                  also sets VACCEL_RPC_ADDRESS=tcp://<host-ip>:<VACCEL_RPC_PORT> (default port: ${VACCEL_RPC_PORT})

Examples:
  $(basename "$0") --host robot --device cpu --run-tag run1
  $(basename "$0") --host edge-asus --device gpu --backend vaccel-local-sol --run-tag iso --no-monitoring
  $(basename "$0") --host robot --device cpu --run-tag run1 --model resnet50,swin_t
  $(basename "$0") --host robot --device cpu --run-tag run1 --no-export
EOF
}

HOST=""
DEVICE=""
RUN_TAG=""
EXPORT_RESULTS=1
RESOURCE_MONITORING=1
EXPERIMENT_DURATION_SEC=120

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        HOST="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --run-tag)     RUN_TAG="$2"; shift 2 ;;
    --backend)     BACKEND="$2"; shift 2 ;;
    --sleep)       SLEEP_SEC="$2"; shift 2 ;;
    --model)       SELECTED_MODELS="$2"; shift 2 ;;
    --duration)    EXPERIMENT_DURATION_SEC="$2"; shift 2 ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --no-monitoring)  RESOURCE_MONITORING=0; shift ;;
    --no-export)      EXPORT_RESULTS=0; RESOURCE_MONITORING=0; shift ;;
    --cpu-affinity)   CPU_AFFINITY="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "[err] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---- Validate required args ----
if [[ -z "${HOST}" || -z "${DEVICE}" || -z "${RUN_TAG}" ]]; then
  echo "[err] Missing required arguments."
  usage; exit 2
fi

# ---- Validate backend ----
case "${BACKEND}" in
  stock|ptc|aoti|sol|\
  vaccel-local-torch|vaccel-remote-torch|\
  vaccel-local-ptc|vaccel-remote-ptc|\
  vaccel-local-sol|vaccel-remote-sol) ;;
  *) echo "[err] invalid backend: ${BACKEND}"; exit 2 ;;
esac

# ---- Host defaults ----
case "${HOST}" in
  robot)       DOCKER_STATS_ENDPOINT="http://192.168.2.2:6000"; SYSTEM_STATS_ENDPOINT="http://192.168.2.2:6001" ;;
  edge-asus)       DOCKER_STATS_ENDPOINT="http://10.5.1.20:6000";  SYSTEM_STATS_ENDPOINT="http://10.5.1.20:6001" ;;
  edge-xtreme)     DOCKER_STATS_ENDPOINT="http://10.5.1.21:6000";  SYSTEM_STATS_ENDPOINT="http://10.5.1.21:6001" ;;
  edge-alienware)  DOCKER_STATS_ENDPOINT="http://10.5.1.22:6000";  SYSTEM_STATS_ENDPOINT="http://10.5.1.22:6001" ;;
  *) echo "[err] Invalid host: ${HOST}"; exit 2 ;;
esac

# ---- Remote host endpoints + RPC address (vaccel-remote-* only) ----
VACCEL_RPC_PORT="${VACCEL_RPC_PORT:-9125}"
case "${REMOTE_HOST}" in
  robot)       DOCKER_STATS_REMOTE_ENDPOINT="http://192.168.2.2:6000"; SYSTEM_STATS_REMOTE_ENDPOINT="http://192.168.2.2:6001"; VACCEL_RPC_ADDRESS="tcp://192.168.2.2:${VACCEL_RPC_PORT}" ;;
  edge-asus)       DOCKER_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6000";  SYSTEM_STATS_REMOTE_ENDPOINT="http://10.5.1.20:6001";  VACCEL_RPC_ADDRESS="tcp://10.5.1.20:${VACCEL_RPC_PORT}" ;;
  edge-xtreme)     DOCKER_STATS_REMOTE_ENDPOINT="http://10.5.1.21:6000";  SYSTEM_STATS_REMOTE_ENDPOINT="http://10.5.1.21:6001";  VACCEL_RPC_ADDRESS="tcp://10.5.1.21:${VACCEL_RPC_PORT}" ;;
  edge-alienware)  DOCKER_STATS_REMOTE_ENDPOINT="http://10.5.1.22:6000";  SYSTEM_STATS_REMOTE_ENDPOINT="http://10.5.1.22:6001";  VACCEL_RPC_ADDRESS="tcp://10.5.1.22:${VACCEL_RPC_PORT}" ;;
  *) echo "[err] Invalid remote host: ${REMOTE_HOST}"; exit 2 ;;
esac

# ---- Runner ----
run_one () {
  local model="$1"

  # Skip if a filter is active and this model is not in it
  if [[ -n "${SELECTED_MODELS}" && ",${SELECTED_MODELS}," != *",${model},"* ]]; then
    return
  fi

  echo "[bench] run: ${model}"

  export EXPORT_RESULTS="${EXPORT_RESULTS}" \
         EXPERIMENT_DURATION_SEC="${EXPERIMENT_DURATION_SEC}" \
         RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
         HOST="${HOST}" \
         DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
         SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
         DEVICE="${DEVICE}" \
         BACKEND="${BACKEND}" \
         MODEL="${model}" \
         RUN_TAG="${RUN_TAG}"

  if [[ "${HOST}" == edge* ]]; then
    local _omp="${OMP_THREADS_OVERRIDE:-12}"
    export OMP_NUM_THREADS="${_omp}"
    # OMP_THREAD_LIMIT is the hard ceiling that num_threads(n) clauses cannot exceed.
    # AOTInductor compiles #pragma omp parallel num_threads(nproc) at compile time,
    # hardcoding nproc=24 when OMP_NUM_THREADS was unset. OMP_NUM_THREADS alone
    # cannot override a num_threads() clause; OMP_THREAD_LIMIT can.
    export OMP_THREAD_LIMIT="${_omp}"
    # For edge loopback remote runs the agent is on the same host; export REMOTE_HOST so
    # model_benchmark_ros.py can embed it in run_id and JSON without defaulting to edge-asus.
    if [[ "${BACKEND}" == vaccel-remote-* ]]; then
      export REMOTE_HOST="${HOST}"
    fi
  elif [[ "${HOST}" == "robot" && "${BACKEND}" != vaccel-remote-* ]]; then
    # N100: cap both OMP vars so .pt2 kernels compiled on a many-core edge machine
    # (which bake num_threads(N) into OMP pragmas) don't oversubscribe the 4-core N100.
    export OMP_NUM_THREADS=$(nproc)
    export OMP_THREAD_LIMIT=$(nproc)
  elif [[ "${HOST}" == "robot" && "${BACKEND}" == vaccel-remote-* ]]; then
    # Client preprocessing still runs on the N100 — cap OMP even for remote backends.
    export OMP_NUM_THREADS=$(nproc)
    export OMP_THREAD_LIMIT=$(nproc)
    export REMOTE_HOST="${REMOTE_HOST}" \
           DOCKER_STATS_REMOTE_ENDPOINT="${DOCKER_STATS_REMOTE_ENDPOINT}" \
           SYSTEM_STATS_REMOTE_ENDPOINT="${SYSTEM_STATS_REMOTE_ENDPOINT}"
  fi

  # Point the vaccel plugin at the correct agent for any remote backend
  if [[ "${BACKEND}" == vaccel-remote-* ]]; then
    export VACCEL_RPC_ADDRESS="${VACCEL_RPC_ADDRESS}"
  fi

  # Force vaccel-torch plugin into CPU mode when device=cpu;
  # without this the plugin always picks CUDA (if available) and crashes
  # when the input tensor is moved to CUDA but the model is CPU-compiled.
  if [[ "${DEVICE}" == "cpu" && ("${BACKEND}" == "vaccel-local-ptc" || "${BACKEND}" == "vaccel-remote-ptc") ]]; then
    export VACCEL_TORCH_CUDA_ENABLED=0
    # Hide CUDA entirely: prevents CUDA init + background threads that inflate and
    # destabilize CPU inference latency even when VACCEL_TORCH_CUDA_ENABLED=0.
    export CUDA_VISIBLE_DEVICES=""
  fi
  # aoti on cpu: hide CUDA to prevent runtime init from inflating CPU latency
  if [[ "${DEVICE}" == "cpu" && "${BACKEND}" == "aoti" ]]; then
    export CUDA_VISIBLE_DEVICES=""
  fi

  # Exit code 134 (SIGABRT) at shutdown is emitted by AOTInductor's CUDA event
  # cleanup after all results are already saved — safe to ignore.
  # 139 (SIGSEGV) is a real crash and must NOT be silently swallowed.
  # ulimit -c 0 prevents core dump files from being written.
  local _rc=0
  if [[ -n "${CPU_AFFINITY}" ]]; then
    ( ulimit -c 0; taskset -c "${CPU_AFFINITY}" python3 -u "${SCRIPT}" ) || _rc=$?
  else
    ( ulimit -c 0; python3 -u "${SCRIPT}" ) || _rc=$?
  fi
  if [[ $_rc -ne 0 && $_rc -ne 134 ]]; then
    echo "[bench] error: ${SCRIPT} exited with code ${_rc}"
    exit "${_rc}"
  fi

  unset OMP_NUM_THREADS OMP_THREAD_LIMIT VACCEL_TORCH_CUDA_ENABLED CUDA_VISIBLE_DEVICES
  echo "[bench] done: ${model} (sleep ${SLEEP_SEC}s)"
  sleep "${SLEEP_SEC}"
}

# ---- Model List ----
MODELS=(
  "resnet50" "swin_t" "swin_s" "swin_v2_b"
  "swin3d_s" "swin3d_b" "r3d_18" "r2plus1d_18"
  "deeplabv3_resnet50" "fcn_resnet50" "deeplabv3_resnet101" "fcn_resnet101"
)

# Execute
for m in "${MODELS[@]}"; do
  run_one "$m"
done

echo "[bench] all done ✅"
