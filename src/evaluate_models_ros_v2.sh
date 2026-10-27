#!/usr/bin/env bash
set -euo pipefail

# v2: runs model_benchmark_ros_v2.py, which reads the GO2 camera directly via
# GStreamer instead of subscribing to a ROS2 topic published by
# go2_camerapublisher -- no go2_camerapublisher/go2-driver container needed
# for this script. Everything else below (model loop, resource monitoring,
# flags) is identical to evaluate_models_ros.sh -- see model_benchmark_ros_v2.py
# for why. GO2_LAN_PORT / IMAGE_WIDTH / IMAGE_HEIGHT (camera interface/capture
# resolution) can be overridden via env var before invoking this script if
# your deployment differs from the defaults (enp2s0, 1280x720).
SCRIPT="model_benchmark_ros_v2.py"
SLEEP_SEC=30
BACKEND="stock"
REMOTE_HOST="edge-asus"   # remote inference target for robot + vaccel-remote-* backends
SELECTED_MODELS=""        # Comma-separated; empty means run all
CPU_AFFINITY=""           # CPU core mask for taskset (e.g. "0-7" for P-cores only); empty = no pinning

# ---- Advanced env-var-only overrides (rarely needed; no CLI flag on purpose,
# to keep the normal CLI surface small -- export these before invoking the
# script if you ever need something other than the default) ----
VACCEL_RPC_PORT="${VACCEL_RPC_PORT:-9125}"               # vaccel agent RPC port
OMP_THREADS_OVERRIDE="${OMP_THREADS_OVERRIDE:-12}"       # OMP thread cap on edge hosts
WIFI_STATS_ENDPOINT="${WIFI_STATS_ENDPOINT:-http://192.168.2.1:6002}"  # miniPC's wifi-stats-collector
WIFI_IFACE="${WIFI_IFACE:-wlp1s0}"                       # miniPC's wireless interface

parse_bool() {
  case "$1" in
    true|1|yes)  echo 1 ;;
    false|0|no)  echo 0 ;;
    *) echo "[err] invalid boolean value: '$1' (use true or false)" >&2; exit 2 ;;
  esac
}

# Cleanup on exit (Ctrl+C or error)
trap 'echo -e "\n[bench] Interrupted. Exiting..."; exit 1' SIGINT SIGTERM

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[err] ${SCRIPT} not found in $(pwd)"
  exit 1
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") --host robot --device <cpu|gpu> --run-tag <tag> [options]

v2-only restriction: --host must be "robot" -- this script reads the GO2
camera directly via GStreamer, local to the robot's own camera network
segment, and cannot reach a remote edge-* host the way go2_camerapublisher's
ROS2/DDS relay could. Use evaluate_models_ros.sh (v1) for --host edge-*.

Required:
  --host        robot (only -- see restriction above)
  --device      cpu|gpu
  --run-tag     run identifier (e.g., run1)

Optional:
  --backend          stock|ptc|aoti|sol|vaccel-local-torch|vaccel-remote-torch\
                     |vaccel-local-ptc|vaccel-remote-ptc\
                     |vaccel-local-sol|vaccel-remote-sol (default: ${BACKEND})
  --sleep            seconds to wait between runs (default: ${SLEEP_SEC})
  --model            comma-separated model(s) to run (default: all)
                     e.g. --model resnet50 or --model resnet50,swin_t,fcn_resnet50
  --duration         experiment duration in seconds per model (default: ${EXPERIMENT_DURATION_SEC})
  --samples          stop after N processed samples per model instead of by duration.
                     Takes priority over --duration when set (default: unset — duration governs stop).
  --export           true|false — export JSON/CSV results (default: true)
  --monitoring       true|false — docker/system resource monitoring (default: true)
  --wifi-monitoring  true|false — WiFi link stats (retries/RSSI) from the miniPC's
                     wifi-stats-collector (default: false). Only takes effect with --host robot,
                     ignored (with a warning) otherwise. Endpoint/interface fixed at
                     ${WIFI_STATS_ENDPOINT} / ${WIFI_IFACE}; override via the
                     WIFI_STATS_ENDPOINT / WIFI_IFACE env vars if ever needed.
  --cpu-affinity     CPU core list for taskset (e.g. "0-7" pins to P-cores only for fair cross-backend comparison)
  --remote-host      robot|edge-asus|edge-xtreme|edge-alienware (default: ${REMOTE_HOST})
                     remote inference target for vaccel-remote-* backends (use --host value for loopback)
                     also sets VACCEL_RPC_ADDRESS=tcp://<host-ip>:<VACCEL_RPC_PORT> (default port: ${VACCEL_RPC_PORT})

Every --<flag> true|false option defaults to the value shown above and is
fully independent of the others -- e.g. --export false --monitoring true
skips result export for a run while still recording resource monitoring.

Examples:
  $(basename "$0") --host robot --device cpu --run-tag run1
  $(basename "$0") --host robot --device cpu --backend vaccel-remote-sol --remote-host edge-asus --run-tag iso --monitoring false
  $(basename "$0") --host robot --device cpu --run-tag run1 --model resnet50,swin_t
  $(basename "$0") --host robot --device cpu --run-tag run1 --export false
  $(basename "$0") --host robot --device cpu --backend vaccel-remote-sol --run-tag wifi1 --wifi-monitoring true
EOF
}

HOST=""
DEVICE=""
RUN_TAG=""
EXPORT_RESULTS=1
RESOURCE_MONITORING=1
WIFI_MONITORING=0
EXPERIMENT_DURATION_SEC=120
MAX_SAMPLES=0   # 0 = disabled; --duration governs stop as today

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)             HOST="$2"; shift 2 ;;
    --device)           DEVICE="$2"; shift 2 ;;
    --run-tag)          RUN_TAG="$2"; shift 2 ;;
    --backend)          BACKEND="$2"; shift 2 ;;
    --sleep)            SLEEP_SEC="$2"; shift 2 ;;
    --model)            SELECTED_MODELS="$2"; shift 2 ;;
    --duration)         EXPERIMENT_DURATION_SEC="$2"; shift 2 ;;
    --samples)          MAX_SAMPLES="$2"; shift 2 ;;
    --remote-host)      REMOTE_HOST="$2"; shift 2 ;;
    --export)           EXPORT_RESULTS="$(parse_bool "$2")"; shift 2 ;;
    --monitoring)        RESOURCE_MONITORING="$(parse_bool "$2")"; shift 2 ;;
    --wifi-monitoring)   WIFI_MONITORING="$(parse_bool "$2")"; shift 2 ;;
    --cpu-affinity)      CPU_AFFINITY="$2"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "[err] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ---- Validate required args ----
if [[ -z "${HOST}" || -z "${DEVICE}" || -z "${RUN_TAG}" ]]; then
  echo "[err] Missing required arguments."
  usage; exit 2
fi

# ---- v2-only restriction ----
# model_benchmark_ros_v2.py reads the GO2 camera's raw UDP/H264 multicast
# stream directly -- local to the robot's own LAN segment only. Unlike
# go2_camerapublisher's ROS2/DDS relay (what v1's INPUT_TOPIC subscription
# actually depends on for --host edge-*), that multicast stream does not
# route to a remote host. --host edge-asus/edge-xtreme/edge-alienware (e.g.
# the "iso" local-backend runs v1 documents above) would just hang for 10s
# per model waiting for a frame that can never arrive. Use
# evaluate_models_ros.sh (v1) for any --host other than robot.
if [[ "${HOST}" != "robot" ]]; then
  echo "[err] evaluate_models_ros_v2.sh only supports --host robot (direct GStreamer" >&2
  echo "      read is local to the robot's own camera network segment -- it cannot" >&2
  echo "      reach ${HOST} the way go2_camerapublisher's ROS2/DDS relay could)." >&2
  echo "      Use evaluate_models_ros.sh for --host ${HOST}." >&2
  exit 2
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

if [[ "${WIFI_MONITORING}" -eq 1 && "${HOST}" != "robot" ]]; then
  echo "[warn] --wifi-monitoring only applies with --host robot; ignoring for --host ${HOST}."
  WIFI_MONITORING=0
fi

# ---- Remote host endpoints + RPC address (vaccel-remote-* only) ----
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
         MAX_SAMPLES="${MAX_SAMPLES}" \
         RESOURCE_MONITORING="${RESOURCE_MONITORING}" \
         HOST="${HOST}" \
         DOCKER_STATS_ENDPOINT="${DOCKER_STATS_ENDPOINT}" \
         SYSTEM_STATS_ENDPOINT="${SYSTEM_STATS_ENDPOINT}" \
         DEVICE="${DEVICE}" \
         BACKEND="${BACKEND}" \
         MODEL="${model}" \
         RUN_TAG="${RUN_TAG}"

  if [[ "${HOST}" == edge* ]]; then
    export OMP_NUM_THREADS="${OMP_THREADS_OVERRIDE}"
    # OMP_THREAD_LIMIT is the hard ceiling that num_threads(n) clauses cannot exceed.
    # AOTInductor compiles #pragma omp parallel num_threads(nproc) at compile time,
    # hardcoding nproc=24 when OMP_NUM_THREADS was unset. OMP_NUM_THREADS alone
    # cannot override a num_threads() clause; OMP_THREAD_LIMIT can.
    export OMP_THREAD_LIMIT="${OMP_THREADS_OVERRIDE}"
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

  # WiFi link stats (miniPC's wifi-stats-collector) bracket the python call
  # directly from here, not threaded through model_benchmark_ros.py -- kept
  # separate on purpose (small/special-case feature, different host).
  # Device/remote-host suffix mirrors model_benchmark_ros.py's own run_id
  # branching exactly (REMOTE_HOST-DEVICE for a remote backend, DEVICE alone
  # otherwise) -- without it, runs that differ only by --device or
  # --remote-host silently overwrote each other's wifi-stats file.
  local _wifi_tag="${RUN_TAG}_${model}_${BACKEND}"
  if [[ "${BACKEND}" == vaccel-remote-* ]]; then
    _wifi_tag="${_wifi_tag}_${REMOTE_HOST}-${DEVICE}"
  else
    _wifi_tag="${_wifi_tag}_${DEVICE}"
  fi
  if [[ "${WIFI_MONITORING}" -eq 1 ]]; then
    curl -s --max-time 5 -X POST "${WIFI_STATS_ENDPOINT}/start" \
      -H "Content-Type: application/json" \
      -d "{\"wifi_interface\": \"${WIFI_IFACE}\", \"tag\": \"${_wifi_tag}\"}" \
      > /dev/null || echo "[warn] wifi-stats /start failed (continuing without it)"
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

  # Always stop wifi monitoring, even if the python run above failed/crashed --
  # this must run before the error-triggered `exit` below, or a crash would
  # leave the miniPC's monitor running forever with nothing left to stop it.
  if [[ "${WIFI_MONITORING}" -eq 1 ]]; then
    mkdir -p "/results/experiments/wifi-stats/${HOST}" 2>/dev/null \
      || echo "[warn] could not create wifi-stats results dir (continuing without it)"
    curl -s --max-time 5 -X POST "${WIFI_STATS_ENDPOINT}/stop" \
      -o "/results/experiments/wifi-stats/${HOST}/${_wifi_tag}_totals.json" \
      || echo "[warn] wifi-stats /stop failed"
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
