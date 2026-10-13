#!/usr/bin/env bash
set -euo pipefail

# Start/stop a fixed-rate iperf3 UDP cross-traffic flow to model uplink
# network congestion for the vAccel-vs-ROS2 experiments.
#
#   edge host:  ./iperf3_congestion.sh server
#   robot host: ./iperf3_congestion.sh client --target <edge-ip> --rate 50M

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
mkdir -p "${RUN_DIR}"

command -v iperf3 >/dev/null 2>&1 || { echo "[err] iperf3 not found on PATH."; exit 1; }

DEFAULT_PORT=5201
DEFAULT_RATE="50M"
DEFAULT_DURATION=60   # finite by default so a forgotten run can't congest the link forever
DEFAULT_PROTO="udp"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

  server [--port PORT] [--bind ADDR]
      Run on the host that receives the extra load (edge).
  client --target HOST [--rate RATE] [--duration SEC] [--port PORT]
         [--proto udp|tcp] [--reverse] [--parallel N] [--bind ADDR] [--force]
      Run on the host whose uplink you want loaded (robot). Returns
      immediately; run your benchmark concurrently.
  stop [server|client|all]   (default: all)
  status

Example:
  edge\$  $(basename "$0") server
  robot\$ $(basename "$0") client --target 10.5.1.20 --rate 50M --duration 120
EOF
}

is_alive() {
  [[ -f "$1" ]] || return 1
  local pid; pid="$(cat "$1" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # Guard against PID reuse: a dead iperf3's PID can get recycled by an
  # unrelated process by the time we check it. Confirm it's still iperf3
  # before trusting the pidfile.
  [[ "$(ps -p "$pid" -o comm= 2>/dev/null)" == "iperf3" ]]
}

cmd_server() {
  local port="$DEFAULT_PORT" bind=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port) port="$2"; shift 2 ;;
      --bind) bind="$2"; shift 2 ;;
      *) echo "[err] Unknown option: $1"; usage; exit 2 ;;
    esac
  done

  is_uint_gt0 "$port" || { echo "[err] --port must be a positive integer."; exit 2; }

  local pidfile="${RUN_DIR}/server.pid"
  if is_alive "$pidfile"; then
    echo "[err] Server already running (PID $(cat "$pidfile"))."; exit 1
  fi
  rm -f "$pidfile"

  local bind_args=()
  [[ -n "$bind" ]] && bind_args=(--bind "$bind")
  iperf3 -s -p "$port" -D -I "$pidfile" "${bind_args[@]}"
  sleep 0.3

  if is_alive "$pidfile"; then
    echo "[ok] Server listening on port ${port} (PID $(cat "$pidfile"))."
  else
    echo "[err] Server failed to start (port ${port} already in use? check: systemctl is-active iperf3)"
    exit 1
  fi
}

cmd_client() {
  local target="" rate="$DEFAULT_RATE" duration="$DEFAULT_DURATION" port="$DEFAULT_PORT"
  local proto="$DEFAULT_PROTO" reverse=0 parallel=1 bind="" force=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target)   target="$2"; shift 2 ;;
      --rate)     rate="$2"; shift 2 ;;
      --duration) duration="$2"; shift 2 ;;
      --port)     port="$2"; shift 2 ;;
      --proto)    proto="$2"; shift 2 ;;
      --reverse)  reverse=1; shift ;;
      --parallel) parallel="$2"; shift 2 ;;
      --bind)     bind="$2"; shift 2 ;;
      --force)    force=1; shift ;;
      *) echo "[err] Unknown option: $1"; usage; exit 2 ;;
    esac
  done

  [[ -n "$target" ]] || { echo "[err] --target HOST is required."; usage; exit 2; }
  case "$proto" in udp|tcp) ;; *) echo "[err] --proto must be udp or tcp"; exit 2 ;; esac
  is_uint_gt0 "$port" || { echo "[err] --port must be a positive integer."; exit 2; }
  is_uint_gt0 "$duration" || { echo "[err] --duration must be a positive integer (seconds)."; exit 2; }
  is_uint_gt0 "$parallel" || { echo "[err] --parallel must be a positive integer."; exit 2; }

  local pidfile="${RUN_DIR}/client.pid"
  if is_alive "$pidfile"; then
    if [[ "$force" -ne 1 ]]; then
      echo "[err] Client already running (PID $(cat "$pidfile")). Use --force to replace it."; exit 1
    fi
    echo "[info] --force: stopping existing client first."
    cmd_stop client
  fi
  rm -f "$pidfile"

  local args=(-c "$target" -p "$port" -b "$rate" -t "$duration" -P "$parallel")
  [[ "$proto" == "udp" ]] && args+=(-u)
  [[ "$reverse" -eq 1 ]] && args+=(-R)
  [[ -n "$bind" ]] && args+=(--bind "$bind")

  echo "[info] ${proto} cross-traffic -> ${target}:${port} @ ${rate} for ${duration}s"
  nohup iperf3 "${args[@]}" >/dev/null 2>&1 &
  local pid=$!
  disown "$pid"
  echo "$pid" > "$pidfile"
  sleep 0.3

  if is_alive "$pidfile"; then
    echo "[ok] Client running (PID ${pid}); stops itself after ${duration}s, or: $(basename "$0") stop client"
  else
    echo "[err] Client failed to start."
    rm -f "$pidfile"
    exit 1
  fi
}

cmd_stop() {
  local target="${1:-all}"
  case "$target" in
    server|client|all) ;;
    *) echo "[err] stop target must be server, client, or all (got: ${target})"; exit 2 ;;
  esac
  for role in server client; do
    [[ "$target" == "$role" || "$target" == "all" ]] || continue
    local pidfile="${RUN_DIR}/${role}.pid"
    if is_alive "$pidfile"; then
      local pid; pid="$(cat "$pidfile")"
      kill "$pid" 2>/dev/null || true
      sleep 0.2
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
      echo "[ok] Stopped ${role} (PID ${pid})."
    else
      echo "[info] No running ${role} tracked."
    fi
    rm -f "$pidfile"
  done
}

cmd_status() {
  for role in server client; do
    local pidfile="${RUN_DIR}/${role}.pid"
    if is_alive "$pidfile"; then
      local pid; pid="$(cat "$pidfile")"
      local etime
      etime="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')" || etime=""
      echo "[${role}] RUNNING  PID=${pid}  uptime=${etime:-?}"
    else
      echo "[${role}] not running"
      rm -f "$pidfile"
    fi
  done
}

is_uint_gt0() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

main() {
  local command="${1:-}"
  [[ $# -ge 1 ]] && shift || true
  case "$command" in
    server) cmd_server "$@" ;;
    client) cmd_client "$@" ;;
    stop)   cmd_stop "$@" ;;
    status) cmd_status "$@" ;;
    -h|--help|"") usage ;;
    *) echo "[err] Unknown command: ${command}"; usage; exit 2 ;;
  esac
}

main "$@"
