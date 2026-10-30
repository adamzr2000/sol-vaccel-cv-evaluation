#!/bin/bash
set -e

# Gives vAccel's ttrpc TCP sockets the same buffer headroom ros-vision-service's
# Fast DDS TCP transport already has.
#
# Fast DDS sets sendBufferSize/receiveBufferSize=10485760 (10 MiB) explicitly via
# setsockopt (start_go2.sh / run_vision_inference.sh); optimize_dds.sh's
# net.core.rmem_max/wmem_max only raise the ceiling for such explicit requests.
# ttrpc doesn't request a size, so its sockets are kernel-autotuned instead --
# capped by the 3rd field of net.ipv4.tcp_rmem/tcp_wmem (stock: 6 MiB / 4 MiB),
# which optimize_dds.sh doesn't touch. A segmentation result (4.2 MB logits) is
# larger than the stock 4 MiB send cap; classification/video payloads aren't.
#
# The kernel doubles an explicit SO_SNDBUF/SO_RCVBUF request (bookkeeping
# overhead), so Fast DDS's 10 MiB = 20 MiB of socket memory; tcp_*mem limits
# are that socket memory directly, so 20 MiB is the matching autotune cap.
# Only the max field changes -- min/default stay as they are. Fast DDS sockets
# are unaffected (explicitly sized sockets ignore tcp_*mem autotuning).
#
# Run on BOTH hosts (robot and edge), then restart the vAccel agent and the
# benchmark -- already-open sockets keep their old limits.
#
# Usage: ./optimize_tcp_vaccel.sh [apply|revert|show]   (default: apply)

TCP_BUF_MAX=20971520   # 20 MiB = 2 x Fast DDS's 10 MiB explicit buffer size
STATE_FILE="/tmp/optimize_tcp_vaccel.orig"   # sysctls reset on reboot, as does /tmp

show() {
  sysctl net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.ipv4.tcp_moderate_rcvbuf \
         net.core.rmem_max net.core.wmem_max
}

need_sudo() {
  # -n: never prompt -- same fail-fast behaviour as optimize_dds.sh.
  if ! sudo -n true 2>/dev/null; then
    echo "Error: sudo requires a password in this session (no cached credential)." >&2
    echo "Run 'sudo -v' first, or re-run this script interactively." >&2
    exit 1
  fi
}

case "${1:-apply}" in
  show)
    show
    ;;

  apply)
    need_sudo
    echo "Raising TCP autotuning max to ${TCP_BUF_MAX} B for vAccel (ttrpc) sockets..."
    # Save originals once, so a second apply doesn't overwrite them with tuned values.
    if [[ ! -f "$STATE_FILE" ]]; then
      { echo "tcp_rmem=\"$(sysctl -n net.ipv4.tcp_rmem | xargs)\""
        echo "tcp_wmem=\"$(sysctl -n net.ipv4.tcp_wmem | xargs)\""
      } > "$STATE_FILE"
      echo "Original values saved to ${STATE_FILE}"
    fi
    read -r rmin rdef _ <<< "$(sysctl -n net.ipv4.tcp_rmem)"
    read -r wmin wdef _ <<< "$(sysctl -n net.ipv4.tcp_wmem)"
    sudo -n sysctl -w net.ipv4.tcp_rmem="${rmin} ${rdef} ${TCP_BUF_MAX}"
    sudo -n sysctl -w net.ipv4.tcp_wmem="${wmin} ${wdef} ${TCP_BUF_MAX}"
    # Receive autotuning must be on for the rmem max to matter (stock default: 1).
    if [[ "$(sysctl -n net.ipv4.tcp_moderate_rcvbuf)" != "1" ]]; then
      echo "[warn] net.ipv4.tcp_moderate_rcvbuf != 1 -- receive autotuning is off, tcp_rmem max has no effect." >&2
    fi
    echo "Settings applied successfully. Restart the vAccel agent and benchmark to pick them up."
    ;;

  revert)
    need_sudo
    if [[ ! -f "$STATE_FILE" ]]; then
      echo "Error: no saved originals (${STATE_FILE}) -- nothing to revert (or the host rebooted, which already reset them)." >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    sudo -n sysctl -w net.ipv4.tcp_rmem="${tcp_rmem}"
    sudo -n sysctl -w net.ipv4.tcp_wmem="${tcp_wmem}"
    rm -f "$STATE_FILE"
    echo "Original values restored."
    ;;

  *)
    echo "Usage: $0 [apply|revert|show]" >&2
    exit 2
    ;;
esac
