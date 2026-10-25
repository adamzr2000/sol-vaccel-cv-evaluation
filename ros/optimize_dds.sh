#!/bin/bash
set -e

echo "Optimizing network buffers for ROS 2/DDS..."

# -n: never prompt -- fail fast with a clear message instead of hanging if
# there's no cached sudo credential (e.g. a stale/non-interactive session).
if ! sudo -n true 2>/dev/null; then
  echo "Error: sudo requires a password in this session (no cached credential)." >&2
  echo "Run 'sudo -v' first, or re-run this script interactively." >&2
  exit 1
fi

# Increase the maximum receive buffer size for network packets
# 2 GiB, default is 208 KiB
sudo -n sysctl -w net.core.rmem_max=2147483647

# Increase the maximum send buffer size for network packets (send-side analog
# of rmem_max above -- only the receive side was tuned previously)
# 2 GiB, default is 208 KiB
sudo -n sysctl -w net.core.wmem_max=2147483647

# IP fragmentation settings
# in seconds, default is 30 s
sudo -n sysctl -w net.ipv4.ipfrag_time=3

# 128 MiB, default is 256 KiB
sudo -n sysctl -w net.ipv4.ipfrag_high_thresh=134217728

echo "Settings applied successfully."