#!/usr/bin/env bash
set -e

# Flags
START_VNC=false
for arg in "$@"; do
  case "$arg" in
    --vnc) START_VNC=true ;;
    -h|--help)
      echo "Usage: $0 [--vnc]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--vnc]"
      exit 1
      ;;
  esac
done

# Get primary host IP
HOST_IP="$(hostname -I | awk '{print $1}')"
if [ -z "$HOST_IP" ]; then
  echo "❌ Could not determine host IP"
  exit 1
fi
echo "Using HOST_IP=$HOST_IP"

# Write .env file for docker compose
cat > .env <<EOF
HOST_IP=$HOST_IP
EOF
echo ".env file written"

# Start compose stack
docker compose up -d

# Compute ROS_MASTER_URI for other containers (host networking)
ROS_MASTER_URI="http://${HOST_IP}:11311"

if [ "$START_VNC" = true ]; then
  # Remove existing ros-vnc container if present
  if docker ps -a --format '{{.Names}}' | grep -qx 'ros-vnc'; then
    docker rm -f ros-vnc >/dev/null
  fi

  docker run --rm -d \
    --name ros-vnc \
    -p 6080:80 \
    --shm-size=512m \
    -e ROS_MASTER_URI="$ROS_MASTER_URI" \
    tiryoh/ros-desktop-vnc:melodic

  echo "ROS VNC desktop started."
  echo "Open: http://localhost:6080 in your browser"
  echo "Inside the desktop, run:"
  echo "  rqt_image_view"
fi