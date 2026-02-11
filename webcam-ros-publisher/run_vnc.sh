#!/usr/bin/env bash
set -euo pipefail

# ROS master
export ROS_MASTER_URI="http://10.5.1.20:11311"

# (Optional but commonly needed) your machine/container IP for ROS nodes
# export ROS_IP="$(hostname -I | awk '{print $1}')"

docker run --rm -d \
  --name ros-vnc \
  -p 6080:80 \
  --shm-size=512m \
  -e ROS_MASTER_URI="$ROS_MASTER_URI" \
  tiryoh/ros-desktop-vnc:melodic

echo "ROS VNC desktop started."
echo "Open: http://localhost:6080 in your browser"
echo "Inside the desktop, run:"
echo " rqt_image_view"