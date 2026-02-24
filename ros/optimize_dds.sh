#!/bin/bash

echo "Optimizing network buffers for ROS 2/DDS..."

# Increase the maximum receive buffer size for network packets
# 2 GiB, default is 208 KiB
sudo sysctl -w net.core.rmem_max=2147483647

# IP fragmentation settings
# in seconds, default is 30 s
sudo sysctl -w net.ipv4.ipfrag_time=3

# 128 MiB, default is 256 KiB
sudo sysctl -w net.ipv4.ipfrag_high_thresh=134217728

echo "Settings applied successfully."