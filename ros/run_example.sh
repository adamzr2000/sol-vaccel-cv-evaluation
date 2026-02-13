#!/usr/bin/env bash
set -euo pipefail

IMAGE="adamzr2000/ros-humble-dev:latest"
DDS="${1:-cyclone}"          # cyclone | fast
IFACE="${2:-enp130s0}"       # host NIC name

# Common (optional but recommended)
DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Paths (host -> container)
HOST_CFG_DIR="$(pwd)/dds"
CONT_CFG_DIR="/etc/dds"

mkdir -p "$HOST_CFG_DIR"

# --- Generate Cyclone config (no deprecated NetworkInterfaceAddress) ---
cat > "$HOST_CFG_DIR/cyclonedds.xml" <<EOF
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="${IFACE}"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>
EOF

# --- Generate Fast DDS config ---
# This is a minimal profile. Interface binding in Fast DDS can get complex depending on version
# and whether you're using UDPv4, shared memory, discovery server, etc.
# Start simple, then tighten if needed.
cat > "$HOST_CFG_DIR/fastdds.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <participant profile_name="ros2_default_participant" is_default_profile="true">
    <rtps>
      <useBuiltinTransports>true</useBuiltinTransports>
      <!-- You can add discovery-server or transport/interface restrictions later if needed -->
    </rtps>
  </participant>
</profiles>
EOF

# Pick RMW + env
ENV_COMMON=(
  -e "ROS_DOMAIN_ID=${DOMAIN_ID}"
)

case "$DDS" in
  cyclone)
    RMW="rmw_cyclonedds_cpp"
    ENV_DDS=(
      -e "RMW_IMPLEMENTATION=${RMW}"
      -e "CYCLONEDDS_URI=file://${CONT_CFG_DIR}/cyclonedds.xml"
    )
    ;;
  fast|fastdds|fastrtps)
    RMW="rmw_fastrtps_cpp"
    ENV_DDS=(
      -e "RMW_IMPLEMENTATION=${RMW}"
      -e "FASTRTPS_DEFAULT_PROFILES_FILE=${CONT_CFG_DIR}/fastdds.xml"
    )
    ;;
  *)
    echo "Usage: $0 [cyclone|fast] [iface]"
    exit 2
    ;;
esac

echo "Launching ROS 2 Humble with ${RMW} (DDS=${DDS}, iface=${IFACE}, domain=${DOMAIN_ID})"

docker run -it --rm \
  --net=host \
  -v "${HOST_CFG_DIR}:${CONT_CFG_DIR}:ro" \
  "${ENV_COMMON[@]}" \
  "${ENV_DDS[@]}" \
  "${IMAGE}" \
  bash
