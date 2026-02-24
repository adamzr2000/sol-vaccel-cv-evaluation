#!/bin/bash
set -e

container_image="adamzr2000/ros-humble-dev"
container_name="ros-dev"
ros_domain_id=0

dds="fast"
iface="lo"   # only applies to cyclone
gui=0        # default: no GUI/X11
gpu=0        # default: no GPU access

HOST_CFG_DIR="./dds"
CONT_CFG_DIR="/etc/dds"

usage() {
  echo "Usage: $0 [--ros-domain-id <id>] [--dds fast|cyclone] [--iface <iface>] [--gui] [--gpu]"
  echo "Defaults: --ros-domain-id 0, --dds fast, --iface lo (cyclone only), --gui off, --gpu off"
}

need_arg() {
  local opt="$1"
  local val="$2"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "ERROR: $opt requires an argument"
    usage
    exit 1
  fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ros-domain-id) need_arg "$1" "${2:-}"; ros_domain_id="$2"; shift 2 ;;
    --dds)           need_arg "$1" "${2:-}"; dds="$2";           shift 2 ;;
    --iface)         need_arg "$1" "${2:-}"; iface="$2";        shift 2 ;;
    --gui)           gui=1; shift ;;
    --gpu)           gpu=1; shift ;; # Added GPU flag
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Validate DDS choice + set RMW
case "$dds" in
  fast)    RMW="rmw_fastrtps_cpp" ;;
  cyclone) RMW="rmw_cyclonedds_cpp" ;;
  *) echo "ERROR: invalid --dds value: '$dds' (use 'fast' or 'cyclone')"; exit 1 ;;
esac


# Extra docker args
EXTRA_VOL_ARGS=()
EXTRA_ENV_ARGS=( --env="RMW_IMPLEMENTATION=${RMW}" )
GUI_ARGS=()
GPU_ARGS=() # Initialize GPU args array

# GPU Configuration
if [[ "$gpu" -eq 1 ]]; then
  GPU_ARGS+=( --gpus all )
fi

# GUI/X11 only if --gui
if [[ "$gui" -eq 1 ]]; then
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null
  else
    echo "WARNING: xhost not found. Install: sudo apt-get install -y x11-xserver-utils"
  fi

  GUI_ARGS+=(
    --env="DISPLAY=$DISPLAY"
    --env="QT_X11_NO_MITSHM=1"
    --env="QT_QPA_PLATFORM=xcb"
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
  )
fi

# Only for cyclone: generate cyclonedds.xml + mount config + set CYCLONEDDS_URI
if [[ "$dds" == "cyclone" ]]; then
  mkdir -p "$HOST_CFG_DIR"

  cat > "$HOST_CFG_DIR/cyclonedds.xml" <<EOF
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="false" name="${iface}" priority="default" multicast="default"/>
      </Interfaces>
      <AllowMulticast>default</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Internal>
      <SocketReceiveBufferSize min="10MB"/>
      <Watermarks>
        <WhcHigh>500kB</WhcHigh>
      </Watermarks>
    </Internal>
  </Domain>
</CycloneDDS>
EOF

  EXTRA_VOL_ARGS+=( --volume="${HOST_CFG_DIR}:${CONT_CFG_DIR}:ro" )
  EXTRA_ENV_ARGS+=( --env="CYCLONEDDS_URI=file://${CONT_CFG_DIR}/cyclonedds.xml" )
fi

echo "Running container with:"
echo "  image:          $container_image"
echo "  name:           $container_name"
echo "  ROS_DOMAIN_ID:  $ros_domain_id"
echo "  DDS:            $dds"
[[ "$dds" == "cyclone" ]] && echo "  iface:          $iface"
echo "  GUI:            $([[ "$gui" -eq 1 ]] && echo on || echo off)"
echo "  GPU:            $([[ "$gpu" -eq 1 ]] && echo on || echo off)"

CONTAINER_CMD='/bin/bash'

docker run \
  -it \
  --rm \
  --name "${container_name}" \
  --env="ROS_DOMAIN_ID=${ros_domain_id}" \
  --net host \
  --privileged \
  "${GPU_ARGS[@]}" \
  "${GUI_ARGS[@]}" \
  "${EXTRA_VOL_ARGS[@]}" \
  "${EXTRA_ENV_ARGS[@]}" \
  "${container_image}" \
  "${CONTAINER_CMD}"
