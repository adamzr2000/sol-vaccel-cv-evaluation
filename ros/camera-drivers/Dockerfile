ARG ROS_DISTRO=melodic
FROM ros:${ROS_DISTRO}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# System deps 
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    cmake \
    build-essential \
    pkg-config \
    libgflags-dev \
    libgoogle-glog-dev \
    libusb-1.0-0-dev \
    libeigen3-dev \
    ros-${ROS_DISTRO}-backward-ros \
    ros-${ROS_DISTRO}-image-geometry \
    ros-${ROS_DISTRO}-camera-info-manager\
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-publisher \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-ros \
 && rm -rf /var/lib/apt/lists/*

# Create a new user with sudo privileges and set a password
RUN useradd -m astra && \
    echo "astra:astra" | chpasswd && adduser astra sudo

# Build and install libuvc
WORKDIR /opt
RUN git clone https://github.com/libuvc/libuvc.git && \
    cd libuvc && mkdir -p build && cd build && \
    cmake .. && make -j"$(nproc)" && make install && ldconfig

# Catkin workspace
USER astra
WORKDIR /home/astra

RUN mkdir -p /home/astra/catkin_ws/src && \
    cd /home/astra/catkin_ws/src && \
    git clone https://github.com/orbbec/ros_astra_camera && \
    cd /home/astra/catkin_ws && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    catkin_make

RUN echo "source /home/astra/catkin_ws/devel/setup.bash" >> /home/astra/.bashrc

USER root
COPY rules/* /etc/udev/rules.d/
RUN chmod 0644 /etc/udev/rules.d/56-orbbec-usb.rules
USER astra

CMD ["/bin/bash","-lc","source /opt/ros/${ROS_DISTRO}/setup.bash && source /home/astra/catkin_ws/devel/setup.bash && roslaunch astra_camera astra.launch --wait"]