ARG ROS_DISTRO=noetic
FROM ros:${ROS_DISTRO}

ENV DEBIAN_FRONTEND=noninteractive
SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-teleop-twist-keyboard \
      ros-${ROS_DISTRO}-teleop-twist-joy \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["/bin/bash","-lc","source /opt/ros/${ROS_DISTRO}/setup.bash && roscore"]