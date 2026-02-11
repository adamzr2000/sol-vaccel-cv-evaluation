ARG ROS_DISTRO=noetic
FROM ros:${ROS_DISTRO}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-sensor-msgs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/ /app/src/
RUN chmod +x /app/src/webcam_pub.py

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["/bin/bash", "-lc", "source /opt/ros/noetic/setup.bash && exec python3 /app/src/webcam_pub.py"]