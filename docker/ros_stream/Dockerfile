FROM ros:indigo

RUN apt-get update && \
    apt-get install -y \
      ros-${ROS_DISTRO}-usb-cam \
      ros-${ROS_DISTRO}-web-video-server && \
    rm -rf /var/lib/apt/lists/*

ENV WORKSPACE /root/ros_stream
ADD . $WORKSPACE
WORKDIR $WORKSPACE
