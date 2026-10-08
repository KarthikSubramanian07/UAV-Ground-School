# The ground school's ROS 2 Jazzy environment: ROS 2 does not run natively on
# macOS or Windows, so everything in ros2_ws/ builds and runs in this image.
#
#   docker build -t ugs-ros:jazzy ros2_ws
#   docker run --rm -it -v "$PWD":/ugs ugs-ros:jazzy
#
# or simply: python -m week05 docker shell
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-example-interfaces \
        ros-jazzy-vision-msgs \
        ros-jazzy-rmw-cyclonedds-cpp \
        python3-numpy \
        python3-pytest \
        tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Every shell sources ROS and, once it is built, the workspace.
RUN printf '%s\n' \
        'source /opt/ros/jazzy/setup.bash' \
        '[ -f /ugs/ros2_ws/install/setup.bash ] && source /ugs/ros2_ws/install/setup.bash' \
        >> /root/.bashrc

WORKDIR /ugs/ros2_ws
CMD ["bash"]
