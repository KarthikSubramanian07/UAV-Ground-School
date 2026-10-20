# Week 6: PX4 SITL with Gazebo Harmonic, the Micro XRCE-DDS Agent and px4_msgs,
# on top of the week 5 ROS 2 Jazzy image. Everything the PX4 guides install on
# Ubuntu 24.04, pinned, so the takeoff action can fly a simulated x500.
#
#   docker build -t ugs-ros:jazzy ros2_ws
#   docker build -t ugs-px4:jazzy ros2_ws/px4
#
# CI builds it on ros:jazzy-ros-base instead (--build-arg BASE=ros:jazzy-ros-base):
# the takeoff needs nothing from the week 5 image.
#
# or simply: python -m week06 docker build
#
# Gazebo runs headless (HEADLESS=1): the container has no display, and the
# x500's sensors (IMU, barometer, magnetometer, GPS) need no rendering.
ARG BASE=ugs-ros:jazzy
FROM ${BASE}

ARG PX4_VERSION=v1.17.0
ARG PX4_MSGS_VERSION=v1.17.0
ARG XRCE_AGENT_VERSION=v2.4.3
ARG JOBS=2
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo git wget lsb-release gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${PX4_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot

# PX4's own setup script, minus the NuttX cross compilers: build tools, the
# Python packages its code generators need, and Gazebo Harmonic from OSRF.
# PIP_IGNORE_INSTALLED: pip cannot upgrade Debian's numpy (no RECORD file), so
# PX4's Python packages go to /usr/local beside it instead.
RUN PIP_IGNORE_INSTALLED=1 bash /opt/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# The SITL firmware and its Gazebo bridge. CCACHE off: the image is built once.
RUN cd /opt/PX4-Autopilot \
    && CCACHE_DISABLE=1 make -j${JOBS} px4_sitl_default \
    && find build/px4_sitl_default -name '*.o' -delete

# px4_msgs matching the firmware, and the agent from the version the PX4 guide
# names for Jazzy. The agent finds ROS 2's Fast DDS instead of building its own.
RUN mkdir -p /opt/px4_ws/src && cd /opt/px4_ws/src \
    && git clone --depth 1 --branch ${PX4_MSGS_VERSION} https://github.com/PX4/px4_msgs.git \
    && git clone --depth 1 --branch ${XRCE_AGENT_VERSION} https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    && cd /opt/px4_ws \
    && . /opt/ros/jazzy/setup.sh \
    && MAKEFLAGS=-j${JOBS} colcon build --parallel-workers 1 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && rm -rf build log

# tcpdump for python -m week06 fly --capture (the week 5 image has it; ros:jazzy-ros-base,
# which CI builds on, does not)
RUN apt-get update && apt-get install -y --no-install-recommends tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Sourced in this order: ROS 2, px4_msgs and the agent, then the repo's workspace.
RUN sed -i '\#/ugs/ros2_ws/install/setup.bash#d' /root/.bashrc \
    && printf '%s\n' \
        'source /opt/px4_ws/install/setup.bash' \
        '[ -f /ugs/ros2_ws/install/setup.bash ] && source /ugs/ros2_ws/install/setup.bash' \
        >> /root/.bashrc

ENV PX4_DIR=/opt/PX4-Autopilot
WORKDIR /ugs/ros2_ws
CMD ["bash"]
