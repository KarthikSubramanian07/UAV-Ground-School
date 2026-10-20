# Copyright 2026 Karthik Subramanian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Everything for week 6 in one launch: PX4 SITL in Gazebo, the agent, the action.

    ros2 launch ugs_px4 sitl.launch.py                      # take off to 5 m and land
    ros2 launch ugs_px4 sitl.launch.py altitude:=10 method:=takeoff_mode
    ros2 launch ugs_px4 sitl.launch.py gui:=true            # with the Gazebo window

Starts, in order:

1. MicroXRCEAgent udp4 -p 8888, the bridge between PX4's uXRCE-DDS client and
   ROS 2 (one agent per link, as the PX4 guide says);
2. PX4 SITL with the x500 quadcopter in Gazebo Harmonic, the same as
   ``make px4_sitl gz_x500`` but without rebuilding, from a fresh parameter set,
   and headless unless gui:=true; PX4 starts the Gazebo server itself. With no
   ground station (gcs:=false, the default) NAV_DLL_ACT is set to 0 once PX4 is
   up, or the x500 airframe refuses to arm;
3. takeoff.launch.py: takeoff_server, takeoff_client and (with record:=) the
   flight recorder. The client waits up to two minutes for the server, which
   waits for PX4's estimator and preflight checks.

The launch ends when the client does; Gazebo is stopped with it. flight:=false
starts only PX4, Gazebo and the agent, for flying them some other way (week06's
pure Python XRCE client, QGroundControl, or ros2 run by hand).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression

PASSED_ON = ('altitude', 'method', 'climb_rate', 'land', 'hover', 'cancel_after', 'record',
             'max_lead')


def generate_launch_description():
    px4_dir = EnvironmentVariable('PX4_DIR', default_value='/opt/PX4-Autopilot')
    gui = LaunchConfiguration('gui')
    rootfs = [px4_dir, '/build/px4_sitl_default/rootfs']
    agent = ExecuteProcess(cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'], output='screen',
                           name='agent',
                           condition=UnlessCondition(LaunchConfiguration('no_agent')))
    # parameters*.bson hold the last run's parameters: remove them for a clean start
    px4 = ExecuteProcess(
        cmd=['bash', '-c', 'rm -f parameters*.bson dataman && exec ../bin/px4 -d'],
        cwd=rootfs, output='screen', name='px4',
        additional_env={
            'PX4_SIM_MODEL': LaunchConfiguration('model'),
            'PX4_GZ_WORLD': LaunchConfiguration('world'),
            'HEADLESS': PythonExpression(["'' if '", gui, "' == 'true' else '1'"]),
            'GZ_IP': '127.0.0.1',  # keep gz-transport discovery on loopback, as make does
        })
    # The x500 airframe sets NAV_DLL_ACT 2 (return on ground station loss), which
    # makes PX4 refuse to arm with no ground station connected. PX4_PARAM_* cannot
    # undo that: setting a parameter to its firmware default (0) leaves it marked
    # default, and the airframe's set-default 2 wins. So wait for the airframe to
    # load, then set it through the running PX4.
    no_gcs = ExecuteProcess(
        cmd=['bash', '-c',
             'for i in $(seq 240); do '
             'if ../bin/px4-param compare NAV_DLL_ACT 2 >/dev/null 2>&1; then '
             'exec ../bin/px4-param set NAV_DLL_ACT 0; fi; sleep 0.5; done; '
             'echo "NAV_DLL_ACT was never 2; left as is"'],
        cwd=rootfs, output='screen', name='no_gcs',
        condition=UnlessCondition(LaunchConfiguration('gcs')))
    flight = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ugs_px4'), 'launch', 'takeoff.launch.py')),
        launch_arguments=[(name, LaunchConfiguration(name)) for name in PASSED_ON]
        + [('server_wait', '120.0')], condition=IfCondition(LaunchConfiguration('flight')))
    stop_gazebo = RegisterEventHandler(OnShutdown(on_shutdown=[
        ExecuteProcess(cmd=['pkill', '-f', 'gz sim'], output='log')]))
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='gz_x500',
                              description='PX4 airframe and Gazebo model'),
        DeclareLaunchArgument('world', default_value='default', description='Gazebo world'),
        DeclareLaunchArgument('gui', default_value='false', description='open the Gazebo GUI'),
        DeclareLaunchArgument('no_agent', default_value='false',
                              description='an agent is already running'),
        DeclareLaunchArgument('gcs', default_value='false',
                              description='true: a ground station will connect, keep NAV_DLL_ACT'),
        DeclareLaunchArgument('flight', default_value='true',
                              description='false: only PX4, Gazebo and the agent'),
        DeclareLaunchArgument('altitude', default_value='5.0'),
        DeclareLaunchArgument('method', default_value='offboard'),
        DeclareLaunchArgument('climb_rate', default_value='0.0'),
        DeclareLaunchArgument('land', default_value='true'),
        DeclareLaunchArgument('hover', default_value='5.0'),
        DeclareLaunchArgument('cancel_after', default_value='0.0'),
        DeclareLaunchArgument('record', default_value=''),
        DeclareLaunchArgument('max_lead', default_value='0.4'),
        agent, px4, no_gcs, flight, stop_gazebo,
    ])
