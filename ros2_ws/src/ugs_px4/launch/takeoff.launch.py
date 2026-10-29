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
The week 6 assignment: the takeoff action server, and a client that sends it.

    ros2 launch ugs_px4 takeoff.launch.py                 # against PX4 (SITL or real)
    ros2 launch ugs_px4 takeoff.launch.py fake:=true      # against ugs_px4's fake PX4
    ros2 launch ugs_px4 takeoff.launch.py altitude:=10 method:=takeoff_mode land:=false

PX4 itself, and the Micro XRCE-DDS Agent between it and ROS 2, are started by
sitl.launch.py (or by hand, following the PX4 ROS 2 guide). The launch ends
when the client exits; its exit code says whether the flight went as asked.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

ARGS = {
    'altitude': ('5.0', 'metres above the takeoff point'),
    'method': ('offboard', 'offboard or takeoff_mode'),
    'climb_rate': ('0.0', "m/s; 0 uses the server's default (1 m/s)"),
    'land': ('true', 'land after the takeoff'),
    'hover': ('3.0', 'seconds to hover before landing'),
    'cancel_after': ('0.0', 'cancel the takeoff this many seconds after lift off (0: never)'),
    'fake': ('false', 'fly ugs_px4 fake_px4 instead of a real PX4'),
    'faults': ('', 'faults for fake_px4: no_preflight, deny_arm, no_offboard, takeover_at=T'),
    'record': ('', 'write the flight to this JSON lines file'),
    'server_wait': ('30.0', 'seconds the client waits for the server'),
    'max_lead': ('0.4', 'metres the climb reference may lead the vehicle (0: a fixed clock)'),
}


def generate_launch_description():
    c = {name: LaunchConfiguration(name) for name in ARGS}
    client_args = [c['altitude'], '--method', c['method'], '--climb-rate', c['climb_rate'],
                   '--cancel-after', c['cancel_after'], '--hover', c['hover'],
                   '--wait', c['server_wait']]
    client = Node(package='ugs_px4', executable='takeoff_client', output='screen',
                  arguments=client_args, condition=UnlessCondition(c['land']))
    client_land = Node(package='ugs_px4', executable='takeoff_client', output='screen',
                       arguments=client_args + ['--land'], condition=IfCondition(c['land']))
    actions = [DeclareLaunchArgument(name, default_value=value, description=text)
               for name, (value, text) in ARGS.items()]
    actions += [
        Node(package='ugs_px4', executable='fake_px4', output='screen',
             parameters=[{'faults': c['faults']}], condition=IfCondition(c['fake'])),
        Node(package='ugs_px4', executable='flight_recorder', output='screen',
             parameters=[{'path': c['record']}],
             condition=IfCondition(PythonExpression(["'", c['record'], "' != ''"]))),
        Node(package='ugs_px4', executable='takeoff_server', output='screen',
             parameters=[{'max_lead': c['max_lead']}]),
        client, client_land,
    ]
    for node in (client, client_land):  # the flight is over when the client exits
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=node, on_exit=[EmitEvent(event=Shutdown(reason='client done'))])))
    return LaunchDescription(actions)
