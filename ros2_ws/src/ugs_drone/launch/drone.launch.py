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
The slides' drone, wired up: camera_driver, detector, planner and px4_bridge.

Run it with ``ros2 launch ugs_drone drone.launch.py``; add ``cancel_after_s:=2.0``
to make an obstacle appear and cancel the first survey leg.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cancel_after = LaunchConfiguration('cancel_after_s')
    target_x = LaunchConfiguration('target_x')
    target_y = LaunchConfiguration('target_y')
    planner = Node(package='ugs_drone', executable='planner', output='screen',
                   parameters=[{'cancel_after_s': cancel_after}])
    return LaunchDescription([
        DeclareLaunchArgument('cancel_after_s', default_value='0.0',
                              description='cancel the first survey leg after this many seconds'),
        DeclareLaunchArgument('target_x', default_value='11.0'),
        DeclareLaunchArgument('target_y', default_value='7.0'),
        Node(package='ugs_drone', executable='px4_bridge', output='screen'),
        Node(package='ugs_drone', executable='camera_driver', output='screen',
             parameters=[{'target_x': target_x, 'target_y': target_y}]),
        Node(package='ugs_drone', executable='detector', output='screen'),
        planner,
        # the mission is over when the planner exits
        RegisterEventHandler(OnProcessExit(target_action=planner,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
