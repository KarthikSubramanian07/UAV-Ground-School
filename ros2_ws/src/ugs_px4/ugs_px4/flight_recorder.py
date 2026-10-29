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
flight_recorder: write a flight to JSON lines, for week06's analysis and site.

Records both directions of the PX4 link (/fmu/out telemetry, /fmu/in setpoints
and commands), the action feedback and status, and every log line, each
stamped with seconds since the recorder started (the first line gives that
start on CLOCK_MONOTONIC). ros2 bag would do the same,
but JSON lines need no ROS to read.
"""

import json
import math
import time

from action_msgs.msg import GoalStatusArray
from px4_msgs.msg import (TrajectorySetpoint, VehicleCommand, VehicleCommandAck,
                          VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
from rcl_interfaces.msg import Log
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from ugs_interfaces.action import Land, Takeoff


def _num(x):
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


class FlightRecorder(Node):

    def __init__(self):
        super().__init__('flight_recorder')
        self.declare_parameter('path', 'flight.jsonl')
        self.file = open(self.get_parameter('path').value, 'w')
        self.t0 = time.monotonic()
        sensor = qos_profile_sensor_data
        for name in ('vehicle_status', 'vehicle_status_v1'):
            self.create_subscription(VehicleStatus, f'/fmu/out/{name}', self.on_status, sensor)
        for name in ('vehicle_local_position', 'vehicle_local_position_v1'):
            self.create_subscription(VehicleLocalPosition, f'/fmu/out/{name}', self.on_position,
                                     sensor)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected',
                                 self.on_land, sensor)
        self.create_subscription(VehicleCommandAck, '/fmu/out/vehicle_command_ack',
                                 self.on_ack, sensor)
        self.create_subscription(TrajectorySetpoint, '/fmu/in/trajectory_setpoint',
                                 self.on_setpoint, 10)
        self.create_subscription(VehicleCommand, '/fmu/in/vehicle_command', self.on_command, 10)
        status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action, kind in (('takeoff', Takeoff), ('land', Land)):
            self.create_subscription(
                kind.Impl.FeedbackMessage, f'/{action}/_action/feedback',
                lambda m, a=action: self.on_feedback(a, m), 10)
            self.create_subscription(
                GoalStatusArray, f'/{action}/_action/status',
                lambda m, a=action: self.on_goal_status(a, m), status_qos)
        self.create_subscription(Log, '/rosout', self.on_log, 100)
        # CLOCK_MONOTONIC is system wide: other recorders (Gazebo's true pose) line up with it
        self.write('start', monotonic=self.t0)
        self.get_logger().info(f'recording to {self.get_parameter("path").value}')

    def write(self, kind, **fields):
        fields = {'t': round(time.monotonic() - self.t0, 4), 'kind': kind, **fields}
        self.file.write(json.dumps(fields) + '\n')

    def on_status(self, m):
        self.write('status', px4_us=m.timestamp, nav_state=m.nav_state,
                   arming_state=m.arming_state, preflight_ok=m.pre_flight_checks_pass,
                   failsafe=m.failsafe)

    def on_position(self, m):
        self.write('position', px4_us=m.timestamp, x=_num(m.x), y=_num(m.y), z=_num(m.z),
                   vx=_num(m.vx), vy=_num(m.vy), vz=_num(m.vz), xy_valid=m.xy_valid,
                   z_valid=m.z_valid, ref_alt=_num(m.ref_alt), heading=_num(m.heading))

    def on_land(self, m):
        self.write('land_detected', landed=m.landed, maybe_landed=m.maybe_landed,
                   ground_contact=m.ground_contact)

    def on_ack(self, m):
        self.write('ack', command=m.command, result=m.result)

    def on_setpoint(self, m):
        self.write('setpoint', x=_num(m.position[0]), y=_num(m.position[1]),
                   z=_num(m.position[2]), vz=_num(m.velocity[2]))

    def on_command(self, m):
        self.write('command', command=m.command,
                   params=[_num(p) for p in (m.param1, m.param2, m.param3, m.param4,
                                             m.param5, m.param6, m.param7)])

    def on_feedback(self, action, m):
        fb = m.feedback
        self.write('feedback', action=action, phase=fb.phase, altitude=_num(fb.altitude),
                   vertical_speed=_num(fb.vertical_speed),
                   remaining=_num(getattr(fb, 'remaining', math.nan)))

    def on_goal_status(self, action, m):
        for s in m.status_list:
            self.write('goal_status', action=action, goal=bytes(s.goal_info.goal_id.uuid).hex(),
                       status=s.status)

    def on_log(self, m):
        if m.name != 'flight_recorder':
            self.write('log', node=m.name, level=m.level, msg=m.msg)
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


def main(args=None):
    rclpy.init(args=args)
    node = FlightRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
