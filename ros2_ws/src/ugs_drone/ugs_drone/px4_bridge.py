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
px4_bridge: the flight controller side of the week 5 drone.

On a real drone this node would translate between ROS 2 and PX4 (or ArduPilot)
over the uXRCE-DDS bridge or MAVLink. Here it wraps a simulated vehicle and
offers the interfaces from the slides:

* topics: /odometry (50 Hz, best effort like PX4's /fmu/out topics),
  /battery (1 Hz) and /mission (published once, transient local);
* services: /arm (std_srvs/SetBool), /set_mode, /reset_odometry, /get_mission;
* action: /fly_to (ugs_interfaces/action/FlyToWaypoint), with feedback and cancel.

The action executes in a reentrant callback group on a multi threaded executor,
so the 50 Hz physics timer keeps running while a goal is in progress.
"""

import time

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState
from std_srvs.srv import SetBool, Trigger
from ugs_interfaces.action import FlyToWaypoint
from ugs_interfaces.msg import Mission
from ugs_interfaces.srv import GetMission, SetMode

from .sim import Vehicle

MODES = ('STABILIZE', 'GUIDED', 'LOITER', 'LAND', 'RTL')
ARRIVED_M = 0.3  # within this distance
SETTLED_MPS = 0.3  # and slower than this: arrived, not passing through
MISSION_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Px4Bridge(Node):
    """A simulated flight controller behind the slides' topics, services and action."""

    def __init__(self):
        super().__init__('px4_bridge')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('max_speed', 4.0)
        self.vehicle = Vehicle(max_speed=self.get_parameter('max_speed').value)
        self.origin = np.zeros(3)
        self.physics = MutuallyExclusiveCallbackGroup()
        self.odom_pub = self.create_publisher(Odometry, 'odometry', qos_profile_sensor_data)
        self.battery_pub = self.create_publisher(BatteryState, 'battery', 10)
        self.mission_pub = self.create_publisher(Mission, 'mission', MISSION_QOS)
        self.mission = Mission(
            waypoints=[Point(x=0.0, y=0.0, z=5.0), Point(x=12.0, y=6.0, z=5.0),
                       Point(x=12.0, y=-6.0, z=5.0)],
            speeds=[2.0, 4.0, 4.0])
        self.mission_pub.publish(self.mission)  # once: late joiners still get it
        period = 1.0 / self.get_parameter('rate_hz').value
        self.last = time.monotonic()
        self.create_timer(period, self.tick, callback_group=self.physics)
        self.create_timer(1.0, self.publish_battery, callback_group=self.physics)
        self.create_service(SetBool, 'arm', self.on_arm)
        self.create_service(SetMode, 'set_mode', self.on_set_mode)
        self.create_service(Trigger, 'reset_odometry', self.on_reset_odometry)
        self.create_service(GetMission, 'get_mission', self.on_get_mission)
        self.fly_to = ActionServer(
            self, FlyToWaypoint, 'fly_to', self.execute,
            goal_callback=self.on_goal, cancel_callback=self.on_cancel,
            callback_group=ReentrantCallbackGroup())
        self.get_logger().info('px4_bridge ready: disarmed, STABILIZE')

    # ---------------------------------------------------------------- topics

    def tick(self):
        if not rclpy.ok():  # shutting down: the context is already gone
            return
        now = time.monotonic()
        self.vehicle.step(min(now - self.last, 0.1))
        self.last = now
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        p = self.vehicle.position - self.origin
        msg.pose.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        msg.pose.pose.orientation.w = 1.0
        v = self.vehicle.velocity
        linear = msg.twist.twist.linear
        linear.x, linear.y, linear.z = (float(c) for c in v)
        self.odom_pub.publish(msg)

    def publish_battery(self):
        if not rclpy.ok():
            return
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = float(self.vehicle.battery)
        msg.voltage = float(13.2 + 3.6 * self.vehicle.battery)  # a 4S LiPo, roughly
        msg.present = True
        self.battery_pub.publish(msg)

    # -------------------------------------------------------------- services

    def on_arm(self, request, response):
        if request.data:
            response.success = True
            response.message = 'armed' if not self.vehicle.armed else 'already armed'
            self.vehicle.armed = True
        elif self.vehicle.in_air:
            response.success = False
            response.message = 'refused: in the air, land first'
        else:
            response.success = True
            response.message = 'disarmed'
            self.vehicle.armed = False
        self.get_logger().info(f'/arm {request.data}: {response.message}')
        return response

    def on_set_mode(self, request, response):
        mode = request.mode.upper()
        response.accepted = mode in MODES
        if response.accepted:
            self.vehicle.mode = mode
            if mode in ('LOITER', 'RTL'):
                if mode == 'RTL':
                    self.vehicle.setpoint = self.origin + np.array([0.0, 0.0, 5.0])
                else:
                    self.vehicle.setpoint = self.vehicle.position.copy()
            response.message = f'mode {mode}'
        else:
            response.message = f'unknown mode {request.mode!r}; try one of {", ".join(MODES)}'
        self.get_logger().info(f'/set_mode {request.mode}: {response.message}')
        return response

    def on_reset_odometry(self, request, response):
        self.origin = self.vehicle.position.copy()
        self.origin[2] = 0.0
        response.success = True
        response.message = 'odometry origin moved to the current position'
        return response

    def on_get_mission(self, request, response):
        response.mission = self.mission
        return response

    # ---------------------------------------------------------------- action

    def on_goal(self, goal_request):
        if not self.vehicle.armed or self.vehicle.mode != 'GUIDED':
            self.get_logger().warn('/fly_to rejected: arm and switch to GUIDED first')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, goal_handle):
        self.get_logger().info('/fly_to cancel requested')
        return CancelResponse.ACCEPT

    def execute(self, goal_handle):
        t = goal_handle.request.target
        target = self.origin + np.array([t.x, t.y, t.z])
        self.get_logger().info(f'/fly_to ({t.x:.1f}, {t.y:.1f}, {t.z:.1f}) m')
        self.vehicle.setpoint = target
        feedback = FlyToWaypoint.Feedback()
        result = FlyToWaypoint.Result()
        while rclpy.ok():
            distance = self.vehicle.distance_to(target)
            if goal_handle.is_cancel_requested:
                self.vehicle.setpoint = self.vehicle.position.copy()  # hold here
                goal_handle.canceled()
                self.get_logger().info(f'/fly_to canceled with {distance:.1f} m to go')
                result.success = False
                return result
            if not self.vehicle.armed or self.vehicle.mode != 'GUIDED':
                goal_handle.abort()
                result.success = False
                return result
            feedback.distance_remaining = float(distance)
            goal_handle.publish_feedback(feedback)
            speed = np.linalg.norm(self.vehicle.velocity)
            if distance < ARRIVED_M and speed < SETTLED_MPS:
                break
            time.sleep(0.1)
        goal_handle.succeed()
        result.success = True
        self.get_logger().info('/fly_to succeeded')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = Px4Bridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
