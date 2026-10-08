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
planner: flies the slides' delivery mission using all three ways nodes talk.

1. Waits for px4_bridge, asks /set_mode for GUIDED and /arm for true (services).
2. Takes off with a /fly_to goal, then flies the legs from /get_mission
   (action: goal, feedback with the distance left, result).
3. While flying it watches /detections (a topic). Three confirmed sightings of
   the target cancel the current leg, and a new goal puts the drone over it.
4. Lands with /set_mode LAND, waits on /odometry until it touches down, disarms.

Every call is asynchronous (call_async and send_goal_async with done
callbacks), so no callback ever blocks the single threaded executor: the
deadlock the slides warn about cannot happen here. Setting ``cancel_after_s``
cancels the first leg partway, the slides' "an obstacle appears" case.
"""

from bisect import bisect_left
from collections import deque

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import SetBool
from ugs_interfaces.action import FlyToWaypoint
from ugs_interfaces.srv import GetMission, SetMode
from vision_msgs.msg import Detection2DArray

from .sim import pixel_to_ground

CRUISE_Z = 5.0  # metres: the camera sees 8 m by 6 m of ground from here
CONFIRMATIONS = 3


class Planner(Node):
    """Event driven mission logic: every step is started from a callback."""

    def __init__(self):
        super().__init__('planner')
        self.declare_parameter('cancel_after_s', 0.0)
        self.set_mode = self.create_client(SetMode, 'set_mode')
        self.arm = self.create_client(SetBool, 'arm')
        self.get_mission = self.create_client(GetMission, 'get_mission')
        self.fly_to = ActionClient(self, FlyToWaypoint, 'fly_to')
        # px4_bridge publishes odometry best effort, like PX4: a default (reliable)
        # subscription would be incompatible and receive nothing.
        self.create_subscription(Odometry, 'odometry', self.on_odometry, qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, 'detections', self.on_detections, 10)
        self.position = None
        self.history = deque(maxlen=200)  # (stamp, position): 4 s of odometry
        self.state = 'waiting'
        self.legs = []
        self.leg_number = 0
        self.goal_handle = None
        self.sightings = []
        self.target = None
        self.last_feedback_log = 0.0
        self.outcome = None
        self.done = False
        self.cancel_timer = None
        self.create_timer(0.2, self.tick)

    # ------------------------------------------------------------- helpers

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def go(self, state):
        self.get_logger().info(f'[{state}]')
        self.state = state

    def call(self, client, request, then):
        client.call_async(request).add_done_callback(lambda future: then(future.result()))

    def send_goal(self, point, label):
        goal = FlyToWaypoint.Goal(target=Point(x=point[0], y=point[1], z=point[2]))
        self.get_logger().info(f'goal {label}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) m')
        future = self.fly_to.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(lambda f: self.on_goal_response(f.result(), label))

    # ------------------------------------------------------------ the flow

    def tick(self):
        ready = (self.set_mode.service_is_ready() and self.arm.service_is_ready() and
                 self.get_mission.service_is_ready() and self.fly_to.server_is_ready())
        if self.state == 'waiting' and ready:
            self.go('guided')
            self.call(self.set_mode, SetMode.Request(mode='GUIDED'), self.on_guided)
        elif self.state == 'landing' and self.position is not None and self.position[2] < 0.05:
            self.go('disarming')
            self.call(self.arm, SetBool.Request(data=False), self.on_disarmed)

    def on_guided(self, response):
        if not response.accepted:
            return self.finish(f'mode change refused: {response.message}')
        self.go('arming')
        self.call(self.arm, SetBool.Request(data=True), self.on_armed)

    def on_armed(self, response):
        if not response.success:
            return self.finish(f'arming refused: {response.message}')
        self.call(self.get_mission, GetMission.Request(), self.on_mission)

    def on_mission(self, response):
        self.legs = [(p.x, p.y, p.z) for p in response.mission.waypoints]
        self.get_logger().info(f'mission: {len(self.legs)} waypoints')
        self.go('takeoff')
        self.send_goal(self.legs.pop(0), 'takeoff')  # the first waypoint is the climb

    def on_goal_response(self, handle, label):
        if not handle.accepted:
            return self.finish(f'goal {label} rejected')
        self.goal_handle = handle
        cancel_after = self.get_parameter('cancel_after_s').value
        if label == 'leg 1' and cancel_after > 0.0:
            self.cancel_timer = self.create_timer(cancel_after, self.obstacle)
        handle.get_result_async().add_done_callback(
            lambda f: self.on_result(f.result(), label))

    def on_feedback(self, feedback_msg):
        now = self.now()
        if now - self.last_feedback_log >= 1.0:
            self.last_feedback_log = now
            d = feedback_msg.feedback.distance_remaining
            self.get_logger().info(f'feedback: {d:.1f} m to go')

    def on_result(self, wrapped, label):
        status = {4: 'succeeded', 5: 'canceled', 6: 'aborted'}.get(wrapped.status, 'unknown')
        self.get_logger().info(f'result {label}: {status}')
        self.goal_handle = None
        if self.state == 'obstacle':
            return self.land('canceled')
        if self.state == 'retarget':
            self.go('to target')
            return self.send_goal((self.target[0], self.target[1], CRUISE_Z), 'target')
        if status != 'succeeded':
            return self.land(status)
        if self.state == 'to target':
            return self.land('succeeded')
        if self.legs:
            self.go('survey')
            self.leg_number += 1
            return self.send_goal(self.legs.pop(0), f'leg {self.leg_number}')
        return self.land('target not found')

    def obstacle(self):
        self.cancel_timer.cancel()
        if self.goal_handle is not None and self.state == 'survey':
            self.get_logger().warn('obstacle ahead: canceling the goal')
            self.go('obstacle')
            self.goal_handle.cancel_goal_async()

    def on_detections(self, msg):
        if self.state != 'survey' or self.position is None or not msg.detections:
            return
        det = msg.detections[0]
        if det.results and det.results[0].hypothesis.score < 0.5:
            return
        # Use where the drone was when the frame was taken, not where it is now:
        # at 4 m/s, 100 ms of pipeline latency is 0.4 m of error.
        pose = self.pose_at(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        c = det.bbox.center.position
        self.sightings.append(pixel_to_ground(c.x, c.y, pose))
        if len(self.sightings) >= CONFIRMATIONS:
            xs = [s[0] for s in self.sightings[-CONFIRMATIONS:]]
            ys = [s[1] for s in self.sightings[-CONFIRMATIONS:]]
            self.target = (sum(xs) / len(xs), sum(ys) / len(ys))
            self.get_logger().info(
                f'target confirmed at ({self.target[0]:.2f}, {self.target[1]:.2f}) m')
            self.go('retarget')
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()

    def on_odometry(self, msg):
        p = msg.pose.pose.position
        self.position = (p.x, p.y, p.z)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.history.append((stamp, self.position))

    def pose_at(self, stamp):
        """Return the odometry sample closest in time to stamp."""
        if not self.history:
            return self.position
        stamps = [s for s, _ in self.history]
        i = min(bisect_left(stamps, stamp), len(stamps) - 1)
        if i > 0 and abs(stamps[i - 1] - stamp) < abs(stamps[i] - stamp):
            i -= 1
        return self.history[i][1]

    def land(self, outcome):
        self.outcome = outcome
        self.go('landing')
        self.call(self.set_mode, SetMode.Request(mode='LAND'), lambda r: None)

    def on_disarmed(self, response):
        if not response.success:
            self.go('landing')  # still descending; tick() will try again
            return
        self.finish(self.outcome)

    def finish(self, outcome):
        self.outcome = outcome
        where = ''
        if self.position is not None:
            where = ' at ({:.2f}, {:.2f}, {:.2f})'.format(*self.position)
        self.get_logger().info(f'MISSION COMPLETE: {outcome}{where}')
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
