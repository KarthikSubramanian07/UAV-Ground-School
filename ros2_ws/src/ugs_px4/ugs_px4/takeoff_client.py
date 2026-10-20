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
takeoff_client: send /takeoff, watch the feedback, then optionally /land.

    ros2 run ugs_px4 takeoff_client                  # 5 m, offboard
    ros2 run ugs_px4 takeoff_client 10 --climb-rate 2 --hover 5 --land
    ros2 run ugs_px4 takeoff_client 5 --method takeoff_mode
    ros2 run ugs_px4 takeoff_client 5 --cancel-after 4

The exit code is 0 when every goal sent succeeded (or, with --cancel-after,
was canceled as asked), so launch tests and scripts can check it.
"""

import argparse
import sys
import time

from action_msgs.msg import GoalStatus
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from ugs_interfaces.action import Land, Takeoff

STATUS = {GoalStatus.STATUS_SUCCEEDED: 'succeeded', GoalStatus.STATUS_ABORTED: 'aborted',
          GoalStatus.STATUS_CANCELED: 'canceled'}


class TakeoffClient(Node):
    """An action client for /takeoff and /land that logs what the server reports."""

    def __init__(self):
        super().__init__('takeoff_client')
        self.takeoff = ActionClient(self, Takeoff, 'takeoff')
        self.land = ActionClient(self, Land, 'land')
        self.last_phase = None
        self.last_print = 0.0
        self.climb_started = None

    def on_feedback(self, msg):
        fb = msg.feedback
        now = time.monotonic()
        # lift off: PX4 spends about 3 s after arming spooling up and ramping
        if fb.phase in ('CLIMB', 'SETTLE') and fb.altitude > 0.3 and self.climb_started is None:
            self.climb_started = now
        if fb.phase != self.last_phase or now - self.last_print >= 1.0:
            extra = f', {fb.remaining:5.2f} m to go' if hasattr(fb, 'remaining') else ''
            self.get_logger().info(f'feedback: [{fb.phase}] {fb.altitude:5.2f} m, '
                                   f'{fb.vertical_speed:+5.2f} m/s{extra}')
            self.last_phase, self.last_print = fb.phase, now

    def send(self, client, goal, cancel_after: float = 0.0, wait: float = 30.0):
        """Send a goal and wait for its result: (status, result) or (None, None)."""
        if not client.wait_for_server(timeout_sec=wait):
            self.get_logger().error(f'no {client._action_name} server after {wait:.0f} s')
            return None, None
        self.last_phase = None
        future = client.send_goal_async(goal, feedback_callback=self.on_feedback)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(f'{client._action_name} goal rejected')
            return GoalStatus.STATUS_UNKNOWN, None
        self.get_logger().info(f'{client._action_name} goal accepted')
        result_future = handle.get_result_async()
        if cancel_after > 0:  # counted from lift off, not from the goal
            self.climb_started = None
            while not result_future.done() and (
                    self.climb_started is None
                    or time.monotonic() - self.climb_started < cancel_after):
                rclpy.spin_once(self, timeout_sec=0.05)
            if not result_future.done():
                self.get_logger().info(f'canceling {cancel_after:.1f} s after lift off')
                rclpy.spin_until_future_complete(self, handle.cancel_goal_async())
        rclpy.spin_until_future_complete(self, result_future)
        response = result_future.result()
        return response.status, response.result


def parse(argv):
    parser = argparse.ArgumentParser(prog='takeoff_client', description=__doc__.split('\n')[1])
    parser.add_argument('altitude', nargs='?', type=float, default=5.0, help='metres (5)')
    parser.add_argument('--climb-rate', type=float, default=0.0,
                        help="m/s; 0 uses the server's default")
    parser.add_argument('--method', default='offboard', choices=('offboard', 'takeoff_mode'))
    parser.add_argument('--cancel-after', type=float, default=0.0,
                        help='cancel the takeoff this many seconds after lift off')
    parser.add_argument('--hover', type=float, default=0.0,
                        help='seconds to hover before landing')
    parser.add_argument('--land', action='store_true', help='land afterwards')
    parser.add_argument('--wait', type=float, default=30.0,
                        help='seconds to wait for the server')
    return parser.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    opts = parse(remove_ros_args(args if args is not None else sys.argv)[1:])
    node = TakeoffClient()
    ok = True
    try:
        goal = Takeoff.Goal(altitude=opts.altitude, climb_rate=opts.climb_rate,
                            method=opts.method)
        node.get_logger().info(f'/takeoff to {opts.altitude:.1f} m ({opts.method})')
        status, result = node.send(node.takeoff, goal, opts.cancel_after, opts.wait)
        expected = GoalStatus.STATUS_CANCELED if opts.cancel_after > 0 \
            else GoalStatus.STATUS_SUCCEEDED
        if result is not None:
            node.get_logger().info(
                f'TAKEOFF {STATUS.get(status, status)}: {result.message} '
                f'(final {result.final_altitude:.2f} m, climb {result.climb_time:.1f} s)')
        ok = status == expected
        if opts.land and status is not None:
            if opts.hover > 0:
                node.get_logger().info(f'hovering {opts.hover:.0f} s')
                end = time.monotonic() + opts.hover
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.1)
            status, result = node.send(node.land, Land.Goal(), wait=opts.wait)
            if result is not None:
                node.get_logger().info(f'LAND {STATUS.get(status, status)}: {result.message}')
            ok = ok and status == GoalStatus.STATUS_SUCCEEDED
        node.get_logger().info('FLIGHT COMPLETE' if ok else 'FLIGHT FAILED')
    except KeyboardInterrupt:
        ok = False
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
