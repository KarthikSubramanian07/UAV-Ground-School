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
camera_driver: publishes the drone's nadir camera on /camera/image.

Frames are rendered from the drone's pose (subscribed on /odometry) and an
orange target on the ground. Both topics use the sensor data profile, best
effort: a late frame is useless anyway.
"""

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .sim import IMAGE_HEIGHT, IMAGE_WIDTH, render


class CameraDriver(Node):
    """Renders and publishes camera frames at a fixed rate."""

    def __init__(self):
        super().__init__('camera_driver')
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('target_x', 14.0)
        self.declare_parameter('target_y', 9.0)
        self.position = np.array([0.0, 0.0, 0.0])
        self.rng = np.random.default_rng(5)
        self.frames = 0
        self.pub = self.create_publisher(Image, 'camera/image', qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odometry', self.on_odometry, qos_profile_sensor_data)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.tick)

    def on_odometry(self, msg):
        p = msg.pose.pose.position
        self.position = np.array([p.x, p.y, p.z])

    def tick(self):
        if not rclpy.ok():  # shutting down: the context is already gone
            return
        target = (self.get_parameter('target_x').value, self.get_parameter('target_y').value)
        frame = render(target, self.position, self.rng)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height, msg.width = IMAGE_HEIGHT, IMAGE_WIDTH
        msg.encoding = 'rgb8'
        msg.step = IMAGE_WIDTH * 3
        msg.data = frame.tobytes()
        self.pub.publish(msg)
        self.frames += 1


def main(args=None):
    rclpy.init(args=args)
    node = CameraDriver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
