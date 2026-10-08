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

"""detector: finds the orange target in /camera/image and publishes /detections."""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .sim import detect


class Detector(Node):
    """One detection message per frame, empty when nothing is in view."""

    def __init__(self):
        super().__init__('detector')
        self.pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.create_subscription(Image, 'camera/image', self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        out = Detection2DArray(header=msg.header)
        found = detect(frame)
        if found is not None:
            u, v, width, height, score = found
            det = Detection2D(header=msg.header, id='target')
            det.bbox.center.position.x = u
            det.bbox.center.position.y = v
            det.bbox.size_x = width
            det.bbox.size_y = height
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'target'
            hypothesis.hypothesis.score = score
            det.results.append(hypothesis)
            out.detections.append(det)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
