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
The code on the week 5 slides, runnable.

* ``altitude_pub``: "A minimal publisher in rclpy" (Float32 on /altitude at 10 Hz)
* ``altitude_sub``: its mirror image, ``on_alt(self, msg)`` once per message
* ``battery_monitor``: "Callbacks: ROS calls you" (warns below 20 percent)
* ``arm_server`` and ``arm_client``: "Service server and client in rclpy",
  with ``call_async`` and a done callback rather than a blocking ``call``
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32
from std_srvs.srv import SetBool


class AltitudePub(Node):

    def __init__(self):
        super().__init__('altitude_pub')
        self.pub = self.create_publisher(Float32, 'altitude', 10)
        self.timer = self.create_timer(0.1, self.tick)

    def tick(self):
        self.pub.publish(Float32(data=12.5))


class AltitudeSub(Node):

    def __init__(self):
        super().__init__('altitude_sub')
        self.create_subscription(Float32, 'altitude', self.on_alt, 10)

    def on_alt(self, msg):
        self.get_logger().info(f'altitude {msg.data:.1f} m')


class BatteryMonitor(Node):

    def __init__(self):
        super().__init__('battery_monitor')
        # 2. hand it to ROS (no parentheses!)
        self.create_subscription(BatteryState, '/battery', self.on_battery, 10)

    # 1. define the callback
    def on_battery(self, msg):
        if msg.percentage < 0.2:
            self.get_logger().warn('Battery low!')
        else:
            self.get_logger().info(f'battery {100 * msg.percentage:.0f} percent')


class ArmServer(Node):

    def __init__(self):
        super().__init__('arm_server')
        self.armed = False
        self.srv = self.create_service(SetBool, 'arm', self.on_arm)

    def on_arm(self, request, response):
        self.armed = request.data
        response.success = True
        response.message = f'armed={self.armed}'
        return response


class ArmClient(Node):

    def __init__(self):
        super().__init__('arm_client')
        self.cli = self.create_client(SetBool, 'arm')
        self.cli.wait_for_service()
        future = self.cli.call_async(SetBool.Request(data=True))
        future.add_done_callback(self.on_reply)
        self.done = False

    def on_reply(self, future):
        reply = future.result()
        self.get_logger().info(f'success={reply.success} message={reply.message!r}')
        self.done = True


def _run(node_class):
    rclpy.init()
    node = node_class()
    try:
        if isinstance(node, ArmClient):
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def altitude_pub():
    _run(AltitudePub)


def altitude_sub():
    _run(AltitudeSub)


def battery_monitor():
    _run(BatteryMonitor)


def arm_server():
    _run(ArmServer)


def arm_client():
    _run(ArmClient)
