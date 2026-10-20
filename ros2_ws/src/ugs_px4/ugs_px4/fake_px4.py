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
fake_px4: PX4's uXRCE-DDS topics, served by ugs_px4.model instead of PX4.

Same names, types and QoS as PX4 v1.17 through the agent (best effort, transient
local, as the capture in docs/week06 shows), and roughly its rates: local
position at 50 Hz, status and land detection at 2 Hz. The takeoff server cannot
tell the difference at the interface; the launch tests use it in CI, where PX4
SITL and Gazebo are too heavy. The `faults` parameter injects failures (see
ugs_px4.model).
"""

import threading
import time

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleCommandAck, VehicleLandDetected, VehicleLocalPosition,
                          VehicleStatus)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .model import Px4Model
from .px4 import ARMING_STATE_ARMED, nav_state_name


class FakePx4(Node):
    """Publish /fmu/out from the model, feed it /fmu/in."""

    def __init__(self):
        super().__init__('fake_px4')
        self.declare_parameter('faults', '')
        faults = self.get_parameter('faults').value
        self.model = Px4Model(faults)
        self.boot = time.monotonic()
        # what PX4's uXRCE-DDS client offers: best effort, transient local
        out = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(VehicleStatus, '/fmu/out/vehicle_status_v1', out)
        self.position_pub = self.create_publisher(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', out)
        self.land_pub = self.create_publisher(
            VehicleLandDetected, '/fmu/out/vehicle_land_detected', out)
        self.ack_pub = self.create_publisher(VehicleCommandAck, '/fmu/out/vehicle_command_ack',
                                             out)
        self.lock = threading.Lock()  # the model, between the loop and the subscriptions
        self.create_subscription(OffboardControlMode, '/fmu/in/offboard_control_mode',
                                 self.on_offboard, 10)
        self.create_subscription(TrajectorySetpoint, '/fmu/in/trajectory_setpoint',
                                 self.on_setpoint, 10)
        self.create_subscription(VehicleCommand, '/fmu/in/vehicle_command', self.on_command, 10)
        self.last = time.monotonic()
        self.ticks = 0
        self.last_status = 0.0
        self.reported = None
        # The 50 Hz loop (PX4's vehicle_local_position rate) runs in its own thread, as
        # in takeoff_server: on an executor timer it starved this node's subscriptions on
        # a busy CI runner, and a missed OffboardControlMode looks like offboard loss.
        self.running = True
        self.loop = threading.Thread(target=self.run_loop, daemon=True)
        self.loop.start()
        self.get_logger().info(f'fake_px4 up (PX4 v1.17 interface), faults: {faults or "none"}')

    def stamp(self) -> int:
        return int((time.monotonic() - self.boot) * 1e6)  # PX4: microseconds since boot

    def run_loop(self):
        next_tick = time.monotonic()
        while self.running and rclpy.ok():
            self.tick()
            next_tick += 0.02
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > 0.5:
                next_tick = time.monotonic()

    def destroy_node(self):
        self.running = False
        if self.loop.is_alive():
            self.loop.join(timeout=1.0)
        return super().destroy_node()

    def on_offboard(self, msg):
        with self.lock:
            self.model.offboard_control_mode()

    def on_setpoint(self, msg):
        with self.lock:
            self.model.trajectory_setpoint(float(msg.position[2]), float(msg.velocity[2]))

    def on_command(self, msg):
        params = (msg.param1, msg.param2, msg.param3, msg.param4, msg.param5, msg.param6,
                  msg.param7)
        with self.lock:
            self.model.vehicle_command(msg.command, params)
            acks, self.model.acks = self.model.acks, []
        for _, command, result in acks:
            ack = VehicleCommandAck()
            ack.timestamp = self.stamp()
            ack.command = command
            ack.result = result
            ack.target_system = msg.source_system
            ack.target_component = msg.source_component
            self.ack_pub.publish(ack)

    def tick(self):
        if not rclpy.ok():
            return
        now = time.monotonic()
        with self.lock:
            self.model.step(min(now - self.last, 0.1))
            s = self.model.snapshot()
            offboard_seen = self.model.t - self.model.offboard_seen
        self.last = now
        self.ticks += 1
        pos = VehicleLocalPosition()
        pos.timestamp = self.stamp()
        pos.xy_valid = pos.z_valid = pos.v_xy_valid = pos.v_z_valid = s.xy_valid
        pos.x, pos.y, pos.z = s.x, s.y, s.z
        pos.vx, pos.vy, pos.vz = 0.0, 0.0, s.vz
        pos.heading = s.heading
        pos.ref_alt = s.ref_alt
        pos.ref_lat, pos.ref_lon = 47.397971, 8.546164  # PX4 SITL's home, Zurich
        pos.dist_bottom = -s.z
        pos.dist_bottom_valid = True
        self.position_pub.publish(pos)
        if now - self.last_status >= 0.5:  # 2 Hz of wall clock, as measured from PX4 SITL
            self.last_status = now
            status = VehicleStatus()
            status.timestamp = self.stamp()
            status.arming_state = s.arming_state
            status.nav_state = status.nav_state_user_intention = s.nav_state
            status.pre_flight_checks_pass = s.preflight_ok
            status.system_id = status.component_id = 1
            self.status_pub.publish(status)
            land = VehicleLandDetected()
            land.timestamp = self.stamp()
            land.landed = s.landed
            land.ground_contact = land.maybe_landed = s.landed
            self.land_pub.publish(land)
        state = (s.nav_state, s.arming_state)
        if state != self.reported and self.reported is not None and self.reported[0] == 14 \
                and s.nav_state != 14:
            self.get_logger().warn(
                f'left offboard {offboard_seen:.2f} s after the last OffboardControlMode')
        if state != self.reported:
            armed = 'armed' if s.arming_state == ARMING_STATE_ARMED else 'disarmed'
            self.get_logger().info(f'{nav_state_name(s.nav_state)}, {armed}, '
                                   f'altitude {-s.z:.2f} m')
            self.reported = state


def main(args=None):
    rclpy.init(args=args)
    node = FakePx4()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
