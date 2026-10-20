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
takeoff_server: the week 6 action server, flying PX4 over uXRCE-DDS.

Actions:

* /takeoff (ugs_interfaces/action/Takeoff): climb to an altitude and hold it,
  in offboard mode (streamed trajectory setpoints) or PX4's takeoff mode;
* /land (ugs_interfaces/action/Land): land and wait for PX4 to disarm.

PX4 side, the topics the uXRCE-DDS client bridges (px4_msgs v1.17):

* in:  /fmu/in/offboard_control_mode, /fmu/in/trajectory_setpoint,
       /fmu/in/vehicle_command
* out: /fmu/out/vehicle_status_v1, /fmu/out/vehicle_local_position_v1,
       /fmu/out/vehicle_land_detected, /fmu/out/vehicle_command_ack

PX4 1.16 started suffixing versioned messages (_v1); the server subscribes to
both names so it works with older firmware too. PX4 publishes best effort, so
the subscriptions use the sensor data QoS profile; a default (reliable)
subscription would match nothing.

A 20 Hz loop, in a thread of its own, runs the logic in ugs_px4.takeoff and
publishes what it asks for; PX4's telemetry is received on a node and executor
of its own, so neither waits on the action servers. Between goals the server keeps streaming the last
setpoint, because PX4 treats a silent offboard link as a failsafe.
"""

import math
import threading
import time

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleCommandAck, VehicleLandDetected, VehicleLocalPosition,
                          VehicleStatus)
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import (ExternalShutdownException, MultiThreadedExecutor,
                             SingleThreadedExecutor)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ugs_interfaces.action import Land, Takeoff

from .px4 import ack_name, command_name, nav_state_name, Snapshot
from .takeoff import Land as LandLogic, Limits, METHODS, Output, Takeoff as TakeoffLogic


class TakeoffServer(Node):
    """Serve /takeoff and /land by streaming setpoints and commands to PX4."""

    def __init__(self):
        super().__init__('takeoff_server')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('default_climb_rate', 1.0)
        self.declare_parameter('max_climb_rate', 3.0)
        self.declare_parameter('min_altitude', 1.0)
        self.declare_parameter('max_altitude', 120.0)  # the usual legal ceiling, in metres
        self.declare_parameter('timeout_s', 90.0)
        self.declare_parameter('preflight_timeout_s', 60.0)
        # how far the climb's reference may run ahead of the vehicle (0: a fixed clock)
        self.declare_parameter('max_lead', 0.4)
        self.declare_parameter('target_system', 1)
        self.lock = threading.Lock()
        self.snapshot = Snapshot()
        self.ground_z = None  # NED z of the ground, learned while landed
        self.active = None  # the running TakeoffLogic or LandLogic
        self.stream = None  # the setpoint streamed to PX4, None for none
        self.last_command = None

        # PX4's telemetry (50 Hz) gets a node and a SingleThreadedExecutor of its own, in
        # its own thread: on the MultiThreadedExecutor beside the action servers it fell
        # over 2 s behind on a loaded CI runner, and stale telemetry aborts the goal.
        sensor = qos_profile_sensor_data
        self.px4 = rclpy.create_node('takeoff_server_px4')
        sub = self.px4.create_subscription
        for name in ('vehicle_status', 'vehicle_status_v1'):
            sub(VehicleStatus, f'/fmu/out/{name}', self.on_status, sensor)
        for name in ('vehicle_local_position', 'vehicle_local_position_v1'):
            sub(VehicleLocalPosition, f'/fmu/out/{name}', self.on_position, sensor)
        sub(VehicleLandDetected, '/fmu/out/vehicle_land_detected', self.on_land_detected, sensor)
        sub(VehicleCommandAck, '/fmu/out/vehicle_command_ack', self.on_ack, sensor)
        self.px4_executor = SingleThreadedExecutor()
        self.px4_executor.add_node(self.px4)
        self.px4_thread = threading.Thread(target=self.spin_px4, daemon=True)
        self.px4_thread.start()
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        # The 20 Hz loop runs in a thread of its own, not on an executor timer: with
        # 50 Hz telemetry and two action servers, rclpy's MultiThreadedExecutor let a
        # timer go 0.4 to 1.2 s between calls, longer than PX4's 1 s offboard loss
        # failsafe (COM_OF_LOSS_T). Publishing from another thread is safe in rclpy.
        self.period = 1.0 / self.get_parameter('rate_hz').value
        self.running = True
        self.loop = threading.Thread(target=self.run_loop, daemon=True)
        self.loop.start()

        actions = ReentrantCallbackGroup()
        self.takeoff_server = ActionServer(
            self, Takeoff, 'takeoff', self.execute_takeoff, goal_callback=self.on_takeoff_goal,
            cancel_callback=self.on_cancel, callback_group=actions)
        self.land_server = ActionServer(
            self, Land, 'land', self.execute_land, goal_callback=self.on_land_goal,
            cancel_callback=self.on_cancel, callback_group=actions)
        self.get_logger().info('takeoff_server ready: /takeoff and /land, waiting for PX4')

    # ---------------------------------------------------------- PX4 telemetry

    def now(self) -> float:
        return time.monotonic()

    def on_status(self, msg: VehicleStatus):
        with self.lock:
            s = self.snapshot
            if s.status_time is None:
                self.get_logger().info(f'PX4 is talking: {nav_state_name(msg.nav_state)}, '
                                       f'{"armed" if msg.arming_state == 2 else "disarmed"}')
            elif msg.nav_state != s.nav_state or msg.arming_state != s.arming_state:
                self.get_logger().info(f'PX4 now {nav_state_name(msg.nav_state)}, '
                                       f'{"armed" if msg.arming_state == 2 else "disarmed"}')
            s.status_time = self.now()
            s.arming_state = msg.arming_state
            s.nav_state = msg.nav_state
            s.preflight_ok = msg.pre_flight_checks_pass

    def on_position(self, msg: VehicleLocalPosition):
        with self.lock:
            s = self.snapshot
            s.position_time = self.now()
            s.xy_valid, s.z_valid = msg.xy_valid, msg.z_valid
            s.x, s.y, s.z, s.vz = msg.x, msg.y, msg.z, msg.vz
            s.heading, s.ref_alt = msg.heading, msg.ref_alt
            if s.landed and s.z_valid and not s.armed:
                self.ground_z = s.z

    def on_land_detected(self, msg: VehicleLandDetected):
        with self.lock:
            self.snapshot.landed = msg.landed

    def on_ack(self, msg: VehicleCommandAck):
        with self.lock:
            self.snapshot.acks.append((self.now(), msg.command, msg.result))
        self.get_logger().info(f'<- PX4 ack {command_name(msg.command)}: {ack_name(msg.result)}')

    # ------------------------------------------------------------- the loop

    def run_loop(self):
        next_tick = time.monotonic()
        while self.running and rclpy.ok():
            self.tick()
            next_tick += self.period
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > 0.5:  # fell far behind: do not burst to catch up
                next_tick = time.monotonic()

    def spin_px4(self):
        try:
            self.px4_executor.spin()
        except (ExternalShutdownException, RuntimeError):
            pass  # shutting down

    def destroy_node(self):
        self.running = False
        if self.loop.is_alive():
            self.loop.join(timeout=1.0)
        self.px4_executor.shutdown(timeout_sec=1.0)
        self.px4.destroy_node()
        return super().destroy_node()

    def tick(self):
        if not rclpy.ok():
            return
        with self.lock:
            now = self.now()
            if self.active is not None and self.active.result is None:
                out = self.active.update(now, self.snapshot)
                self.snapshot.acks = []
                self.stream = out.setpoint
                for command in out.commands:
                    self.send(command)
            setpoint = self.stream
        if setpoint is not None:
            self.publish_setpoint(setpoint)

    def publish_setpoint(self, sp):
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp = stamp
        mode.position = True
        mode.velocity = not math.isnan(sp.vz)  # feed forward rides along
        self.offboard_pub.publish(mode)
        msg = TrajectorySetpoint()
        msg.timestamp = stamp
        msg.position = [float(sp.x), float(sp.y), float(sp.z)]
        msg.velocity = [math.nan, math.nan, float(sp.vz)]
        msg.acceleration = [math.nan] * 3
        msg.jerk = [math.nan] * 3
        msg.yaw = float(sp.yaw)
        msg.yawspeed = math.nan
        self.setpoint_pub.publish(msg)

    def send(self, command):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command.command
        (msg.param1, msg.param2, msg.param3, msg.param4, msg.param5, msg.param6,
         msg.param7) = (float(p) for p in command.params)
        target = self.get_parameter('target_system').value
        msg.target_system = target
        msg.target_component = 1
        msg.source_system = target
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)
        self.get_logger().info(f'-> PX4 {command}')

    # -------------------------------------------------------------- actions

    def _busy(self) -> bool:
        return self.active is not None and self.active.result is None

    def on_takeoff_goal(self, goal):
        lo = self.get_parameter('min_altitude').value
        hi = self.get_parameter('max_altitude').value
        max_rate = self.get_parameter('max_climb_rate').value
        method = goal.method or 'offboard'
        why = None
        if not math.isfinite(goal.altitude) or not lo <= goal.altitude <= hi:
            why = f'altitude {goal.altitude} m is outside {lo:g} to {hi:g} m'
        elif not math.isfinite(goal.climb_rate) or not 0 <= goal.climb_rate <= max_rate:
            why = f'climb rate {goal.climb_rate} m/s is outside 0 to {max_rate:g} m/s'
        elif method not in METHODS:
            why = f'method {goal.method!r} is not one of {", ".join(METHODS)}'
        elif self._busy():
            why = 'another goal is running'
        if why:
            self.get_logger().warn(f'/takeoff rejected: {why}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_land_goal(self, goal):
        if self._busy():
            self.get_logger().warn('/land rejected: another goal is running')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, goal_handle):
        self.get_logger().info('cancel requested')
        return CancelResponse.ACCEPT

    def execute_takeoff(self, goal_handle):
        goal = goal_handle.request
        rate = goal.climb_rate or self.get_parameter('default_climb_rate').value
        method = goal.method or 'offboard'
        limits = Limits(climb_rate=rate, timeout=self.get_parameter('timeout_s').value,
                        preflight_timeout=self.get_parameter('preflight_timeout_s').value,
                        max_lead=self.get_parameter('max_lead').value)
        with self.lock:
            logic = TakeoffLogic(goal.altitude, self.now(), self.ground_z, method, limits)
            self.active = logic
        self.get_logger().info(f'/takeoff to {goal.altitude:.1f} m at {rate:.1f} m/s ({method})')

        def feedback(fb):
            s = self.snapshot
            fb.phase = logic.phase
            fb.altitude = float(logic.altitude_of(s))
            fb.vertical_speed = float(-s.vz)
            fb.remaining = float(goal.altitude - fb.altitude)

        out = self._run(goal_handle, logic, Takeoff.Feedback(), feedback)
        result = Takeoff.Result()
        result.success = bool(out.done)
        with self.lock:
            result.final_altitude = float(logic.altitude_of(self.snapshot))
        result.climb_time = float(logic.climb_time)
        result.message = out.message
        return result

    def execute_land(self, goal_handle):
        with self.lock:
            logic = LandLogic(self.now(), self.ground_z,
                              Limits(timeout=self.get_parameter('timeout_s').value))
            self.active = logic
            self.stream = None  # PX4's land mode flies the descent
        self.get_logger().info('/land')

        def feedback(fb):
            s = self.snapshot
            fb.phase = logic.phase
            fb.altitude = float(logic.altitude_of(s))
            fb.vertical_speed = float(-s.vz)

        start = self.now()
        out = self._run(goal_handle, logic, Land.Feedback(), feedback)
        result = Land.Result()
        result.success = bool(out.done)
        result.descent_time = float(self.now() - start)
        result.message = out.message
        return result

    def _run(self, goal_handle, logic, fb, fill):
        """Publish feedback at 10 Hz until the logic finishes or the goal is canceled."""
        while rclpy.ok():
            with self.lock:
                if goal_handle.is_cancel_requested and logic.result is None:
                    out = logic.cancel(self.now(), self.snapshot)
                    self.stream = out.setpoint
                    for command in out.commands:
                        self.send(command)
                out = logic.result
                fill(fb)
            if out is not None:
                break
            goal_handle.publish_feedback(fb)
            time.sleep(0.1)
        else:
            goal_handle.abort()
            return logic.result or Output(logic.phase, done=False, message='shut down')
        if out.done:
            goal_handle.succeed()
            self.get_logger().info(f'succeeded: {out.message}')
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info(out.message)
        else:
            goal_handle.abort()
            self.get_logger().error(f'aborted: {out.message}')
        return out


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
