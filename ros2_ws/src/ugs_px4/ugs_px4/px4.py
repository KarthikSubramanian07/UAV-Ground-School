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
The parts of PX4's uORB interface the takeoff server uses, with no ROS in it.

The numbers are copied from px4_msgs v1.17.0 (VehicleStatus, VehicleCommand,
VehicleCommandAck), so the takeoff logic can be unit tested without px4_msgs
installed; test_px4_constants.py checks them against the real messages.
"""

from dataclasses import dataclass, field
import math

# VehicleStatus.arming_state
ARMING_STATE_DISARMED = 1
ARMING_STATE_ARMED = 2

# VehicleStatus.nav_state
NAV_STATES = {
    0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'AUTO_MISSION', 4: 'AUTO_LOITER',
    5: 'AUTO_RTL', 6: 'POSITION_SLOW', 8: 'ALTITUDE_CRUISE', 10: 'ACRO', 12: 'DESCEND',
    13: 'TERMINATION', 14: 'OFFBOARD', 15: 'STAB', 17: 'AUTO_TAKEOFF', 18: 'AUTO_LAND',
    19: 'AUTO_FOLLOW_TARGET', 20: 'AUTO_PRECLAND', 21: 'ORBIT', 22: 'AUTO_VTOL_TAKEOFF',
}
NAV_POSCTL = 2
NAV_AUTO_LOITER = 4
NAV_AUTO_RTL = 5
NAV_DESCEND = 12
NAV_OFFBOARD = 14
NAV_AUTO_TAKEOFF = 17
NAV_AUTO_LAND = 18

# VehicleCommand.command (the MAVLink MAV_CMD numbers)
CMD_NAV_LAND = 21
CMD_NAV_TAKEOFF = 22
CMD_DO_SET_MODE = 176
CMD_COMPONENT_ARM_DISARM = 400

# VehicleCommandAck.result (MAVLink MAV_RESULT)
ACK_RESULTS = {
    0: 'ACCEPTED', 1: 'TEMPORARILY_REJECTED', 2: 'DENIED', 3: 'UNSUPPORTED',
    4: 'FAILED', 5: 'IN_PROGRESS', 6: 'CANCELLED',
}
ACK_ACCEPTED = 0
ACK_IN_PROGRESS = 5

# PX4 custom modes, the parameters of DO_SET_MODE: param1 = 1 (custom mode
# enabled), param2 = main mode, param3 = sub mode.
MAIN_MODE_AUTO = 4
MAIN_MODE_OFFBOARD = 6
SUB_MODE_AUTO_LOITER = 3


def nav_state_name(state: int) -> str:
    """Name a VehicleStatus.nav_state value the way PX4's code does."""
    return NAV_STATES.get(state, f'NAV_STATE_{state}')


def ack_name(result: int) -> str:
    """Name a VehicleCommandAck.result value."""
    return ACK_RESULTS.get(result, f'RESULT_{result}')


def command_name(command: int) -> str:
    """Name the VehicleCommand values the takeoff server sends."""
    return {CMD_NAV_LAND: 'NAV_LAND', CMD_NAV_TAKEOFF: 'NAV_TAKEOFF',
            CMD_DO_SET_MODE: 'DO_SET_MODE',
            CMD_COMPONENT_ARM_DISARM: 'COMPONENT_ARM_DISARM'}.get(command, f'CMD_{command}')


STATUS_MAX_AGE = 5.0  # seconds VehicleStatus may be quiet before PX4 counts as lost


@dataclass
class Command:
    """A VehicleCommand to send: the command number and its seven parameters."""

    command: int
    params: tuple = (0.0,) * 7

    @staticmethod
    def arm() -> 'Command':
        return Command(CMD_COMPONENT_ARM_DISARM, (1.0, 0, 0, 0, 0, 0, 0))

    @staticmethod
    def offboard() -> 'Command':
        return Command(CMD_DO_SET_MODE, (1.0, float(MAIN_MODE_OFFBOARD), 0, 0, 0, 0, 0))

    @staticmethod
    def hold() -> 'Command':
        return Command(CMD_DO_SET_MODE, (1.0, float(MAIN_MODE_AUTO),
                                         float(SUB_MODE_AUTO_LOITER), 0, 0, 0, 0))

    @staticmethod
    def takeoff(altitude_amsl: float) -> 'Command':
        # lat, lon and yaw NaN: take off where the vehicle is, keeping its heading
        nan = math.nan
        return Command(CMD_NAV_TAKEOFF, (0, 0, 0, nan, nan, nan, float(altitude_amsl)))

    @staticmethod
    def land() -> 'Command':
        nan = math.nan
        return Command(CMD_NAV_LAND, (0, 0, 0, nan, nan, nan, nan))

    def __str__(self) -> str:
        return command_name(self.command)


@dataclass
class Setpoint:
    """A TrajectorySetpoint in PX4's local NED frame: metres, m/s, radians."""

    x: float
    y: float
    z: float
    vz: float = math.nan  # feed forward; NaN lets PX4 derive it
    yaw: float = math.nan


@dataclass
class Snapshot:
    """
    What PX4 has said most recently, gathered from its /fmu/out topics.

    Times are seconds on the caller's clock; None means never heard.
    """

    status_time: float | None = None
    position_time: float | None = None
    arming_state: int = ARMING_STATE_DISARMED
    nav_state: int = NAV_POSCTL
    preflight_ok: bool = False
    xy_valid: bool = False
    z_valid: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # NED: down positive
    vz: float = 0.0  # NED: down positive
    heading: float = 0.0
    ref_alt: float = math.nan  # AMSL of the local origin, metres
    landed: bool = True
    acks: list = field(default_factory=list)  # (time, command, result) since last read

    @property
    def armed(self) -> bool:
        return self.arming_state == ARMING_STATE_ARMED

    def fresh(self, now: float, max_age: float = 2.0) -> bool:
        """
        Tell whether PX4 is still talking: a position within max_age seconds.

        Liveness comes from vehicle_local_position (50 Hz). VehicleStatus is sent on
        change and at about 2 Hz, and SITL went 2.5 s without one while Gazebo ran
        at half real time, so it may be quiet for up to STATUS_MAX_AGE.
        """
        return (self.status_time is not None and self.position_time is not None
                and now - self.position_time < max_age
                and now - self.status_time < max(max_age, STATUS_MAX_AGE))
