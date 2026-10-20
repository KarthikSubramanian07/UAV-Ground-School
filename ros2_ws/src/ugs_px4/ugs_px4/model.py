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
A stand in for PX4: the commander's rules and a vertical flight model, no ROS.

Real PX4 SITL in Gazebo needs a 3 GB image and half a minute to boot, which is
too much for every launch test. This model reproduces the behaviour the takeoff
server depends on, each rule taken from PX4 v1.17 and its default parameters:

* the estimator needs a few seconds before positions are valid, and arming is
  refused until the preflight checks pass;
* offboard mode is refused unless OffboardControlMode has been streaming, and
  losing the stream for COM_OF_LOSS_T (1 s) is a failsafe: PX4 switches to hold;
* NAV_TAKEOFF switches to takeoff mode, which climbs at MPC_TKO_SPEED (1.5 m/s)
  once armed, then holds (AUTO_LOITER);
* NAV_LAND descends at MPC_LAND_SPEED (0.7 m/s); PX4 disarms COM_DISARM_LAND (2 s)
  after touchdown, or COM_DISARM_PRFLT (10 s) after arming if it never took off;
* after arming the motors spool up for COM_SPOOLUP_TIME (1 s), then the takeoff
  ramps the climb speed limit up over MPC_TKO_RAMP_T (3 s); the x500 in SITL
  leaves the ground about 3 s after arming, and so does the model;
* vertical motion: a P position loop (MPC_Z_P = 1) with velocity feed forward,
  limited to MPC_Z_VEL_MAX_UP (3 m/s) and MPC_Z_VEL_MAX_DN (1.5 m/s), and a
  first order velocity response limited to MPC_ACC_UP_MAX (4 m/s^2).

Faults can be injected to test the server's failure paths: no_preflight (the
checks never pass), deny_arm (they pass but arming is refused), no_offboard (the
mode switch is always refused), mute (nothing is acknowledged), and takeover_at=T
(a pilot switches to position mode T seconds after the climb starts).
"""

import math

from .px4 import (ACK_ACCEPTED, ARMING_STATE_ARMED, ARMING_STATE_DISARMED,
                  CMD_COMPONENT_ARM_DISARM, CMD_DO_SET_MODE, CMD_NAV_LAND, CMD_NAV_TAKEOFF,
                  MAIN_MODE_AUTO, MAIN_MODE_OFFBOARD, NAV_AUTO_LAND, NAV_AUTO_LOITER,
                  NAV_AUTO_TAKEOFF, NAV_OFFBOARD, NAV_POSCTL, Snapshot, SUB_MODE_AUTO_LOITER)

ACK_TEMPORARILY_REJECTED = 1
ACK_DENIED = 2

EKF_DELAY = 3.0  # s until local position is valid
PREFLIGHT_DELAY = 5.0  # s until arming checks pass
COM_OF_LOSS_T = 1.0
MPC_TKO_SPEED = 1.5
MPC_LAND_SPEED = 0.7
MPC_Z_P = 1.0
MPC_Z_VEL_MAX_UP = 3.0
MPC_Z_VEL_MAX_DN = 1.5
MPC_ACC_UP_MAX = 4.0
VELOCITY_TAU = 0.25  # s, the velocity loop's response
COM_SPOOLUP_TIME = 1.0
MPC_TKO_RAMP_T = 3.0
RAMP_DEAD = 1.7  # s into the ramp before thrust beats weight (fitted to the SITL flights)
COM_DISARM_LAND = 2.0
COM_DISARM_PRFLT = 10.0
REF_ALT = 488.0  # m AMSL of the local origin: PX4 SITL's default home (Zurich)


class Px4Model:
    """PX4's commander and a vertical point mass, stepped by the caller's clock."""

    def __init__(self, faults: str = ''):
        self.faults = {f.split('=')[0]: (f.split('=')[1] if '=' in f else '')
                       for f in faults.replace(',', ' ').split()}
        self.t = 0.0
        self.z = 0.0  # NED, ground at 0
        self.vz = 0.0
        self.arming_state = ARMING_STATE_DISARMED
        self.nav_state = NAV_AUTO_LOITER  # PX4 v1.17 boots in hold
        self.offboard_seen = -math.inf  # last OffboardControlMode
        self.offboard_since = None  # start of the current unbroken stream
        self.setpoint_z = None
        self.setpoint_vz = 0.0
        self.takeoff_z = None
        self.hold_z = 0.0
        self.armed_at = None
        self.touchdown_at = None
        self.climb_started = None
        self.took_over = False
        self.ramp_start = None  # takeoff ramp, once the setpoint asks to leave the ground
        self.flew = False  # has left the ground since arming
        self.acks = []  # (time, command, result)

    # ----------------------------------------------------------------- inputs

    def offboard_control_mode(self) -> None:
        if self.t - self.offboard_seen > COM_OF_LOSS_T:
            self.offboard_since = self.t
        self.offboard_seen = self.t

    def trajectory_setpoint(self, z: float, vz: float = math.nan) -> None:
        self.setpoint_z = z
        self.setpoint_vz = 0.0 if math.isnan(vz) else vz

    def vehicle_command(self, command: int, params) -> None:
        result = self._handle(command, params)
        if 'mute' not in self.faults:
            self.acks.append((self.t, command, result))

    def _handle(self, command: int, p) -> int:
        valid = self.t >= EKF_DELAY
        if command == CMD_COMPONENT_ARM_DISARM:
            if p[0] < 0.5:
                if self.landed:
                    self._disarm()
                    return ACK_ACCEPTED
                return ACK_DENIED
            if {'deny_arm', 'no_preflight'} & self.faults.keys() or self.t < PREFLIGHT_DELAY:
                return ACK_DENIED
            if self.arming_state != ARMING_STATE_ARMED:
                self.arming_state = ARMING_STATE_ARMED
                self.armed_at = self.t
                self.climb_started = None
            return ACK_ACCEPTED
        if command == CMD_DO_SET_MODE:
            main, sub = int(p[1]), int(p[2])
            if main == MAIN_MODE_OFFBOARD:
                streaming = (self.offboard_since is not None
                             and self.t - self.offboard_seen < COM_OF_LOSS_T
                             and self.t - self.offboard_since > 0.5)
                if 'no_offboard' in self.faults or not streaming or not valid:
                    return ACK_TEMPORARILY_REJECTED
                self.nav_state = NAV_OFFBOARD
                return ACK_ACCEPTED
            if main == MAIN_MODE_AUTO and sub == SUB_MODE_AUTO_LOITER and valid:
                self.nav_state = NAV_AUTO_LOITER
                self.hold_z = self.z
                return ACK_ACCEPTED
            return ACK_TEMPORARILY_REJECTED
        if command == CMD_NAV_TAKEOFF:
            if not valid:
                return ACK_TEMPORARILY_REJECTED
            self.takeoff_z = -(float(p[6]) - REF_ALT) if not math.isnan(p[6]) else -2.5
            self.nav_state = NAV_AUTO_TAKEOFF
            return ACK_ACCEPTED
        if command == CMD_NAV_LAND:
            if not valid or self.arming_state != ARMING_STATE_ARMED:
                return ACK_TEMPORARILY_REJECTED
            self.nav_state = NAV_AUTO_LAND
            return ACK_ACCEPTED
        return 3  # UNSUPPORTED

    def _disarm(self) -> None:
        self.arming_state = ARMING_STATE_DISARMED
        self.armed_at = None
        self.touchdown_at = None
        self.ramp_start = None
        self.flew = False

    # ---------------------------------------------------------------- physics

    @property
    def landed(self) -> bool:
        return self.z > -0.05 and abs(self.vz) < 0.1

    def step(self, dt: float) -> None:
        self.t += dt
        armed = self.arming_state == ARMING_STATE_ARMED
        if self.nav_state == NAV_OFFBOARD and self.t - self.offboard_seen > COM_OF_LOSS_T:
            self.nav_state = NAV_AUTO_LOITER  # offboard loss failsafe: hold
            self.hold_z = self.z
        takeover = self.faults.get('takeover_at')
        if takeover and not self.took_over and self.climb_started is not None \
                and self.t - self.climb_started > float(takeover):
            self.took_over = True
            self.nav_state = NAV_POSCTL
            self.hold_z = self.z
        # the reference the position controller tracks
        target, feed_forward, up_limit = self.z, 0.0, MPC_Z_VEL_MAX_UP
        if armed and self.nav_state == NAV_OFFBOARD and self.setpoint_z is not None:
            target, feed_forward = self.setpoint_z, self.setpoint_vz
        elif armed and self.nav_state == NAV_AUTO_TAKEOFF and self.takeoff_z is not None:
            target, up_limit = self.takeoff_z, MPC_TKO_SPEED
            if abs(self.z - self.takeoff_z) < 0.1:
                self.nav_state = NAV_AUTO_LOITER
                self.hold_z = self.takeoff_z
        elif armed and self.nav_state in (NAV_AUTO_LOITER, NAV_POSCTL):
            target = self.hold_z
        if armed and target < -0.2 and self.climb_started is None:
            self.climb_started = self.t
        if armed and self.landed and self.climb_started is not None and self.ramp_start is None \
                and self.armed_at is not None and self.t - self.armed_at >= COM_SPOOLUP_TIME:
            self.ramp_start = self.t
        if armed and self.ramp_start is None and self.landed:
            up_limit = 0.0  # spooling up, or nothing asked of it yet
        elif armed and self.ramp_start is not None:
            ramp = (self.t - self.ramp_start - RAMP_DEAD) / (MPC_TKO_RAMP_T - RAMP_DEAD)
            up_limit = up_limit * max(0.0, min(1.0, ramp))
        desired = feed_forward + MPC_Z_P * (target - self.z) if armed else 0.0
        if armed and self.nav_state == NAV_AUTO_LAND:
            desired = MPC_LAND_SPEED  # land mode descends at a fixed speed
        desired = max(-up_limit, min(MPC_Z_VEL_MAX_DN, desired))
        dv = (desired - self.vz) * min(1.0, dt / VELOCITY_TAU)
        dv = max(-MPC_ACC_UP_MAX * dt, min(MPC_ACC_UP_MAX * dt, dv))
        self.vz += dv
        self.z += self.vz * dt
        if self.z > 0.0:  # the ground
            self.z, self.vz = 0.0, min(0.0, self.vz)
        if self.z < -0.3:
            self.flew = True
        if armed and self.landed and (self.nav_state == NAV_AUTO_LAND or self.flew):
            if self.touchdown_at is None:
                self.touchdown_at = self.t
            elif self.t - self.touchdown_at > COM_DISARM_LAND:
                self._disarm()
                self.nav_state = NAV_AUTO_LOITER
        elif armed and not self.landed:
            self.touchdown_at = None
        if armed and self.climb_started is None and self.armed_at is not None \
                and self.t - self.armed_at > COM_DISARM_PRFLT:
            self._disarm()

    # ---------------------------------------------------------------- outputs

    def snapshot(self, now: float | None = None) -> Snapshot:
        """Return what PX4 would be publishing, stamped with `now` (default: model time)."""
        now = self.t if now is None else now
        valid = self.t >= EKF_DELAY
        acks, self.acks = self.acks, []
        return Snapshot(
            status_time=now, position_time=now, arming_state=self.arming_state,
            nav_state=self.nav_state, preflight_ok=self.t >= PREFLIGHT_DELAY
            and 'no_preflight' not in self.faults,
            xy_valid=valid, z_valid=valid, x=0.0, y=0.0, z=self.z, vz=self.vz, heading=0.0,
            ref_alt=REF_ALT, landed=self.landed, acks=acks)
