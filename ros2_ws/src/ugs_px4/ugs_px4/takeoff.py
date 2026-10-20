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
The takeoff and landing logic behind the week 6 action server, with no ROS in it.

Each controller is a state machine fed a Snapshot of PX4 (from its /fmu/out
topics) and the time, which answers with what to send back on /fmu/in: a
trajectory setpoint to stream, commands, and a phase for the action feedback.
The same code drives PX4 SITL in Gazebo through rclpy, the fake PX4 in the launch
tests, and the unit tests, so what is tested is what flies.

Offboard takeoff, the PX4 ROS 2 guide's way:

    WAIT_FOR_PX4 -> PREFLIGHT -> OFFBOARD -> ARMING -> CLIMB -> SETTLE -> done

PX4 only accepts offboard mode once setpoints are already streaming, so they
start in PREFLIGHT, holding the vehicle where it is, and never stop: offboard
loss is a failsafe. The climb follows a trapezoidal velocity profile (limited
acceleration, then the climb rate, then braking) rather than a step, so PX4's
position controller tracks a reference it can actually fly, and the reference
waits whenever it gets more than 0.4 m ahead (PX4's spool up and takeoff ramp).

Takeoff mode, PX4's own: NAV_TAKEOFF to an AMSL altitude, then arm. PX4 climbs at
MPC_TKO_SPEED and switches itself to hold (AUTO_LOITER) at the altitude.
"""

from dataclasses import dataclass
import math

from .px4 import (ack_name, Command, NAV_AUTO_LAND, NAV_AUTO_LOITER, NAV_AUTO_TAKEOFF,
                  NAV_OFFBOARD, nav_state_name, Setpoint, Snapshot)

METHODS = ('offboard', 'takeoff_mode')


@dataclass
class Limits:
    """Tunables of the takeoff, all in metres and seconds."""

    climb_rate: float = 1.0  # m/s, the cruise speed of the profile
    accel: float = 1.0  # m/s^2, at both ends of the profile
    tolerance: float = 0.15  # within this of the goal ...
    settled_speed: float = 0.15  # ... and slower than this (m/s) ...
    settle_time: float = 1.0  # ... for this long: reached
    max_lead: float = 0.4  # never let the setpoint get further ahead than this (0: no limit)
    stream_before_switch: float = 1.0  # setpoints streamed before asking for offboard
    retry_every: float = 1.0  # resend a command PX4 has not acted on
    max_tries: int = 5
    # vehicle_local_position (50 Hz) silent this long: PX4 is lost (see Snapshot.fresh)
    telemetry_timeout: float = 2.0
    preflight_timeout: float = 60.0  # PX4 SITL needs about 10 s for the estimator and GPS
    timeout: float = 90.0  # the whole goal


@dataclass
class Output:
    """What a controller wants sent to PX4 this cycle, and how the goal stands."""

    phase: str
    setpoint: Setpoint | None = None  # None: stop streaming
    commands: tuple = ()
    done: bool | None = None  # None while running, then True (succeeded) or False
    message: str = ''


class Profile:
    """
    A trapezoidal velocity profile from z0 to z1 (any frame, any direction).

    Accelerate at a, cruise at v, brake at a; a triangle if the move is too short
    to reach v. sample(t) is (position, velocity) t seconds after the start.
    """

    def __init__(self, z0: float, z1: float, v: float, a: float):
        self.z0, self.z1 = z0, z1
        self.sign = 1.0 if z1 >= z0 else -1.0
        d = abs(z1 - z0)
        self.a = a
        self.v = min(v, math.sqrt(d * a))  # the peak speed: v, or less for a triangle
        self.t_acc = self.v / a if a > 0 else 0.0
        d_acc = 0.5 * a * self.t_acc ** 2
        self.t_cruise = (d - 2 * d_acc) / self.v if self.v > 0 else 0.0
        self.duration = 2 * self.t_acc + self.t_cruise

    def sample(self, t: float) -> tuple[float, float]:
        if t <= 0:
            return self.z0, 0.0
        if t >= self.duration:
            return self.z1, 0.0
        a, v, ta, tc = self.a, self.v, self.t_acc, self.t_cruise
        if t < ta:
            s, speed = 0.5 * a * t * t, a * t
        elif t < ta + tc:
            s, speed = 0.5 * a * ta * ta + v * (t - ta), v
        else:
            tb = self.duration - t  # time left
            s, speed = abs(self.z1 - self.z0) - 0.5 * a * tb * tb, a * tb
        return self.z0 + self.sign * s, self.sign * speed


class _Commander:
    """Sends one command, resends it until PX4 does what it asked, and gives up."""

    def __init__(self, command: Command, limits: Limits):
        self.command = command
        self.limits = limits
        self.sent_at = None
        self.tries = 0
        self.last_ack = None

    def poll(self, now: float, acks) -> tuple[Command | None, str | None]:
        """(command to send now or None, error or None)."""
        for _, command, result in acks:
            if command == self.command.command:
                self.last_ack = result
        if self.sent_at is None or now - self.sent_at >= self.limits.retry_every:
            if self.tries >= self.limits.max_tries:
                why = f'last ack {ack_name(self.last_ack)}' if self.last_ack is not None \
                    else 'no ack'
                return None, f'PX4 did not act on {self.command} after {self.tries} tries ({why})'
            self.sent_at = now
            self.tries += 1
            return self.command, None
        return None, None


class Takeoff:
    """Take off to `altitude` metres above `ground_z` (NED) and hold it."""

    def __init__(self, altitude: float, now: float, ground_z: float | None = None,
                 method: str = 'offboard', limits: Limits | None = None):
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}, not {method!r}')
        self.altitude = altitude
        self.method = method
        self.limits = limits or Limits()
        self.ground_z = ground_z
        self.start = now
        self.phase = 'WAIT_FOR_PX4'
        self.phase_since = now
        self.hold = None  # Setpoint held while getting ready
        self.profile = None
        self.climb_start = None
        self.tau = 0.0  # how far along the profile the reference is, seconds
        self.last_update = None
        self.left_ground = None
        self.settled_since = None
        self.commander = None
        self.result = None  # the final Output, once done
        self.climb_time = math.nan

    # ------------------------------------------------------------------ helpers

    def _enter(self, phase: str, now: float) -> None:
        self.phase = phase
        self.phase_since = now
        self.commander = None

    def target_z(self) -> float:
        return self.ground_z - self.altitude

    def altitude_of(self, s: Snapshot) -> float:
        base = self.ground_z if self.ground_z is not None else s.z
        return base - s.z

    def _finish(self, done: bool, message: str, setpoint: Setpoint | None) -> Output:
        self.result = Output(self.phase, setpoint, (), done, message)
        return self.result

    def _command(self, command: Command, now: float, s: Snapshot):
        if self.commander is None or self.commander.command.command != command.command:
            self.commander = _Commander(command, self.limits)
        return self.commander.poll(now, s.acks)

    # --------------------------------------------------------------------- step

    def update(self, now: float, s: Snapshot) -> Output:
        """Advance with the latest snapshot; returns what to send to PX4."""
        if self.result is not None:  # done: keep holding what we ended on
            return Output(self.phase, self.result.setpoint, (), self.result.done,
                          self.result.message)
        if now - self.start > self.limits.timeout:
            return self._finish(False, f'timed out after {self.limits.timeout:.0f} s in '
                                f'{self.phase}', self.hold if self.method == 'offboard' else None)
        if self.phase != 'WAIT_FOR_PX4' and not s.fresh(now, self.limits.telemetry_timeout):
            return self._finish(False, 'lost PX4 telemetry', None)

        if self.phase == 'WAIT_FOR_PX4':
            if not (s.fresh(now, self.limits.telemetry_timeout) and s.xy_valid and s.z_valid):
                return Output(self.phase)
            if self.ground_z is None or s.landed:
                self.ground_z = s.z  # take off from here
            self.hold = Setpoint(s.x, s.y, s.z, yaw=s.heading)
            if s.armed and not s.landed:
                return self._airborne(now, s)
            self._enter('PREFLIGHT', now)

        stream = self.hold if self.method == 'offboard' else None

        if self.phase == 'PREFLIGHT':
            if s.preflight_ok or s.armed:
                self._enter('OFFBOARD' if self.method == 'offboard' else 'TAKEOFF_MODE', now)
            elif now - self.phase_since > self.limits.preflight_timeout:
                return self._finish(False, "PX4's preflight checks did not pass within "
                                    f'{self.limits.preflight_timeout:.0f} s', stream)
            else:
                return Output(self.phase, stream)

        if self.phase == 'OFFBOARD':
            if s.nav_state == NAV_OFFBOARD:
                self._enter('ARMING', now)
            elif now - self.phase_since < self.limits.stream_before_switch:
                return Output(self.phase, stream)  # PX4 wants to see setpoints first
            else:
                command, error = self._command(Command.offboard(), now, s)
                if error:
                    return self._finish(False, error, stream)
                return Output(self.phase, stream, (command,) if command else ())

        if self.phase == 'TAKEOFF_MODE':
            if s.nav_state == NAV_AUTO_TAKEOFF:
                self._enter('ARMING', now)
            else:
                amsl = (s.ref_alt - self.ground_z) + self.altitude
                if math.isnan(amsl):
                    return self._finish(False, 'PX4 has no altitude reference (ref_alt)', None)
                command, error = self._command(Command.takeoff(amsl), now, s)
                if error:
                    return self._finish(False, error, None)
                return Output(self.phase, None, (command,) if command else ())

        if self.phase == 'ARMING':
            wanted = NAV_OFFBOARD if self.method == 'offboard' else NAV_AUTO_TAKEOFF
            if s.nav_state != wanted and not s.armed:
                # PX4 left the mode before arming (another client, or a failsafe)
                return self._finish(False, f'PX4 switched to {nav_state_name(s.nav_state)} '
                                    'before arming', stream)
            if s.armed:
                return self._begin_climb(now, s)
            command, error = self._command(Command.arm(), now, s)
            if error:
                return self._finish(False, error, stream)
            return Output(self.phase, stream, (command,) if command else ())

        return self._fly(now, s)

    def _airborne(self, now: float, s: Snapshot) -> Output:
        """Change altitude from where we are: a goal while already flying."""
        if self.method == 'offboard' and s.nav_state != NAV_OFFBOARD:
            self._enter('OFFBOARD', now)
            self.phase_since = now - self.limits.stream_before_switch  # already streaming
            return Output(self.phase, self.hold)
        return self._begin_climb(now, s)

    def _begin_climb(self, now: float, s: Snapshot) -> Output:
        self._enter('CLIMB', now)
        self.climb_start = now
        self.tau, self.last_update = 0.0, now
        self.profile = Profile(s.z, self.target_z(), self.limits.climb_rate, self.limits.accel)
        return self._fly(now, s)

    def _fly(self, now: float, s: Snapshot) -> Output:
        if not s.armed:
            return self._finish(False, 'PX4 disarmed during the climb', None)
        expected = (NAV_AUTO_TAKEOFF, NAV_AUTO_LOITER)
        if self.method == 'offboard' and s.nav_state != NAV_OFFBOARD:
            return self._finish(False, f'PX4 left offboard for {nav_state_name(s.nav_state)} '
                                '(a failsafe or a pilot took over)', None)
        if self.method == 'takeoff_mode' and s.nav_state not in expected:
            return self._finish(False, f'PX4 left takeoff for {nav_state_name(s.nav_state)}',
                                None)
        if self.left_ground is None and (not s.landed or self.altitude_of(s) > 0.3):
            self.left_ground = now
        setpoint = None
        if self.method == 'offboard':
            # The reference runs on its own clock, which stops whenever it is max_lead
            # ahead of the vehicle: PX4 spends about 3 s after arming spooling up and
            # ramping its takeoff, and a reference that ran on would be metres up by
            # then, leaving PX4 to chase it at full speed.
            tau = self.tau + (now - self.last_update)
            self.last_update = now
            z, vz = self.profile.sample(tau)
            if self.limits.max_lead > 0 and (z - s.z) * self.profile.sign > self.limits.max_lead:
                tau = self.tau
                z, vz = self.profile.sample(tau)
            self.tau = tau
            setpoint = Setpoint(self.hold.x, self.hold.y, z, vz, self.hold.yaw)
            self.hold = Setpoint(self.hold.x, self.hold.y, self.target_z(), yaw=self.hold.yaw)
            if self.tau >= self.profile.duration:
                self.phase = 'SETTLE'
        elif s.nav_state == NAV_AUTO_LOITER:
            self.phase = 'SETTLE'
        error = abs(self.altitude_of(s) - self.altitude)
        close = error < self.limits.tolerance and abs(s.vz) < self.limits.settled_speed
        if self.phase == 'SETTLE' and close:
            if self.settled_since is None:
                self.settled_since = now
            if now - self.settled_since >= self.limits.settle_time:
                start = self.left_ground if self.left_ground is not None else self.climb_start
                self.climb_time = self.settled_since - start
                return self._finish(True, f'holding {self.altitude_of(s):.2f} m '
                                    f'({self.climb_time:.1f} s climb)', setpoint and self.hold)
        else:
            self.settled_since = None
        return Output(self.phase, setpoint)

    # ------------------------------------------------------------------- cancel

    def cancel(self, now: float, s: Snapshot) -> Output:
        """Stop climbing and hold the current altitude."""
        if self.method == 'offboard':
            if self.hold is None:
                return self._finish(False, 'canceled before PX4 was ready', None)
            self.hold = Setpoint(self.hold.x, self.hold.y, s.z, yaw=self.hold.yaw)
            out = self._finish(False, f'canceled at {self.altitude_of(s):.2f} m, holding',
                               self.hold if s.armed else None)
            return out
        out = self._finish(False, f'canceled at {self.altitude_of(s):.2f} m, switching to hold',
                           None)
        if s.armed:
            out.commands = (Command.hold(),)
        return out


class Land:
    """Land where the vehicle is, and wait for PX4 to detect it and disarm."""

    def __init__(self, now: float, ground_z: float | None = None,
                 limits: Limits | None = None):
        self.limits = limits or Limits(timeout=120.0)
        self.ground_z = ground_z
        self.start = now
        self.phase = 'DESCEND'
        self.commander = None
        self.touchdown = None
        self.result = None

    def altitude_of(self, s: Snapshot) -> float:
        base = self.ground_z if self.ground_z is not None else s.z
        return base - s.z

    def update(self, now: float, s: Snapshot) -> Output:
        if self.result is not None:
            return self.result
        if now - self.start > self.limits.timeout:
            self.result = Output(self.phase, None, (), False, 'timed out landing')
            return self.result
        if not s.fresh(now, self.limits.telemetry_timeout):
            self.result = Output(self.phase, None, (), False, 'lost PX4 telemetry')
            return self.result
        if not s.armed:
            self.phase = 'DISARMED'
            took = now - self.start
            message = f'landed and disarmed in {took:.1f} s' if took > 0.5 \
                else 'already on the ground and disarmed'
            self.result = Output(self.phase, None, (), s.landed, message)
            return self.result
        if s.landed:
            self.phase = 'TOUCHDOWN'  # PX4 disarms itself after COM_DISARM_LAND
            if self.touchdown is None:
                self.touchdown = now
            return Output(self.phase)
        if s.nav_state == NAV_AUTO_LAND:
            return Output(self.phase)
        if self.commander is None:
            self.commander = _Commander(Command.land(), self.limits)
        command, error = self.commander.poll(now, s.acks)
        if error:
            self.result = Output(self.phase, None, (), False, error)
            return self.result
        return Output(self.phase, None, (command,) if command else ())

    def cancel(self, now: float, s: Snapshot) -> Output:
        self.result = Output(self.phase, None, (Command.hold(),) if s.armed else (), False,
                             f'canceled at {self.altitude_of(s):.2f} m, switching to hold')
        return self.result
