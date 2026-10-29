"""The week 6 takeoff and landing logic against the PX4 model, no ROS needed.

The logic lives in the ROS package (ros2_ws/src/ugs_px4) so the action server
and these tests run the same code; it imports nothing from ROS.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "ugs_px4"))

from ugs_px4 import px4  # noqa: E402
from ugs_px4.model import COM_OF_LOSS_T, MPC_TKO_SPEED, PREFLIGHT_DELAY, REF_ALT, Px4Model  # noqa: E402
from ugs_px4.takeoff import Land, Limits, Profile, Takeoff  # noqa: E402

DT = 0.02


class Flight:
    """The server's loop, minus ROS: the logic at 50 Hz, setpoints streamed at 10 Hz."""

    def __init__(self, faults="", **limits):
        self.model = Px4Model(faults)
        self.t = 0.0
        self.stream = None
        self.limits = Limits(**limits) if limits else None
        self.log = []  # (t, phase, altitude, setpoint altitude or None)
        self.sent = []  # command numbers, in order

    def run(self, logic, until=120.0, cancel_at=None, stream_ok=lambda t: True):
        out = None
        while self.t < until:
            s = self.model.snapshot(self.t)
            if cancel_at is not None and self.t >= cancel_at and logic.result is None:
                out = logic.cancel(self.t, s)
            else:
                out = logic.update(self.t, s)
            for c in out.commands:
                self.sent.append(c.command)
                self.model.vehicle_command(c.command, c.params)
            self.stream = out.setpoint
            self.log.append((self.t, out.phase, -self.model.z, None if out.setpoint is None else -out.setpoint.z))
            if self.stream is not None and round(self.t / DT) % 5 == 0 and stream_ok(self.t):
                self.model.offboard_control_mode()
                self.model.trajectory_setpoint(self.stream.z, self.stream.vz)
            if out.done is not None:
                return out
            self.model.step(DT)
            self.t += DT
        return out

    def hold(self, seconds):
        end = self.t + seconds
        while self.t < end:
            if self.stream is not None and round(self.t / DT) % 5 == 0:
                self.model.offboard_control_mode()
                self.model.trajectory_setpoint(self.stream.z, self.stream.vz)
            self.model.step(DT)
            self.t += DT

    @property
    def altitude(self):
        return -self.model.z


def takeoff(flight, altitude=5.0, method="offboard"):
    return Takeoff(altitude, flight.t, method=method, limits=flight.limits)


# ------------------------------------------------------------------- profile


def test_profile_is_trapezoidal_and_ends_at_rest():
    p = Profile(0.0, -5.0, 1.0, 1.0)  # NED: up is negative
    assert p.duration == pytest.approx(6.0)  # 1 s up to speed, 4 s cruise, 1 s braking
    assert p.sample(0.5) == pytest.approx((-0.125, -0.5))
    assert p.sample(3.0) == pytest.approx((-2.5, -1.0))
    assert p.sample(6.0) == (-5.0, 0.0)
    assert p.sample(-1.0) == (0.0, 0.0)


def test_profile_short_move_is_a_triangle():
    p = Profile(0.0, 1.0, 3.0, 1.0)
    assert p.v == pytest.approx(1.0)  # sqrt(d a): never reaches 3 m/s
    assert p.duration == pytest.approx(2.0)
    assert p.sample(1.0) == pytest.approx((0.5, 1.0))


def test_profile_is_continuous():
    p = Profile(-2.0, -12.0, 2.0, 1.0)
    previous = p.sample(0.0)[0]
    for i in range(1, 1000):
        z, _ = p.sample(i * p.duration / 1000)
        assert abs(z - previous) <= 2.0 * p.duration / 1000 + 1e-9
        previous = z


# ------------------------------------------------------------------ takeoff


def test_offboard_takeoff_reaches_5_m_and_holds():
    f = Flight()
    out = f.run(takeoff(f))
    assert out.done is True, out.message
    assert f.altitude == pytest.approx(5.0, abs=0.15)
    phases = [phase for _, phase, _, _ in f.log]
    order = ["WAIT_FOR_PX4", "PREFLIGHT", "OFFBOARD", "ARMING", "CLIMB", "SETTLE"]
    assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == order
    # offboard was requested before arming, as PX4's offboard example does
    assert f.sent.index(px4.CMD_DO_SET_MODE) < f.sent.index(px4.CMD_COMPONENT_ARM_DISARM)
    f.hold(10.0)  # the server keeps streaming: PX4 stays in offboard at 5 m
    assert f.model.nav_state == px4.NAV_OFFBOARD
    assert f.altitude == pytest.approx(5.0, abs=0.05)


def test_offboard_not_requested_before_setpoints_stream():
    f = Flight()
    f.run(takeoff(f))
    first = next(t for t, phase, _, _ in f.log if phase == "OFFBOARD")
    switch = next(t for t, phase, _, sp in f.log if phase == "ARMING")
    assert switch - first >= 1.0
    assert all(sp is not None for t, phase, _, sp in f.log if phase in ("PREFLIGHT", "OFFBOARD"))


def test_offboard_climb_tracks_the_profile_without_overshooting():
    f = Flight()
    f.run(takeoff(f))
    climb = [(alt, sp) for _, phase, alt, sp in f.log if phase in ("CLIMB", "SETTLE")]
    assert max(alt for alt, _ in climb) < 5.0 + 0.25  # PX4 lags the profile a little
    # the reference never runs more than max_lead (0.4 m) ahead, even while PX4 spools up
    assert max(sp - alt for alt, sp in climb) <= 0.4 + 0.05
    airborne = [(alt, sp) for alt, sp in climb if alt > 0.3]
    assert max(abs(alt - sp) for alt, sp in airborne) < 0.45


def test_reference_waits_for_px4s_takeoff_ramp():
    f = Flight()
    f.run(takeoff(f))
    arm = next(t for t, phase, _, _ in f.log if phase == "CLIMB")
    liftoff = next(t for t, _, alt, _ in f.log if alt > 0.1)
    assert 2.5 < liftoff - arm < 3.6  # spool up and ramp, as in SITL (about 3 s)
    held = [sp for t, phase, alt, sp in f.log if phase == "CLIMB" and t < liftoff]
    assert max(held) <= 0.6  # not metres up while PX4 is still on the ground


def test_climb_rate_is_respected():
    f = Flight(climb_rate=2.0)
    out = f.run(takeoff(f, 20.0))
    assert out.done is True
    sps = [sp for _, phase, _, sp in f.log if phase == "CLIMB"]
    rates = [(b - a) / DT for a, b in zip(sps, sps[1:])]
    assert max(rates) == pytest.approx(2.0, abs=0.05)


def test_takeoff_mode_uses_px4s_own_climb():
    f = Flight()
    out = f.run(takeoff(f, method="takeoff_mode"))
    assert out.done is True, out.message
    assert f.model.nav_state == px4.NAV_AUTO_LOITER
    assert px4.CMD_NAV_TAKEOFF in f.sent
    assert all(sp is None for _, _, _, sp in f.log)  # no offboard stream at all
    climb_speeds = []
    previous = None
    for _, phase, alt, _ in f.log:
        if previous is not None and phase == "CLIMB":
            climb_speeds.append((alt - previous) / DT)
        previous = alt
    assert max(climb_speeds) <= MPC_TKO_SPEED + 0.05


def test_takeoff_mode_sends_an_amsl_altitude():
    f = Flight()
    logic = takeoff(f, 7.0, method="takeoff_mode")
    commands = []
    while not commands:
        out = logic.update(f.t, f.model.snapshot(f.t))
        commands = [c for c in out.commands if c.command == px4.CMD_NAV_TAKEOFF]
        f.model.step(DT)
        f.t += DT
    assert commands[0].params[6] == pytest.approx(REF_ALT + 7.0)
    assert all(math.isnan(p) for p in commands[0].params[3:6])  # here, keep the heading


def test_waits_for_preflight_checks():
    f = Flight()
    f.run(takeoff(f))
    arm = next(t for t, phase, _, _ in f.log if phase == "ARMING")
    assert arm >= PREFLIGHT_DELAY


def test_goal_while_flying_changes_altitude():
    f = Flight()
    assert f.run(takeoff(f, 5.0)).done is True
    f.hold(2.0)
    ground = 0.0
    out = f.run(Takeoff(12.0, f.t, ground_z=ground, limits=f.limits))
    assert out.done is True, out.message
    assert f.altitude == pytest.approx(12.0, abs=0.15)
    assert f.sent.count(px4.CMD_COMPONENT_ARM_DISARM) == 1  # armed once, not again
    out = f.run(Takeoff(3.0, f.t, ground_z=ground, limits=f.limits))  # and down
    assert out.done is True and f.altitude == pytest.approx(3.0, abs=0.15)


# ------------------------------------------------------------- failure paths


def test_arming_refused_aborts_with_the_reason():
    f = Flight("deny_arm")
    out = f.run(takeoff(f))
    assert out.done is False
    assert "COMPONENT_ARM_DISARM after 5 tries (last ack DENIED)" in out.message
    assert f.sent.count(px4.CMD_COMPONENT_ARM_DISARM) == 5
    assert f.altitude == 0.0


def test_offboard_refused_aborts():
    f = Flight("no_offboard")
    out = f.run(takeoff(f))
    assert out.done is False and "DO_SET_MODE" in out.message
    assert "TEMPORARILY_REJECTED" in out.message


def test_preflight_never_passing_times_out_early():
    f = Flight("no_preflight", preflight_timeout=20.0)
    out = f.run(takeoff(f))
    assert out.done is False and "preflight checks did not pass within 20 s" in out.message
    assert f.t < 30.0


def test_pilot_takeover_aborts_instead_of_fighting():
    f = Flight("takeover_at=2")
    out = f.run(takeoff(f, 10.0))
    assert out.done is False and "left offboard for POSCTL" in out.message
    assert out.setpoint is None  # stop streaming: the pilot has it


def test_offboard_loss_is_a_failsafe_the_server_reports():
    f = Flight()
    climb = {}

    def stream_ok(t):  # the link drops 3 s into the climb
        if "start" not in climb and f.log and f.log[-1][1] == "CLIMB":
            climb["start"] = t
        return "start" not in climb or t - climb["start"] < 3.0

    out = f.run(takeoff(f, 10.0), stream_ok=stream_ok)
    assert out.done is False and "left offboard for AUTO_LOITER" in out.message
    lost = climb["start"] + 3.0
    assert f.t - lost == pytest.approx(COM_OF_LOSS_T, abs=0.25)


def test_lost_telemetry_aborts():
    f = Flight()
    logic = takeoff(f)
    f.run(logic, until=7.0)
    stale = f.model.snapshot(f.t)
    out = logic.update(f.t + 3.0, stale)  # no position for 3 s
    assert out.done is False and out.message == "lost PX4 telemetry"


def test_timeout():
    f = Flight("no_preflight", timeout=15.0)
    out = f.run(takeoff(f))
    assert out.done is False and "timed out after 15 s in PREFLIGHT" in out.message


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        Takeoff(5.0, 0.0, method="teleport")


# --------------------------------------------------------------------- cancel


def test_cancel_holds_where_it_is():
    f = Flight()
    out = f.run(takeoff(f, 10.0), cancel_at=12.0)
    assert out.done is False and out.message.startswith("canceled at")
    held = f.altitude
    assert 1.0 < held < 9.0
    f.hold(5.0)
    assert f.altitude == pytest.approx(held, abs=0.5)  # braked, then held
    assert f.model.nav_state == px4.NAV_OFFBOARD


def test_cancel_in_takeoff_mode_switches_to_hold():
    f = Flight()
    out = f.run(takeoff(f, 10.0, method="takeoff_mode"), cancel_at=8.0)
    assert out.done is False
    assert [c.command for c in out.commands] == [px4.CMD_DO_SET_MODE]
    f.hold(0.1)
    assert f.model.nav_state == px4.NAV_AUTO_LOITER


# ----------------------------------------------------------------------- land


def test_land_waits_for_touchdown_and_disarm():
    f = Flight()
    f.run(takeoff(f))
    f.hold(2.0)
    out = f.run(Land(f.t, ground_z=0.0))
    assert out.done is True, out.message
    assert f.altitude == 0.0
    assert f.model.arming_state == px4.ARMING_STATE_DISARMED
    phases = {phase for _, phase, _, _ in f.log}
    assert {"DESCEND", "TOUCHDOWN"} <= phases


def test_land_on_the_ground_is_immediate():
    f = Flight()
    f.hold(4.0)
    out = Land(f.t).update(f.t, f.model.snapshot(f.t))
    assert out.done is True and out.message == "already on the ground and disarmed"


def test_cancel_landing_switches_to_hold():
    f = Flight()
    f.run(takeoff(f, 8.0))
    out = f.run(Land(f.t, ground_z=0.0), cancel_at=f.t + 3.0)
    assert out.done is False and [c.command for c in out.commands] == [px4.CMD_DO_SET_MODE]
    f.model.vehicle_command(out.commands[0].command, out.commands[0].params)
    f.hold(3.0)
    assert f.altitude > 3.0 and f.model.nav_state == px4.NAV_AUTO_LOITER


# ---------------------------------------------------------------------- model


def test_model_refuses_offboard_without_a_stream():
    m = Px4Model()
    for _ in range(int(6 / DT)):
        m.step(DT)
    m.vehicle_command(px4.CMD_DO_SET_MODE, px4.Command.offboard().params)
    assert m.acks[-1][2] == 1  # TEMPORARILY_REJECTED
    assert m.nav_state != px4.NAV_OFFBOARD


def test_model_disarms_if_it_never_takes_off():
    m = Px4Model()
    for _ in range(int(6 / DT)):
        m.step(DT)
    m.vehicle_command(px4.CMD_COMPONENT_ARM_DISARM, px4.Command.arm().params)
    assert m.arming_state == px4.ARMING_STATE_ARMED
    for _ in range(int(11 / DT)):
        m.step(DT)
    assert m.arming_state == px4.ARMING_STATE_DISARMED  # COM_DISARM_PRFLT


def test_names():
    assert px4.nav_state_name(14) == "OFFBOARD"
    assert px4.nav_state_name(99) == "NAV_STATE_99"
    assert px4.ack_name(2) == "DENIED"
    assert str(px4.Command.land()) == "NAV_LAND"


def test_quiet_status_is_not_lost_telemetry():
    """VehicleStatus may go quiet for seconds in SITL; the 50 Hz position is the heartbeat."""
    f = Flight()
    logic = takeoff(f)
    f.run(logic, until=7.0)
    s = f.model.snapshot(f.t)
    s.status_time = f.t - 3.0  # status 3 s old, position current
    assert logic.update(f.t, s).message != "lost PX4 telemetry"
    s.status_time = f.t - 6.0
    assert logic.update(f.t, s).message == "lost PX4 telemetry"
