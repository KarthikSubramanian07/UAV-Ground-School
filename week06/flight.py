"""Read a recorded flight (flight_recorder's JSON lines) and measure the takeoff.

The numbers a controls engineer would ask of a step: when PX4 was ready, how
long arming took, rise time (10 to 90 percent), peak climb rate, overshoot,
settling time into a band around the goal, hover error once settled, how far
the vehicle drifted sideways on the way up, and how closely PX4 followed the
server's setpoints. Gazebo's true pose, when recorded, gives the error of PX4's
own altitude estimate as well.

Altitudes are metres above the takeoff point, up positive (PX4 reports NED, z
down, so they are negated here).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

NAV_OFFBOARD = 14
BAND = 0.25  # m: "settled" means within this of the goal from then on (5 percent of 5 m)


@dataclass
class Flight:
    """A flight's samples, grouped by kind and sorted by time."""

    records: list[dict]
    kinds: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Flight:
        """Read NAME.jsonl, and NAME.truth.jsonl (Gazebo's true pose) if it is there."""
        path = Path(path)
        records = _lines(path)
        truth = path.with_suffix(".truth.jsonl")
        start = next((r for r in records if r["kind"] == "start"), None)
        if truth.is_file() and start is not None:
            for r in _lines(truth):
                r["t"] = round(r.pop("monotonic") - start["monotonic"], 4)
                records.append(r)
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict], sim_time: bool = True) -> Flight:
        records = sorted(records, key=lambda r: r["t"])
        if sim_time:
            records = to_sim_time(records)
        kinds: dict[str, list[dict]] = {}
        for r in records:
            kinds.setdefault(r["kind"], []).append(r)
        return cls(records, kinds)

    def of(self, kind: str) -> list[dict]:
        return self.kinds.get(kind, [])

    def logs(self, text: str) -> list[dict]:
        return [r for r in self.of("log") if text in r["msg"]]

    def first_log(self, text: str, exact: bool = False) -> dict | None:
        found = [r for r in self.of("log") if r["msg"] == text] if exact else self.logs(text)
        return found[0] if found else None


def to_sim_time(records: list[dict]) -> list[dict]:
    """Restamp every record in PX4's clock (simulated seconds) instead of the recorder's.

    PX4 SITL runs in lockstep with Gazebo, so when the host is loaded the whole
    simulation slows down (the M1 went to half real time during some flights):
    the vehicle's physics stay right, but wall clock durations stretch. Each
    vehicle_local_position carries PX4's timestamp; the recorder's clock is mapped
    onto it piecewise linearly, and every record (setpoints, feedback, logs,
    Gazebo's pose) is restamped. Outside the samples time runs at real time.
    """
    pairs = [(r["t"], r["px4_us"] * 1e-6) for r in records if r["kind"] == "position" and r.get("px4_us")]
    pairs = [p for i, p in enumerate(pairs) if i == 0 or (p[0] > pairs[i - 1][0] and p[1] > pairs[i - 1][1])]
    if len(pairs) < 10:
        return records
    (w0, s0), (w1, s1) = pairs[0], pairs[-1]
    out = []
    for r in records:
        w = r["t"]
        if w <= w0:
            t = w
        elif w >= w1:
            t = w0 + (s1 - s0) + (w - w1)
        else:
            t = w0 + (_interp(pairs, w) - s0)
        out.append({**r, "t": round(t, 4), "wall_t": w})
    return out


def _lines(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short when the recorder was stopped
    return records


def _interp(series: list[tuple[float, float]], t: float) -> float:
    if not series:
        return math.nan
    if t <= series[0][0]:
        return series[0][1]
    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, v0), (t1, v1) = series[lo], series[hi]
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v0


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else math.nan


def _round(value, digits=3):
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def phases(flight: Flight, action: str = "takeoff") -> list[dict]:
    """[{phase, start, end}] from the action feedback."""
    out: list[dict] = []
    for r in flight.of("feedback"):
        if r["action"] != action:
            continue
        if not out or out[-1]["phase"] != r["phase"]:
            if out:
                out[-1]["end"] = r["t"]
            out.append({"phase": r["phase"], "start": r["t"], "end": r["t"]})
        else:
            out[-1]["end"] = r["t"]
    return out


def ground_z(flight: Flight) -> float:
    """PX4's z (NED) on the ground before the takeoff."""
    positions = [r for r in flight.of("position") if r["z"] is not None and r.get("z_valid")]
    arm = arm_time(flight)
    before = [r["z"] for r in positions if arm is None or r["t"] <= arm]
    if not before:
        before = [r["z"] for r in positions[:20]]
    tail = before[-25:]
    return sum(tail) / len(tail) if tail else 0.0


def arm_time(flight: Flight) -> float | None:
    for r in flight.of("status"):
        if r["arming_state"] == 2:
            return r["t"]
    return None


def altitude_series(flight: Flight) -> list[tuple[float, float]]:
    z0 = ground_z(flight)
    return [(r["t"], z0 - r["z"]) for r in flight.of("position") if r["z"] is not None]


def setpoint_series(flight: Flight) -> list[tuple[float, float]]:
    z0 = ground_z(flight)
    return [(r["t"], z0 - r["z"]) for r in flight.of("setpoint") if r["z"] is not None]


def takeoff_goal(flight: Flight) -> tuple[float, float, str] | None:
    """(time, altitude, method) of the takeoff goal, from the server's log."""
    r = flight.first_log("/takeoff to ")
    if r is None:
        return None
    words = r["msg"].split()
    altitude = float(words[2])
    method = words[-1].strip("()")
    return r["t"], altitude, method


def analyze(flight: Flight) -> dict:
    """Every number on the week 6 page for one flight."""
    goal = takeoff_goal(flight)
    if goal is None:
        raise ValueError("no /takeoff goal in this flight")
    t_goal, target, method = goal
    alt = altitude_series(flight)
    sp = setpoint_series(flight)
    statuses = flight.of("status")
    t_arm = arm_time(flight)
    out: dict = {"target": target, "method": method, "goal_time": t_goal}
    first_status = statuses[0]["t"] if statuses else None
    out["px4_first_heard"] = first_status
    ready = next((r["t"] for r in statuses if r["preflight_ok"]), None)
    out["preflight_ok_at"] = ready
    out["arm_time"] = t_arm
    offboard = next((r["t"] for r in statuses if r["nav_state"] == NAV_OFFBOARD), None)
    out["offboard_at"] = offboard
    takeoff_result = flight.first_log("succeeded: holding") or flight.first_log("aborted:") or flight.first_log("canceled at")
    out["result"] = takeoff_result["msg"] if takeoff_result else None
    out["succeeded"] = bool(takeoff_result and takeoff_result["msg"].startswith("succeeded"))
    t_done = takeoff_result["t"] if takeoff_result else (alt[-1][0] if alt else t_goal)
    out["result_time"] = t_done
    climb = [(t, a) for t, a in alt if t_arm is not None and t >= t_arm and t <= t_done + 10.0]
    if climb and t_arm is not None:
        liftoff = next((t for t, a in climb if a > 0.1), None)
        out["liftoff"] = liftoff
        t10 = next((t for t, a in climb if a >= 0.1 * target), None)
        t90 = next((t for t, a in climb if a >= 0.9 * target), None)
        out["rise_time"] = t90 - t10 if t10 is not None and t90 is not None else None
        peak = max(a for t, a in climb)
        out["peak_altitude"] = peak
        out["overshoot"] = max(0.0, peak - target)
        out["overshoot_percent"] = 100.0 * max(0.0, peak - target) / target
        rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(climb, climb[1:]) if b[0] - a[0] > 1e-3]
        # smooth over 0.2 s so EKF noise does not read as speed
        smooth = []
        for i in range(len(climb)):
            j = i
            while j < len(climb) - 1 and climb[j][0] - climb[i][0] < 0.2:
                j += 1
            if climb[j][0] > climb[i][0]:
                smooth.append((climb[j][1] - climb[i][1]) / (climb[j][0] - climb[i][0]))
        out["max_climb_rate"] = max(smooth) if smooth else (max(rates) if rates else None)
        # settled: the first time after which the altitude never leaves the band (until the result)
        settled = None
        for t, a in reversed([(t, a) for t, a in climb if t <= t_done]):
            if abs(a - target) > BAND:
                break
            settled = t
        out["settling_time"] = settled - liftoff if settled is not None and liftoff is not None else None
        out["arm_to_liftoff"] = liftoff - t_arm if liftoff is not None else None
    hover_end = flight.first_log("/land", exact=True)
    t_hover_end = hover_end["t"] if hover_end else (alt[-1][0] if alt else t_done)
    # hover error only means something once the goal has been reached
    hover = [a - target for t, a in alt if t_done <= t <= t_hover_end] if out["succeeded"] else []
    out["hover_seconds"] = t_hover_end - t_done
    out["hover_rms_error"] = _rms(hover)
    out["hover_mean_error"] = sum(hover) / len(hover) if hover else None
    if out["result"] and "canceled" in out["result"] and alt:
        at_cancel = _interp(alt, t_done)
        after = [a for t, a in alt if t_done <= t <= t_hover_end]
        held = [a - at_cancel for t, a in alt if t_done + 2.0 <= t <= t_hover_end]
        out["cancel_altitude"] = at_cancel
        out["cancel_coast"] = max(after) - at_cancel if after else None  # braking after the cancel
        out["cancel_hold_rms"] = _rms(held)
    positions = [r for r in flight.of("position") if r["x"] is not None and (t_arm is None or r["t"] >= t_arm) and r["t"] <= t_hover_end]
    if positions:
        x0, y0 = positions[0]["x"], positions[0]["y"]
        out["max_drift"] = max(math.hypot(r["x"] - x0, r["y"] - y0) for r in positions)
    if sp and method == "offboard":
        climbing = [(t, a) for t, a in alt if t_arm is not None and t_arm <= t <= t_done]
        errors = [a - _interp(sp, t) for t, a in climbing if sp[0][0] <= t]
        out["tracking_rms"] = _rms(errors)
        out["tracking_max"] = max((abs(e) for e in errors), default=None)
    landed = flight.first_log("LAND succeeded") or flight.first_log("succeeded: landed")
    out["landed"] = landed is not None
    land_start = flight.first_log("/land", exact=True)
    if landed and land_start:
        out["land_time"] = landed["t"] - land_start["t"]
    commands = flight.of("command")
    out["commands"] = [{"t": r["t"], "command": r["command"]} for r in commands]
    acks = flight.of("ack")
    out["acks"] = [{"t": r["t"], "command": r["command"], "result": r["result"]} for r in acks]
    truth = flight.of("truth")
    if truth and alt:
        tz0 = truth[0]["z"]
        errors = [(_interp(alt, r["t"]) - (r["z"] - tz0)) for r in truth if alt[0][0] <= r["t"] <= alt[-1][0]]
        out["estimate_vs_truth_rms"] = _rms(errors)
        out["estimate_vs_truth_max"] = max((abs(e) for e in errors), default=None)
        hover_truth = [r["z"] - tz0 for r in truth if t_done <= r["t"] <= t_hover_end]
        if hover_truth:
            out["hover_truth_mean"] = sum(hover_truth) / len(hover_truth)  # where the drone really was
    return {k: _round(v) for k, v in out.items()}


def timeline(flight: Flight, every: float = 0.1) -> dict:
    """Downsampled series for the site's replay: altitude, setpoint, speed, phase."""
    alt = altitude_series(flight)
    sp = setpoint_series(flight)
    if not alt:
        return {}
    t0, t1 = alt[0][0], alt[-1][0]
    fb = phases(flight)
    lands = phases(flight, "land")
    nav = [(r["t"], r["nav_state"], r["arming_state"]) for r in flight.of("status")]
    truth_rows = flight.of("truth")
    truth = [(r["t"], r["z"] - truth_rows[0]["z"]) for r in truth_rows] if truth_rows else []
    rows = []
    t = t0
    while t <= t1:
        phase = next((p["phase"] for p in reversed(fb) if p["start"] <= t), None)
        land = next((p["phase"] for p in reversed(lands) if p["start"] <= t), None)
        state = next(((n, a) for tt, n, a in reversed(nav) if tt <= t), (None, None))
        a = _interp(alt, t)
        a2 = _interp(alt, t + every)
        rows.append(
            [
                round(t - t0, 2),
                round(a, 3),
                round(_interp(sp, t), 3) if sp and t >= sp[0][0] else None,
                round((a2 - a) / every, 3),
                land or phase,
                state[0],
                state[1],
                round(_interp(truth, t), 3) if truth and truth[0][0] <= t <= truth[-1][0] else None,
            ]
        )
        t += every
    events = [
        {"t": round(r["t"] - t0, 2), "node": r["node"], "msg": r["msg"]}
        for r in flight.of("log")
        if r["node"] in ("takeoff_server", "takeoff_client") and r["t"] >= t0 and not r["msg"].startswith("feedback:")
    ]
    columns = ["t", "altitude", "setpoint", "climb_rate", "phase", "nav_state", "arming_state", "truth"]
    return {"columns": columns, "rows": rows, "events": events}
