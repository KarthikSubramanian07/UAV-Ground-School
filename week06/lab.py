"""The takeoff lab's simulation, in Python: the reference for the page's JavaScript.

``simulate`` flies the PX4 model (ugs_px4.model) with the action server's logic
(ugs_px4.takeoff) the way the server and fake_px4 run together: logic every
20 ms, setpoints streamed every 5th step (10 Hz), a hover, then the land goal.
site/js/week6.js has a line by line port; tests/test_site.py runs both on the
same scenarios and requires identical traces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "ugs_px4"))

from ugs_px4.model import Px4Model  # noqa: E402
from ugs_px4.px4 import ack_name, command_name  # noqa: E402
from ugs_px4.takeoff import Land, Limits, Takeoff  # noqa: E402

DT = 0.02

SCENARIOS = {
    "takeoff": {},
    "takeoff_mode": {"method": "takeoff_mode"},
    "fast": {"altitude": 20, "climb_rate": 3},
    "cancel": {"altitude": 10, "cancel_at": 12},
    "deny_arm": {"faults": "deny_arm"},
    "no_offboard": {"faults": "no_offboard"},
    "takeover": {"altitude": 10, "faults": "takeover_at=6"},
    "link_drop": {"altitude": 10, "link_drop_at": 6},
    "no_preflight": {"faults": "no_preflight", "until": 70},
}


def simulate(
    altitude=5.0, method="offboard", climb_rate=1.0, faults="", cancel_at=None, link_drop_at=None, hover=3.0, land=True, until=90.0
) -> dict:
    m = Px4Model(faults)
    logic = Takeoff(altitude, 0.0, None, method, Limits(climb_rate=climb_rate))
    stage, hover_until, climb_at = "takeoff", None, None
    trace = {"t": [], "altitude": [], "setpoint": [], "phase": [], "nav": [], "events": [], "result": None, "land": None}
    last_phase, stream, t = None, None, 0.0
    for i in range(round(until / DT)):
        s = m.snapshot(t)
        for _, c, r in s.acks:
            trace["events"].append([t, "ack", f"<- {command_name(c)}: {ack_name(r)}"])
        if stage == "takeoff" and cancel_at is not None and t >= cancel_at and logic.result is None:
            out = logic.cancel(t, s)
        else:
            out = logic.update(t, s)
        for c in out.commands:
            trace["events"].append([t, "cmd", f"-> {command_name(c.command)}"])
            m.vehicle_command(c.command, c.params)
        stream = out.setpoint
        if stage == "takeoff" and out.phase == "CLIMB" and climb_at is None:
            climb_at = t
        dropped = link_drop_at is not None and climb_at is not None and t - climb_at >= link_drop_at
        if stream is not None and i % 5 == 0 and not dropped:
            m.offboard_control_mode()
            m.trajectory_setpoint(stream.z, stream.vz)
        if i % 5 == 0:
            trace["t"].append(t)
            trace["altitude"].append(-m.z)
            trace["setpoint"].append(-stream.z if stream else None)
            trace["phase"].append(out.phase)
            trace["nav"].append(m.nav_state)
        if out.phase != last_phase:
            trace["events"].append([t, "phase", f"[{'land ' if stage == 'land' else ''}{out.phase}]"])
            last_phase = out.phase
        if stage == "takeoff" and out.done is not None:
            trace["result"] = {"t": t, "done": out.done, "message": out.message, "altitude": -m.z}
            if out.done:
                label = "TAKEOFF succeeded: "
            elif cancel_at is not None and out.message.startswith("canceled"):
                label = "TAKEOFF canceled: "
            else:
                label = "TAKEOFF aborted: "
            trace["events"].append([t, "ok" if out.done else "fail", label + out.message])
            if not land or (not out.done and not out.message.startswith("canceled")):
                stage, hover_until = "after", t + 4
            else:
                stage, hover_until = "hover", t + hover
        elif stage == "hover" and t >= hover_until:
            stage, logic = "land", Land(t, 0.0)
            trace["events"].append([t, "cmd", "/land"])
        elif stage == "land" and out.done is not None:
            trace["land"] = {"t": t, "done": out.done, "message": out.message}
            trace["events"].append([t, "ok" if out.done else "fail", ("LAND succeeded: " if out.done else "LAND failed: ") + out.message])
            break
        elif stage == "after" and t >= hover_until:
            break
        m.step(DT)
        t += DT
    return trace
