"""week06.flight: the takeoff measurements, on a flight recorded from the PX4 model.

The model flight is written in flight_recorder's format, so the analysis is held
to exact answers here; the real SITL flights in docs/week06 are checked too.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "ugs_px4"))

from ugs_px4.model import Px4Model  # noqa: E402
from ugs_px4.takeoff import Land, Takeoff  # noqa: E402

from week06.flight import Flight, analyze, phases, timeline  # noqa: E402

DT = 0.02
FLIGHTS = Path(__file__).resolve().parents[1] / "docs" / "week06" / "flights"


def record_model_flight(altitude=5.0, method="offboard", hover=5.0):
    """Fly the model like the server would and write what flight_recorder would."""
    m = Px4Model()
    t = 0.0
    out_records = [{"t": 0.0, "kind": "start", "monotonic": 100.0}]

    def log(msg, node="takeoff_server"):
        out_records.append({"t": round(t, 4), "kind": "log", "node": node, "level": 20, "msg": msg})

    log(f"/takeoff to {altitude:.1f} m at 1.0 m/s ({method})")
    logic = Takeoff(altitude, t, method=method)
    stream = None
    stage = "takeoff"
    hover_until = None
    for i in range(int(90 / DT)):
        s = m.snapshot(t)
        out = logic.update(t, s)
        for c in out.commands:
            m.vehicle_command(c.command, c.params)
            out_records.append({"t": round(t, 4), "kind": "command", "command": c.command, "params": [0.0] * 7})
        stream = out.setpoint
        if i % 5 == 0:
            if stream is not None:
                m.offboard_control_mode()
                m.trajectory_setpoint(stream.z, stream.vz)
                out_records.append({"t": round(t, 4), "kind": "setpoint", "x": 0.0, "y": 0.0, "z": stream.z, "vz": None})
            out_records.append(
                {
                    "t": round(t, 4),
                    "kind": "feedback",
                    "action": stage,
                    "phase": out.phase,
                    "altitude": -m.z,
                    "vertical_speed": -m.vz,
                    "remaining": None,
                }
            )
            out_records.append(
                {
                    "t": round(t, 4),
                    "kind": "status",
                    "px4_us": int(t * 1e6),
                    "nav_state": m.nav_state,
                    "arming_state": m.arming_state,
                    "preflight_ok": s.preflight_ok,
                    "failsafe": False,
                }
            )
        out_records.append(
            {
                "t": round(t, 4),
                "kind": "position",
                "px4_us": int(t * 1e6),
                "x": 0.0,
                "y": 0.0,
                "z": m.z,
                "vx": 0.0,
                "vy": 0.0,
                "vz": m.vz,
                "xy_valid": s.xy_valid,
                "z_valid": s.z_valid,
                "ref_alt": 488.0,
                "heading": 0.0,
            }
        )
        if out.done is not None and stage == "takeoff":
            log(f"succeeded: {out.message}")
            stage, hover_until = "hover", t + hover  # the logic keeps returning its hold setpoint
        elif stage == "hover" and t >= hover_until:
            log("/land")
            stage, logic = "land", Land(t, ground_z=0.0)
        elif stage == "land" and out.done is not None:
            log(f"succeeded: {out.message}")
            log(f"LAND succeeded: {out.message}", "takeoff_client")
            break
        m.step(DT)
        t += DT
    return out_records


@pytest.fixture(scope="module")
def model_flight():
    return Flight.from_records(record_model_flight())


def test_measures_the_model_takeoff(model_flight):
    r = analyze(model_flight)
    assert r["target"] == 5.0 and r["method"] == "offboard" and r["succeeded"]
    assert r["peak_altitude"] == pytest.approx(5.14, abs=0.05)  # the model lags the profile a little
    assert r["overshoot"] == pytest.approx(r["peak_altitude"] - 5.0, abs=1e-3)
    # the profile: 1 s up to 1 m/s, 4 s at it, 1 s braking; 10 to 90 percent is 4 m at 1 m/s
    assert r["rise_time"] == pytest.approx(4.0, abs=0.5)
    assert r["max_climb_rate"] == pytest.approx(1.0, abs=0.3)  # catching up after the takeoff ramp
    assert r["settling_time"] < 8.0
    assert abs(r["hover_mean_error"]) < 0.05 and r["hover_rms_error"] < 0.05
    assert r["tracking_rms"] < 0.4
    assert r["landed"] and r["land_time"] > 5.0
    assert r["arm_time"] >= 5.0  # PREFLIGHT_DELAY
    assert r["max_drift"] == 0.0


def test_phases_in_order(model_flight):
    names = [p["phase"] for p in phases(model_flight)]
    # ARMING lasts one cycle against the model, too short for 10 Hz feedback to catch
    assert [n for n in names if n != "ARMING"] == ["WAIT_FOR_PX4", "PREFLIGHT", "OFFBOARD", "CLIMB", "SETTLE"]
    assert [p["phase"] for p in phases(model_flight, "land")][-1] in ("TOUCHDOWN", "DISARMED")


def test_takeoff_mode_has_no_setpoints():
    f = Flight.from_records(record_model_flight(method="takeoff_mode"))
    r = analyze(f)
    assert r["method"] == "takeoff_mode" and r["succeeded"]
    assert "tracking_rms" not in r
    assert r["max_climb_rate"] == pytest.approx(1.5, abs=0.1)  # MPC_TKO_SPEED


def test_timeline_for_the_site(model_flight):
    tl = timeline(model_flight, every=0.5)
    assert tl["columns"][:3] == ["t", "altitude", "setpoint"]
    assert tl["rows"][0][0] == 0.0
    assert max(row[1] for row in tl["rows"]) == pytest.approx(5.1, abs=0.1)
    assert any(e["msg"].startswith("/takeoff to") for e in tl["events"])


def test_truth_lines_up_by_monotonic_clock(tmp_path):
    records = record_model_flight(hover=1.0)
    path = tmp_path / "f.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n{broken")
    truth = [
        {"monotonic": 100.0 + r["t"], "kind": "truth", "x": 0.0, "y": 0.0, "z": 0.25 - r["z"] + 0.02}
        for r in records
        if r["kind"] == "position"
    ]
    path.with_suffix(".truth.jsonl").write_text("\n".join(json.dumps(r) for r in truth))
    r = analyze(Flight.load(path))
    assert r["estimate_vs_truth_rms"] == pytest.approx(0.0, abs=0.03)


def test_no_goal_is_an_error():
    with pytest.raises(ValueError):
        analyze(Flight.from_records([{"t": 0.0, "kind": "start", "monotonic": 0.0}]))


@pytest.mark.parametrize("path", sorted(FLIGHTS.glob("*.jsonl")) if FLIGHTS.is_dir() else [], ids=lambda p: p.stem)
def test_committed_sitl_flights_parse(path):
    if path.name.endswith(".truth.jsonl"):
        pytest.skip("truth companion")
    r = analyze(Flight.load(path))
    assert r["target"] > 0
