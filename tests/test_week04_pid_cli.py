"""The PID simulator (and its JavaScript twin), the command line and the docs generator."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from week04 import cli, docs, pid

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ PID ----


def run(g, plant=pid.DEFAULT_PLANT, sc=pid.DEFAULT_SCENARIO):
    return pid.metrics(pid.simulate(g, plant, sc), sc)


def test_slide_characters_behave_like_the_slides():
    p_only, pd, too_much_i, all_three = (run(pid.PRESETS[k]) for k in ("preeti", "preeti_darren", "isabelle_too_much", "all_three"))
    assert pid.damping_ratio(pid.PRESETS["preeti"]) < 0  # P alone with motor lag is unstable
    assert p_only.overshoot > 1.0
    assert pd.overshoot < 1e-3 and pd.settling_time is not None  # critically damped step
    assert abs(pd.disturbance_error) > 0.02  # but the gust leaves an offset
    assert too_much_i.overshoot > 0.5  # Isabelle, please stop
    assert abs(all_three.disturbance_error) < abs(pd.disturbance_error) / 4  # the integral removes the offset


def test_critical_kd_is_the_edge_of_overshoot():
    kd = pid.critical_kd(1.5)
    sc = pid.Scenario(disturbance_at=10.0, duration=2.0)
    plant = pid.Plant(disturbance=0.0)
    assert run(pid.Gains(1.5, 0, kd), plant, sc).overshoot <= 1e-4
    assert run(pid.Gains(1.5, 0, 0.9 * kd), plant, sc).overshoot > 1e-4


def test_linear_model_agrees_with_simulation_on_stability():
    for kp, kd in ((0.5, 0.1), (1.5, 0.27), (3.0, 0.1), (1.5, 0.0)):
        g = pid.Gains(kp, 0.0, kd)
        unstable = pid.damping_ratio(g) < 0
        sc = pid.Scenario(duration=6.0, disturbance_at=10.0)
        tr = pid.simulate(g, pid.Plant(disturbance=0.0), sc)
        late = max(abs(a - sc.target) for a in tr.angle[-400:])  # the last second
        assert (late > 0.05) if unstable else (late < 0.02), (kp, kd, late)


def test_ziegler_nichols_and_tuning_improve_on_nothing():
    ku, tu = pid.ultimate_gain()
    assert ku > 0 and tu > 0
    tuned = pid.tune(iterations=40)
    start = pid.Gains(0.8, 0.8, 0.12)
    assert pid.cost(tuned, pid.Plant(), pid.NOISY) <= pid.cost(start, pid.Plant(), pid.NOISY)


def test_cascade_tuning_respects_firmware_ranges():
    plant = pid.Plant(inertia=0.0148, tau_max=3.456)
    g = pid.tune_cascade(plant)
    for name, (lo, hi) in pid.ARDUPILOT_RATE_RANGES.items():
        assert lo * plant.tau_max * 0.999 <= getattr(g, name) <= hi * plant.tau_max * 1.001


def test_noise_is_deterministic_and_bounded():
    n1, n2 = pid._Noise(7), pid._Noise(7)
    a = [n1() for _ in range(1000)]
    assert a == [n2() for _ in range(1000)]
    assert all(-1 <= v <= 1 for v in a) and abs(sum(a) / len(a)) < 0.05


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_simulator_matches_python_exactly():
    cases = [
        ({"kp": 1.5, "ki": 2.0, "kd": 0.274}, {}, {}),
        (
            {"kp": 0.12, "ki": 0.1, "kd": 0.003, "mode": "cascade", "angle_p": 4.5},
            {"inertia": 0.0148, "tau_max": 3.456},
            {"gyro_noise": 0.03, "angle_noise": 0.002},
        ),
        ({"kp": 1.5, "ki": 10.0, "kd": 0.274}, {"disturbance": 0.2}, {"duration": 2.0}),
    ]
    script = """
const PID = require(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
const out = cases.map(([g, p, s]) => { const tr = PID.simulate(g, p, s); return {angle: tr.angle, command: tr.command, metrics: PID.metrics(tr, s)}; });
process.stdout.write(JSON.stringify(out));
"""
    res = subprocess.run(
        [NODE, "-e", script, str(ROOT / "site" / "js" / "pid.js"), json.dumps(cases)], capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    for (g, p, s), got in zip(cases, js):
        gains = pid.Gains(**{**{"kp": 0.0}, **g})
        plant = pid.Plant(**p)
        sc = pid.Scenario(**s)
        tr = pid.simulate(gains, plant, sc)
        assert got["angle"] == tr.angle  # bit for bit: same operations in the same order
        assert got["command"] == tr.command
        m = pid.metrics(tr, sc).as_dict()
        for k, v in m.items():
            if v is None:
                assert got["metrics"][k] is None
            else:
                assert math.isclose(got["metrics"][k], v, rel_tol=1e-12, abs_tol=1e-15), k


# ------------------------------------------------------------------ CLI ----


def test_cli_commands(tmp_path, capsys):
    assert cli.main(["check"]) == 0
    assert "0 errors" in capsys.readouterr().out
    assert cli.main(["harness"]) == 0
    assert "TX" in capsys.readouterr().out
    assert cli.main(["params", "--out", str(tmp_path / "p.param")]) == 0
    assert (tmp_path / "p.param").read_text().count("\n") > 60
    assert cli.main(["diagram", "--out", str(tmp_path / "w.svg")]) == 0
    assert (tmp_path / "w.svg").read_text().startswith("<svg")
    assert cli.main(["budget"]) == 0
    assert "telemetry" in capsys.readouterr().out
    assert cli.main(["journey"]) == 0
    assert "delivered intact: True" in capsys.readouterr().out
    assert cli.main(["frame", "can", "--message", "0x123", "--values", "0102"]) == 0
    assert "stuff bits" in capsys.readouterr().out
    assert cli.main(["frame", "mavlink", "--message", "HEARTBEAT", "--values", '{"type": 2}']) == 0
    assert capsys.readouterr().out.startswith("fd 05")  # 9 byte payload, trailing zeros trimmed to 5
    for proto in ("crsf", "sbus", "dshot"):
        assert cli.main(["frame", proto]) == 0
    assert cli.main(["pid", "--preset", "preeti_darren"]) == 0
    assert "overshoot" in capsys.readouterr().out
    assert cli.main(["decode", str(ROOT / "docs" / "week04" / "sitl_flight.tlog"), "--top", "3"]) == 0
    assert "0 CRC errors" in capsys.readouterr().out


def test_check_exit_code_reflects_errors(tmp_path, capsys):
    design = json.loads((ROOT / "week04" / "data" / "design.json").read_text())
    next(link for link in design["links"] if link["id"] == "rc")["baud"] = 115200
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(design))
    assert cli.main(["--design", str(path), "check", "--json", str(tmp_path / "r.json")]) == 1
    assert any(f["level"] == "error" for f in json.loads((tmp_path / "r.json").read_text()))


def test_sitl_command_explains_a_missing_build(monkeypatch, capsys):
    from week04 import sitl

    monkeypatch.setattr(sitl, "ardupilot_dir", lambda: None)
    assert cli.main(["sitl"]) == 2
    assert "ARDUPILOT_DIR" in capsys.readouterr().err


# ----------------------------------------------------------------- docs ----


def test_docs_build_without_sitl(tmp_path):
    shutil.copy(ROOT / "docs" / "week04" / "sitl.json", tmp_path / "sitl.json")
    docs.build(tmp_path)
    for name in (
        "wiring.svg",
        "check.json",
        "harness.md",
        "protocol_pro.param",
        "bom.md",
        "journey.json",
        "scope.json",
        "pid.json",
        "pid.svg",
        "site.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["errors"] == 0 and summary["sitl"]["params_accepted"] >= 70
    bom = json.loads((tmp_path / "bom.json").read_text())
    assert bom["total_usd"] == pytest.approx(sum((r["unit_usd"] or 0) * r["count"] for r in bom["rows"]))
    assert all(r["unit_usd"] == 0 for r in bom["rows"] if r["included_in"])


def test_scope_can_stuff_bits_are_where_stuffing_put_them():
    from week04.can import CanFrame

    s = docs.scope()["can"]
    frame = CanFrame(0x1004260A, bytes.fromhex("4fea03d300133e80"), extended=True)
    stuffed = frame.stuffed_bits()
    for i in s["stuff_bits"]:
        assert stuffed[i] != stuffed[i - 1]
        assert len(set(stuffed[i - 5 : i])) == 1  # five equal bits before every stuff bit
    assert s["stuffed_length"] == len(stuffed)


def test_committed_docs_are_current(tmp_path):
    """The docs in the repo must be what the code produces today."""
    shutil.copy(ROOT / "docs" / "week04" / "sitl.json", tmp_path / "sitl.json")
    docs.build(tmp_path)
    for name in ("check.json", "harness.json", "journey.json", "scope.json", "site.json", "bom.json", "wiring.svg", "protocol_pro.param"):
        assert (tmp_path / name).read_text() == (ROOT / "docs" / "week04" / name).read_text(), (
            f"docs/week04/{name} is stale: run python -m week04 docs"
        )
