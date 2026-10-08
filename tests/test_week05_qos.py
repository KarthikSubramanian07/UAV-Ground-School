"""QoS compatibility, against rclpy.qos.qos_check_compatible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from week05.qos import PRESETS, QoSProfile, check_compatible, matrix

DATA = Path(__file__).parent / "data" / "week05"


def test_the_slides_rules():
    best_effort, reliable = QoSProfile(reliability="best_effort"), QoSProfile(reliability="reliable")
    assert check_compatible(best_effort, reliable) == ("error", "ERROR: Best effort publisher and reliable subscription;")
    assert check_compatible(reliable, best_effort)[0] != "error"  # a warning: liveliness is system_default
    volatile, durable = QoSProfile(durability="volatile"), QoSProfile(durability="transient_local")
    assert check_compatible(volatile, durable)[0] == "error"
    assert check_compatible(durable, volatile)[0] != "error"


def test_px4_odometry_needs_sensor_data():
    # PX4's /fmu/out topics are best effort: a default subscription gets nothing
    assert check_compatible(PRESETS["sensor_data"], PRESETS["default"])[0] == "error"
    assert check_compatible(PRESETS["sensor_data"], PRESETS["sensor_data"])[0] != "error"


def test_deadlines_and_leases():
    assert check_compatible(QoSProfile(), QoSProfile(deadline_ns=10**9))[0] == "error"
    assert check_compatible(QoSProfile(deadline_ns=2 * 10**9), QoSProfile(deadline_ns=10**9))[0] == "error"
    assert "deadline" not in check_compatible(QoSProfile(deadline_ns=10**9), QoSProfile(deadline_ns=2 * 10**9))[1]


def test_matrix_covers_every_pair():
    rows = matrix()
    assert len(rows) == 25
    assert {r["compatibility"] for r in rows} == {"ok", "warning", "error"}


@pytest.mark.skipif(not (DATA / "qos_cases.json").is_file(), reason="fixture not generated")
def test_identical_to_rclpy():
    cases = json.loads((DATA / "qos_cases.json").read_text())["cases"]
    assert len(cases) >= 1000
    for pub, sub, level, reason in cases:
        mine = check_compatible(
            QoSProfile(reliability=pub[0], durability=pub[1], liveliness=pub[2], deadline_ns=pub[3], lease_ns=pub[4]),
            QoSProfile(reliability=sub[0], durability=sub[1], liveliness=sub[2], deadline_ns=sub[3], lease_ns=sub[4]),
        )
        assert mine == (level, reason), (pub, sub)
