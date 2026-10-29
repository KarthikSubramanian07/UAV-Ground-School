"""Interface definitions and REP 2011 type hashes against what ROS 2 Jazzy generated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from week05.idl import Registry, Type, parse_action, parse_message, parse_service, parse_type

DATA = Path(__file__).parent / "data" / "week05"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry()


def test_field_types():
    assert parse_type("int32", "p") == Type("int32")
    assert parse_type("string<=8", "p") == Type("string", string_bound=8)
    assert parse_type("float64[3]", "p") == Type("float64", array="array", size=3)
    assert parse_type("uint8[<=4]", "p") == Type("uint8", array="bounded", size=4)
    assert parse_type("string<=5[]", "p") == Type("string", 5, "sequence")
    assert parse_type("Header", "p").base == "std_msgs/msg/Header"
    assert parse_type("geometry_msgs/Point", "p").base == "geometry_msgs/msg/Point"
    assert parse_type("Mission", "ugs_interfaces").base == "ugs_interfaces/msg/Mission"
    with pytest.raises(ValueError):
        parse_type("int32<=4", "p")


def test_constants_defaults_and_comments():
    msg = parse_message(
        'int32 ANSWER=42\nstring GREETING="hi # not a comment"  # but this is\nfloat64[] seq [1.5, -2.0]  # default\nbool flag true\n',
        "p/msg/M",
    )
    assert [(c.name, c.value) for c in msg.constants] == [("ANSWER", "42"), ("GREETING", '"hi # not a comment"')]
    assert [(f.name, f.default) for f in msg.fields] == [("seq", "[1.5, -2.0]"), ("flag", "true")]


def test_empty_message_gets_a_placeholder():
    msg = parse_message("# nothing here\n", "std_srvs/srv/Trigger_Request")
    assert [f.name for f in msg.fields] == ["structure_needs_at_least_one_member"]


def test_service_and_action_expansion():
    srv = parse_service("int64 a\nint64 b\n---\nint64 sum\n", "example_interfaces/srv/AddTwoInts")
    assert srv.request.name == "example_interfaces/srv/AddTwoInts_Request"
    assert [f.name for f in srv.event.fields] == ["info", "request", "response"]
    act = parse_action(
        "geometry_msgs/Point target\n---\nbool success\n---\nfloat32 distance_remaining\n", "ugs_interfaces/action/FlyToWaypoint"
    )
    assert act.send_goal.request.name == "ugs_interfaces/action/FlyToWaypoint_SendGoal_Request"
    assert [f.name for f in act.send_goal.request.fields] == ["goal_id", "goal"]
    assert [f.name for f in act.get_result.response.fields] == ["status", "result"]
    assert [f.name for f in act.feedback_message.fields] == ["goal_id", "feedback"]
    with pytest.raises(ValueError):
        parse_action("int32 a\n---\nint32 b\n", "p/action/Broken")


def test_known_hashes(registry):
    assert registry.type_hash("std_msgs/msg/String") == "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18"
    # the request and reply topic hashes that rmw_fastrtps announced in the capture
    assert (
        registry.type_hash("example_interfaces/srv/AddTwoInts_Request")
        == "RIHS01_000c5fd92d6b2e1a05949348f584d6d652adea1e92d691792011ac2273508302"
    )
    assert (
        registry.type_hash("example_interfaces/srv/AddTwoInts_Response")
        == "RIHS01_de5c030d4af33cba2749310b249737b631594703f9300495f48bffb2b44dcc2f"
    )


@pytest.mark.skipif(not (DATA / "type_hashes.json").is_file(), reason="fixture not generated")
def test_every_bundled_type_hash_matches_rosidl(registry):
    hashes = json.loads((DATA / "type_hashes.json").read_text())["hashes"]
    assert len(hashes) > 250
    wrong = {name: h for name, h in hashes.items() if registry.type_hash(name) != h}
    assert not wrong


def test_show_expands_nested_types(registry):
    text = registry.show("ugs_interfaces/msg/Mission")
    assert "geometry_msgs/Point[] waypoints" in text
    assert "\tfloat64 x" in text
