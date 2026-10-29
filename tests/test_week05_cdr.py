"""CDR serialization, byte for byte against rclpy.serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from week05 import cdr
from week05.idl import Registry

DATA = Path(__file__).parent / "data" / "week05"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry()


def _from_json(value):
    if isinstance(value, dict) and set(value) == {"bytes"}:
        return bytes.fromhex(value["bytes"])
    if isinstance(value, dict):
        return {k: _from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    return value


def test_string_layout():
    payload = cdr.serialize("std_msgs/msg/String", {"data": "Hello World: 0"})
    assert payload.hex() == "000100000f00000048656c6c6f20576f726c643a203000"
    assert cdr.deserialize("std_msgs/msg/String", payload) == {"data": "Hello World: 0"}


def test_alignment_is_counted_after_the_header():
    # int64 fields align to 8 from the start of the body, not of the header
    payload = cdr.serialize("example_interfaces/srv/AddTwoInts_Request", {"a": 2, "b": 3})
    assert payload.hex() == "0001000002000000000000000300000000000000"


def test_padding_on_the_wire_is_tolerated():
    padded = cdr.serialize("std_msgs/msg/String", {"data": "Hello World: 7"}, pad=True)
    assert len(padded) % 4 == 0
    assert cdr.deserialize("std_msgs/msg/String", padded) == {"data": "Hello World: 7"}


def test_big_endian_round_trip(registry):
    value = {"x": 1.5, "y": -2.0, "z": 5.0}
    payload = cdr.serialize("geometry_msgs/msg/Point", value, registry, big_endian=True)
    assert payload[:2] == b"\x00\x00"
    assert cdr.deserialize("geometry_msgs/msg/Point", payload, registry) == value


def test_defaults_fill_missing_fields(registry):
    payload = cdr.serialize("geometry_msgs/msg/Point", {"x": 1.0}, registry)
    assert cdr.deserialize("geometry_msgs/msg/Point", payload, registry) == {"x": 1.0, "y": 0.0, "z": 0.0}
    defaults = cdr.message_defaults(registry.message("ugs_interfaces/msg/AllTypes"), registry)
    assert defaults["with_default"] == 7 and defaults["seq_default"] == [1.5, -2.0]


@pytest.mark.parametrize(
    "value, message",
    [
        ({"i8": 200}, "out of range"),
        ({"u32": -1}, "out of range"),
        ({"bounded_s": "too long for eight"}, "bound"),
        ({"fixed_i32": [1, 2]}, "exactly 3"),
        ({"bounded_f32": [1.0] * 5}, "at most 4"),
        ({"nope": 1}, "no field"),
        ({"i32": "7"}, "needs an int"),
    ],
)
def test_bad_values_are_rejected(registry, value, message):
    with pytest.raises(cdr.CdrError, match=message):
        cdr.serialize("ugs_interfaces/msg/AllTypes", value, registry)


def test_truncated_payload_is_an_error():
    payload = cdr.serialize("std_msgs/msg/String", {"data": "hello"})
    with pytest.raises(cdr.CdrError):
        cdr.deserialize("std_msgs/msg/String", payload[:-3])


@pytest.mark.skipif(not (DATA / "cdr_cases.json").is_file(), reason="fixture not generated")
def test_identical_to_rclpy(registry):
    cases = json.loads((DATA / "cdr_cases.json").read_text())["cases"]
    assert len(cases) >= 60
    tails = 0
    for case in cases:
        value = _from_json(case["value"])
        theirs = bytes.fromhex(case["cdr"])
        ours, padding = cdr.serialize_with_padding(case["type"], value, registry)
        # rclpy returns rmw_fastrtps's size *estimate* as the length, which can run past
        # the end of the CDR it wrote; and Fast CDR leaves alignment padding unwritten.
        assert len(theirs) >= len(ours), case["type"]
        assert cdr.same_except_padding(ours, theirs[: len(ours)], padding), case["type"]
        tails += len(theirs) > len(ours)
        decoded = cdr.deserialize(case["type"], theirs[: len(ours)], registry)
        assert cdr.serialize(case["type"], decoded, registry) == ours
    assert tails > 0  # the estimate really does overshoot for AllTypes
    # the simple types have no padding and match exactly
    simple = [c for c in cases if c["type"] != "ugs_interfaces/msg/AllTypes"]
    assert all(cdr.serialize(c["type"], _from_json(c["value"]), registry).hex() == c["cdr"] for c in simple)
