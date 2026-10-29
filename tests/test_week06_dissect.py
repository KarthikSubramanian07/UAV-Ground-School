"""The XRCE dissector, on a synthetic session and on the real PX4 capture in docs/week06."""

from pathlib import Path

import pytest

from week05 import cdr
from week05.pcap import Datagram, write
from week06 import dissect, xrce
from week06.docs import trim_capture
from week06.xrce_client import dds_type, registry

KEY = b"\x11\x22\x33\x44"
CAPTURE = Path(__file__).resolve().parents[1] / "docs" / "week06" / "flights" / "offboard_5m.xrce.pcap"


def synthetic(path: Path) -> None:
    grams = []

    def add(t, to_agent, data):
        grams.append(Datagram(t, "127.0.0.1", 50000 if to_agent else 8888, "127.0.0.1", 8888 if to_agent else 50000, data))

    add(0.0, True, xrce.message(xrce.header(0x80, 0, 0, KEY), xrce.submessage(0, 0, xrce.create_client_payload(KEY, 0x81))))
    add(0.01, False, xrce.message(xrce.header(0x81, 0, 0, b""), xrce.submessage(4, 0, b"\x00\x00XRCE\x01\x00\x01\x0f\x00")))
    pid = xrce.object_id(1, "PARTICIPANT")
    add(
        0.02,
        True,
        xrce.message(
            xrce.header(0x81, 0x80, 0, b""),
            xrce.submessage(1, 6, xrce.create_payload(1, pid, "PARTICIPANT", xrce.encode_binary("PARTICIPANT"), origin=8)),
        ),
    )
    tid = xrce.object_id(5, "TOPIC")
    blob = xrce.encode_binary("TOPIC", name="rt/fmu/out/vehicle_status_v1", type_name=dds_type("px4_msgs/msg/VehicleStatus"))
    add(
        0.03,
        True,
        xrce.message(xrce.header(0x81, 0x80, 1, b""), xrce.submessage(1, 6, xrce.create_payload(2, tid, "TOPIC", blob, pid, origin=8))),
    )
    wid = xrce.object_id(5, "DATAWRITER")
    blob = xrce.encode_binary("DATAWRITER", topic_id=tid, qos={"reliable": False, "depth": 1})
    add(
        0.04,
        True,
        xrce.message(
            xrce.header(0x81, 0x80, 2, b""),
            xrce.submessage(1, 6, xrce.create_payload(3, wid, "DATAWRITER", blob, xrce.object_id(5, "PUBLISHER"), origin=8)),
        ),
    )
    add(0.05, False, xrce.message(xrce.header(0x81, 0x80, 0, b""), xrce.submessage(5, 0, b"\x00\x03" + wid + b"\x00\x00")))
    sample = cdr.serialize("px4_msgs/msg/VehicleStatus", {"nav_state": 14, "arming_state": 2}, registry())[4:]
    for i in range(20):
        add(0.1 + i * 0.2, True, xrce.message(xrce.header(0x81, 1, i, b""), xrce.submessage(7, 0, b"\x00\x00" + wid + sample)))
    # a sample split over two FRAGMENTs on the reliable stream
    whole = xrce.submessage(7, 0, b"\x00\x00" + wid + sample)
    add(4.5, True, xrce.message(xrce.header(0x81, 0x80, 3, b""), xrce.submessage(13, 0, whole[:40])))
    add(4.6, True, xrce.message(xrce.header(0x81, 0x80, 4, b""), xrce.submessage(13, xrce.FLAG_LAST_FRAGMENT, whole[40:])))
    add(30.0, True, xrce.message(xrce.header(0x81, 0x80, 5, b""), xrce.submessage(11, 0, xrce.Writer().u16(3).u16(5).u8(0x80).bytes())))
    write(path, grams)


@pytest.fixture()
def capture(tmp_path):
    path = tmp_path / "s.pcap"
    synthetic(path)
    return path


def test_session_reconstructs_topics_and_traffic(capture):
    s = dissect.session(capture)
    assert s["client"]["client_key"] == KEY.hex() and s["client"]["session_id"] == 0x81
    assert s["agent"]["status"] == "OK"
    assert s["participants"] == 1 and s["created"] == 3
    (topic,) = s["topics"]
    assert topic["topic"] == "/fmu/out/vehicle_status_v1"
    assert topic["type"] == "px4_msgs/msg/VehicleStatus"
    assert topic["direction"] == "out" and topic["qos"]["reliable"] is False
    assert topic["samples"] == 21  # 20 plain, 1 reassembled from fragments
    assert topic["created"] == "OK"


def test_dissect_decodes_samples_with_px4_names(capture):
    lines = list(dissect.dissect(capture))
    assert "CREATE_CLIENT key 11223344 session 0x81" in lines[0]
    assert any("CREATE #2 topic 5 rt/fmu/out/vehicle_status_v1 type px4_msgs::msg::dds_::VehicleStatus_" in line for line in lines)
    write_lines = [line for line in lines if "WRITE_DATA" in line]
    assert len(write_lines) == 21 and "arming_state=2" in write_lines[0]
    assert list(dissect.dissect(capture, limit=3)) == lines[:3]


def test_spans_cover_the_message(capture):
    from week05.pcap import read

    for d in read(capture):
        spans = dissect.spans(d.payload)
        assert spans[0]["label"] == "session id"
        assert max(s["end"] for s in spans) <= len(d.payload)
        assert all(s["start"] < s["end"] for s in spans)


def test_trim_keeps_setup_and_a_little_traffic(capture, tmp_path):
    kept = trim_capture(capture, tmp_path / "t.pcap")
    s = dissect.session(tmp_path / "t.pcap")
    assert s["created"] == 3
    assert kept < 26  # the late HEARTBEAT is dropped


def test_type_names():
    assert dissect.ros_type("px4_msgs::msg::dds_::VehicleStatus_") == "px4_msgs/msg/VehicleStatus"
    assert dissect.ros_topic("rt/fmu/in/vehicle_command") == "/fmu/in/vehicle_command"


@pytest.mark.skipif(not CAPTURE.is_file(), reason="no SITL capture committed yet")
def test_real_px4_session():
    """The capture of the 5 m SITL flight: PX4 v1.17 and the v2.4.3 agent, as recorded."""
    s = dissect.session(CAPTURE)
    names = {t["topic"] for t in s["topics"]}
    assert {
        "/fmu/out/vehicle_status_v1",
        "/fmu/out/vehicle_local_position_v1",
        "/fmu/in/vehicle_command",
        "/fmu/in/trajectory_setpoint",
    } <= names
    assert all(t["created"] and t["created"].startswith("OK") for t in s["topics"])
    out = {t["topic"]: t for t in s["topics"] if t["direction"] == "out"}
    assert out["/fmu/out/vehicle_status_v1"]["qos"]["reliable"] is False  # why default subscriptions hear nothing
    # every sample decodes as its px4_msgs type
    lines = [line for line in dissect.dissect(CAPTURE) if "WRITE_DATA" in line]
    assert lines and not any("Error" in line for line in lines)
