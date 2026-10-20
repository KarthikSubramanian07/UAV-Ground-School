"""DDS-XRCE encoding and decoding, without an agent."""

import struct

import pytest

from week06 import xrce


def test_object_ids_pack_12_bits_of_number_and_4_of_kind():
    raw = xrce.object_id(0x123, "DATAWRITER")
    assert raw == bytes((0x12, 0x35))
    assert xrce.split_object_id(raw) == (0x123, "DATAWRITER")
    assert xrce.describe_object(xrce.object_id(1, "PARTICIPANT")) == "participant 1"
    with pytest.raises(xrce.XrceError):
        xrce.object_id(4096, "TOPIC")


def test_header_carries_the_client_key_only_below_0x80():
    key = b"\xaa\xbb\xcc\xdd"
    assert xrce.header(0x01, 0x80, 7, key) == b"\x01\x80\x07\x00" + key
    assert xrce.header(0x81, 0x01, 0x1234, key) == b"\x81\x01\x34\x12"
    m = xrce.parse(
        xrce.message(
            xrce.header(0x01, 0x80, 7, key), xrce.submessage(xrce.SUB["HEARTBEAT"], 0, xrce.Writer().u16(3).u16(9).u8(0x80).bytes())
        )
    )
    assert m.client_key == key and m.sequence == 7 and m.reliable
    assert m.submessages[0].name == "HEARTBEAT"
    assert m.submessages[0].body == {"first_unacked": 3, "last_unacked": 9, "stream": 0x80}


def test_create_client_round_trip():
    payload = xrce.create_client_payload(b"\x01\x02\x03\x04", 0x81, mtu=512)
    assert payload[:8] == b"XRCE\x01\x00\x01\x0f"
    m = xrce.parse(xrce.message(xrce.header(0x80, 0, 0, b""), xrce.submessage(xrce.SUB["CREATE_CLIENT"], 0, payload)))
    c = m.submessages[0].body["client"]
    assert c == {"cookie": "XRCE", "version": "0100", "vendor": "010f", "client_key": "01020304", "session_id": 0x81, "mtu": 512}


def _create(kind, number, blob, parent=None):
    raw = xrce.object_id(number, kind)
    payload = xrce.create_payload(42, raw, kind, blob, parent, origin=8)
    msg = xrce.message(xrce.header(0x81, 1, 0, b""), xrce.submessage(xrce.SUB["CREATE"], xrce.CREATE_REUSE, payload))
    return xrce.parse(msg).submessages[0].body


def test_create_topic_and_endpoints_round_trip():
    participant = xrce.object_id(1, "PARTICIPANT")
    topic = _create(
        "TOPIC",
        5,
        xrce.encode_binary("TOPIC", name="rt/fmu/out/vehicle_status_v1", type_name="px4_msgs::msg::dds_::VehicleStatus_"),
        participant,
    )
    assert topic["request_id"] == 42 and topic["reuse"] and not topic["replace"]
    rep = topic["representation"]
    assert rep["kind"] == "TOPIC" and rep["format"] == "binary"
    assert rep["name"] == "rt/fmu/out/vehicle_status_v1"
    assert rep["type_name"] == "px4_msgs::msg::dds_::VehicleStatus_"
    assert rep["parent"] == "participant 1"
    qos = {"reliable": False, "history": "keep_last", "depth": 5}
    reader = _create(
        "DATAREADER", 5, xrce.encode_binary("DATAREADER", topic_id=xrce.object_id(5, "TOPIC"), qos=qos), xrce.object_id(5, "SUBSCRIBER")
    )
    r = reader["representation"]
    assert r["topic"] == "topic 5" and r["parent"] == "subscriber 5"
    assert r["qos"] == {"reliable": False, "history": "keep_last", "durability": "volatile", "depth": 5}
    writer = _create(
        "DATAWRITER",
        7,
        xrce.encode_binary("DATAWRITER", topic_id=xrce.object_id(7, "TOPIC"), qos={"reliable": True, "durability": "transient_local"}),
        xrce.object_id(7, "PUBLISHER"),
    )
    assert writer["representation"]["qos"]["reliable"] is True
    assert writer["representation"]["qos"]["durability"] == "transient_local"


def test_participant_carries_a_domain():
    p = _create("PARTICIPANT", 1, xrce.encode_binary("PARTICIPANT"))
    assert p["representation"]["domain_id"] == 0 and p["representation"]["kind"] == "PARTICIPANT"


def test_submessages_are_padded_to_4_bytes():
    a = xrce.submessage(xrce.SUB["ACKNACK"], 0, xrce.acknack_payload(3, 0x80))  # 4 + 5 bytes
    b = xrce.submessage(xrce.SUB["ACKNACK"], 0, xrce.acknack_payload(4, 0x81))
    msg = xrce.message(xrce.header(0x81, 0, 0, b""), a, b)
    m = xrce.parse(msg)
    assert [s.offset for s in m.submessages] == [4, 16]
    assert [s.body["first_unacked"] for s in m.submessages] == [3, 4]
    assert m.submessages[0].body["nack_bitmap"] == "0000"


def test_read_data_requests_unlimited_samples():
    reader = xrce.object_id(3, "DATAREADER")
    m = xrce.parse(
        xrce.message(xrce.header(0x81, 1, 0, b""), xrce.submessage(xrce.SUB["READ_DATA"], 0, xrce.read_data_payload(9, reader, 1)))
    )
    b = m.submessages[0].body
    assert b["object"] == "datareader 3" and b["preferred_stream"] == 1
    assert b["delivery_control"]["max_samples"] == 0xFFFF


def test_write_data_keeps_the_sample_bytes():
    writer = xrce.object_id(2, "DATAWRITER")
    sample = struct.pack("<Qf", 123456, 1.5)
    m = xrce.parse(xrce.message(xrce.header(0x81, 1, 5, b""), xrce.submessage(xrce.SUB["WRITE_DATA"], 0, b"\x00\x09" + writer + sample)))
    b = m.submessages[0].body
    assert b["data"] == sample and b["object"] == "datawriter 2" and b["request_id"] == 9


def test_truncated_messages_are_errors():
    good = xrce.message(xrce.header(0x81, 1, 0, b""), xrce.submessage(xrce.SUB["HEARTBEAT"], 0, b"\x00" * 5))
    with pytest.raises(xrce.XrceError):
        xrce.parse(good[:-2])
    with pytest.raises(xrce.XrceError):
        xrce.parse(b"\x01\x80\x00")
    with pytest.raises(xrce.XrceError):
        xrce.parse(b"\x01\x80\x00\x00\xaa")  # a key was promised


def test_status_and_timestamps():
    w = xrce.Writer().raw(b"\x00\x2a").raw(xrce.object_id(4, "TOPIC")).u8(0x82).u8(0)
    m = xrce.parse(xrce.message(xrce.header(0x81, 0x80, 0, b""), xrce.submessage(xrce.SUB["STATUS"], 0, w.bytes())))
    assert m.submessages[0].body["status"] == "ERR_ALREADY_EXISTS"
    t = xrce.Writer().raw(struct.pack("<iI", 12, 500_000_000)).bytes()
    m = xrce.parse(xrce.message(xrce.header(0x81, 0, 0, b""), xrce.submessage(xrce.SUB["TIMESTAMP"], 0, t)))
    assert m.submessages[0].body["transmit"] == pytest.approx(12.5)
