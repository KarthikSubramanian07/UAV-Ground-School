"""CAN, DroneCAN, MAVLink, RTCM and the RTK journey, checked against the reference libraries where they are installed."""

import json
import os
import random
from pathlib import Path

import pytest

from week04 import can, dronecan, journey, links, mavlink, rtcm

DOCS = Path(__file__).resolve().parents[1] / "docs" / "week04"

# ------------------------------------------------------------------ CAN ----


def _random_frames(rng, n):
    frames = []
    for _ in range(n):
        ext = rng.random() < 0.5
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(9)))
        frames.append(can.CanFrame(rng.randrange(1 << (29 if ext else 11)), data, ext))
    return frames


def test_can_round_trip_and_stuffing_rule():
    rng = random.Random(7)
    for f in _random_frames(rng, 300):
        wire = f.wire_bits()
        stuffed = f.stuffed_bits()
        run, prev = 0, None
        for b in stuffed:
            run = run + 1 if b == prev else 1
            prev = b
            assert run <= 5, "six equal bits inside the stuffed region"
        assert can.decode(wire) == f
        assert len(wire) <= can.worst_case_bits(len(f.data), f.extended) + 3


def test_can_worst_case_is_reached_by_adversarial_data():
    zeros = can.CanFrame(0, bytes(8))
    assert len(zeros.wire_bits()) > len(can.CanFrame(0x555, bytes([0x55] * 8)).wire_bits())


def test_can_detects_stuff_and_crc_errors_and_truncation():
    f = can.CanFrame(0x123, b"\xde\xad\xbe\xef")
    wire = f.wire_bits()
    stuffed = can.stuff([0, 0, 0, 0, 0])
    assert stuffed == [0, 0, 0, 0, 0, 1]
    bad = list(wire)
    bad[30] ^= 1
    with pytest.raises((can.CrcError, can.StuffError)):
        can.decode(bad)
    with pytest.raises(ValueError):
        can.decode(wire[:40])
    with pytest.raises(can.StuffError):
        can.destuff([1, 1, 1, 1, 1, 1])


def test_can_frame_validation():
    with pytest.raises(ValueError):
        can.CanFrame(0x800, b"")
    with pytest.raises(ValueError):
        can.CanFrame(1, bytes(9))
    with pytest.raises(ValueError):
        can.CanFrame(1, b"\x01", remote=True)
    with pytest.raises(ValueError):
        can.arbitrate([])


def test_arbitration_lowest_identifier_wins_without_losing_a_bit():
    rng = random.Random(3)
    for _ in range(100):
        ids = rng.sample(range(1 << 11), 4)
        frames = [can.CanFrame(i, bytes([i & 0xFF])) for i in ids]
        result = can.arbitrate(frames)
        assert frames[result.winner].can_id == min(ids)
        assert result.bus == frames[result.winner].wire_bits(interframe=False)
        assert all(t <= 13 for t in result.lost_at.values())  # decided inside the arbitration field


def test_standard_frame_beats_extended_with_the_same_base_id():
    std = can.CanFrame(0x100, b"")
    ext = can.CanFrame(0x100 << 18, b"", extended=True)
    assert can.arbitrate([ext, std]).winner == 1
    with pytest.raises(ValueError):
        can.arbitrate([std, can.CanFrame(0x100, b"\x01")])


def test_bus_load_is_linear():
    one = can.bus_load([(8, True, 100)])
    assert abs(can.bus_load([(8, True, 200)]) - 2 * one) < 1e-12
    assert 0.01 < one < 0.02


# ------------------------------------------------------------- DroneCAN ----

TYPES = [
    "uavcan.protocol.NodeStatus",
    "uavcan.equipment.gnss.Fix2",
    "uavcan.equipment.gnss.RTCMStream",
    "uavcan.equipment.ahrs.MagneticFieldStrength2",
    "uavcan.equipment.air_data.StaticPressure",
    "uavcan.equipment.air_data.StaticTemperature",
    "uavcan.equipment.esc.RawCommand",
    "uavcan.equipment.esc.Status",
    "uavcan.equipment.power.BatteryInfo",
    "uavcan.equipment.safety.ArmingStatus",
    "uavcan.equipment.indication.BeepCommand",
    "ardupilot.gnss.Status",
    "ardupilot.indication.SafetyState",
]


def test_known_signatures():
    # published values (pydronecan and libcanard headers agree)
    assert dronecan.load("uavcan.protocol.NodeStatus").signature() == 0x0F0868D0C1A7C6F1
    assert dronecan.load("uavcan.equipment.gnss.Fix2").signature() == 0xCA41E7000F37435F
    assert dronecan.Signature().add(b"123456789").value == 0x62EC59E3F1A4F00A


def test_signatures_and_serialisation_match_pydronecan():
    pydronecan = pytest.importorskip("dronecan")
    rng = random.Random(11)
    for name in TYPES:
        ours = dronecan.load(name)
        ref = pydronecan.TYPENAMES[name]
        assert ours.signature() == ref.get_data_type_signature(), name
        assert ours.type_id == ref.default_dtid
        for _ in range(5):
            msg = ref()
            values = _random_values(ours, rng)
            _assign(msg, values)
            transfer = pydronecan.transport.Transfer(
                payload=msg, source_node_id=rng.randrange(1, 128), transfer_id=rng.randrange(32), transfer_priority=rng.randrange(32)
            )
            assert ours.encode(values) == transfer.payload, name
            frames = dronecan.encode_message(ours, values, transfer.source_node_id, transfer.transfer_id, transfer.transfer_priority)
            assert [(f.can_id, f.data) for f in frames] == [(f.message_id, bytes(f.bytes)) for f in transfer.to_frames()], name


def _random_values(t, rng):
    out = {}
    for name, ft in t.fields:
        if name is not None:
            out[name] = _random_value(ft, rng)
    return out


def _random_value(ft, rng):
    if isinstance(ft, dronecan.Primitive):
        if ft.kind == "float":
            return float(rng.choice([0.0, 1.5, -2.25, 100.0, 0.125]))
        if ft.kind == "bool":
            return rng.random() < 0.5
        lo, hi = (0, (1 << ft.bits) - 1) if ft.kind == "uint" else (-(1 << (ft.bits - 1)), (1 << (ft.bits - 1)) - 1)
        return rng.randint(lo, hi)
    if isinstance(ft, dronecan.Array):
        n = rng.randint(0, min(ft.size, 12)) if ft.dynamic else ft.size
        return [_random_value(ft.element, rng) for _ in range(n)]
    return _random_values(ft, rng)


def _assign(msg, values):
    for k, v in values.items():
        if isinstance(v, dict):
            _assign(getattr(msg, k), v)
        elif isinstance(v, list):
            arr = getattr(msg, k)
            static = len(arr) == len(v) and len(v) > 0
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    if not static:
                        arr.append(arr.new_item())
                    _assign(arr[i], item)
                elif static:
                    arr[i] = item
                else:
                    arr.append(item)
        else:
            setattr(msg, k, v)


def test_decode_inverts_encode_including_tail_array_optimisation():
    fix = dronecan.load("uavcan.equipment.gnss.Fix2")
    values = dict(
        fix.default(),
        longitude_deg_1e8=-12225000000,
        latitude_deg_1e8=3787000000,
        sats_used=23,
        status=3,
        covariance=[0.5, 0.25],
        pdop=1.25,
    )
    values["ecef_position_velocity"] = [{"velocity_xyz": [1.0, 2.0, 3.0], "position_xyz_mm": [1, -2, 3], "covariance": [0.5]}]
    decoded = fix.decode(fix.encode(values))
    assert decoded["latitude_deg_1e8"] == 3787000000 and decoded["ecef_position_velocity"][0]["covariance"] == [0.5]
    rtcm_t = dronecan.load("uavcan.equipment.gnss.RTCMStream")
    payload = rtcm_t.encode({"protocol_id": 3, "data": [1, 2, 3]})
    assert payload == bytes([3, 1, 2, 3])  # no length prefix on the last array


def test_transfer_reassembly_checks_toggle_and_crc():
    t = dronecan.load("uavcan.equipment.gnss.RTCMStream")
    frames = dronecan.encode_message(t, {"protocol_id": 3, "data": list(range(100))}, 10, 5)
    assert len(frames) > 1 and frames[0].data[-1] & 0x80 and frames[-1].data[-1] & 0x40
    assert dronecan.reassemble(frames, t.signature()) == t.encode({"protocol_id": 3, "data": list(range(100))})
    with pytest.raises(dronecan.TransferError, match="CRC"):
        dronecan.reassemble(frames, t.signature() ^ 1)
    swapped = [frames[0], frames[2], frames[1]] + frames[3:]
    with pytest.raises(dronecan.TransferError):
        dronecan.reassemble(swapped, t.signature())
    with pytest.raises(ValueError):
        dronecan.message_id(16, 341, 0)
    assert dronecan.parse_id(dronecan.message_id(16, 1062, 10)) == {"priority": 16, "service": False, "type_id": 1062, "source_node": 10}


# -------------------------------------------------------------- MAVLink ----


@pytest.fixture(scope="module")
def dialect():
    return mavlink.Dialect.default()


def test_crc_extra_known_values(dialect):
    known = {
        "HEARTBEAT": 50,
        "SYS_STATUS": 124,
        "ATTITUDE": 39,
        "GLOBAL_POSITION_INT": 104,
        "GPS_RAW_INT": 24,
        "COMMAND_LONG": 152,
        "PARAM_SET": 168,
        "GPS_RTCM_DATA": 35,
        "STATUSTEXT": 83,
    }
    for name, value in known.items():
        assert dialect[name].crc_extra == value == dialect[name].compute_crc_extra()


def test_wire_order_is_by_size_then_extensions(dialect):
    msg = dialect["GPS_RAW_INT"]
    base = [f for f in msg.wire_fields if not f.extension]
    sizes = [mavlink._TYPES[f.type][1] for f in base]
    assert sizes == sorted(sizes, reverse=True)
    assert msg.wire_fields[: len(base)] == base and all(f.extension for f in msg.wire_fields[len(base) :])


def test_every_message_matches_pymavlink_byte_for_byte():
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v20 import ardupilotmega as mav2

    xml = mavlink.xml_dir()
    d = mavlink.Dialect.from_xml(xml / "ardupilotmega.xml") if xml else mavlink.Dialect.default()

    class Sink:
        def write(self, b):
            pass

    mav = mav2.MAVLink(Sink(), srcSystem=7, srcComponent=9)
    rng = random.Random(5)
    compared = 0
    for msg in d.messages.values():
        cls = mav2.mavlink_map.get(msg.id)
        if cls is None or cls.msgname != msg.name or list(cls.fieldnames) != [f.name for f in msg.fields]:
            continue  # the installed pymavlink was generated from a different revision of this message
        assert cls.crc_extra == msg.crc_extra, msg.name
        values = {f.name: _mav_value(f, rng) for f in msg.fields}
        m = cls(*[values[n].encode() if isinstance(values[n], str) else values[n] for n in cls.fieldnames])
        mav.seq = 42
        assert mavlink.encode(msg, values, seq=42, sysid=7, compid=9) == bytes(m.pack(mav)), msg.name
        compared += 1
    assert compared > 250


def test_json_dialect_matches_ardupilot_xml_when_available():
    xml = mavlink.xml_dir()
    if xml is None:
        pytest.skip("no MAVLink XML (install pymavlink or set MAVLINK_XML_DIR)")
    ours = mavlink.Dialect.default()
    ref = mavlink.Dialect.from_xml(xml / "ardupilotmega.xml")
    shared = set(ours.messages) & set(ref.messages)
    assert len(shared) > 280
    for mid in shared:
        if ours[mid].name == ref[mid].name and [f.name for f in ours[mid].fields] == [f.name for f in ref[mid].fields]:
            assert ours[mid].crc_extra == ref[mid].crc_extra


def _mav_value(f, rng):
    import struct

    if f.type == "char":
        return "".join(rng.choice("ABCxyz09") for _ in range(rng.randint(0, max(f.array, 1))))

    def one():
        if f.type == "float":
            return struct.unpack("<f", struct.pack("<f", rng.uniform(-1e4, 1e4)))[0]
        if f.type == "double":
            return rng.uniform(-1e9, 1e9)
        bits = mavlink._TYPES[f.type][1] * 8
        signed = f.type.startswith("int")
        return rng.randint(-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else rng.randint(0, (1 << bits) - 1)

    return [one() for _ in range(f.array)] if f.array else one()


def test_v1_and_signing_match_pymavlink(dialect):
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v10 import ardupilotmega as mav1
    from pymavlink.dialects.v20 import ardupilotmega as mav2

    class Sink:
        def write(self, b):
            pass

    m1 = mav1.MAVLink(Sink(), 1, 1)
    ref = m1.attitude_encode(1000, 0.1, -0.2, 0.3, 0.01, 0.02, 0.03)
    ours = mavlink.encode(
        dialect["ATTITUDE"],
        dict(time_boot_ms=1000, roll=0.1, pitch=-0.2, yaw=0.3, rollspeed=0.01, pitchspeed=0.02, yawspeed=0.03),
        version=1,
    )
    assert ours == bytes(ref.pack(m1))

    m2 = mav2.MAVLink(Sink(), 255, 190)
    key = bytes(range(32))
    m2.signing.secret_key = bytearray(key)
    m2.signing.sign_outgoing = True
    m2.signing.link_id = 3
    m2.signing.timestamp = 123456789
    msg = m2.param_set_encode(1, 1, b"MAV2_EXTRA1", 4.0, 9)
    ref_bytes = bytes(msg.pack(m2))
    ours = mavlink.encode(
        dialect["PARAM_SET"],
        dict(target_system=1, target_component=1, param_id="MAV2_EXTRA1", param_value=4.0, param_type=9),
        sysid=255,
        compid=190,
        key=key,
        link_id=3,
        timestamp=123456789,
    )
    assert ours == ref_bytes
    packet = next(mavlink.Parser(dialect).feed(ours))
    assert mavlink.verify_signature(packet, key) and not mavlink.verify_signature(packet, bytes(32))


def test_truncation_and_zero_payload(dialect):
    hb = mavlink.encode(dialect["HEARTBEAT"], {})
    assert hb[1] == 1  # an all zero payload keeps one byte
    p = mavlink.Parser(dialect)
    name, values = p.decode(next(p.feed(hb)))
    assert name == "HEARTBEAT" and values["type"] == 0
    with pytest.raises(ValueError):
        mavlink.encode(dialect["HEARTBEAT"], {}, version=1, key=bytes(32))
    with pytest.raises(ValueError):
        mavlink.encode(dialect["SETUP_SIGNING"], {}, version=1)  # id 256 needs the 24 bit ids of MAVLink 2


def test_parser_statistics(dialect):
    msg = dialect["ATTITUDE"]
    frames = [mavlink.encode(msg, {"time_boot_ms": i}, seq=s) for i, s in enumerate([0, 1, 1, 4])]
    corrupt = bytearray(frames[0])
    corrupt[12] ^= 0x40
    flagged = bytearray(mavlink.encode(msg, {"time_boot_ms": 9}, seq=5))
    flagged[2] = 0x02
    p = mavlink.Parser(dialect)
    # a corrupted frame can leave a byte that looks like a start marker with a long length; like pymavlink,
    # the parser waits for that many bytes before the CRC fails, so give the stream some tail
    out = list(p.feed(b"\x00\x55" + bytes(corrupt) + b"".join(frames) + bytes(flagged) + bytes(300)))
    assert len(out) == 4
    assert p.crc_errors >= 1 and p.duplicates == 1 and p.lost == 2 and p.bad_flags == 1 and p.dropped_bytes >= 2


def test_tlog_fixture_decodes_like_pymavlink():
    data = (DOCS / "sitl_flight.tlog").read_bytes()
    entries, parser = mavlink.read_tlog(data)
    assert len(entries) > 5000 and parser.crc_errors == 0 and parser.lost == 0
    pymav = pytest.importorskip("pymavlink.mavutil")
    m = pymav.mavlink_connection(str(DOCS / "sitl_flight.tlog"), dialect="ardupilotmega")
    count = 0
    while m.recv_match() is not None:
        count += 1
    assert count == len(entries)


# ----------------------------------------------------------------- RTCM ----


def test_crc24q_and_1005_round_trip():
    assert rtcm.crc24q(b"123456789") == 0xCDE703
    x, y, z = rtcm.geodetic_to_ecef(37.8719, -122.2585, 72.0)
    frame = rtcm.encode_1005(rtcm.StationPosition(7, x, y, z))
    assert len(frame) == 25 and rtcm.message_number(frame) == 1005
    p = rtcm.decode_1005(frame)
    lat, lon, h = rtcm.ecef_to_geodetic(p.x, p.y, p.z)
    assert abs(lat - 37.8719) < 1e-8 and abs(lon + 122.2585) < 1e-8 and abs(h - 72.0) < 1e-3
    with pytest.raises(rtcm.RtcmError):
        rtcm.unframe(frame[:-1] + bytes([frame[-1] ^ 1]))


def test_rtcm_matches_pyrtcm():
    pyrtcm = pytest.importorskip("pyrtcm")
    x, y, z = rtcm.geodetic_to_ecef(37.8719, -122.2585, 72.0)
    parsed = pyrtcm.RTCMReader.parse(rtcm.encode_1005(rtcm.StationPosition(7, x, y, z)))
    assert parsed.DF003 == 7 and abs(parsed.DF025 - x) < 1e-3 and abs(parsed.DF026 - y) < 1e-3 and abs(parsed.DF027 - z) < 1e-3
    for message, sats in ((1074, 10), (1084, 7), (1094, 8), (1124, 9)):
        frame = rtcm.msm4_placeholder(message, sats, 2)
        parsed = pyrtcm.RTCMReader.parse(frame)
        assert parsed.identity == str(message) and parsed.NSat == sats and parsed.NSig == 2
        assert len(frame) == links.rtcm_msm4_bytes(sats, 2)


def test_split_skips_garbage():
    frames = journey.base_burst()
    stream = b"\x00\xd3\x00" + b"".join(frames[:2]) + b"\xd3\xff" + frames[2]
    assert rtcm.split(stream) == frames[:3]


# -------------------------------------------------------------- journey ----


def test_journey_delivers_every_byte():
    r = journey.run()
    assert r["delivered_intact"] and r["base_position"]["station_id"] == 7
    hops = {h["name"]: h for h in r["hops"]}
    assert hops["Ground station to radio"]["units"] == 4  # 629 bytes need four 180 byte fragments
    assert hops["CAN bus"]["units"] > 80
    assert all(h["wire_bytes"] >= h["payload_bytes"] for h in r["hops"])
    assert 0.1 < r["latency_s"] < 0.5


def test_gps_rtcm_data_fragmentation_rules():
    parts = journey.to_gps_rtcm_data(bytes(400), sequence=5)
    assert [p["flags"] & 1 for p in parts] == [1, 1, 1]
    assert [(p["flags"] >> 1) & 3 for p in parts] == [0, 1, 2]
    assert {p["flags"] >> 3 for p in parts} == {5}
    assert journey.from_gps_rtcm_data(parts) == bytes(400)
    with pytest.raises(ValueError):
        journey.from_gps_rtcm_data([parts[1], parts[0]])
    with pytest.raises(ValueError):
        journey.to_gps_rtcm_data(bytes(721), 0)
    single = journey.to_gps_rtcm_data(bytes(100), 2)
    assert len(single) == 1 and single[0]["flags"] == 2 << 3


@pytest.mark.skipif(not os.environ.get("CI") and not (DOCS / "journey.json").exists(), reason="docs not built")
def test_docs_journey_is_current():
    stored = json.loads((DOCS / "journey.json").read_text())
    assert stored == json.loads(json.dumps(journey.run()))
