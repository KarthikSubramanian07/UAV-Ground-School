"""One second of RTK corrections, followed from the base station to the drone's GPS.

This is the "think about communication protocols" part made concrete. The
same bytes cross five protocols on four kinds of wire, and every hop is
encoded with this week's own code and decoded again to prove nothing was
lost:

1. **RTCM 3** frames from the Here4 base station (USB to the laptop).
2. **MAVLink 2** ``GPS_RTCM_DATA`` messages from the ground station, at most
   180 bytes each, fragmented with a sequence number when a burst is longer.
3. **UART** at 57600 baud 8N1 from the drone's RFD900x into TELEM1 (the
   radio link is transparent: the same bytes come out as went in).
4. **DroneCAN** ``uavcan.equipment.gnss.RTCMStream`` transfers of up to 128
   bytes, split across CAN frames with tail bytes and a transfer CRC.
5. **CAN** frames at 1 Mbit/s, with bit stuffing, CRC-15 and arbitration IDs.

The Here4 rover then reassembles the RTCM stream and checks each frame's
CRC-24Q, which is exactly what its u-blox receiver does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import can, dronecan, mavlink, rtcm, uart

BASE = (37.8719, -122.2585, 72.0)  # the Memorial Glade, where the base station sits


@dataclass
class Hop:
    name: str
    protocol: str
    medium: str
    units: int  # frames or messages on this hop
    payload_bytes: int  # RTCM bytes carried
    wire_bytes: int  # bytes (or bit equivalent) on the medium
    seconds: float  # time on the wire at the link rate
    sample: dict = field(default_factory=dict)  # first unit, for drawing

    @property
    def overhead(self) -> float:
        return self.wire_bytes / self.payload_bytes - 1


def base_burst(station_id: int = 7) -> list[bytes]:
    """One epoch from a dual band base: position (1005) and four MSM4 messages."""
    x, y, z = rtcm.geodetic_to_ecef(*BASE)
    frames = [rtcm.encode_1005(rtcm.StationPosition(station_id, x, y, z))]
    for i, (msg, sats) in enumerate(((1074, 10), (1084, 7), (1094, 8), (1124, 9))):
        frames.append(rtcm.msm4_placeholder(msg, sats, 2, station_id=station_id, seed=i + 1))
    return frames


def to_gps_rtcm_data(stream: bytes, sequence: int) -> list[dict]:
    """How a ground station packs RTCM into GPS_RTCM_DATA (the MAVLink message definition's rules)."""
    chunks = [stream[i : i + 180] for i in range(0, len(stream), 180)]
    if len(chunks) > 4:
        raise ValueError("a GPS_RTCM_DATA burst holds at most 4 fragments (720 bytes)")
    fragmented = len(chunks) > 1
    out = []
    for i, chunk in enumerate(chunks):
        flags = (int(fragmented) | (i << 1) | ((sequence & 0x1F) << 3)) if fragmented else (sequence & 0x1F) << 3
        out.append({"flags": flags, "len": len(chunk), "data": list(chunk)})
    return out


def from_gps_rtcm_data(messages: list[dict]) -> bytes:
    """Reassemble, checking fragment order and sequence the way ArduPilot's AP_GPS does."""
    if not messages:
        return b""
    seqs = {m["flags"] >> 3 for m in messages}
    if len(seqs) != 1:
        raise ValueError("fragments from different bursts")
    if messages[0]["flags"] & 1:
        ids = [(m["flags"] >> 1) & 3 for m in messages]
        if ids != list(range(len(messages))):
            raise ValueError(f"fragments out of order: {ids}")
    return b"".join(bytes(m["data"][: m["len"]]) for m in messages)


def run(sequence: int = 3, telemetry_baud: int = 57600, air_kbps: int = 64, can_bitrate: int = 1_000_000) -> dict:
    frames = base_burst()
    stream = b"".join(frames)
    hops: list[Hop] = []

    # 1. RTCM on USB
    hops.append(
        Hop(
            "Base station to laptop", "RTCM 3", "USB", len(frames), len(stream), len(stream), 0.0, {"hex": frames[0].hex(), "message": 1005}
        )
    )

    # 2. MAVLink 2 GPS_RTCM_DATA from the ground station
    dialect = mavlink.Dialect.default()
    msg = dialect["GPS_RTCM_DATA"]
    parts = []
    for burst_start in range(0, len(stream), 720):
        parts += to_gps_rtcm_data(stream[burst_start : burst_start + 720], sequence + burst_start // 720)
    mav_frames = [mavlink.encode(msg, p, seq=i, sysid=255, compid=190) for i, p in enumerate(parts)]
    mav_bytes = b"".join(mav_frames)
    hops.append(
        Hop(
            "Ground station to radio",
            "MAVLink 2 GPS_RTCM_DATA",
            "USB serial",
            len(mav_frames),
            len(stream),
            len(mav_bytes),
            0.0,
            {"hex": mav_frames[0].hex(), "flags": parts[0]["flags"], "fragments": len(parts)},
        )
    )

    # 3. over the air and into TELEM1 as UART characters
    cfg = uart.UartConfig(telemetry_baud)
    levels = uart.encode(mav_bytes, cfg)
    assert uart.decode(levels, cfg) == mav_bytes
    air_seconds = len(mav_bytes) * 8 / (air_kbps * 1000)
    hops.append(
        Hop(
            "Radio link, then TELEM1",
            "SiK over 915 MHz, then UART 8N1",
            f"{air_kbps} kbit/s air, {telemetry_baud} baud wire",
            len(mav_frames),
            len(stream),
            len(levels) // 8,
            len(levels) / telemetry_baud + air_seconds,
            {"bits": levels[: 10 * 12], "bytes": mav_bytes[:12].hex(), "config": cfg.label()},
        )
    )

    # the autopilot parses MAVLink with the same parser a real one would use
    parser = mavlink.Parser(dialect)
    received = [parser.decode(p)[1] for p in parser.feed(mav_bytes)]
    rebuilt = b"".join(from_gps_rtcm_data(received[i : i + 4]) for i in range(0, len(received), 4))
    assert rebuilt == stream, "MAVLink hop corrupted the stream"

    # 4 and 5. DroneCAN RTCMStream to the Here4 on CAN1
    dtype = dronecan.load("uavcan.equipment.gnss.RTCMStream")
    can_frames: list[can.CanFrame] = []
    transfers = 0
    for i in range(0, len(rebuilt), 128):
        chunk = rebuilt[i : i + 128]
        can_frames += dronecan.encode_message(
            dtype, {"protocol_id": 3, "data": list(chunk)}, source_node=10, transfer_id=transfers, priority=16
        )
        transfers += 1
    wire_bits = sum(len(f.wire_bits()) for f in can_frames)
    first = can_frames[0]
    hops.append(
        Hop(
            "Autopilot to Here4",
            "DroneCAN RTCMStream",
            "CAN transfers",
            transfers,
            len(stream),
            sum(len(f.data) for f in can_frames),
            0.0,
            {
                "id": f"0x{first.can_id:08X}",
                "fields": dronecan.parse_id(first.can_id),
                "frames_first_transfer": sum(1 for f in can_frames if f.data[-1] & 0x1F == 0),
            },
        )
    )
    hops.append(
        Hop(
            "CAN bus",
            "CAN 2.0B, 29 bit identifiers",
            f"{can_bitrate // 1000} kbit/s differential pair",
            len(can_frames),
            len(stream),
            wire_bits // 8,
            wire_bits / can_bitrate,
            {"bits": first.wire_bits(), "stuffed": len(first.stuffed_bits()) - len(first.header_bits()) - 15, "data": first.data.hex()},
        )
    )

    # the Here4 side: decode CAN, reassemble transfers, then RTCM
    decoded = [can.decode(f.wire_bits()) for f in can_frames]
    by_transfer: dict[int, list[can.CanFrame]] = {}
    for f in decoded:
        by_transfer.setdefault(f.data[-1] & 0x1F, []).append(f)
    out = b""
    for tid in range(transfers):
        payload = dronecan.reassemble(by_transfer[tid], dtype.signature())
        out += bytes(dtype.decode(payload)["data"])
    delivered = rtcm.split(out)
    assert delivered == frames, "CAN hop corrupted the stream"
    position = rtcm.decode_1005(delivered[0])
    lat, lon, h = rtcm.ecef_to_geodetic(position.x, position.y, position.z)

    return {
        "rtcm_bytes": len(stream),
        "rtcm_frames": [{"message": rtcm.message_number(f), "bytes": len(f)} for f in frames],
        "hops": [dict(asdict(h), overhead=round(h.overhead, 4)) for h in hops],
        "delivered_intact": True,
        "base_position": {"lat": round(lat, 7), "lon": round(lon, 7), "height_m": round(h, 3), "station_id": position.station_id},
        "latency_s": round(sum(h.seconds for h in hops), 4),
    }
