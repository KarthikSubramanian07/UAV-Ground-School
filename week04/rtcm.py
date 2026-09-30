"""RTCM 3: how an RTK base station tells the drone's GPS what errors it sees.

A frame is::

    0xD3 | 6 reserved zero bits, 10 bit length | payload | CRC-24Q (3 bytes)

and the payload is a bit packed message whose first 12 bits are its number.
Message 1005 carries the base antenna's position in Earth centred, Earth
fixed coordinates to 0.1 mm; the MSM messages (1074 GPS, 1084 GLONASS, 1094
Galileo, 1124 BeiDou) carry the measurements. Field widths follow RTCM
10403.3, and :func:`encode_1005` is checked against ``pyrtcm`` in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PREAMBLE = 0xD3


def crc24q(data: bytes) -> int:
    """CRC-24Q (poly 0x1864CFB, init 0), the check value for "123456789" is 0xCDE703."""
    crc = 0
    for b in data:
        crc ^= b << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF


class _Bits:
    def __init__(self):
        self.value, self.n = 0, 0

    def put(self, value: int, width: int, signed: bool = False):
        if signed:
            value &= (1 << width) - 1
        elif value >> width:
            raise ValueError(f"{value} does not fit in {width} bits")
        self.value = (self.value << width) | value
        self.n += width

    def to_bytes(self) -> bytes:
        pad = -self.n % 8
        return (self.value << pad).to_bytes((self.n + pad) // 8, "big")


def _get(data: bytes, start: int, width: int, signed: bool = False) -> int:
    value = int.from_bytes(data, "big") >> (8 * len(data) - start - width) & ((1 << width) - 1)
    if signed and value >> (width - 1):
        value -= 1 << width
    return value


def frame(payload: bytes) -> bytes:
    if len(payload) > 1023:
        raise ValueError("RTCM 3 payloads are at most 1023 bytes")
    head = bytes([PREAMBLE, len(payload) >> 8 & 0x03, len(payload) & 0xFF]) + payload
    return head + crc24q(head).to_bytes(3, "big")


class RtcmError(ValueError):
    pass


def unframe(data: bytes) -> bytes:
    if len(data) < 6 or data[0] != PREAMBLE:
        raise RtcmError("no RTCM 3 preamble")
    n = (data[1] & 0x03) << 8 | data[2]
    if len(data) < n + 6:
        raise RtcmError("truncated frame")
    if crc24q(data[: n + 3]) != int.from_bytes(data[n + 3 : n + 6], "big"):
        raise RtcmError("CRC-24Q mismatch")
    return data[3 : n + 3]


def split(stream: bytes) -> list[bytes]:
    """Cut a byte stream into whole frames, skipping anything that fails the CRC."""
    out, i = [], 0
    while i + 6 <= len(stream):
        if stream[i] != PREAMBLE:
            i += 1
            continue
        n = (stream[i + 1] & 0x03) << 8 | stream[i + 2]
        try:
            unframe(stream[i : i + n + 6])
        except RtcmError:
            i += 1
            continue
        out.append(stream[i : i + n + 6])
        i += n + 6
    return out


@dataclass
class StationPosition:
    """RTCM 1005: stationary antenna reference point."""

    station_id: int
    x: float  # metres, ECEF
    y: float
    z: float
    gps: bool = True
    glonass: bool = True
    galileo: bool = True


def encode_1005(p: StationPosition) -> bytes:
    b = _Bits()
    b.put(1005, 12)
    b.put(p.station_id, 12)
    b.put(0, 6)  # ITRF realisation year (reserved for future use)
    b.put(int(p.gps), 1)
    b.put(int(p.glonass), 1)
    b.put(int(p.galileo), 1)
    b.put(0, 1)  # reference station indicator: real station
    b.put(round(p.x * 10000), 38, signed=True)
    b.put(0, 1)  # single receiver oscillator
    b.put(0, 1)  # reserved
    b.put(round(p.y * 10000), 38, signed=True)
    b.put(0, 2)  # quarter cycle indicator
    b.put(round(p.z * 10000), 38, signed=True)
    return frame(b.to_bytes())


def decode_1005(data: bytes) -> StationPosition:
    payload = unframe(data)
    if _get(payload, 0, 12) != 1005:
        raise RtcmError("not a 1005 message")
    return StationPosition(
        _get(payload, 12, 12),
        _get(payload, 34, 38, True) / 10000,
        _get(payload, 74, 38, True) / 10000,
        _get(payload, 114, 38, True) / 10000,
        bool(_get(payload, 30, 1)),
        bool(_get(payload, 31, 1)),
        bool(_get(payload, 32, 1)),
    )


def message_number(data: bytes) -> int:
    return _get(unframe(data), 0, 12)


def msm4_placeholder(message: int, satellites: int, signals: int, station_id: int = 0, seed: int = 0) -> bytes:
    """A structurally valid MSM4 frame (header, masks, field widths) with pseudo random observations.

    Real observations need a receiver; for moving bytes through the protocol
    stack only the size and framing matter. The size equals
    :func:`week04.links.rtcm_msm4_bytes`.
    """
    state = seed * 2654435761 + 1

    def rand(width: int) -> int:
        nonlocal state
        value = 0
        for _ in range(0, width, 16):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            value = (value << 16) | (state >> 8 & 0xFFFF)
        return value & ((1 << width) - 1)

    b = _Bits()
    b.put(message, 12)
    b.put(station_id, 12)
    b.put(rand(30), 30)  # epoch time
    b.put(0, 1)  # last message of this epoch? (0 = yes)
    b.put(0, 3)  # IODS
    b.put(0, 7)  # reserved
    b.put(0, 2)  # clock steering
    b.put(0, 2)  # external clock
    b.put(0, 1)  # divergence free smoothing
    b.put(0, 3)  # smoothing interval
    b.put(((1 << satellites) - 1) << (64 - satellites), 64)  # satellite mask
    b.put(((1 << signals) - 1) << (32 - signals), 32)  # signal mask
    b.put((1 << (satellites * signals)) - 1, satellites * signals)  # cell mask: every satellite on every signal
    for _ in range(satellites):
        b.put(rand(8), 8)  # rough range, whole milliseconds
    for _ in range(satellites):
        b.put(rand(10), 10)  # rough range, modulo 1 ms
    for width in (15, 22, 4, 1, 6):  # fine pseudorange, fine phase range, lock time, half cycle, CNR
        for _ in range(satellites * signals):
            b.put(rand(width), width)
    return frame(b.to_bytes())


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> tuple[float, float, float]:
    """WGS 84."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return ((n + h) * math.cos(lat) * math.cos(lon), (n + h) * math.cos(lat) * math.sin(lon), (n * (1 - e2) + h) * math.sin(lat))


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - e2 * n / (n + h)))
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n
