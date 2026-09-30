"""RC receiver protocols: SBUS (Futaba, FrSky) and CRSF (TBS Crossfire, ExpressLRS).

Both carry sixteen 11 bit channels packed least significant bit first into
22 bytes, and both use the same raw scale (172 to 1811, centre 992), but they
frame it very differently:

SBUS, 25 bytes at 100000 baud 8E2 *inverted*, one frame every 7 or 14 ms::

    0x0F | 22 bytes of channels | flags | 0x00
    flags: bit 0 channel 17, bit 1 channel 18, bit 2 frame lost, bit 3 failsafe

CRSF, variable length at 420000 baud 8N1, not inverted, and bidirectional so
the receiver can send telemetry back on the same UART::

    address | length | type | payload | CRC-8/DVB-S2(type + payload)

CRSF has a checksum and SBUS does not; SBUS only has a fixed header and
footer to catch misalignment.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from .crc import dvb_s2

RAW_MIN, RAW_MID, RAW_MAX = 172, 992, 1811


def raw_to_us(raw: int) -> float:
    """The mapping ArduPilot and Betaflight use: 172 is 987.5 us, 992 is 1500 us, 1811 is 2011.9 us."""
    return 1500.0 + (raw - RAW_MID) * 5.0 / 8.0


def us_to_raw(us: float) -> int:
    return max(0, min(2047, round(RAW_MID + (us - 1500.0) * 8.0 / 5.0)))


def pack11(channels: Sequence[int]) -> bytes:
    """Sixteen 11 bit values, LSB first, into 22 bytes."""
    if len(channels) != 16:
        raise ValueError("need exactly 16 channels")
    acc = 0
    for i, value in enumerate(channels):
        if not 0 <= value < 2048:
            raise ValueError(f"channel {i + 1} value {value} does not fit in 11 bits")
        acc |= value << (11 * i)
    return acc.to_bytes(22, "little")


def unpack11(data: bytes) -> list[int]:
    acc = int.from_bytes(data[:22], "little")
    return [(acc >> (11 * i)) & 0x7FF for i in range(16)]


# ----------------------------------------------------------------- SBUS ----

SBUS_HEADER, SBUS_FOOTER, SBUS_LEN = 0x0F, 0x00, 25


@dataclass
class SbusFrame:
    channels: list[int]
    ch17: bool = False
    ch18: bool = False
    frame_lost: bool = False
    failsafe: bool = False

    def encode(self) -> bytes:
        flags = self.ch17 | self.ch18 << 1 | self.frame_lost << 2 | self.failsafe << 3
        return bytes([SBUS_HEADER]) + pack11(self.channels) + bytes([flags, SBUS_FOOTER])

    @classmethod
    def decode(cls, data: bytes) -> SbusFrame:
        if len(data) != SBUS_LEN or data[0] != SBUS_HEADER or data[24] != SBUS_FOOTER:
            raise ValueError("not an SBUS frame (bad length, header or footer)")
        flags = data[23]
        return cls(unpack11(data[1:23]), bool(flags & 1), bool(flags & 2), bool(flags & 4), bool(flags & 8))


# ----------------------------------------------------------------- CRSF ----

CRSF_SYNC = 0xC8  # "flight controller" address, also used as the sync byte
CRSF_MAX_FRAME = 64


class CrsfType:
    GPS = 0x02
    BATTERY = 0x08
    LINK_STATISTICS = 0x14
    RC_CHANNELS = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21


def crsf_frame(frame_type: int, payload: bytes, address: int = CRSF_SYNC) -> bytes:
    body = bytes([frame_type]) + payload
    if len(body) + 3 > CRSF_MAX_FRAME:
        raise ValueError("CRSF frames are at most 64 bytes")
    return bytes([address, len(body) + 1]) + body + bytes([dvb_s2(body)])


def crsf_rc(channels: Sequence[int]) -> bytes:
    return crsf_frame(CrsfType.RC_CHANNELS, pack11(channels))


def crsf_battery(volts: float, amps: float, used_mah: int, percent: int) -> bytes:
    payload = struct.pack(">HH", round(volts * 10), round(amps * 10)) + int(used_mah).to_bytes(3, "big") + bytes([percent])
    return crsf_frame(CrsfType.BATTERY, payload)


def crsf_attitude(pitch_rad: float, roll_rad: float, yaw_rad: float) -> bytes:
    return crsf_frame(CrsfType.ATTITUDE, struct.pack(">hhh", *(round(a * 10000) for a in (pitch_rad, roll_rad, yaw_rad))))


def crsf_gps(lat: float, lon: float, speed_kmh: float, heading_deg: float, alt_m: float, sats: int) -> bytes:
    payload = struct.pack(
        ">iiHHHB", round(lat * 1e7), round(lon * 1e7), round(speed_kmh * 10), round(heading_deg * 100), round(alt_m + 1000), sats
    )
    return crsf_frame(CrsfType.GPS, payload)


def crsf_link_statistics(
    rssi1: int, rssi2: int, lq: int, snr: int, antenna: int, rf_mode: int, tx_power: int, d_rssi: int, d_lq: int, d_snr: int
) -> bytes:
    """RSSI values are sent as positive dBm magnitudes (a byte of 70 means -70 dBm)."""
    return crsf_frame(
        CrsfType.LINK_STATISTICS, struct.pack(">BBBbBBBBBb", rssi1, rssi2, lq, snr, antenna, rf_mode, tx_power, d_rssi, d_lq, d_snr)
    )


def crsf_flight_mode(mode: str) -> bytes:
    return crsf_frame(CrsfType.FLIGHT_MODE, mode.encode("ascii") + b"\0")


@dataclass
class CrsfPacket:
    address: int
    type: int
    payload: bytes

    def decoded(self) -> dict:
        p = self.payload
        if self.type == CrsfType.RC_CHANNELS:
            return {"channels": unpack11(p)}
        if self.type == CrsfType.BATTERY:
            v, a = struct.unpack(">HH", p[:4])
            return {"volts": v / 10, "amps": a / 10, "used_mah": int.from_bytes(p[4:7], "big"), "percent": p[7]}
        if self.type == CrsfType.ATTITUDE:
            pitch, roll, yaw = struct.unpack(">hhh", p)
            return {"pitch": pitch / 10000, "roll": roll / 10000, "yaw": yaw / 10000}
        if self.type == CrsfType.GPS:
            lat, lon, spd, hdg, alt, sats = struct.unpack(">iiHHHB", p)
            return {"lat": lat / 1e7, "lon": lon / 1e7, "speed_kmh": spd / 10, "heading_deg": hdg / 100, "alt_m": alt - 1000, "sats": sats}
        if self.type == CrsfType.LINK_STATISTICS:
            keys = ("rssi1", "rssi2", "lq", "snr", "antenna", "rf_mode", "tx_power", "d_rssi", "d_lq", "d_snr")
            return dict(zip(keys, struct.unpack(">BBBbBBBBBb", p)))
        if self.type == CrsfType.FLIGHT_MODE:
            return {"mode": p.split(b"\0")[0].decode("ascii")}
        return {"raw": p.hex()}


@dataclass
class CrsfParser:
    """Byte stream parser that resynchronises after noise, as a flight controller must."""

    buffer: bytearray = field(default_factory=bytearray)
    crc_errors: int = 0
    dropped_bytes: int = 0

    def feed(self, data: bytes) -> Iterator[CrsfPacket]:
        self.buffer += data
        while len(self.buffer) >= 2:
            length = self.buffer[1]
            if self.buffer[0] not in (CRSF_SYNC, 0xEA, 0xEE, 0xEC) or not 2 <= length <= CRSF_MAX_FRAME - 2:
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            if len(self.buffer) < length + 2:
                return
            frame = bytes(self.buffer[: length + 2])
            body, crc = frame[2:-1], frame[-1]
            if dvb_s2(body) != crc:
                self.crc_errors += 1
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            del self.buffer[: length + 2]
            yield CrsfPacket(frame[0], body[0], body[1:])
