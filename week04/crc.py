"""The checksums every drone protocol in this week leans on.

Each one is written out bit by bit from its catalogue definition (Greg Cook's
CRC RevEng catalogue names them; the "check" value is the CRC of the ASCII
string ``123456789`` and is asserted in the tests), then sped up with a table
where the protocol runs at line rate.

========================  ===========  ==============================  ======
Name                      Width, poly  Used by                         Check
========================  ===========  ==============================  ======
CRC-16/MCRF4XX (X.25)     16, 0x1021   MAVLink v1 and v2 frames        0x6F91
                          reflected
CRC-16/CCITT-FALSE        16, 0x1021   DroneCAN multi frame transfers  0x29B1
CRC-8/DVB-S2              8, 0xD5      CRSF (ExpressLRS, Crossfire)    0xBC
CRC-15/CAN                15, 0x4599   Every classic CAN frame         0x059E
DShot nibble XOR          4            DShot throttle frames           n/a
========================  ===========  ==============================  ======
"""

from __future__ import annotations

from collections.abc import Iterable

# --------------------------------------------------------------- CRC-16 ----


def _x25_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_X25 = _x25_table()


def x25_accumulate(data: Iterable[int], crc: int = 0xFFFF) -> int:
    """CRC-16/MCRF4XX, the checksum MAVLink calls ``crc_accumulate``.

    MAVLink feeds it every byte after the start marker, then the message's
    one byte ``CRC_EXTRA`` seed so that sender and receiver must agree on the
    message layout, not just on the bytes.
    """
    for b in data:
        crc = (crc >> 8) ^ _X25[(crc ^ b) & 0xFF]
    return crc


def x25_bitwise(data: Iterable[int], crc: int = 0xFFFF) -> int:
    """The same CRC the way MAVLink's C header computes it, one byte at a time."""
    for b in data:
        tmp = (b ^ (crc & 0xFF)) & 0xFF
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def ccitt_false(data: Iterable[int], crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, MSB first): DroneCAN transfer CRC."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------- CRC-8 ----


def _dvb_s2_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_DVB_S2 = _dvb_s2_table()


def dvb_s2(data: Iterable[int], crc: int = 0) -> int:
    """CRC-8/DVB-S2, the CRSF frame checksum (covers type and payload, not sync or length)."""
    if not 0 <= crc <= 0xFF:
        raise ValueError("CRC-8 state must fit in a byte")
    for b in data:
        crc = _DVB_S2[crc ^ b]
    return crc


# --------------------------------------------------------------- CRC-15 ----


def can15(bits: Iterable[int]) -> int:
    """CRC-15/CAN over a *bit* sequence, exactly as ISO 11898-1 describes the shift register.

    The CAN CRC covers start of frame, arbitration, control and data fields
    before bit stuffing, so it takes bits, not bytes.
    """
    crc = 0
    for bit in bits:
        nxt = (bit & 1) ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if nxt:
            crc ^= 0x4599
    return crc


def bytes_to_bits(data: Iterable[int]) -> list[int]:
    """MSB first, the order CAN, I2C and SPI put bytes on the wire."""
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


# --------------------------------------------------------------- DShot ----


def dshot_crc(value12: int, inverted: bool = False) -> int:
    """XOR of the three nibbles of the 12 bit DShot payload; bidirectional DShot inverts it."""
    crc = (value12 ^ (value12 >> 4) ^ (value12 >> 8)) & 0x0F
    return (~crc) & 0x0F if inverted else crc


CHECK_INPUT = b"123456789"
CHECK_VALUES = {"x25": 0x6F91, "ccitt_false": 0x29B1, "dvb_s2": 0xBC, "can15": 0x059E}
