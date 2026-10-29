"""Read UDP datagrams out of a pcap file, with nothing but the standard library.

Enough of libpcap's format to replay the RTPS captures in ``docs/week05``:
classic pcap in either byte order, microsecond or nanosecond timestamps, and
the link layers ``tcpdump`` produces on Linux (Ethernet, raw IP, and the
"cooked" SLL and SLL2 headers used for ``-i any``). IPv4 only, fragments
skipped, since RTPS keeps every datagram under the MTU unless told otherwise.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228
LINKTYPE_LINUX_SLL2 = 276

_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}


@dataclass(frozen=True)
class Datagram:
    """One UDP datagram: when it was seen, where it went, and its payload."""

    time: float
    src: str
    sport: int
    dst: str
    dport: int
    payload: bytes


def _ip_payload(link: int, frame: bytes) -> bytes | None:
    if link == LINKTYPE_ETHERNET:
        ethertype, offset = struct.unpack_from(">H", frame, 12)[0], 14
        while ethertype == 0x8100:  # 802.1Q VLAN tag
            ethertype, offset = struct.unpack_from(">H", frame, offset + 2)[0], offset + 4
        return frame[offset:] if ethertype == 0x0800 else None
    if link == LINKTYPE_LINUX_SLL:
        return frame[16:] if struct.unpack_from(">H", frame, 14)[0] == 0x0800 else None
    if link == LINKTYPE_LINUX_SLL2:
        return frame[20:] if struct.unpack_from(">H", frame, 0)[0] == 0x0800 else None
    if link in (LINKTYPE_RAW, LINKTYPE_IPV4):
        return frame if frame and frame[0] >> 4 == 4 else None
    raise ValueError(f"unsupported pcap link type {link}")


def _udp(packet: bytes) -> tuple[str, int, str, int, bytes] | None:
    if len(packet) < 20 or packet[0] >> 4 != 4:
        return None
    ihl = (packet[0] & 0x0F) * 4
    total = struct.unpack_from(">H", packet, 2)[0]
    flags_frag = struct.unpack_from(">H", packet, 6)[0]
    if packet[9] != 17 or flags_frag & 0x3FFF:  # not UDP, or a fragment
        return None
    src = ".".join(map(str, packet[12:16]))
    dst = ".".join(map(str, packet[16:20]))
    sport, dport, length = struct.unpack_from(">HHH", packet, ihl)
    end = min(len(packet), total, ihl + length)
    return src, sport, dst, dport, bytes(packet[ihl + 8 : end])


def read(path: str | Path) -> Iterator[Datagram]:
    """Yield every IPv4 UDP datagram in a pcap file, in capture order."""
    data = Path(path).read_bytes()
    if data[:4] not in _MAGIC:
        raise ValueError(f"{path}: not a classic pcap file (pcapng is not supported)")
    order, tick = _MAGIC[data[:4]]
    link = struct.unpack_from(order + "I", data, 20)[0] & 0x0FFFFFFF
    offset = 24
    while offset + 16 <= len(data):
        sec, frac, incl, _orig = struct.unpack_from(order + "IIII", data, offset)
        frame = data[offset + 16 : offset + 16 + incl]
        offset += 16 + incl
        ip = _ip_payload(link, frame)
        if ip is None:
            continue
        udp = _udp(ip)
        if udp is not None:
            yield Datagram(sec + frac * tick, *udp)


def write(path: str | Path, datagrams: list[Datagram]) -> None:
    """Write datagrams as a raw IPv4 pcap (used by tests to build fixtures)."""
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_RAW))
    for d in datagrams:
        udp = struct.pack(">HHHH", d.sport, d.dport, 8 + len(d.payload), 0) + d.payload
        ip = struct.pack(
            ">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, bytes(map(int, d.src.split("."))), bytes(map(int, d.dst.split(".")))
        )
        frame = ip + udp
        sec = int(d.time)
        out += struct.pack("<IIII", sec, int(round((d.time - sec) * 1e6)), len(frame), len(frame)) + frame
    Path(path).write_bytes(bytes(out))
