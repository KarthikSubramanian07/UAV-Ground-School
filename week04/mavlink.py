"""MAVLink v1 and v2 from the XML message definitions: no generated code, no pymavlink.

MAVLink is the "drone language" of the slides. What is on the wire::

    v2: 0xFD len incompat compat seq sysid compid msgid(3, LE) payload crc(2) [signature(13)]
    v1: 0xFE len seq sysid compid msgid(1) payload crc(2)

Details that matter when you implement it yourself (all tested against
pymavlink, message by message, across the whole ``ardupilotmega`` dialect):

* **Field order.** Fields are sent sorted by type size, largest first, so
  every field is naturally aligned; extension fields (added later, after the
  ``<extensions/>`` tag) are appended in declaration order.
* **CRC_EXTRA.** The checksum (CRC-16/MCRF4XX) also covers one extra byte
  derived from the message name and field layout. A sender and receiver that
  disagree on a message definition therefore reject each other's frames
  instead of silently misreading them.
* **Truncation.** MAVLink 2 drops trailing zero bytes from the payload
  (never the first byte). The receiver pads them back.
* **Signing.** MAVLink 2 can append a 13 byte signature: link id, a 48 bit
  timestamp in 10 us units, and the first 6 bytes of
  ``SHA-256(secret key + header + payload + crc + link id + timestamp)``.

The dialect can be loaded from the XML files (``Dialect.from_xml``) or from
the compact JSON this module writes (``week04/data/mavlink_ardupilotmega.json``),
which is what the tests and the showcase page use.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .crc import x25_accumulate

DATA = Path(__file__).with_name("data")
DEFAULT_JSON = DATA / "mavlink_ardupilotmega.json"

STX_V1, STX_V2 = 0xFE, 0xFD
INCOMPAT_SIGNED = 0x01

_TYPES = {
    "int8_t": ("b", 1),
    "uint8_t": ("B", 1),
    "int16_t": ("h", 2),
    "uint16_t": ("H", 2),
    "int32_t": ("i", 4),
    "uint32_t": ("I", 4),
    "int64_t": ("q", 8),
    "uint64_t": ("Q", 8),
    "float": ("f", 4),
    "double": ("d", 8),
    "char": ("s", 1),
    "uint8_t_mavlink_version": ("B", 1),
}


@dataclass
class Field:
    name: str
    type: str  # base C type, e.g. "uint16_t"
    array: int = 0  # 0 for scalars
    extension: bool = False
    enum: str | None = None
    units: str | None = None

    @property
    def size(self) -> int:
        return _TYPES[self.type][1] * max(self.array, 1)

    @property
    def fmt(self) -> str:
        code = _TYPES[self.type][0]
        if self.type == "char":
            return f"{max(self.array, 1)}s"
        return f"{self.array}{code}" if self.array else code


@dataclass
class Message:
    id: int
    name: str
    fields: list[Field]  # declaration order
    crc_extra: int = 0

    @property
    def wire_fields(self) -> list[Field]:
        base = [f for f in self.fields if not f.extension]
        ext = [f for f in self.fields if f.extension]
        return sorted(base, key=lambda f: -_TYPES[f.type][1]) + ext

    @property
    def length(self) -> int:
        return sum(f.size for f in self.wire_fields)

    @property
    def base_length(self) -> int:
        return sum(f.size for f in self.fields if not f.extension)

    def compute_crc_extra(self) -> int:
        crc = x25_accumulate(f"{self.name} ".encode())
        for f in self.wire_fields:
            if f.extension:
                continue
            ctype = "uint8_t" if f.type == "uint8_t_mavlink_version" else f.type
            crc = x25_accumulate(f"{ctype} {f.name} ".encode(), crc)
            if f.array:
                crc = x25_accumulate([f.array], crc)
        return (crc & 0xFF) ^ (crc >> 8)

    # ------------------------------------------------------- payload ----

    def pack(self, values: dict) -> bytes:
        out = bytearray()
        for f in self.wire_fields:
            v = values.get(f.name)
            if f.type == "char":
                raw = v.encode() if isinstance(v, str) else bytes(v or b"")
                out += struct.pack("<" + f.fmt, raw)
            elif f.array:
                items = list(v) if v is not None else []
                items += [0] * (f.array - len(items))
                out += struct.pack("<" + f.fmt, *items[: f.array])
            else:
                out += struct.pack("<" + f.fmt, 0 if v is None else v)
        return bytes(out)

    def unpack(self, payload: bytes) -> dict:
        payload = payload + bytes(max(0, self.length - len(payload)))  # undo v2 truncation
        values: dict = {}
        offset = 0
        for f in self.wire_fields:
            (value, *rest) = struct.unpack_from("<" + f.fmt, payload, offset)
            offset += f.size
            if f.type == "char":
                value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
            elif f.array:
                value = [value, *rest]
            values[f.name] = value
        return values


@dataclass
class Dialect:
    name: str
    messages: dict[int, Message]
    enums: dict[str, dict[int, str]] = field(default_factory=dict)

    def __post_init__(self):
        self.by_name = {m.name: m for m in self.messages.values()}

    def __getitem__(self, key: int | str) -> Message:
        return self.by_name[key] if isinstance(key, str) else self.messages[key]

    # ------------------------------------------------------------- I/O ----

    @classmethod
    def from_xml(cls, path: str | Path) -> Dialect:
        path = Path(path)
        messages: dict[int, Message] = {}
        enums: dict[str, dict[int, str]] = {}
        seen: set[Path] = set()

        def visit(p: Path):
            if p in seen:
                return
            seen.add(p)
            root = ET.parse(p).getroot()
            for inc in root.findall("include"):
                visit(p.parent / inc.text.strip())
            for e in root.iter("enum"):
                entries = enums.setdefault(e.get("name"), {})
                for entry in e.findall("entry"):
                    value = entry.get("value")
                    if value is not None:
                        entries[int(value, 0)] = entry.get("name")
            for m in root.iter("message"):
                fields, ext = [], False
                for child in m:
                    if child.tag == "extensions":
                        ext = True
                    elif child.tag == "field":
                        ctype, array = child.get("type"), 0
                        if "[" in ctype:
                            ctype, n = ctype[:-1].split("[")
                            array = int(n)
                        fields.append(Field(child.get("name"), ctype, array, ext, child.get("enum"), child.get("units")))
                msg = Message(int(m.get("id")), m.get("name"), fields)
                msg.crc_extra = msg.compute_crc_extra()
                messages[msg.id] = msg

        visit(path)
        return cls(path.stem, dict(sorted(messages.items())), enums)

    def to_json(self) -> dict:
        return {
            "dialect": self.name,
            "messages": [
                {
                    "id": m.id,
                    "name": m.name,
                    "crc_extra": m.crc_extra,
                    "fields": [[f.name, f.type, f.array, int(f.extension)] + ([f.enum] if f.enum else []) for f in m.fields],
                }
                for m in self.messages.values()
            ],
            "enums": {name: {str(k): v for k, v in entries.items()} for name, entries in sorted(self.enums.items())},
        }

    @classmethod
    def from_json(cls, data: dict) -> Dialect:
        messages = {}
        for m in data["messages"]:
            fields = [Field(f[0], f[1], f[2], bool(f[3]), f[4] if len(f) > 4 else None) for f in m["fields"]]
            messages[m["id"]] = Message(m["id"], m["name"], fields, m["crc_extra"])
        enums = {name: {int(k): v for k, v in entries.items()} for name, entries in data.get("enums", {}).items()}
        return cls(data["dialect"], messages, enums)

    @classmethod
    def default(cls) -> Dialect:
        global _DEFAULT
        if _DEFAULT is None:
            _DEFAULT = cls.from_json(json.loads(DEFAULT_JSON.read_text()))
        return _DEFAULT


_DEFAULT: Dialect | None = None


def xml_dir() -> Path | None:
    """Where to find the official XML: $MAVLINK_XML_DIR, else the copy inside an installed pymavlink."""
    env = os.environ.get("MAVLINK_XML_DIR")
    if env:
        return Path(env) if (Path(env) / "ardupilotmega.xml").exists() else None
    try:
        import pymavlink  # noqa: F401  (only used to locate its XML)
    except ImportError:
        return None
    path = Path(pymavlink.__file__).parent / "message_definitions" / "v1.0"
    return path if (path / "ardupilotmega.xml").exists() else None  # wheels on some platforms leave the XML out


# ----------------------------------------------------------- framing ----


@dataclass
class Packet:
    version: int
    seq: int
    sysid: int
    compid: int
    msgid: int
    payload: bytes
    incompat: int = 0
    compat: int = 0
    signature: bytes | None = None  # link id (1) + timestamp (6) + signature (6)
    raw: bytes = b""

    @property
    def signed(self) -> bool:
        return self.signature is not None

    @property
    def link_id(self) -> int | None:
        return self.signature[0] if self.signature else None

    @property
    def timestamp(self) -> int | None:
        return int.from_bytes(self.signature[1:7], "little") if self.signature else None


def _signature(key: bytes, signed_part: bytes, link_id: int, timestamp: int) -> bytes:
    tail = bytes([link_id]) + timestamp.to_bytes(6, "little")
    return tail + hashlib.sha256(key + signed_part + tail).digest()[:6]


def encode(
    msg: Message,
    values: dict,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 1,
    version: int = 2,
    key: bytes | None = None,
    link_id: int = 0,
    timestamp: int = 0,
) -> bytes:
    payload = msg.pack(values)
    if version == 1:
        if key is not None:
            raise ValueError("MAVLink 1 has no signing; use version 2")
        if msg.id > 255:
            raise ValueError(f"{msg.name} (id {msg.id}) needs MAVLink 2")
        payload = payload[: msg.base_length]  # v1 has no extension fields
        header = bytes([STX_V1, len(payload), seq & 0xFF, sysid, compid, msg.id])
        crc = x25_accumulate([*header[1:], *payload, msg.crc_extra])
        return header + payload + crc.to_bytes(2, "little")
    payload = payload.rstrip(b"\0") or payload[:1]
    incompat = INCOMPAT_SIGNED if key is not None else 0
    header = bytes([STX_V2, len(payload), incompat, 0, seq & 0xFF, sysid, compid]) + msg.id.to_bytes(3, "little")
    crc = x25_accumulate([*header[1:], *payload, msg.crc_extra])
    frame = header + payload + crc.to_bytes(2, "little")
    if key is not None:
        frame += _signature(key, frame, link_id, timestamp)
    return frame


def verify_signature(packet: Packet, key: bytes) -> bool:
    if not packet.signed:
        return False
    body = packet.raw[: len(packet.raw) - 13]
    return _signature(key, body, packet.link_id, packet.timestamp) == packet.signature


@dataclass
class Parser:
    """Byte stream to packets, with the statistics a ground station shows as "link quality"."""

    dialect: Dialect = field(default_factory=Dialect.default)
    buffer: bytearray = field(default_factory=bytearray)
    crc_errors: int = 0
    unknown: int = 0
    duplicates: int = 0
    bad_flags: int = 0
    dropped_bytes: int = 0
    lost: int = 0
    last_seq: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)

    def feed(self, data: bytes) -> Iterator[Packet]:
        self.buffer += data
        while self.buffer:
            stx = self.buffer[0]
            if stx not in (STX_V1, STX_V2):
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            header_len = 10 if stx == STX_V2 else 6
            if len(self.buffer) < header_len:
                return
            n = self.buffer[1]
            signed = stx == STX_V2 and self.buffer[2] & INCOMPAT_SIGNED
            total = header_len + n + 2 + (13 if signed else 0)
            if len(self.buffer) < total:
                return
            frame = bytes(self.buffer[:total])
            if stx == STX_V2:
                seq, sysid, compid = frame[4], frame[5], frame[6]
                msgid = int.from_bytes(frame[7:10], "little")
                incompat, compat = frame[2], frame[3]
            else:
                seq, sysid, compid, msgid = frame[2], frame[3], frame[4], frame[5]
                incompat = compat = 0
            if incompat & ~INCOMPAT_SIGNED:
                # MAVLink 2 rule: a receiver must drop frames with incompatibility flags it does not understand
                self.bad_flags += 1
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            msg = self.dialect.messages.get(msgid)
            if msg is None:
                self.unknown += 1
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            payload = frame[header_len : header_len + n]
            crc = int.from_bytes(frame[header_len + n : header_len + n + 2], "little")
            if x25_accumulate([*frame[1:header_len], *payload, msg.crc_extra]) != crc:
                self.crc_errors += 1
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            del self.buffer[:total]
            key = (sysid, compid)
            if key in self.last_seq:
                gap = (seq - self.last_seq[key]) & 0xFF
                if gap == 0:
                    self.duplicates += 1
                else:
                    self.lost += gap - 1
            self.last_seq[key] = seq
            self.counts[msg.name] += 1
            yield Packet(
                1 if stx == STX_V1 else 2,
                seq,
                sysid,
                compid,
                msgid,
                payload,
                incompat,
                compat,
                frame[-13:] if signed else None,
                frame,
            )

    def decode(self, packet: Packet) -> tuple[str, dict]:
        msg = self.dialect.messages[packet.msgid]
        return msg.name, msg.unpack(packet.payload)


def frame_bytes(msg: Message, version: int = 2, signed: bool = False) -> int:
    """Worst case (untruncated) bytes on the wire for one message."""
    if version == 1:
        return 8 + msg.base_length
    return 12 + msg.length + (13 if signed else 0)


def read_tlog(data: bytes, dialect: Dialect | None = None) -> tuple[list[tuple[int, Packet]], Parser]:
    """A .tlog is what ground stations record: each frame preceded by an 8 byte big endian microsecond timestamp."""
    parser = Parser(dialect or Dialect.default())
    out: list[tuple[int, Packet]] = []
    i = 0
    while i + 8 + 8 <= len(data):
        usec = int.from_bytes(data[i : i + 8], "big")
        stx, n = data[i + 8], data[i + 9]
        if stx == STX_V2:
            size = 12 + n + (13 if data[i + 10] & INCOMPAT_SIGNED else 0)
        elif stx == STX_V1:
            size = 8 + n
        else:
            raise ValueError(f"not a tlog: byte {i + 8} is 0x{stx:02X}, expected a MAVLink start marker")
        for packet in parser.feed(data[i + 8 : i + 8 + size]):
            out.append((usec, packet))
        i += 8 + size
    return out, parser
