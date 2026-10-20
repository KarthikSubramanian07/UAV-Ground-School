"""DDS-XRCE, the protocol between PX4 and the Micro XRCE-DDS Agent, from the spec.

PX4 does not speak DDS. Its ``uxrce_dds_client`` module speaks DDS-XRCE (OMG
DDS for eXtremely Resource Constrained Environments 1.0) to an agent, and the
agent creates the real DDS participant, topics, writers and readers on PX4's
behalf. Every PX4 topic you see in ``ros2 topic list`` is an object the client
asked the agent to create, in binary, over UDP port 8888.

A message is a header and a run of submessages, each aligned to 4 bytes::

    session_id u8 | stream_id u8 | sequence_nr u16 LE | client_key[4] (only if session_id < 0x80)
    submessage_id u8 | flags u8 | length u16 LE | payload, CDR (flags bit 0: little endian)

Streams: 0 is the unordered "none" stream used to log in, 1..127 are best
effort, 128..255 reliable (sequence numbers, HEARTBEAT and ACKNACK, FRAGMENT
for anything bigger than the MTU). Objects are named by a 16 bit id: 12 bits
chosen by the client and 4 bits of kind (participant, topic, datawriter, ...).

The object representations PX4 uses are eProsima's binary ones (smaller than
XML): each is a CDR structure serialized into its own byte sequence, aligned
from its own start, inside the CREATE submessage. Field orders follow
Micro-XRCE-DDS-Client's ``xrce_types.c``, which the tests check against bytes
captured between PX4 v1.17 SITL and the v2.4.3 agent.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

SESSION_ID_WITHOUT_CLIENT_KEY = 0x80
COOKIE = b"XRCE"
VERSION = b"\x01\x00"
VENDOR_EPROSIMA = b"\x01\x0f"

SUBMESSAGES = {
    0: "CREATE_CLIENT",
    1: "CREATE",
    2: "GET_INFO",
    3: "DELETE",
    4: "STATUS_AGENT",
    5: "STATUS",
    6: "INFO",
    7: "WRITE_DATA",
    8: "READ_DATA",
    9: "DATA",
    10: "ACKNACK",
    11: "HEARTBEAT",
    12: "RESET",
    13: "FRAGMENT",
    14: "TIMESTAMP",
    15: "TIMESTAMP_REPLY",
}
SUB = {name: number for number, name in SUBMESSAGES.items()}

OBJECT_KINDS = {
    0x01: "PARTICIPANT",
    0x02: "TOPIC",
    0x03: "PUBLISHER",
    0x04: "SUBSCRIBER",
    0x05: "DATAWRITER",
    0x06: "DATAREADER",
    0x07: "REQUESTER",
    0x08: "REPLIER",
    0x0A: "TYPE",
    0x0B: "QOSPROFILE",
    0x0C: "APPLICATION",
    0x0D: "AGENT",
    0x0E: "CLIENT",
    0x0F: "OTHER",
}
KIND = {name: number for number, name in OBJECT_KINDS.items()}

REPRESENTATION_BY_REFERENCE = 0x01
REPRESENTATION_AS_XML = 0x02
REPRESENTATION_IN_BINARY = 0x03

STATUS = {
    0x00: "OK",
    0x01: "OK_MATCHED",
    0x80: "ERR_DDS_ERROR",
    0x81: "ERR_MISMATCH",
    0x82: "ERR_ALREADY_EXISTS",
    0x83: "ERR_DENIED",
    0x84: "ERR_UNKNOWN_REFERENCE",
    0x85: "ERR_INVALID_DATA",
    0x86: "ERR_INCOMPATIBLE",
    0x87: "ERR_RESOURCES",
}

# CREATE flags: what to do if the object exists
CREATE_REUSE = 0x02
CREATE_REPLACE = 0x04

# Endpoint QoS flags (OBJK_Endpoint_QosBinary.qos_flags)
QOS_RELIABLE = 0x01
QOS_KEEP_LAST = 0x02
QOS_OWNERSHIP_EXCLUSIVE = 0x04
QOS_TRANSIENT_LOCAL = 0x08
QOS_TRANSIENT = 0x10
QOS_PERSISTENT = 0x20

FORMAT_DATA = 0x00
FLAG_LAST_FRAGMENT = 0x02


class XrceError(ValueError):
    """A malformed or truncated DDS-XRCE message."""


# ------------------------------------------------------------------- objects


def object_id(number: int, kind: str | int) -> bytes:
    """The 2 byte object id: 12 bits of number, then 4 bits of kind."""
    k = KIND[kind] if isinstance(kind, str) else kind
    if not 0 <= number < 4096:
        raise XrceError(f"object number {number} does not fit in 12 bits")
    return bytes(((number >> 4) & 0xFF, ((number & 0x0F) << 4) | k))


def split_object_id(raw: bytes) -> tuple[int, str]:
    """(number, kind name) of a 2 byte object id."""
    number = (raw[0] << 4) | (raw[1] >> 4)
    return number, OBJECT_KINDS.get(raw[1] & 0x0F, f"KIND_{raw[1] & 0x0F}")


def describe_object(raw: bytes) -> str:
    number, kind = split_object_id(raw)
    return f"{kind.lower()} {number}"


# ---------------------------------------------------------------------- CDR


class Reader:
    """CDR primitives, aligned relative to `origin` (the start of the enclosing buffer)."""

    def __init__(self, data: bytes, pos: int = 0, little: bool = True, origin: int = 0, end: int | None = None):
        self.data = data
        self.pos = pos
        self.e = "<" if little else ">"
        self.origin = origin
        self.end = len(data) if end is None else end

    def align(self, n: int) -> None:
        self.pos += (-(self.pos - self.origin)) % n

    def take(self, n: int) -> bytes:
        if self.pos + n > self.end:
            raise XrceError(f"truncated: wanted {n} bytes at {self.pos}, have {self.end - self.pos}")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def boolean(self) -> bool:
        return self.take(1)[0] != 0

    def _num(self, fmt: str, size: int):
        self.align(size)
        return struct.unpack(self.e + fmt, self.take(size))[0]

    def u16(self) -> int:
        return self._num("H", 2)

    def i16(self) -> int:
        return self._num("h", 2)

    def u32(self) -> int:
        return self._num("I", 4)

    def i32(self) -> int:
        return self._num("i", 4)

    def u64(self) -> int:
        return self._num("Q", 8)

    def string(self) -> str:
        n = self.u32()
        raw = self.take(n)
        return raw[:-1].decode("utf-8", "replace") if n else ""

    def octets(self) -> bytes:
        return self.take(self.u32())

    def remaining(self) -> bytes:
        return self.take(self.end - self.pos)


class Writer:
    """CDR primitives, little endian, aligned relative to `origin`."""

    def __init__(self, origin: int = 0):
        self.buf = bytearray()
        self.origin = origin

    def align(self, n: int) -> None:
        self.buf += b"\x00" * ((-(len(self.buf) + self.origin)) % n)

    def raw(self, data: bytes) -> Writer:
        self.buf += data
        return self

    def u8(self, v: int) -> Writer:
        self.buf.append(v & 0xFF)
        return self

    def boolean(self, v: bool) -> Writer:
        return self.u8(1 if v else 0)

    def _num(self, fmt: str, size: int, v) -> Writer:
        self.align(size)
        self.buf += struct.pack("<" + fmt, v)
        return self

    def u16(self, v: int) -> Writer:
        return self._num("H", 2, v)

    def i16(self, v: int) -> Writer:
        return self._num("h", 2, v)

    def u32(self, v: int) -> Writer:
        return self._num("I", 4, v)

    def u64(self, v: int) -> Writer:
        return self._num("Q", 8, v)

    def string(self, s: str) -> Writer:
        b = s.encode() + b"\x00"
        self.u32(len(b))
        return self.raw(b)

    def octets(self, b: bytes) -> Writer:
        self.u32(len(b))
        return self.raw(b)

    def bytes(self) -> bytes:
        return bytes(self.buf)


# ------------------------------------------------------- binary representations


def _endpoint_qos(r: Reader) -> dict:
    flags = r.u16()
    qos = {
        "reliable": bool(flags & QOS_RELIABLE),
        "history": "keep_last" if flags & QOS_KEEP_LAST else "keep_all",
        "durability": "transient_local"
        if flags & QOS_TRANSIENT_LOCAL
        else "transient"
        if flags & QOS_TRANSIENT
        else "persistent"
        if flags & QOS_PERSISTENT
        else "volatile",
    }
    if flags & QOS_OWNERSHIP_EXCLUSIVE:
        qos["ownership"] = "exclusive"
    if r.boolean():
        qos["depth"] = r.u16()
    if r.boolean():
        qos["deadline_ms"] = r.u32()
    if r.boolean():
        qos["lifespan_ms"] = r.u32()
    if r.boolean():
        qos["user_data"] = r.octets().hex()
    return qos


def decode_binary(kind: str, blob: bytes) -> dict:
    """Decode the eProsima binary representation of an object of `kind`."""
    r = Reader(blob)  # its own buffer: aligned from its own start
    out: dict = {}
    if kind == "PARTICIPANT":
        if r.boolean():
            out["domain_reference"] = r.string()
        if r.boolean():
            out["qos_profile_reference"] = r.string()
    elif kind == "TOPIC":
        out["name"] = r.string()
        if r.boolean():
            out["type_reference"] = r.string()
        if r.boolean():
            out["type_name"] = r.string()
    elif kind in ("PUBLISHER", "SUBSCRIBER"):
        if r.boolean():
            out["name"] = r.string()
        if r.boolean():  # qos: partitions, group data
            if r.boolean():
                out["partitions"] = [r.string() for _ in range(r.u32())]
            if r.boolean():
                out["group_data"] = r.octets().hex()
    elif kind in ("DATAWRITER", "DATAREADER"):
        out["topic"] = describe_object(r.take(2))
        out["topic_id"] = r.data[r.pos - 2 : r.pos].hex()
        if r.boolean():
            out["qos"] = _endpoint_qos(r)
            if kind == "DATAWRITER":
                if r.boolean():
                    out["qos"]["ownership_strength"] = r.u64()
            else:
                if r.boolean():
                    out["qos"]["time_based_filter_ms"] = r.u64()
                if r.boolean():
                    out["qos"]["content_filter"] = r.string()
    elif kind in ("REQUESTER", "REPLIER"):  # a ROS 2 service, e.g. PX4's /fmu/vehicle_command
        out["service_name"] = r.string()
        out["request_type"] = r.string()
        out["reply_type"] = r.string()
        if r.boolean():
            out["request_topic"] = r.string()
        if r.boolean():
            out["reply_topic"] = r.string()
    else:
        out["raw"] = blob.hex()
    return out


def encode_binary(kind: str, **f) -> bytes:
    """The binary representation of a participant, topic, publisher, subscriber, writer or reader."""
    w = Writer()
    if kind == "PARTICIPANT":
        w.boolean(False).boolean(False)
    elif kind == "TOPIC":
        w.string(f["name"]).boolean(False).boolean(True).string(f["type_name"])
    elif kind in ("PUBLISHER", "SUBSCRIBER"):
        w.boolean(False).boolean(False)
    elif kind in ("DATAWRITER", "DATAREADER"):
        w.raw(f["topic_id"]).boolean(True)
        q = f.get("qos", {})
        flags = (QOS_RELIABLE if q.get("reliable") else 0) | (QOS_KEEP_LAST if q.get("history", "keep_last") == "keep_last" else 0)
        if q.get("durability") == "transient_local":
            flags |= QOS_TRANSIENT_LOCAL
        w.u16(flags)
        depth = q.get("depth")
        w.boolean(depth is not None)
        if depth is not None:
            w.u16(depth)
        w.boolean(False).boolean(False).boolean(False)  # deadline, lifespan, user data
        w.boolean(False)  # ownership strength / time based filter
        if kind == "DATAREADER":
            w.boolean(False)  # content filter
    else:
        raise XrceError(f"no binary representation for {kind}")
    return w.bytes()


# ----------------------------------------------------------------- payloads


def _representation3(r: Reader, kind: str) -> dict:
    fmt = r.u8()
    if fmt == REPRESENTATION_IN_BINARY:
        return {"format": "binary", **decode_binary(kind, r.octets())}
    if fmt == REPRESENTATION_AS_XML:
        return {"format": "xml", "xml": r.string()}
    if fmt == REPRESENTATION_BY_REFERENCE:
        return {"format": "reference", "reference": r.string()}
    raise XrceError(f"unknown representation format {fmt}")


def _properties(r: Reader) -> dict:
    return {r.string(): r.string() for _ in range(r.u32())}


def _client_representation(r: Reader) -> dict:
    out = {
        "cookie": r.take(4).decode("ascii", "replace"),
        "version": r.take(2).hex(),
        "vendor": r.take(2).hex(),
        "client_key": r.take(4).hex(),
        "session_id": r.u8(),
    }
    if r.boolean():
        out["properties"] = _properties(r)
    out["mtu"] = r.u16()
    return out


def _object_variant(r: Reader) -> dict:
    kind_number = r.u8()
    kind = OBJECT_KINDS.get(kind_number, f"KIND_{kind_number}")
    out: dict = {"kind": kind}
    if kind == "CLIENT":
        out.update(_client_representation(r))
    elif kind == "PARTICIPANT":
        out.update(_representation3(r, kind))
        out["domain_id"] = r.i16()
    elif kind in ("TOPIC", "PUBLISHER", "SUBSCRIBER", "DATAWRITER", "DATAREADER", "REQUESTER", "REPLIER"):
        out.update(_representation3(r, kind))
        out["parent"] = describe_object(r.take(2))
    else:
        out.update(_representation3(r, kind))
    return out


def _request(r: Reader) -> dict:
    request_id = r.take(2)
    raw = r.take(2)
    return {"request_id": int.from_bytes(request_id, "big"), "object": describe_object(raw), "object_id": raw.hex()}


def _result(r: Reader) -> dict:
    status = r.u8()
    return {"status": STATUS.get(status, f"0x{status:02x}"), "implementation_status": r.u8()}


def _time(r: Reader) -> float:
    return r.i32() + r.u32() * 1e-9


def decode_payload(sub_id: int, flags: int, r: Reader) -> dict:
    """Decode one submessage payload; unknown parts are returned as hex."""
    name = SUBMESSAGES.get(sub_id)
    if name == "CREATE_CLIENT":
        return {"client": _client_representation(r)}
    if name == "STATUS_AGENT":
        out = _result(r)
        out["cookie"] = r.take(4).decode("ascii", "replace")
        out["version"] = r.take(2).hex()
        out["vendor"] = r.take(2).hex()
        if r.boolean() and r.end - r.pos >= 4:
            out["properties"] = _properties(r)
        return out
    if name == "CREATE":
        out = _request(r)
        out["reuse"], out["replace"] = bool(flags & CREATE_REUSE), bool(flags & CREATE_REPLACE)
        out["representation"] = _object_variant(r)
        return out
    if name == "DELETE":
        return _request(r)
    if name == "STATUS":
        return {**_request(r), **_result(r)}
    if name == "READ_DATA":
        out = _request(r)
        out["preferred_stream"] = r.u8()
        out["data_format"] = r.u8()
        if r.boolean():
            out["content_filter"] = r.string()
        if r.boolean():
            out["delivery_control"] = {
                "max_samples": r.u16(),
                "max_elapsed_time": r.u16(),
                "max_bytes_per_second": r.u16(),
                "min_pace_period": r.u16(),
            }
        return out
    if name in ("WRITE_DATA", "DATA"):
        out = _request(r)
        out["format"] = flags & 0x0E
        out["data"] = r.remaining()
        return out
    if name == "HEARTBEAT":
        return {"first_unacked": r.u16(), "last_unacked": r.u16(), "stream": r.u8()}
    if name == "ACKNACK":
        return {"first_unacked": r.u16(), "nack_bitmap": r.take(2).hex(), "stream": r.u8()}
    if name == "FRAGMENT":
        return {"last": bool(flags & FLAG_LAST_FRAGMENT), "data": r.remaining()}
    if name == "TIMESTAMP":
        return {"transmit": _time(r)}
    if name == "TIMESTAMP_REPLY":
        return {"transmit": _time(r), "receive": _time(r), "originate": _time(r)}
    if name == "GET_INFO":
        return {**_request(r), "info_mask": r.u32()}
    return {"raw": r.remaining().hex()}


# ------------------------------------------------------------------ messages


@dataclass
class Submessage:
    id: int
    flags: int
    offset: int  # of the submessage header in the message
    length: int
    body: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return SUBMESSAGES.get(self.id, f"SUBMESSAGE_{self.id}")


@dataclass
class Message:
    session_id: int
    stream_id: int
    sequence: int
    client_key: bytes | None
    submessages: list[Submessage]
    size: int

    @property
    def reliable(self) -> bool:
        return self.stream_id >= 0x80


def parse(data: bytes, decode: bool = True) -> Message:
    """Split a datagram into its header and submessages (and decode their payloads)."""
    if len(data) < 4:
        raise XrceError("shorter than a message header")
    session_id, stream_id, sequence = data[0], data[1], struct.unpack_from("<H", data, 2)[0]
    pos, key = 4, None
    if session_id < SESSION_ID_WITHOUT_CLIENT_KEY:
        if len(data) < 8:
            raise XrceError("header says a client key follows, but the message ends")
        key, pos = data[4:8], 8
    subs = []
    while pos + 4 <= len(data):
        sub_id, flags, length = data[pos], data[pos + 1], struct.unpack_from("<H", data, pos + 2)[0]
        start, end = pos + 4, pos + 4 + length
        if end > len(data):
            raise XrceError(f"{SUBMESSAGES.get(sub_id, sub_id)} says {length} bytes but {len(data) - start} remain")
        sub = Submessage(sub_id, flags, pos, length)
        if decode:
            # payload CDR is aligned from the start of the message
            sub.body = decode_payload(sub_id, flags, Reader(data, start, bool(flags & 1), 0, end))
        subs.append(sub)
        pos = end + (-end) % 4
    return Message(session_id, stream_id, sequence, key, subs, len(data))


def header(session_id: int, stream_id: int, sequence: int, client_key: bytes) -> bytes:
    out = bytes((session_id, stream_id)) + struct.pack("<H", sequence & 0xFFFF)
    return out + (client_key if session_id < SESSION_ID_WITHOUT_CLIENT_KEY else b"")


def submessage(sub_id: int, flags: int, payload: bytes) -> bytes:
    return bytes((sub_id, flags | 0x01)) + struct.pack("<H", len(payload)) + payload


def message(head: bytes, *subs: bytes) -> bytes:
    """Join a header and submessages, padding each submessage to 4 bytes."""
    out = bytearray(head)
    for s in subs:
        out += b"\x00" * ((-len(out)) % 4)
        out += s
    return bytes(out)


def create_client_payload(client_key: bytes, session_id: int, mtu: int = 512) -> bytes:
    w = Writer(origin=0)
    w.raw(COOKIE).raw(VERSION).raw(VENDOR_EPROSIMA).raw(client_key).u8(session_id).boolean(False)
    w.u16(mtu)
    return w.bytes()


def create_payload(
    request_id: int, raw_id: bytes, kind: str, blob: bytes, parent: bytes | None = None, domain: int = 0, origin: int = 0
) -> bytes:
    """A CREATE payload; `origin` is the payload's offset in the message, for alignment."""
    w = Writer(origin=origin)
    w.raw(request_id.to_bytes(2, "big")).raw(raw_id).u8(KIND[kind]).u8(REPRESENTATION_IN_BINARY).octets(blob)
    if kind == "PARTICIPANT":
        w.i16(domain)
    else:
        w.raw(parent)
    return w.bytes()


def read_data_payload(request_id: int, reader_id: bytes, stream_id: int) -> bytes:
    w = Writer()
    w.raw(request_id.to_bytes(2, "big")).raw(reader_id).u8(stream_id).u8(FORMAT_DATA)
    w.boolean(False)  # no content filter
    w.boolean(True).u16(0xFFFF).u16(0).u16(0).u16(0)  # unlimited samples, as fast as they come
    return w.bytes()


def acknack_payload(first_unacked: int, stream_id: int, bitmap: int = 0) -> bytes:
    return Writer().u16(first_unacked).raw(bitmap.to_bytes(2, "big")).u8(stream_id).bytes()
