"""The RTPS wire protocol (DDSI-RTPS 2.5), the bytes underneath ROS 2.

ROS 2 does not define a network protocol of its own: every rmw implementation
hands messages to a DDS library, and DDS libraries talk to each other with the
OMG's Real-Time Publish-Subscribe protocol over UDP. This module encodes and
decodes it from the specification:

* the 20 byte message header and the submessages a ROS 2 graph uses: DATA,
  DATA_FRAG, HEARTBEAT, ACKNACK, GAP, INFO_TS, INFO_DST and INFO_SRC (spec 8.3,
  9.4);
* parameter lists, the self describing encoding used for discovery and inline
  QoS (9.4.2.11, 9.6.2);
* the participant and endpoint data that SPDP and SEDP exchange (8.5, 9.6.2.2),
  with the QoS policies ROS 2 maps onto them;
* the well known UDP port mapping (9.6.1.1);
* how ROS 2 names DDS topics and types (rmw's ``rt/``, ``rq/`` and ``rr/``
  prefixes and ``pkg::msg::dds_::Name_``).

Nothing here touches a socket; :mod:`week05.participant` does that.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass, field

PROTOCOL_VERSION = (2, 3)  # what Fast DDS 2.14 announces; 2.3+ framing is compatible
VENDOR_UGS = b"\x01\xbe"  # not in the OMG registry: this repository's own vendor id
VENDORS = {
    b"\x01\x01": "RTI Connext",
    b"\x01\x02": "OpenSplice",
    b"\x01\x03": "OpenDDS",
    b"\x01\x0f": "eProsima Fast DDS",
    b"\x01\x10": "Eclipse Cyclone DDS",
    b"\x01\x12": "GurumDDS",
    b"\x01\x13": "RustDDS",
    b"\x01\x14": "Zenoh",
    VENDOR_UGS: "week05 (this repository)",
}

# --------------------------------------------------------------- submessages --

PAD, ACKNACK, HEARTBEAT, GAP, INFO_TS = 0x01, 0x06, 0x07, 0x08, 0x09
INFO_SRC, INFO_REPLY_IP4, INFO_DST, INFO_REPLY = 0x0C, 0x0D, 0x0E, 0x0F
NACK_FRAG, HEARTBEAT_FRAG, DATA, DATA_FRAG = 0x12, 0x13, 0x15, 0x16
SUBMESSAGE_NAMES = {
    PAD: "PAD",
    ACKNACK: "ACKNACK",
    HEARTBEAT: "HEARTBEAT",
    GAP: "GAP",
    INFO_TS: "INFO_TS",
    INFO_SRC: "INFO_SRC",
    INFO_REPLY_IP4: "INFO_REPLY_IP4",
    INFO_DST: "INFO_DST",
    INFO_REPLY: "INFO_REPLY",
    NACK_FRAG: "NACK_FRAG",
    HEARTBEAT_FRAG: "HEARTBEAT_FRAG",
    DATA: "DATA",
    DATA_FRAG: "DATA_FRAG",
}

FLAG_E = 0x01  # little endian submessage
DATA_Q, DATA_D, DATA_K = 0x02, 0x04, 0x08  # inline QoS, data, key
FRAG_Q, FRAG_K = 0x02, 0x04
HB_F, HB_L = 0x02, 0x04  # final (no reply needed), liveliness
ACK_F = 0x02  # final
INFO_TS_I = 0x02  # invalidate: no timestamp

# ------------------------------------------------------------------ entities --

ENTITYID_UNKNOWN = 0x00000000
ENTITYID_PARTICIPANT = 0x000001C1
ENTITYID_SEDP_TOPICS_WRITER = 0x000002C2
ENTITYID_SEDP_TOPICS_READER = 0x000002C7
ENTITYID_SEDP_PUBLICATIONS_WRITER = 0x000003C2
ENTITYID_SEDP_PUBLICATIONS_READER = 0x000003C7
ENTITYID_SEDP_SUBSCRIPTIONS_WRITER = 0x000004C2
ENTITYID_SEDP_SUBSCRIPTIONS_READER = 0x000004C7
ENTITYID_SPDP_WRITER = 0x000100C2
ENTITYID_SPDP_READER = 0x000100C7
ENTITYID_P2P_MESSAGE_WRITER = 0x000200C2
ENTITYID_P2P_MESSAGE_READER = 0x000200C7
ENTITY_NAMES = {
    ENTITYID_UNKNOWN: "unknown",
    ENTITYID_PARTICIPANT: "participant",
    ENTITYID_SEDP_TOPICS_WRITER: "SEDP topics writer",
    ENTITYID_SEDP_TOPICS_READER: "SEDP topics reader",
    ENTITYID_SEDP_PUBLICATIONS_WRITER: "SEDP publications writer",
    ENTITYID_SEDP_PUBLICATIONS_READER: "SEDP publications reader",
    ENTITYID_SEDP_SUBSCRIPTIONS_WRITER: "SEDP subscriptions writer",
    ENTITYID_SEDP_SUBSCRIPTIONS_READER: "SEDP subscriptions reader",
    ENTITYID_SPDP_WRITER: "SPDP writer",
    ENTITYID_SPDP_READER: "SPDP reader",
    ENTITYID_P2P_MESSAGE_WRITER: "participant message writer",
    ENTITYID_P2P_MESSAGE_READER: "participant message reader",
}
# The low byte of a user entity id says what it is (spec 9.3.1.2).
KIND_WRITER_WITH_KEY, KIND_WRITER_NO_KEY = 0x02, 0x03
KIND_READER_NO_KEY, KIND_READER_WITH_KEY = 0x04, 0x07


def entity_name(entity_id: int) -> str:
    if entity_id in ENTITY_NAMES:
        return ENTITY_NAMES[entity_id]
    kind = entity_id & 0xFF
    role = {0x02: "writer", 0x03: "writer", 0x04: "reader", 0x07: "reader"}.get(kind & 0x3F, "entity")
    return f"user {role} {entity_id >> 8:#x}"


def is_writer(entity_id: int) -> bool:
    return entity_id & 0x0F in (0x02, 0x03)


# --------------------------------------------------------------------- ports --

PORT_BASE, DOMAIN_GAIN, PARTICIPANT_GAIN = 7400, 250, 2
D0, D1, D2, D3 = 0, 10, 1, 11
SPDP_MULTICAST_ADDRESS = "239.255.0.1"


def spdp_multicast_port(domain: int) -> int:
    return PORT_BASE + DOMAIN_GAIN * domain + D0


def metatraffic_unicast_port(domain: int, participant: int) -> int:
    return PORT_BASE + DOMAIN_GAIN * domain + D1 + PARTICIPANT_GAIN * participant


def user_multicast_port(domain: int) -> int:
    return PORT_BASE + DOMAIN_GAIN * domain + D2


def user_unicast_port(domain: int, participant: int) -> int:
    return PORT_BASE + DOMAIN_GAIN * domain + D3 + PARTICIPANT_GAIN * participant


def describe_port(port: int) -> str:
    """Invert the port mapping: which domain, and which kind of traffic."""
    offset = port - PORT_BASE
    if offset < 0:
        return "not RTPS"
    domain, rest = divmod(offset, DOMAIN_GAIN)
    if rest == D0:
        return f"domain {domain} discovery multicast"
    if rest == D2:
        return f"domain {domain} user multicast"
    if rest >= D1 and (rest - D1) % 2 == 0:
        return f"domain {domain} participant {(rest - D1) // 2} discovery unicast"
    if rest >= D3:
        return f"domain {domain} participant {(rest - D3) // 2} user unicast"
    return "not RTPS"


# --------------------------------------------------------------- primitives --


@dataclass(frozen=True, order=True)
class Guid:
    """A 12 byte participant prefix plus a 4 byte entity id."""

    prefix: bytes
    entity: int

    def __bytes__(self) -> bytes:
        return self.prefix + self.entity.to_bytes(4, "big")

    @classmethod
    def from_bytes(cls, raw: bytes) -> Guid:
        return cls(bytes(raw[:12]), int.from_bytes(raw[12:16], "big"))

    def __str__(self) -> str:
        p = self.prefix.hex()
        return f"{p[0:8]}.{p[8:16]}.{p[16:24]}|{self.entity:08x}"


@dataclass(frozen=True)
class Locator:
    kind: int  # 1 UDPv4, 2 UDPv6, 16 Fast DDS shared memory
    port: int
    address: bytes  # 16 bytes

    KINDS = {1: "udpv4", 2: "udpv6", 16: "shm", 0x01000000: "tcpv4"}

    @classmethod
    def udpv4(cls, ip: str, port: int) -> Locator:
        return cls(1, port, bytes(12) + bytes(map(int, ip.split("."))))

    @property
    def ip(self) -> str:
        if self.kind == 1:
            return ".".join(map(str, self.address[12:]))
        return self.address.hex()

    def __str__(self) -> str:
        return f"{self.KINDS.get(self.kind, self.kind)}:{self.ip}:{self.port}"


def seq_pack(n: int, e: str = "<") -> bytes:
    return struct.pack(e + "iI", n >> 32, n & 0xFFFFFFFF)


def seq_unpack(raw: bytes, offset: int, e: str = "<") -> int:
    high, low = struct.unpack_from(e + "iI", raw, offset)
    return (high << 32) | low


@dataclass(frozen=True)
class Duration:
    """RTPS Duration_t: seconds and a 2^-32 fraction. Infinite is (0x7fffffff, 0xffffffff)."""

    seconds: int
    fraction: int

    INFINITE_SECONDS, INFINITE_FRACTION = 0x7FFFFFFF, 0xFFFFFFFF

    @classmethod
    def infinite(cls) -> Duration:
        return cls(cls.INFINITE_SECONDS, cls.INFINITE_FRACTION)

    @classmethod
    def from_seconds(cls, value: float) -> Duration:
        whole = int(value)
        return cls(whole, min(0xFFFFFFFF, round((value - whole) * 2**32)))

    @property
    def is_infinite(self) -> bool:
        return self.seconds == self.INFINITE_SECONDS and self.fraction == self.INFINITE_FRACTION

    def to_seconds(self) -> float:
        return float("inf") if self.is_infinite else self.seconds + self.fraction / 2**32

    def pack(self, e: str = "<") -> bytes:
        return struct.pack(e + "iI", self.seconds, self.fraction)


# ----------------------------------------------------------- parameter lists --

PID_PAD = 0x0000
PID_SENTINEL = 0x0001
PID_PARTICIPANT_LEASE_DURATION = 0x0002
PID_TIME_BASED_FILTER = 0x0004
PID_TOPIC_NAME = 0x0005
PID_OWNERSHIP_STRENGTH = 0x0006
PID_TYPE_NAME = 0x0007
PID_DOMAIN_ID = 0x000F
PID_PROTOCOL_VERSION = 0x0015
PID_VENDORID = 0x0016
PID_RELIABILITY = 0x001A
PID_LIVELINESS = 0x001B
PID_DURABILITY = 0x001D
PID_DURABILITY_SERVICE = 0x001E
PID_OWNERSHIP = 0x001F
PID_PRESENTATION = 0x0021
PID_DEADLINE = 0x0023
PID_DESTINATION_ORDER = 0x0025
PID_LATENCY_BUDGET = 0x0027
PID_PARTITION = 0x0029
PID_LIFESPAN = 0x002B
PID_USER_DATA = 0x002C
PID_GROUP_DATA = 0x002D
PID_TOPIC_DATA = 0x002E
PID_UNICAST_LOCATOR = 0x002F
PID_MULTICAST_LOCATOR = 0x0030
PID_DEFAULT_UNICAST_LOCATOR = 0x0031
PID_METATRAFFIC_UNICAST_LOCATOR = 0x0032
PID_METATRAFFIC_MULTICAST_LOCATOR = 0x0033
PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT = 0x0034
PID_CONTENT_FILTER_PROPERTY = 0x0035
PID_HISTORY = 0x0040
PID_RESOURCE_LIMITS = 0x0041
PID_EXPECTS_INLINE_QOS = 0x0043
PID_DEFAULT_MULTICAST_LOCATOR = 0x0048
PID_TRANSPORT_PRIORITY = 0x0049
PID_PARTICIPANT_GUID = 0x0050
PID_GROUP_GUID = 0x0052
PID_BUILTIN_ENDPOINT_SET = 0x0058
PID_PROPERTY_LIST = 0x0059
PID_ENDPOINT_GUID = 0x005A
PID_TYPE_MAX_SIZE_SERIALIZED = 0x0060
PID_ENTITY_NAME = 0x0062
PID_KEY_HASH = 0x0070
PID_STATUS_INFO = 0x0071
PID_DATA_REPRESENTATION = 0x0073
PID_TYPE_CONSISTENCY_ENFORCEMENT = 0x0074
PID_TYPE_INFORMATION = 0x0075
PID_BUILTIN_ENDPOINT_QOS = 0x0077
PID_RELATED_SAMPLE_IDENTITY = 0x0083  # DDS-RPC
PID_DOMAIN_TAG = 0x4014
# eProsima vendor specific parameters (only meaningful when the vendor is Fast DDS)
PID_FASTDDS_PERSISTENCE_GUID = 0x8002
PID_FASTDDS_DISABLE_POSITIVE_ACKS = 0x8005
PID_FASTDDS_DATASHARING = 0x8006
PID_FASTDDS_NETWORK_CONFIGURATION_SET = 0x8007
PID_FASTDDS_RELATED_SAMPLE_IDENTITY = 0x800F

PID_NAMES = {v: k[4:] for k, v in globals().items() if k.startswith("PID_") and isinstance(v, int)}

BUILTIN_ENDPOINTS = {
    0: "participant announcer",
    1: "participant detector",
    2: "publications announcer",
    3: "publications detector",
    4: "subscriptions announcer",
    5: "subscriptions detector",
    10: "participant message writer",
    11: "participant message reader",
    28: "type lookup request writer",
    29: "type lookup request reader",
    30: "type lookup reply writer",
    31: "type lookup reply reader",
}


@dataclass
class ParameterList:
    """An ordered list of (parameter id, raw value). Repeated ids are allowed (locators)."""

    items: list[tuple[int, bytes]] = field(default_factory=list)

    def get(self, pid: int, default: bytes | None = None) -> bytes | None:
        for p, v in self.items:
            if p == pid:
                return v
        return default

    def all(self, pid: int) -> list[bytes]:
        return [v for p, v in self.items if p == pid]

    def add(self, pid: int, value: bytes) -> ParameterList:
        self.items.append((pid, value))
        return self

    def encode(self, e: str = "<") -> bytes:
        out = bytearray()
        for pid, value in self.items:
            padded = value + bytes(-len(value) % 4)
            out += struct.pack(e + "HH", pid, len(padded)) + padded
        out += struct.pack(e + "HH", PID_SENTINEL, 0)
        return bytes(out)

    @classmethod
    def decode(cls, raw: bytes, offset: int = 0, e: str = "<") -> tuple[ParameterList, int]:
        """Decode from ``offset``; returns the list and the offset after the sentinel."""
        items = []
        while offset + 4 <= len(raw):
            pid, length = struct.unpack_from(e + "HH", raw, offset)
            offset += 4
            if pid == PID_SENTINEL:
                return cls(items), offset
            if pid != PID_PAD:
                items.append((pid & 0x3FFF if pid & 0x4000 and pid != PID_DOMAIN_TAG else pid, bytes(raw[offset : offset + length])))
            offset += length
        raise ValueError("parameter list has no sentinel")


def cdr_string(text: str, e: str = "<") -> bytes:
    raw = text.encode() + b"\0"
    return struct.pack(e + "I", len(raw)) + raw


def read_cdr_string(raw: bytes, offset: int = 0, e: str = "<") -> str:
    (n,) = struct.unpack_from(e + "I", raw, offset)
    return raw[offset + 4 : offset + 4 + n].rstrip(b"\0").decode("utf-8", "replace")


def pack_locator(loc: Locator, e: str = "<") -> bytes:
    return struct.pack(e + "iI", loc.kind, loc.port) + loc.address


def unpack_locator(raw: bytes, e: str = "<") -> Locator:
    kind, port = struct.unpack_from(e + "iI", raw, 0)
    return Locator(kind, port, bytes(raw[8:24]))


def pack_properties(props: dict[str, str], e: str = "<") -> bytes:
    out = bytearray(struct.pack(e + "I", len(props)))
    for k, v in props.items():
        for s in (k, v):
            out += cdr_string(s, e)
            out += bytes(-len(out) % 4)
    return bytes(out)


def unpack_properties(raw: bytes, e: str = "<") -> dict[str, str]:
    (n,) = struct.unpack_from(e + "I", raw, 0)
    offset, props = 4, {}
    for _ in range(n):
        strings = []
        for _ in range(2):
            offset += -offset % 4
            (length,) = struct.unpack_from(e + "I", raw, offset)
            strings.append(raw[offset + 4 : offset + 4 + length].rstrip(b"\0").decode("utf-8", "replace"))
            offset += 4 + length
        props[strings[0]] = strings[1]
    return props


# ------------------------------------------------------------------------ QoS --
# The QoS policies that ROS 2 sets, as DDS sends them in discovery (DDS 1.4 2.2.3).

RELIABILITY = {1: "best_effort", 2: "reliable"}
DURABILITY = {0: "volatile", 1: "transient_local", 2: "transient", 3: "persistent"}
LIVELINESS = {0: "automatic", 1: "manual_by_participant", 2: "manual_by_topic"}
HISTORY = {0: "keep_last", 1: "keep_all"}


@dataclass
class EndpointQos:
    reliability: str = "best_effort"  # DDS default for readers; writers default to reliable
    durability: str = "volatile"
    history: str = "keep_last"
    depth: int = 1
    deadline: Duration = field(default_factory=Duration.infinite)
    liveliness: str = "automatic"
    lease_duration: Duration = field(default_factory=Duration.infinite)
    lifespan: Duration = field(default_factory=Duration.infinite)
    max_blocking_time: Duration = field(default_factory=lambda: Duration.from_seconds(0.1))

    def encode_into(self, plist: ParameterList, e: str = "<", writer: bool = True) -> None:
        inv = {v: k for k, v in RELIABILITY.items()}
        plist.add(PID_RELIABILITY, struct.pack(e + "I", inv[self.reliability]) + self.max_blocking_time.pack(e))
        plist.add(PID_DURABILITY, struct.pack(e + "I", {v: k for k, v in DURABILITY.items()}[self.durability]))
        if not self.deadline.is_infinite:
            plist.add(PID_DEADLINE, self.deadline.pack(e))
        lv = {v: k for k, v in LIVELINESS.items()}[self.liveliness]
        plist.add(PID_LIVELINESS, struct.pack(e + "I", lv) + self.lease_duration.pack(e))
        if writer and not self.lifespan.is_infinite:
            plist.add(PID_LIFESPAN, self.lifespan.pack(e))
        plist.add(PID_HISTORY, struct.pack(e + "Ii", {v: k for k, v in HISTORY.items()}[self.history], self.depth))

    @classmethod
    def decode_from(cls, plist: ParameterList, e: str = "<", writer: bool = True) -> EndpointQos:
        q = cls(reliability="reliable" if writer else "best_effort")
        if (raw := plist.get(PID_RELIABILITY)) is not None:
            kind = struct.unpack_from(e + "I", raw, 0)[0]
            q.reliability = RELIABILITY.get(kind, f"unknown({kind})")
            if len(raw) >= 12:
                q.max_blocking_time = Duration(*struct.unpack_from(e + "iI", raw, 4))
        if (raw := plist.get(PID_DURABILITY)) is not None:
            kind = struct.unpack_from(e + "I", raw, 0)[0]
            q.durability = DURABILITY.get(kind, f"unknown({kind})")
        if (raw := plist.get(PID_DEADLINE)) is not None:
            q.deadline = Duration(*struct.unpack_from(e + "iI", raw, 0))
        if (raw := plist.get(PID_LIVELINESS)) is not None:
            kind = struct.unpack_from(e + "I", raw, 0)[0]
            q.liveliness = LIVELINESS.get(kind, f"unknown({kind})")
            q.lease_duration = Duration(*struct.unpack_from(e + "iI", raw, 4))
        if (raw := plist.get(PID_LIFESPAN)) is not None:
            q.lifespan = Duration(*struct.unpack_from(e + "iI", raw, 0))
        if (raw := plist.get(PID_HISTORY)) is not None:
            kind, depth = struct.unpack_from(e + "Ii", raw, 0)
            q.history, q.depth = HISTORY.get(kind, f"unknown({kind})"), depth
        return q


# ------------------------------------------------------------------- message --


@dataclass
class Submessage:
    kind: int
    flags: int
    body: bytes  # the raw bytes after the 4 byte submessage header

    @property
    def endian(self) -> str:
        return "<" if self.flags & FLAG_E else ">"

    @property
    def name(self) -> str:
        return SUBMESSAGE_NAMES.get(self.kind, f"0x{self.kind:02x}")


@dataclass
class Data:
    """DATA (spec 9.4.5.3): one sample, or a disposal, from a writer."""

    reader: int
    writer: int
    seq: int
    inline_qos: ParameterList | None
    payload: bytes | None  # serialized payload, encapsulation header included
    key_only: bool = False


@dataclass
class DataFrag:
    reader: int
    writer: int
    seq: int
    first_fragment: int  # 1 based
    fragments_in_submessage: int
    fragment_size: int
    sample_size: int
    inline_qos: ParameterList | None
    payload: bytes


@dataclass
class Heartbeat:
    reader: int
    writer: int
    first: int
    last: int
    count: int
    final: bool
    liveliness: bool


@dataclass
class AckNack:
    reader: int
    writer: int
    base: int  # every sequence number below base is acknowledged
    missing: list[int]
    count: int
    final: bool


@dataclass
class Gap:
    reader: int
    writer: int
    start: int
    list_base: int
    irrelevant: list[int]


def _seqset_pack(base: int, members: Iterable[int], e: str = "<") -> bytes:
    members = sorted(m for m in members if m >= base)
    num_bits = (members[-1] - base + 1) if members else 0
    if num_bits > 256:
        raise ValueError("a sequence number set spans at most 256 numbers")
    words = [0] * ((num_bits + 31) // 32)
    for m in members:
        i = m - base
        words[i // 32] |= 1 << (31 - i % 32)
    return seq_pack(base, e) + struct.pack(e + "I", num_bits) + b"".join(struct.pack(e + "I", w) for w in words)


def _seqset_unpack(raw: bytes, offset: int, e: str = "<") -> tuple[int, list[int], int]:
    base = seq_unpack(raw, offset, e)
    (num_bits,) = struct.unpack_from(e + "I", raw, offset + 8)
    words = struct.unpack_from(e + "I" * ((num_bits + 31) // 32), raw, offset + 12)
    members = [base + i for i in range(num_bits) if words[i // 32] >> (31 - i % 32) & 1]
    return base, members, offset + 12 + 4 * len(words)


def decode_submessage(sub: Submessage):
    """Turn a raw submessage into its dataclass, or return it unchanged if not modelled."""
    e, b = sub.endian, sub.body
    if sub.kind == DATA:
        _extra, to_qos = struct.unpack_from(e + "HH", b, 0)
        reader, writer = struct.unpack_from(">II", b, 4)
        seq = seq_unpack(b, 12, e)
        offset = 4 + to_qos  # counted from the end of the octetsToInlineQos field
        qos = None
        if sub.flags & DATA_Q:
            qos, offset = ParameterList.decode(b, offset, e)
        payload = bytes(b[offset:]) if sub.flags & (DATA_D | DATA_K) else None
        return Data(reader, writer, seq, qos, payload, key_only=bool(sub.flags & DATA_K and not sub.flags & DATA_D))
    if sub.kind == DATA_FRAG:
        _extra, to_qos = struct.unpack_from(e + "HH", b, 0)
        reader, writer = struct.unpack_from(">II", b, 4)
        seq = seq_unpack(b, 12, e)
        first, count, size, total = struct.unpack_from(e + "IHHI", b, 20)
        offset = 4 + to_qos  # counted from the end of the octetsToInlineQos field
        qos = None
        if sub.flags & FRAG_Q:
            qos, offset = ParameterList.decode(b, offset, e)
        return DataFrag(reader, writer, seq, first, count, size, total, qos, bytes(b[offset:]))
    if sub.kind == HEARTBEAT:
        reader, writer = struct.unpack_from(">II", b, 0)
        first, last = seq_unpack(b, 8, e), seq_unpack(b, 16, e)
        (count,) = struct.unpack_from(e + "i", b, 24)
        return Heartbeat(reader, writer, first, last, count, bool(sub.flags & HB_F), bool(sub.flags & HB_L))
    if sub.kind == ACKNACK:
        reader, writer = struct.unpack_from(">II", b, 0)
        base, missing, offset = _seqset_unpack(b, 8, e)
        (count,) = struct.unpack_from(e + "i", b, offset)
        return AckNack(reader, writer, base, missing, count, bool(sub.flags & ACK_F))
    if sub.kind == GAP:
        reader, writer = struct.unpack_from(">II", b, 0)
        start = seq_unpack(b, 8, e)
        base, members, _ = _seqset_unpack(b, 16, e)
        return Gap(reader, writer, start, base, members)
    return sub


@dataclass
class Message:
    """A whole RTPS datagram: header plus submessages."""

    version: tuple[int, int]
    vendor: bytes
    prefix: bytes
    submessages: list[Submessage]

    @property
    def vendor_name(self) -> str:
        return VENDORS.get(self.vendor, f"vendor {self.vendor.hex()}")


def parse(datagram: bytes) -> Message | None:
    """Parse a datagram; ``None`` if it is not RTPS. Truncated submessages raise ``ValueError``."""
    if len(datagram) < 20 or datagram[:4] != b"RTPS":
        return None
    version, vendor, prefix = (datagram[4], datagram[5]), datagram[6:8], datagram[8:20]
    subs, offset = [], 20
    while offset + 4 <= len(datagram):
        kind, flags = datagram[offset], datagram[offset + 1]
        e = "<" if flags & FLAG_E else ">"
        (length,) = struct.unpack_from(e + "H", datagram, offset + 2)
        start = offset + 4
        # octetsToNextHeader 0 means "to the end of the message" for anything but PAD/INFO_TS
        end = len(datagram) if length == 0 and kind not in (PAD, INFO_TS) else start + length
        if end > len(datagram):
            raise ValueError(f"truncated {SUBMESSAGE_NAMES.get(kind, kind)} submessage")
        subs.append(Submessage(kind, flags, bytes(datagram[start:end])))
        offset = end
    return Message(version, vendor, bytes(prefix), subs)


@dataclass
class Received:
    """A decoded submessage with the context the interpreter state machine gives it (8.3.4)."""

    source_prefix: bytes
    dest_prefix: bytes | None
    timestamp: float | None
    submessage: object


def interpret(msg: Message) -> list[Received]:
    """Apply INFO_TS, INFO_SRC and INFO_DST to the submessages that follow them."""
    source, dest, ts = msg.prefix, None, None
    out = []
    for sub in msg.submessages:
        e = sub.endian
        if sub.kind == INFO_TS:
            ts = None if sub.flags & INFO_TS_I else (lambda s, f: s + f / 2**32)(*struct.unpack_from(e + "iI", sub.body, 0))
        elif sub.kind == INFO_DST:
            dest = bytes(sub.body[:12])
            dest = None if dest == bytes(12) else dest
        elif sub.kind == INFO_SRC:
            source = bytes(sub.body[8:20])
        elif sub.kind != PAD:
            out.append(Received(source, dest, ts, decode_submessage(sub)))
    return out


class MessageBuilder:
    """Assemble an outgoing RTPS message, little endian, one submessage at a time."""

    def __init__(self, prefix: bytes, vendor: bytes = VENDOR_UGS) -> None:
        self.buf = bytearray(b"RTPS" + bytes(PROTOCOL_VERSION) + vendor + prefix)

    def _sub(self, kind: int, flags: int, body: bytes) -> MessageBuilder:
        body += bytes(-len(body) % 4)
        self.buf += struct.pack("<BBH", kind, flags | FLAG_E, len(body)) + body
        return self

    def info_ts(self, t: float) -> MessageBuilder:
        sec = int(t)
        return self._sub(INFO_TS, 0, struct.pack("<iI", sec, int((t - sec) * 2**32) & 0xFFFFFFFF))

    def info_dst(self, prefix: bytes) -> MessageBuilder:
        return self._sub(INFO_DST, 0, prefix)

    def data(self, reader: int, writer: int, seq: int, payload: bytes | None, inline_qos: ParameterList | None = None) -> MessageBuilder:
        flags = (DATA_Q if inline_qos else 0) | (DATA_D if payload is not None else 0)
        body = struct.pack("<HH", 0, 16) + struct.pack(">II", reader, writer) + seq_pack(seq)
        if inline_qos:
            body += inline_qos.encode()
        if payload is not None:
            body += payload
        return self._sub(DATA, flags, body)

    def data_frag(
        self, reader: int, writer: int, seq: int, first: int, fragment_size: int, sample: bytes, count: int = 1
    ) -> MessageBuilder:
        start = (first - 1) * fragment_size
        chunk = sample[start : start + fragment_size * count]
        body = struct.pack("<HH", 0, 28) + struct.pack(">II", reader, writer) + seq_pack(seq)
        body += struct.pack("<IHHI", first, count, fragment_size, len(sample)) + chunk
        return self._sub(DATA_FRAG, 0, body)

    def heartbeat(self, reader: int, writer: int, first: int, last: int, count: int, final: bool = False) -> MessageBuilder:
        body = struct.pack(">II", reader, writer) + seq_pack(first) + seq_pack(last) + struct.pack("<i", count)
        return self._sub(HEARTBEAT, HB_F if final else 0, body)

    def acknack(self, reader: int, writer: int, base: int, missing: Iterable[int], count: int, final: bool = True) -> MessageBuilder:
        body = struct.pack(">II", reader, writer) + _seqset_pack(base, missing) + struct.pack("<i", count)
        return self._sub(ACKNACK, ACK_F if final else 0, body)

    def gap(self, reader: int, writer: int, start: int, list_base: int, members: Iterable[int] = ()) -> MessageBuilder:
        body = struct.pack(">II", reader, writer) + seq_pack(start) + _seqset_pack(list_base, members)
        return self._sub(GAP, 0, body)

    def __bytes__(self) -> bytes:
        return bytes(self.buf)

    def __len__(self) -> int:
        return len(self.buf)


# ------------------------------------------------------------------ discovery --


@dataclass
class ParticipantData:
    """What SPDP announces about a participant (spec 8.5.3.2, 9.6.2.2.1)."""

    guid_prefix: bytes
    protocol_version: tuple[int, int] = PROTOCOL_VERSION
    vendor: bytes = VENDOR_UGS
    metatraffic_unicast: list[Locator] = field(default_factory=list)
    metatraffic_multicast: list[Locator] = field(default_factory=list)
    default_unicast: list[Locator] = field(default_factory=list)
    default_multicast: list[Locator] = field(default_factory=list)
    builtin_endpoints: int = 0
    lease_duration: Duration = field(default_factory=lambda: Duration.from_seconds(20))
    entity_name: str = ""
    user_data: bytes = b""
    properties: dict[str, str] = field(default_factory=dict)
    domain_id: int | None = None
    extra: list[tuple[int, bytes]] = field(default_factory=list)  # parameters this module does not model

    @property
    def vendor_name(self) -> str:
        return VENDORS.get(self.vendor, f"vendor {self.vendor.hex()}")

    @property
    def enclave(self) -> str:
        """ROS 2 puts ``enclave=/;`` in the participant's user data."""
        text = self.user_data.decode("utf-8", "replace")
        for part in text.split(";"):
            if part.startswith("enclave="):
                return part[len("enclave=") :]
        return ""

    def builtin_endpoint_names(self) -> list[str]:
        return [name for bit, name in BUILTIN_ENDPOINTS.items() if self.builtin_endpoints >> bit & 1]

    def encode(self) -> bytes:
        e = "<"
        pl = ParameterList()
        pl.add(PID_PROTOCOL_VERSION, bytes(self.protocol_version))
        pl.add(PID_VENDORID, self.vendor)
        pl.add(PID_PARTICIPANT_GUID, bytes(Guid(self.guid_prefix, ENTITYID_PARTICIPANT)))
        if self.domain_id is not None:
            pl.add(PID_DOMAIN_ID, struct.pack(e + "I", self.domain_id))
        for pid, locs in (
            (PID_METATRAFFIC_UNICAST_LOCATOR, self.metatraffic_unicast),
            (PID_METATRAFFIC_MULTICAST_LOCATOR, self.metatraffic_multicast),
            (PID_DEFAULT_UNICAST_LOCATOR, self.default_unicast),
            (PID_DEFAULT_MULTICAST_LOCATOR, self.default_multicast),
        ):
            for loc in locs:
                pl.add(pid, pack_locator(loc, e))
        pl.add(PID_PARTICIPANT_LEASE_DURATION, self.lease_duration.pack(e))
        pl.add(PID_BUILTIN_ENDPOINT_SET, struct.pack(e + "I", self.builtin_endpoints))
        if self.entity_name:
            pl.add(PID_ENTITY_NAME, cdr_string(self.entity_name, e))
        if self.user_data:
            pl.add(PID_USER_DATA, struct.pack(e + "I", len(self.user_data)) + self.user_data)
        if self.properties:
            pl.add(PID_PROPERTY_LIST, pack_properties(self.properties, e))
        return b"\x00\x03\x00\x00" + pl.encode(e)

    @classmethod
    def decode(cls, payload: bytes) -> ParticipantData:
        e = _payload_endian(payload)
        pl, _ = ParameterList.decode(payload, 4, e)
        guid = pl.get(PID_PARTICIPANT_GUID)
        if guid is None:
            raise ValueError("participant data without PID_PARTICIPANT_GUID")
        p = cls(guid_prefix=bytes(guid[:12]))
        modelled = {
            PID_PARTICIPANT_GUID,
            PID_PROTOCOL_VERSION,
            PID_VENDORID,
            PID_METATRAFFIC_UNICAST_LOCATOR,
            PID_METATRAFFIC_MULTICAST_LOCATOR,
            PID_DEFAULT_UNICAST_LOCATOR,
            PID_DEFAULT_MULTICAST_LOCATOR,
            PID_BUILTIN_ENDPOINT_SET,
            PID_PARTICIPANT_LEASE_DURATION,
            PID_ENTITY_NAME,
            PID_USER_DATA,
            PID_PROPERTY_LIST,
            PID_DOMAIN_ID,
        }
        for pid, raw in pl.items:
            if pid == PID_PROTOCOL_VERSION:
                p.protocol_version = (raw[0], raw[1])
            elif pid == PID_VENDORID:
                p.vendor = bytes(raw[:2])
            elif pid == PID_METATRAFFIC_UNICAST_LOCATOR:
                p.metatraffic_unicast.append(unpack_locator(raw, e))
            elif pid == PID_METATRAFFIC_MULTICAST_LOCATOR:
                p.metatraffic_multicast.append(unpack_locator(raw, e))
            elif pid == PID_DEFAULT_UNICAST_LOCATOR:
                p.default_unicast.append(unpack_locator(raw, e))
            elif pid == PID_DEFAULT_MULTICAST_LOCATOR:
                p.default_multicast.append(unpack_locator(raw, e))
            elif pid == PID_BUILTIN_ENDPOINT_SET:
                p.builtin_endpoints = struct.unpack_from(e + "I", raw, 0)[0]
            elif pid == PID_PARTICIPANT_LEASE_DURATION:
                p.lease_duration = Duration(*struct.unpack_from(e + "iI", raw, 0))
            elif pid == PID_ENTITY_NAME:
                p.entity_name = read_cdr_string(raw, 0, e)
            elif pid == PID_USER_DATA:
                (n,) = struct.unpack_from(e + "I", raw, 0)
                p.user_data = bytes(raw[4 : 4 + n])
            elif pid == PID_PROPERTY_LIST:
                p.properties = unpack_properties(raw, e)
            elif pid == PID_DOMAIN_ID:
                p.domain_id = struct.unpack_from(e + "I", raw, 0)[0]
            elif pid not in modelled:
                p.extra.append((pid, raw))
        return p


@dataclass
class EndpointData:
    """What SEDP announces about a writer (DiscoveredWriterData) or reader (DiscoveredReaderData)."""

    guid: Guid
    topic: str
    type_name: str
    qos: EndpointQos
    writer: bool
    unicast: list[Locator] = field(default_factory=list)
    multicast: list[Locator] = field(default_factory=list)
    user_data: bytes = b""
    properties: dict[str, str] = field(default_factory=dict)
    expects_inline_qos: bool = False
    type_information: bytes = b""
    extra: list[tuple[int, bytes]] = field(default_factory=list)

    @property
    def type_hash(self) -> str:
        """ROS 2 Jazzy puts ``typehash=RIHS01_...;`` in every endpoint's user data."""
        for part in self.user_data.decode("utf-8", "replace").split(";"):
            if part.startswith("typehash="):
                return part[len("typehash=") :]
        return ""

    def encode(self) -> bytes:
        e = "<"
        pl = ParameterList()
        pl.add(PID_ENDPOINT_GUID, bytes(self.guid))
        pl.add(PID_PARTICIPANT_GUID, bytes(Guid(self.guid.prefix, ENTITYID_PARTICIPANT)))
        pl.add(PID_TOPIC_NAME, cdr_string(self.topic, e))
        pl.add(PID_TYPE_NAME, cdr_string(self.type_name, e))
        self.qos.encode_into(pl, e, writer=self.writer)
        for loc in self.unicast:
            pl.add(PID_UNICAST_LOCATOR, pack_locator(loc, e))
        for loc in self.multicast:
            pl.add(PID_MULTICAST_LOCATOR, pack_locator(loc, e))
        if self.user_data:
            pl.add(PID_USER_DATA, struct.pack(e + "I", len(self.user_data)) + self.user_data)
        if self.properties:
            pl.add(PID_PROPERTY_LIST, pack_properties(self.properties, e))
        if not self.writer:
            pl.add(PID_EXPECTS_INLINE_QOS, b"\x01" if self.expects_inline_qos else b"\x00")
        return b"\x00\x03\x00\x00" + pl.encode(e)

    @classmethod
    def decode(cls, payload: bytes, writer: bool) -> EndpointData:
        e = _payload_endian(payload)
        pl, _ = ParameterList.decode(payload, 4, e)
        raw_guid = pl.get(PID_ENDPOINT_GUID)
        if raw_guid is None:
            raise ValueError("endpoint data without PID_ENDPOINT_GUID")
        ep = cls(
            guid=Guid.from_bytes(raw_guid),
            topic=read_cdr_string(pl.get(PID_TOPIC_NAME, b"\0\0\0\0"), 0, e),
            type_name=read_cdr_string(pl.get(PID_TYPE_NAME, b"\0\0\0\0"), 0, e),
            qos=EndpointQos.decode_from(pl, e, writer=writer),
            writer=writer,
        )
        modelled = {
            PID_ENDPOINT_GUID,
            PID_TOPIC_NAME,
            PID_TYPE_NAME,
            PID_RELIABILITY,
            PID_DURABILITY,
            PID_DEADLINE,
            PID_LIVELINESS,
            PID_LIFESPAN,
            PID_HISTORY,
            PID_PARTICIPANT_GUID,
            PID_UNICAST_LOCATOR,
            PID_MULTICAST_LOCATOR,
            PID_USER_DATA,
            PID_PROPERTY_LIST,
            PID_EXPECTS_INLINE_QOS,
            PID_TYPE_INFORMATION,
        }
        for pid, raw in pl.items:
            if pid == PID_UNICAST_LOCATOR:
                ep.unicast.append(unpack_locator(raw, e))
            elif pid == PID_MULTICAST_LOCATOR:
                ep.multicast.append(unpack_locator(raw, e))
            elif pid == PID_USER_DATA:
                (n,) = struct.unpack_from(e + "I", raw, 0)
                ep.user_data = bytes(raw[4 : 4 + n])
            elif pid == PID_PROPERTY_LIST:
                ep.properties = unpack_properties(raw, e)
            elif pid == PID_EXPECTS_INLINE_QOS:
                ep.expects_inline_qos = bool(raw[0])
            elif pid == PID_TYPE_INFORMATION:
                ep.type_information = raw
            elif pid not in modelled:
                ep.extra.append((pid, raw))
        return ep


def _payload_endian(payload: bytes) -> str:
    scheme = payload[:2]
    if scheme in (b"\x00\x02", b"\x00\x00"):
        return ">"
    if scheme in (b"\x00\x03", b"\x00\x01"):
        return "<"
    raise ValueError(f"unknown encapsulation {scheme.hex()}")


# ----------------------------------------------------------------- ROS names --

TOPIC_PREFIX, REQUEST_PREFIX, REPLY_PREFIX = "rt", "rq", "rr"


def ros_topic_to_dds(topic: str) -> str:
    """``/chatter`` is the DDS topic ``rt/chatter`` (rmw_dds_common's convention)."""
    return TOPIC_PREFIX + "/" + topic.lstrip("/")


def ros_service_to_dds(service: str) -> tuple[str, str]:
    name = service.lstrip("/")
    return f"{REQUEST_PREFIX}/{name}Request", f"{REPLY_PREFIX}/{name}Reply"


def dds_to_ros(dds_topic: str) -> tuple[str, str] | None:
    """Map a DDS topic back to ``(kind, ROS name)``; ``None`` for non-ROS topics."""
    prefix, _, rest = dds_topic.partition("/")
    if prefix == TOPIC_PREFIX:
        return "topic", "/" + rest
    if prefix == REQUEST_PREFIX and rest.endswith("Request"):
        return "service request", "/" + rest[: -len("Request")]
    if prefix == REPLY_PREFIX and rest.endswith("Reply"):
        return "service reply", "/" + rest[: -len("Reply")]
    return None


def ros_type_to_dds(type_name: str) -> str:
    """``std_msgs/msg/String`` is ``std_msgs::msg::dds_::String_`` on the wire."""
    pkg, kind, name = type_name.split("/")
    return f"{pkg}::{kind}::dds_::{name}_"


def dds_type_to_ros(dds_type: str) -> str:
    parts = dds_type.split("::")
    if len(parts) == 4 and parts[2] == "dds_" and parts[3].endswith("_"):
        return f"{parts[0]}/{parts[1]}/{parts[3][:-1]}"
    return dds_type
