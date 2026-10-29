"""Dissect RTPS datagrams into labelled byte ranges, like Wireshark's RTPS dissector.

Used by ``python -m week05 decode`` and to build the packet inspector on the
showcase page: every field gets a start and end offset, so the page can
highlight the bytes that mean "this is a HEARTBEAT" or "this is the topic name".
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import cdr, rtps
from .idl import Registry
from .pcap import read

_SEDP = {rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER: True, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_WRITER: False}


@dataclass
class Span:
    start: int
    end: int
    label: str
    value: str = ""
    depth: int = 0

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label, "value": self.value, "depth": self.depth}


@dataclass
class Dissection:
    spans: list[Span] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)

    def add(self, start: int, end: int, label: str, value: object = "", depth: int = 0) -> None:
        self.spans.append(Span(start, end, label, str(value), depth))


def _plist(d: Dissection, raw: bytes, base: int, e: str, depth: int) -> int:
    """Label each parameter of a parameter list starting at raw[base]."""
    offset = base
    while offset + 4 <= len(raw):
        pid, length = struct.unpack_from(e + "HH", raw, offset)
        name = rtps.PID_NAMES.get(pid, f"0x{pid:04x}")
        value = raw[offset + 4 : offset + 4 + length]
        shown = ""
        if pid in (rtps.PID_TOPIC_NAME, rtps.PID_TYPE_NAME, rtps.PID_ENTITY_NAME):
            shown = rtps.read_cdr_string(value, 0, e)
        elif pid in (rtps.PID_PARTICIPANT_GUID, rtps.PID_ENDPOINT_GUID, rtps.PID_KEY_HASH):
            shown = str(rtps.Guid.from_bytes(value))
        elif pid in (
            rtps.PID_UNICAST_LOCATOR,
            rtps.PID_DEFAULT_UNICAST_LOCATOR,
            rtps.PID_METATRAFFIC_UNICAST_LOCATOR,
            rtps.PID_MULTICAST_LOCATOR,
            rtps.PID_METATRAFFIC_MULTICAST_LOCATOR,
            rtps.PID_DEFAULT_MULTICAST_LOCATOR,
        ):
            shown = str(rtps.unpack_locator(value, e))
        elif pid == rtps.PID_RELIABILITY:
            shown = rtps.RELIABILITY.get(struct.unpack_from(e + "I", value)[0], "?")
        elif pid == rtps.PID_DURABILITY:
            shown = rtps.DURABILITY.get(struct.unpack_from(e + "I", value)[0], "?")
        elif pid == rtps.PID_HISTORY:
            kind, depth_n = struct.unpack_from(e + "Ii", value)
            shown = f"{rtps.HISTORY.get(kind, '?')} {depth_n}"
        elif pid == rtps.PID_USER_DATA:
            (n,) = struct.unpack_from(e + "I", value)
            shown = value[4 : 4 + n].rstrip(b"\0").decode("utf-8", "replace")
        elif pid == rtps.PID_VENDORID:
            shown = rtps.VENDORS.get(bytes(value[:2]), value[:2].hex())
        elif pid == rtps.PID_PROTOCOL_VERSION:
            shown = f"{value[0]}.{value[1]}"
        elif pid in (rtps.PID_RELATED_SAMPLE_IDENTITY, rtps.PID_FASTDDS_RELATED_SAMPLE_IDENTITY) and len(value) >= 24:
            seq = rtps.seq_unpack(value, 16, e)
            shown = f"{rtps.Guid.from_bytes(value)} seq {'unknown' if seq < 0 else seq}"
        elif pid == rtps.PID_PARTICIPANT_LEASE_DURATION:
            shown = f"{rtps.Duration(*struct.unpack_from(e + 'iI', value)).to_seconds():g} s"
        d.add(offset, offset + 4 + length, name, shown, depth)
        offset += 4 + length
        if pid == rtps.PID_SENTINEL:
            break
    return offset


def dissect(datagram: bytes, registry: Registry | None = None, types: dict[str, str] | None = None) -> Dissection:
    """Label every byte range of one RTPS datagram.

    ``types`` maps writer GUID strings to ROS types so user payloads can be
    decoded; it is filled in as SEDP announcements go by in :func:`dissect_pcap`.
    """
    d = Dissection()
    msg = rtps.parse(datagram)
    if msg is None:
        d.summary.append("not RTPS")
        return d
    d.add(0, 4, "magic", "RTPS")
    d.add(4, 6, "protocol version", f"{msg.version[0]}.{msg.version[1]}")
    d.add(6, 8, "vendor", msg.vendor_name)
    d.add(8, 20, "GUID prefix", msg.prefix.hex())
    offset = 20
    source = msg.prefix
    for sub in msg.submessages:
        e = sub.endian
        start, body = offset, offset + 4
        end = body + len(sub.body)
        d.add(start, body, f"{sub.name} header", f"flags 0x{sub.flags:02x}, {len(sub.body)} bytes")
        d.kinds.append(sub.name)
        decoded = rtps.decode_submessage(sub)
        if sub.kind == rtps.INFO_TS and not sub.flags & rtps.INFO_TS_I:
            sec, frac = struct.unpack_from(e + "iI", sub.body)
            d.add(body, end, "timestamp", f"{sec + frac / 2**32:.6f}", 1)
        elif sub.kind == rtps.INFO_DST:
            d.add(body, end, "destination GUID prefix", sub.body[:12].hex(), 1)
        elif isinstance(decoded, rtps.Data):
            d.add(body + 4, body + 8, "reader", rtps.entity_name(decoded.reader), 1)
            d.add(body + 8, body + 12, "writer", rtps.entity_name(decoded.writer), 1)
            d.add(body + 12, body + 20, "sequence number", decoded.seq, 1)
            pos = body + 4 + struct.unpack_from(e + "H", sub.body, 2)[0]
            if sub.flags & rtps.DATA_Q:
                pos = _plist(d, datagram, pos, e, 2)
            what = _describe_data(decoded, source, registry, types)
            d.summary.append(what)
            if decoded.payload is not None:
                d.add(pos, pos + 4, "encapsulation", decoded.payload[:2].hex(), 1)
                if decoded.writer in (rtps.ENTITYID_SPDP_WRITER, *_SEDP):
                    _plist(d, datagram, pos + 4, rtps._payload_endian(decoded.payload), 2)
                else:
                    d.add(pos + 4, end, "serialized payload (CDR)", what.split(": ", 1)[-1], 1)
        elif isinstance(decoded, rtps.Heartbeat):
            d.add(body, body + 8, "reader, writer", f"{rtps.entity_name(decoded.reader)}, {rtps.entity_name(decoded.writer)}", 1)
            d.add(body + 8, body + 24, "first, last", f"{decoded.first}, {decoded.last}", 1)
            d.add(body + 24, end, "count", decoded.count, 1)
            d.summary.append(
                f"HEARTBEAT {rtps.entity_name(decoded.writer)} has [{decoded.first}, {decoded.last}]{' final' if decoded.final else ''}"
            )
        elif isinstance(decoded, rtps.AckNack):
            d.add(body, body + 8, "reader, writer", f"{rtps.entity_name(decoded.reader)}, {rtps.entity_name(decoded.writer)}", 1)
            d.add(body + 8, end - 4, "sequence number set", f"base {decoded.base}, missing {decoded.missing}", 1)
            d.add(end - 4, end, "count", decoded.count, 1)
            missing = f", missing {decoded.missing}" if decoded.missing else ""
            d.summary.append(f"ACKNACK {rtps.entity_name(decoded.reader)}: have everything below {decoded.base}{missing}")
        elif isinstance(decoded, rtps.Gap):
            d.add(body, end, "gap", f"{decoded.start} to {decoded.list_base} {decoded.irrelevant}", 1)
            d.summary.append(f"GAP {decoded.start}..{decoded.list_base - 1}")
        elif sub.kind >= 0x80:
            d.add(body, end, "vendor specific submessage", "skipped, as the spec requires", 1)
        offset = end
    return d


def _describe_data(data: rtps.Data, source: bytes, registry, types) -> str:
    if data.payload is None:
        return f"DATA {rtps.entity_name(data.writer)} (no payload)"
    if data.writer == rtps.ENTITYID_SPDP_WRITER:
        pd = rtps.ParticipantData.decode(data.payload)
        loc = pd.default_unicast[0] if pd.default_unicast else ""
        return f"SPDP: participant {pd.guid_prefix.hex()[:8]} ({pd.vendor_name}) at {loc}"
    if data.writer in _SEDP:
        ep = rtps.EndpointData.decode(data.payload, _SEDP[data.writer])
        if types is not None:
            types[str(ep.guid)] = rtps.dds_type_to_ros(ep.type_name)
        role = "publication" if ep.writer else "subscription"
        return f"SEDP: {role} {ep.topic} [{rtps.dds_type_to_ros(ep.type_name)}] {ep.qos.reliability}, {ep.qos.durability}"
    guid = str(rtps.Guid(source, data.writer))
    ros_type = (types or {}).get(guid)
    if ros_type and registry is not None:
        try:
            value = cdr.deserialize(ros_type, data.payload, registry)
            text = str(value)
            return f"DATA {ros_type} #{data.seq}: {text if len(text) < 100 else text[:97] + '...'}"
        except (cdr.CdrError, KeyError, ValueError, struct.error, UnicodeDecodeError):
            pass
    return f"DATA {rtps.entity_name(data.writer)} #{data.seq}: {len(data.payload)} bytes"


def dissect_pcap(path, verbose: bool = False, limit: int = 0) -> Iterator[str]:
    registry = Registry()
    types: dict[str, str] = {}
    t0 = None
    for n, dg in enumerate(read(path)):
        if limit and n >= limit:
            break
        t0 = dg.time if t0 is None else t0
        d = dissect(dg.payload, registry, types)
        what = "; ".join(d.summary) or ", ".join(k for k in d.kinds if k not in ("INFO_TS", "INFO_DST"))
        yield f"{dg.time - t0:8.3f}  {dg.src}:{dg.sport} -> {dg.dst}:{dg.dport} ({rtps.describe_port(dg.dport)})  {what}"
        if verbose:
            for s in d.spans:
                yield f"{'':10}{'  ' * s.depth}[{s.start:4}:{s.end:4}] {s.label}: {s.value}"
