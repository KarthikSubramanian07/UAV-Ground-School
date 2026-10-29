"""Dissect a capture of the PX4 to agent link (UDP 8888) into a readable session.

``dissect`` prints one line per submessage, like ``tcpdump`` would if it knew
DDS-XRCE; ``session`` reconstructs what PX4 asked the agent to create (every
topic with its DDS type, direction and QoS) and how much each topic carried;
``spans`` labels the bytes of a message for the site's packet inspector.

Samples are decoded with the px4_msgs definitions bundled in week06/interfaces,
using week05's CDR, so a WRITE_DATA of vehicle_status reads as PX4's own fields.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from week05.pcap import read

from . import xrce

PORT = 8888
ENCAPSULATION = b"\x00\x01\x00\x00"


def _registry():
    from .xrce_client import registry

    return registry()


def ros_type(dds: str) -> str:
    """px4_msgs::msg::dds_::VehicleStatus_ -> px4_msgs/msg/VehicleStatus."""
    parts = dds.split("::")
    if len(parts) == 4 and parts[2] == "dds_":
        return f"{parts[0]}/{parts[1]}/{parts[3].rstrip('_')}"
    return dds


def ros_topic(dds: str) -> str:
    return dds[2:] if dds.startswith("rt/") else dds


@dataclass
class Event:
    time: float
    direction: str  # "px4>agent" or "agent>px4"
    message: xrce.Message
    sub: xrce.Submessage
    raw: bytes


@dataclass
class Session:
    """The objects of one XRCE session and the traffic on them."""

    objects: dict[str, dict] = field(default_factory=dict)  # object id hex -> representation
    order: list[str] = field(default_factory=list)
    status: dict[int, str] = field(default_factory=dict)  # request id -> result
    requests: dict[int, str] = field(default_factory=dict)  # request id -> object id hex
    samples: dict[str, list[tuple[float, int]]] = field(default_factory=dict)  # endpoint id -> (time, bytes)

    def topic_of(self, endpoint_hex: str) -> dict | None:
        rep = self.objects.get(endpoint_hex)
        if not rep or "topic_id" not in rep:
            return None
        return self.objects.get(rep["topic_id"])


def events(path: str | Path) -> Iterator[Event]:
    """Every submessage in the capture, fragments reassembled, in order."""
    pending: dict[tuple[str, int], bytearray] = {}
    for d in read(path):
        if PORT not in (d.sport, d.dport):
            continue
        direction = "px4>agent" if d.dport == PORT else "agent>px4"
        try:
            msg = xrce.parse(d.payload)
        except xrce.XrceError:
            continue
        for sub in msg.submessages:
            if sub.name == "FRAGMENT":
                key = (direction, msg.stream_id)
                pending.setdefault(key, bytearray()).extend(sub.body["data"])
                if sub.body["last"]:
                    whole = bytes(pending.pop(key))
                    head = xrce.header(msg.session_id, msg.stream_id, msg.sequence, msg.client_key or b"")
                    try:
                        inner = xrce.parse(xrce.message(head, whole))
                    except xrce.XrceError:
                        continue
                    for s in inner.submessages:
                        yield Event(d.time, direction, msg, s, d.payload)
                continue
            yield Event(d.time, direction, msg, sub, d.payload)


def build_session(path: str | Path) -> tuple[Session, list[Event]]:
    s = Session()
    evs = list(events(path))
    for e in evs:
        body = e.sub.body
        if e.sub.name == "CREATE":
            oid = body["object_id"]
            rep = dict(body["representation"])
            if rep.get("format") == "binary" and "topic_id" in rep:
                rep["topic_id"] = rep["topic_id"]
            if oid not in s.objects:
                s.order.append(oid)
            s.objects[oid] = rep
            s.requests[body["request_id"]] = oid
        elif e.sub.name == "STATUS":
            s.status[body["request_id"]] = body["status"]
        elif e.sub.name in ("WRITE_DATA", "DATA"):
            s.samples.setdefault(body["object_id"], []).append((e.time, len(body["data"])))
    return s, evs


def _summary(type_name: str, data: bytes, registry) -> str:
    from week05 import cdr

    try:
        value = cdr.deserialize(type_name, ENCAPSULATION + data, registry)
    except Exception as error:  # noqa: BLE001 - a dissector shows what it can
        return f"<{len(data)} bytes, {type(error).__name__}>"
    items = []
    for k, v in value.items():
        if isinstance(v, float):
            v = round(v, 3)
        elif isinstance(v, list):
            v = [round(x, 3) if isinstance(x, float) else x for x in v[:4]]
        items.append(f"{k}={v}")
        if len(items) == 6:
            break
    return ", ".join(items)


def describe(e: Event, s: Session, registry=None, decode_samples: bool = True) -> str:
    b = e.sub.body
    name = e.sub.name
    if name == "CREATE_CLIENT":
        c = b["client"]
        return f"CREATE_CLIENT key {c['client_key']} session 0x{c['session_id']:02x} mtu {c['mtu']}"
    if name == "STATUS_AGENT":
        return f"STATUS_AGENT {b['status']} (XRCE {b['version']}, vendor {b['vendor']})"
    if name == "CREATE":
        rep = b["representation"]
        what = rep.get("name") or rep.get("topic") or rep.get("service_name") or ""
        extra = f" type {rep['type_name']}" if "type_name" in rep else ""
        qos = rep.get("qos")
        if qos:
            extra += f" {'reliable' if qos['reliable'] else 'best effort'}, {qos['durability']}, depth {qos.get('depth', '-')}"
        return f"CREATE #{b['request_id']} {b['object']} {what}{extra}".rstrip()
    if name == "STATUS":
        return f"STATUS #{b['request_id']} {b['object']}: {b['status']}"
    if name == "READ_DATA":
        return f"READ_DATA #{b['request_id']} {b['object']} into stream {b['preferred_stream']}"
    if name in ("WRITE_DATA", "DATA"):
        topic = s.topic_of(b["object_id"])
        label = ros_topic(topic["name"]) if topic else b["object"]
        text = f"{name} {label} ({len(b['data'])} bytes)"
        if decode_samples and topic and registry is not None:
            text += ": " + _summary(ros_type(topic.get("type_name", "")), b["data"], registry)
        return text
    if name == "HEARTBEAT":
        return f"HEARTBEAT stream {b['stream']} unacked {b['first_unacked']}..{b['last_unacked']}"
    if name == "ACKNACK":
        return f"ACKNACK stream {b['stream']} first unacked {b['first_unacked']} nack {b['nack_bitmap']}"
    if name == "TIMESTAMP":
        return f"TIMESTAMP {b['transmit']:.6f}"
    if name == "TIMESTAMP_REPLY":
        return f"TIMESTAMP_REPLY originate {b['originate']:.6f} receive {b['receive']:.6f} transmit {b['transmit']:.6f}"
    if name == "DELETE":
        return f"DELETE {b['object']}"
    return name


def dissect(path: str | Path, limit: int = 0, decode_samples: bool = True) -> Iterator[str]:
    s, evs = build_session(path)
    registry = _registry() if decode_samples else None
    t0 = evs[0].time if evs else 0.0
    for i, e in enumerate(evs):
        if limit and i >= limit:
            return
        m = e.message
        stream = "none" if m.stream_id == 0 else ("reliable" if m.reliable else "best effort") + f" {m.stream_id}"
        yield f"{e.time - t0:9.4f}  {e.direction}  [{stream} #{m.sequence}]  {describe(e, s, registry, decode_samples)}"


def session(path: str | Path) -> dict:
    """What the capture says PX4 created on the agent, and what each topic carried."""
    s, evs = build_session(path)
    t0 = evs[0].time if evs else 0.0
    duration = (evs[-1].time - t0) if evs else 0.0
    login = next((e for e in evs if e.sub.name == "CREATE_CLIENT"), None)
    agent = next((e for e in evs if e.sub.name == "STATUS_AGENT"), None)
    topics = []
    for oid in s.order:
        rep = s.objects[oid]
        if rep["kind"] not in ("DATAWRITER", "DATAREADER"):
            continue
        topic = s.objects.get(rep.get("topic_id", ""), {})
        samples = s.samples.get(oid, [])
        span = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
        topics.append(
            {
                "topic": ros_topic(topic.get("name", "?")),
                "type": ros_type(topic.get("type_name", "?")),
                "direction": "out" if rep["kind"] == "DATAWRITER" else "in",
                "qos": rep.get("qos", {}),
                "samples": len(samples),
                "bytes": sum(n for _, n in samples),
                "sample_bytes": samples[0][1] if samples else None,
                "rate_hz": round((len(samples) - 1) / span, 1) if span > 0 else None,
                "created": rep and s.status.get(next((r for r, o in s.requests.items() if o == oid), -1)),
            }
        )
    counts: dict[str, int] = {}
    sizes = {"px4>agent": 0, "agent>px4": 0}
    seen = set()
    for e in evs:
        counts[e.sub.name] = counts.get(e.sub.name, 0) + 1
        key = (e.time, e.direction, len(e.raw))
        if key not in seen:
            seen.add(key)
            sizes[e.direction] += len(e.raw) + 28  # IPv4 and UDP headers
    return {
        "duration_s": round(duration, 3),
        "client": login.sub.body["client"] if login else None,
        "agent": {k: v for k, v in agent.sub.body.items()} if agent else None,
        "created": sum(1 for r in s.objects.values()),
        "participants": sum(1 for r in s.objects.values() if r["kind"] == "PARTICIPANT"),
        "topics": topics,
        "submessages": counts,
        "bytes": sizes,
        "kbps": {k: round(v * 8 / 1000 / duration, 1) if duration else None for k, v in sizes.items()},
    }


def spans(raw: bytes) -> list[dict]:
    """Labelled byte ranges of one message, for the packet inspector."""
    out = []
    session_id, stream_id = raw[0], raw[1]
    out.append(
        {
            "start": 0,
            "end": 1,
            "label": "session id",
            "value": f"0x{session_id:02x}" + (" (key follows)" if session_id < 0x80 else ""),
            "depth": 0,
        }
    )
    stream = "none" if stream_id == 0 else ("reliable" if stream_id >= 0x80 else "best effort")
    out.append({"start": 1, "end": 2, "label": "stream id", "value": f"{stream_id} ({stream})", "depth": 0})
    out.append(
        {
            "start": 2,
            "end": 4,
            "label": "sequence number",
            "value": str(struct.unpack_from("<H", raw, 2)[0]) + " (little endian)",
            "depth": 0,
        }
    )
    if session_id < 0x80:
        out.append({"start": 4, "end": 8, "label": "client key", "value": raw[4:8].hex(), "depth": 0})
    msg = xrce.parse(raw)
    for sub in msg.submessages:
        o = sub.offset
        out.append({"start": o, "end": o + 1, "label": "submessage id", "value": f"{sub.id} {sub.name}", "depth": 1})
        out.append(
            {
                "start": o + 1,
                "end": o + 2,
                "label": "flags",
                "value": f"0x{sub.flags:02x}" + (" little endian" if sub.flags & 1 else ""),
                "depth": 1,
            }
        )
        out.append({"start": o + 2, "end": o + 4, "label": "length", "value": f"{sub.length} bytes", "depth": 1})
        start = o + 4
        b = sub.body
        if "request_id" in b:
            out.append({"start": start, "end": start + 2, "label": "request id", "value": str(b["request_id"]), "depth": 2})
            out.append({"start": start + 2, "end": start + 4, "label": "object id", "value": b["object"], "depth": 2})
            rest = start + 4
            if sub.name in ("WRITE_DATA", "DATA"):
                out.append(
                    {
                        "start": rest,
                        "end": start + sub.length,
                        "label": "sample (CDR, no encapsulation)",
                        "value": f"{len(b['data'])} bytes",
                        "depth": 2,
                    }
                )
            elif sub.name == "CREATE":
                rep = b["representation"]
                value = ", ".join(f"{k}: {v}" for k, v in rep.items() if k not in ("format", "qos", "topic_id"))
                out.append(
                    {
                        "start": rest,
                        "end": start + sub.length,
                        "label": f"{rep['kind'].lower()} representation ({rep.get('format', '?')})",
                        "value": value,
                        "depth": 2,
                    }
                )
            elif start + sub.length > rest:
                out.append({"start": rest, "end": start + sub.length, "label": "payload", "value": "", "depth": 2})
        elif sub.length:
            value = ", ".join(f"{k}: {v}" for k, v in b.items() if not isinstance(v, (bytes, dict)))
            out.append({"start": start, "end": start + sub.length, "label": sub.name.lower().replace("_", " "), "value": value, "depth": 2})
    return out
