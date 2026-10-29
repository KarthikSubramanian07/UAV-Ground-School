"""A ROS 2 node with no ROS 2: a small RTPS participant in pure Python.

It speaks just enough DDSI-RTPS to join a live ROS 2 graph as a peer:

* **SPDP**: announces itself on the discovery multicast group and learns every
  other participant's locators and lease;
* **SEDP**: reliably exchanges endpoint announcements (topic, type, QoS, type
  hash) with each participant, so ROS 2 nodes see this one's publishers and
  subscriptions and it sees theirs;
* **reliable and best effort user data**: writers keep a history, send DATA and
  HEARTBEAT and answer ACKNACK with retransmissions or GAP; readers acknowledge,
  reorder, drop duplicates and reassemble DATA_FRAG;
* **ROS 2 conventions on top**: ``rt/`` topic names, ``pkg::msg::dds_::Name_``
  types, the ``typehash=`` user data, CDR payloads, and services as request and
  reply topics correlated by ``RELATED_SAMPLE_IDENTITY``, the way rmw_fastrtps
  does it.

Run it next to ROS 2 (shared memory off for the ROS nodes is not needed: Fast DDS
falls back to UDP for a peer that offers no shared memory locator)::

    from week05.participant import Participant
    with Participant() as p:
        p.create_subscription("/topic", "std_msgs/msg/String", print)
        p.spin(5.0)

There is deliberately no security, no ownership, no content filtering and no
writer liveliness protocol: ROS 2's defaults use none of them.
"""

from __future__ import annotations

import contextlib
import os
import random
import selectors
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import cdr, rtps
from .idl import Registry
from .rtps import Duration, EndpointData, EndpointQos, Guid, Locator, MessageBuilder, ParameterList, ParticipantData

MAX_DATAGRAM = 1400  # stay under a typical MTU; larger samples go out as DATA_FRAG
FRAGMENT_SIZE = 1024
INITIAL_PEERS = 6  # participant ids probed by unicast on this host
SEQUENCE_UNKNOWN = (-1 << 32) | 0  # SEQUENCENUMBER_UNKNOWN as (high=-1, low=0)

BUILTIN_ENDPOINTS = (
    (1 << 0)  # participant announcer
    | (1 << 1)  # participant detector
    | (1 << 2)  # publications announcer
    | (1 << 3)  # publications detector
    | (1 << 4)  # subscriptions announcer
    | (1 << 5)  # subscriptions detector
)


def local_ipv4() -> str:
    """The address other hosts reach us on (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def qos(reliability: str = "reliable", durability: str = "volatile", depth: int = 10) -> EndpointQos:
    """A ROS 2 style profile: the rclpy default is ``qos()``, sensor data is ``qos("best_effort", depth=5)``."""
    return EndpointQos(reliability=reliability, durability=durability, history="keep_last", depth=depth)


def compatible(writer: EndpointQos, reader: EndpointQos) -> bool:
    """DDS request versus offered matching for the policies ROS 2 uses."""
    rank_r = {"best_effort": 0, "reliable": 1}
    rank_d = {"volatile": 0, "transient_local": 1, "transient": 2, "persistent": 3}
    if rank_r.get(writer.reliability, -1) < rank_r.get(reader.reliability, 9):
        return False
    if rank_d.get(writer.durability, -1) < rank_d.get(reader.durability, 9):
        return False
    if writer.deadline.to_seconds() > reader.deadline.to_seconds():
        return False
    rank_l = {"automatic": 0, "manual_by_participant": 1, "manual_by_topic": 2}
    if rank_l.get(writer.liveliness, -1) < rank_l.get(reader.liveliness, 9):
        return False
    return writer.lease_duration.to_seconds() <= reader.lease_duration.to_seconds()


# --------------------------------------------------------------- endpoints --


@dataclass
class Sample:
    """One received sample: its decoded value, raw payload and where it came from."""

    value: dict | None
    payload: bytes
    writer: Guid
    seq: int
    timestamp: float | None
    inline_qos: ParameterList | None

    def related_identity(self) -> tuple[Guid, int] | None:
        if self.inline_qos is None:
            return None
        raw = self.inline_qos.get(rtps.PID_RELATED_SAMPLE_IDENTITY) or self.inline_qos.get(rtps.PID_FASTDDS_RELATED_SAMPLE_IDENTITY)
        if raw is None or len(raw) < 24:
            return None
        return Guid.from_bytes(raw[:16]), rtps.seq_unpack(raw, 16)


def _identity(guid: Guid, seq: int) -> bytes:
    return bytes(guid) + rtps.seq_pack(seq)


@dataclass
class _ReaderProxy:
    """A writer's view of one matched remote reader."""

    guid: Guid
    locators: list[Locator]
    reliable: bool
    acked: int = 0  # every seq <= acked is acknowledged
    last_acknack_count: int = -1
    first_relevant: int = 1  # a volatile reader matched later never sees older samples


@dataclass
class _WriterProxy:
    """A reader's view of one matched remote writer."""

    guid: Guid
    locators: list[Locator]
    reliable: bool
    next_seq: int = 1  # next sequence number to deliver in order
    received: dict[int, object] = field(default_factory=dict)  # out of order samples waiting
    seen_highest: int = 0
    acknack_count: int = 0
    fragments: dict[int, dict[int, bytes]] = field(default_factory=dict)


class Writer:
    """A DDS data writer: a keep last history plus one proxy per matched reader."""

    def __init__(
        self,
        participant: Participant,
        entity: int,
        topic: str,
        type_name: str,
        qos: EndpointQos,
        type_hash: str = "",
        builtin: bool = False,
    ) -> None:
        self.p = participant
        self.guid = Guid(participant.prefix, entity)
        self.topic, self.type_name, self.qos = topic, type_name, qos
        self.type_hash = type_hash
        self.builtin = builtin
        self.seq = 0
        self.history: dict[int, tuple[bytes, ParameterList | None]] = {}
        self.readers: dict[Guid, _ReaderProxy] = {}
        self.hb_count = 0
        self.ros_type = ""
        self.sedp_seq = 0

    @property
    def matched(self) -> int:
        return len(self.readers)

    def endpoint_data(self) -> EndpointData:
        user = f"typehash={self.type_hash};".encode() if self.type_hash else b""
        return EndpointData(self.guid, self.topic, self.type_name, self.qos, writer=True, unicast=self.p.default_locators, user_data=user)

    def write_payload(self, payload: bytes, inline_qos: ParameterList | None = None) -> int:
        with self.p.lock:
            self.seq += 1
            self.history[self.seq] = (payload, inline_qos)
            depth = self.qos.depth if self.qos.history == "keep_last" and not self.builtin else 10**9
            for old in [s for s in self.history if s <= self.seq - depth]:
                del self.history[old]
            for proxy in self.readers.values():
                self._send_data(proxy, [self.seq])
            return self.seq

    def _send_data(self, proxy: _ReaderProxy, seqs: list[int]) -> None:
        """Send the given samples to one reader, batched into datagrams, then a HEARTBEAT."""

        def fresh() -> MessageBuilder:
            return MessageBuilder(self.p.prefix).info_ts(time.time()).info_dst(proxy.guid.prefix)

        msg, used = fresh(), False
        for s in seqs:
            if s not in self.history or s < proxy.first_relevant:  # gone, or not for this reader
                msg.gap(proxy.guid.entity, self.guid.entity, s, s + 1)
                used = True
                continue
            payload, inline = self.history[s]
            if len(payload) > MAX_DATAGRAM - 100:
                for first in range(1, (len(payload) + FRAGMENT_SIZE - 1) // FRAGMENT_SIZE + 1):
                    frag = fresh().data_frag(proxy.guid.entity, self.guid.entity, s, first, FRAGMENT_SIZE, payload)
                    self.p._send(bytes(frag), proxy.locators)
                continue
            if used and len(msg) + len(payload) + 64 > MAX_DATAGRAM:
                self.p._send(bytes(msg), proxy.locators)
                msg, used = fresh(), False
            msg.data(proxy.guid.entity, self.guid.entity, s, payload, inline)
            used = True
        if proxy.reliable:
            self.hb_count += 1
            msg.heartbeat(proxy.guid.entity, self.guid.entity, self._first_for(proxy), self.seq, self.hb_count)
            used = True
        if used:
            self.p._send(bytes(msg), proxy.locators)

    def _first_for(self, proxy: _ReaderProxy) -> int:
        """The first sequence number this reader may still ask for."""
        first = min(self.history) if self.history else self.seq + 1
        return max(first, proxy.first_relevant)

    def heartbeat(self, proxy: _ReaderProxy) -> None:
        self.hb_count += 1
        msg = (
            MessageBuilder(self.p.prefix)
            .info_dst(proxy.guid.prefix)
            .heartbeat(proxy.guid.entity, self.guid.entity, self._first_for(proxy), self.seq, self.hb_count)
        )
        self.p._send(bytes(msg), proxy.locators)

    def on_acknack(self, src: Guid, ack: rtps.AckNack) -> None:
        proxy = self.readers.get(src)
        if proxy is None or ack.count <= proxy.last_acknack_count:
            return
        proxy.last_acknack_count = ack.count
        proxy.acked = max(proxy.acked, ack.base - 1)
        if ack.missing:
            self._send_data(proxy, ack.missing)
        elif proxy.acked < self.seq and not ack.final:
            self.heartbeat(proxy)

    def add_reader(self, guid: Guid, locators: list[Locator], reliable: bool, durable: bool = False) -> None:
        if guid in self.readers:
            self.readers[guid].locators = locators
            return
        proxy = _ReaderProxy(guid, locators, reliable)
        durable_match = self.builtin or (durable and self.qos.durability != "volatile")
        if not durable_match:
            proxy.first_relevant = self.seq + 1
        self.readers[guid] = proxy
        # Only a durable reader of a durable writer gets the history: a volatile
        # late joiner starts with the next sample (DDS 2.2.3.4).
        if durable_match and self.history:
            self._send_data(proxy, sorted(self.history))
        elif reliable:
            self.heartbeat(proxy)


class Reader:
    """A DDS data reader: one proxy per matched writer, delivering samples in order."""

    def __init__(
        self,
        participant: Participant,
        entity: int,
        topic: str,
        type_name: str,
        qos: EndpointQos,
        callback: Callable[[Sample], None],
        type_hash: str = "",
        builtin: bool = False,
    ) -> None:
        self.p = participant
        self.guid = Guid(participant.prefix, entity)
        self.topic, self.type_name, self.qos = topic, type_name, qos
        self.callback = callback
        self.type_hash = type_hash
        self.builtin = builtin
        self.writers: dict[Guid, _WriterProxy] = {}
        self.sedp_seq = 0

    @property
    def matched(self) -> int:
        return len(self.writers)

    def endpoint_data(self) -> EndpointData:
        user = f"typehash={self.type_hash};".encode() if self.type_hash else b""
        return EndpointData(self.guid, self.topic, self.type_name, self.qos, writer=False, unicast=self.p.default_locators, user_data=user)

    def add_writer(self, guid: Guid, locators: list[Locator], reliable: bool) -> None:
        if guid in self.writers:
            self.writers[guid].locators = locators
            return
        self.writers[guid] = _WriterProxy(guid, locators, reliable and self.qos.reliability == "reliable")

    def on_data(self, src: Guid, seq: int, payload: bytes | None, inline_qos, ts: float | None) -> None:
        proxy = self.writers.get(src)
        if proxy is None:
            if not self.builtin:
                return
            proxy = self.writers.setdefault(src, _WriterProxy(src, [], True))
        if payload is None:
            return
        sample = (payload, inline_qos, ts)
        if not proxy.reliable:
            if seq >= proxy.next_seq:
                proxy.next_seq = seq + 1
                self._deliver(src, seq, sample)
            return
        if seq < proxy.next_seq or seq in proxy.received:
            return
        proxy.received[seq] = sample
        proxy.seen_highest = max(proxy.seen_highest, seq)
        self._flush(src, proxy)

    def on_fragment(self, src: Guid, frag: rtps.DataFrag, ts: float | None) -> None:
        proxy = self.writers.get(src)
        if proxy is None or frag.seq < proxy.next_seq:
            return
        parts = proxy.fragments.setdefault(frag.seq, {})
        for i in range(frag.fragments_in_submessage):
            parts[frag.first_fragment + i] = frag.payload[i * frag.fragment_size : (i + 1) * frag.fragment_size]
        total = (frag.sample_size + frag.fragment_size - 1) // frag.fragment_size
        if len(parts) == total:
            payload = b"".join(parts[i] for i in range(1, total + 1))[: frag.sample_size]
            del proxy.fragments[frag.seq]
            self.on_data(src, frag.seq, payload, frag.inline_qos, ts)

    def _flush(self, src: Guid, proxy: _WriterProxy) -> None:
        while proxy.next_seq in proxy.received:
            sample = proxy.received.pop(proxy.next_seq)
            seq = proxy.next_seq
            proxy.next_seq += 1
            if sample is not None:
                self._deliver(src, seq, sample)

    def _deliver(self, src: Guid, seq: int, sample) -> None:
        payload, inline_qos, ts = sample
        self.p._dispatch(self, Sample(None, payload, src, seq, ts, inline_qos))

    def on_heartbeat(self, src: Guid, hb: rtps.Heartbeat) -> None:
        proxy = self.writers.get(src)
        if proxy is None:
            if not self.builtin:
                return
            proxy = self.writers.setdefault(src, _WriterProxy(src, [], True))
        if not proxy.reliable:
            return
        # The writer tailors HEARTBEAT and GAP to each reader (a volatile reader
        # is never offered samples from before it matched), so the reader only
        # skips what the writer says is gone: below HEARTBEAT.first, or in a GAP.
        if hb.first > proxy.next_seq:  # older samples are gone: skip them
            for s in list(proxy.received):
                if s < hb.first:
                    del proxy.received[s]
            proxy.next_seq = hb.first
            self._flush(src, proxy)
        missing = [s for s in range(proxy.next_seq, min(hb.last, proxy.next_seq + 255) + 1) if s not in proxy.received]
        if hb.final and not missing:
            return
        proxy.acknack_count += 1
        msg = (
            MessageBuilder(self.p.prefix)
            .info_dst(src.prefix)
            .acknack(self.guid.entity, src.entity, proxy.next_seq, missing, proxy.acknack_count, final=not missing)
        )
        self.p._send(bytes(msg), self.p._locators_for_writer(src, proxy))

    def on_gap(self, src: Guid, gap: rtps.Gap) -> None:
        proxy = self.writers.get(src)
        if proxy is None:
            return
        for s in list(range(gap.start, gap.list_base)) + gap.irrelevant:
            if s >= proxy.next_seq and s not in proxy.received:
                proxy.received[s] = None
        self._flush(src, proxy)


# --------------------------------------------------------- the participant --


@dataclass
class RemoteParticipant:
    data: ParticipantData
    last_seen: float
    address: str


class Participant:
    """One RTPS participant on a DDS domain, with ROS 2 style topics and services on top."""

    def __init__(
        self, domain: int = 0, name: str = "week05", ip: str | None = None, registry: Registry | None = None, announce_period: float = 1.0
    ) -> None:
        self.domain = domain
        self.name = name
        self.ip = ip or os.environ.get("WEEK05_IP") or local_ipv4()
        self.registry = registry or Registry()
        self.announce_period = announce_period
        rng = random.Random()
        self.prefix = rtps.VENDOR_UGS + bytes(rng.getrandbits(8) for _ in range(10))
        self.lock = threading.RLock()
        self.participants: dict[bytes, RemoteParticipant] = {}
        self.remote_writers: dict[Guid, EndpointData] = {}
        self.remote_readers: dict[Guid, EndpointData] = {}
        self.writers: dict[int, Writer] = {}
        self.readers: dict[int, Reader] = {}
        self.stats = {"datagrams_in": 0, "datagrams_out": 0, "bytes_in": 0, "bytes_out": 0, "bad": 0}
        self.on_discovery: list[Callable[[str, object], None]] = []
        self._next_entity = 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_sockets()
        self._make_builtin()

    # ----------------------------------------------------------- sockets --

    def _open_sockets(self) -> None:
        self.sel = selectors.DefaultSelector()
        mc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        mc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            mc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        mc.bind(("", rtps.spdp_multicast_port(self.domain)))
        group = socket.inet_aton(rtps.SPDP_MULTICAST_ADDRESS)
        self.interfaces = sorted({self.ip, "127.0.0.1"})
        for iface in self.interfaces:
            with contextlib.suppress(OSError):
                mc.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + socket.inet_aton(iface))
        self.mc = mc
        for pid in range(120):
            try:
                meta = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                meta.bind(("", rtps.metatraffic_unicast_port(self.domain, pid)))
                user = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                user.bind(("", rtps.user_unicast_port(self.domain, pid)))
            except OSError:
                meta.close()
                continue
            self.participant_id = pid
            self.meta, self.user = meta, user
            break
        else:
            raise OSError("no free RTPS participant id on this host")
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self.out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        for s in (self.mc, self.meta, self.user):
            s.setblocking(False)
            self.sel.register(s, selectors.EVENT_READ)
        self.meta_locators = [Locator.udpv4(self.ip, rtps.metatraffic_unicast_port(self.domain, self.participant_id))]
        self.default_locators = [Locator.udpv4(self.ip, rtps.user_unicast_port(self.domain, self.participant_id))]

    def _send(self, datagram: bytes, locators: list[Locator]) -> None:
        sent = set()
        for loc in locators:
            if loc.kind != 1 or (loc.ip, loc.port) in sent:
                continue
            sent.add((loc.ip, loc.port))
            try:
                self.out.sendto(datagram, (loc.ip, loc.port))
                self.stats["datagrams_out"] += 1
                self.stats["bytes_out"] += len(datagram)
            except OSError:
                pass

    def _send_multicast(self, datagram: bytes) -> None:
        for iface in self.interfaces:
            try:
                self.out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface))
                self.out.sendto(datagram, (rtps.SPDP_MULTICAST_ADDRESS, rtps.spdp_multicast_port(self.domain)))
                self.stats["datagrams_out"] += 1
                self.stats["bytes_out"] += len(datagram)
            except OSError:
                pass

    # ---------------------------------------------------- builtin endpoints --

    def _make_builtin(self) -> None:
        durable = EndpointQos(reliability="reliable", durability="transient_local", depth=1)
        self.sedp_pub_writer = Writer(
            self, rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER, "DCPSPublication", "PublicationBuiltinTopicData", durable, builtin=True
        )
        self.sedp_sub_writer = Writer(
            self, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_WRITER, "DCPSSubscription", "SubscriptionBuiltinTopicData", durable, builtin=True
        )
        self.sedp_pub_reader = Reader(
            self, rtps.ENTITYID_SEDP_PUBLICATIONS_READER, "DCPSPublication", "", durable, self._on_sedp, builtin=True
        )
        self.sedp_sub_reader = Reader(
            self, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_READER, "DCPSSubscription", "", durable, self._on_sedp, builtin=True
        )
        for w in (self.sedp_pub_writer, self.sedp_sub_writer):
            self.writers[w.guid.entity] = w
        for r in (self.sedp_pub_reader, self.sedp_sub_reader):
            self.readers[r.guid.entity] = r

    def participant_data(self) -> ParticipantData:
        return ParticipantData(
            guid_prefix=self.prefix,
            metatraffic_unicast=self.meta_locators,
            default_unicast=self.default_locators,
            builtin_endpoints=BUILTIN_ENDPOINTS,
            lease_duration=Duration.from_seconds(20),
            entity_name=self.name,
            user_data=b"enclave=/;",
        )

    def _announce(self, to: list[Locator] | None = None) -> None:
        pd = self.participant_data()
        key = ParameterList().add(rtps.PID_KEY_HASH, bytes(Guid(self.prefix, rtps.ENTITYID_PARTICIPANT)))
        msg = (
            MessageBuilder(self.prefix).info_ts(time.time()).data(rtps.ENTITYID_SPDP_READER, rtps.ENTITYID_SPDP_WRITER, 1, pd.encode(), key)
        )
        if to is None:
            self._send_multicast(bytes(msg))
            # Initial peers, as Fast DDS does: the first few participant ids on this
            # host, so discovery also works where multicast is filtered.
            peers = [
                Locator.udpv4(ip, rtps.metatraffic_unicast_port(self.domain, pid))
                for ip in self.interfaces
                for pid in range(INITIAL_PEERS)
                if pid != self.participant_id
            ]
            for rp in self.participants.values():
                peers += rp.data.metatraffic_unicast
            self._send(bytes(msg), peers)
        else:
            self._send(bytes(msg), to)

    def _new_entity(self, kind: int) -> int:
        with self.lock:
            self._next_entity += 1
            return (self._next_entity << 8) | kind

    def _announce_endpoint(self, ep: EndpointData) -> int:
        """Publish an endpoint over SEDP; returns the announcement's sequence number."""
        payload = ep.encode()
        inline = ParameterList().add(rtps.PID_KEY_HASH, bytes(ep.guid))
        return (self.sedp_pub_writer if ep.writer else self.sedp_sub_writer).write_payload(payload, inline)

    def announced_to(self, prefix: bytes, endpoint: Reader | Writer) -> bool:
        """Has participant ``prefix`` acknowledged our SEDP announcement of ``endpoint``?"""
        sedp = self.sedp_pub_writer if isinstance(endpoint, Writer) else self.sedp_sub_writer
        remote = rtps.ENTITYID_SEDP_PUBLICATIONS_READER if isinstance(endpoint, Writer) else rtps.ENTITYID_SEDP_SUBSCRIPTIONS_READER
        proxy = sedp.readers.get(Guid(prefix, remote))
        return proxy is not None and proxy.acked >= endpoint.sedp_seq

    # ------------------------------------------------------------ receive --

    def _locators_for_writer(self, guid: Guid, proxy: _WriterProxy) -> list[Locator]:
        if proxy.locators:
            return proxy.locators
        rp = self.participants.get(guid.prefix)
        if rp is None:
            return []
        if guid.entity & 0xC0 == 0xC0:  # builtin entities reply to metatraffic
            return rp.data.metatraffic_unicast
        return rp.data.default_unicast

    def _handle(self, datagram: bytes, addr) -> None:
        self.stats["datagrams_in"] += 1
        self.stats["bytes_in"] += len(datagram)
        try:
            msg = rtps.parse(datagram)
        except ValueError:
            self.stats["bad"] += 1
            return
        if msg is None or msg.prefix == self.prefix:
            return
        with self.lock:
            try:
                for rec in rtps.interpret(msg):
                    if rec.dest_prefix is not None and rec.dest_prefix != self.prefix:
                        continue
                    self._handle_one(rec, addr)
            except (ValueError, struct.error):
                self.stats["bad"] += 1

    def _handle_one(self, rec: rtps.Received, addr) -> None:
        s = rec.submessage
        if isinstance(s, rtps.Data):
            src = Guid(rec.source_prefix, s.writer)
            if s.writer == rtps.ENTITYID_SPDP_WRITER:
                self._on_spdp(s, addr)
                return
            for reader in self._readers_for(src, s.reader):
                reader.on_data(src, s.seq, s.payload if not s.key_only else None, s.inline_qos, rec.timestamp)
        elif isinstance(s, rtps.DataFrag):
            src = Guid(rec.source_prefix, s.writer)
            for reader in self._readers_for(src, s.reader):
                reader.on_fragment(src, s, rec.timestamp)
        elif isinstance(s, rtps.Heartbeat):
            src = Guid(rec.source_prefix, s.writer)
            for reader in self._readers_for(src, s.reader):
                reader.on_heartbeat(src, s)
        elif isinstance(s, rtps.Gap):
            src = Guid(rec.source_prefix, s.writer)
            for reader in self._readers_for(src, s.reader):
                reader.on_gap(src, s)
        elif isinstance(s, rtps.AckNack):
            writer = self.writers.get(s.writer)
            if writer is not None:
                writer.on_acknack(Guid(rec.source_prefix, s.reader), s)

    def _readers_for(self, src: Guid, reader_entity: int) -> list[Reader]:
        if reader_entity != rtps.ENTITYID_UNKNOWN:
            r = self.readers.get(reader_entity)
            return [r] if r is not None else []
        return [r for r in self.readers.values() if src in r.writers or (r.builtin and (src.entity == r.guid.entity - 5))]

    def _on_spdp(self, data: rtps.Data, addr) -> None:
        if data.payload is None:
            return
        pd = ParticipantData.decode(data.payload)
        if pd.guid_prefix == self.prefix:
            return
        new = pd.guid_prefix not in self.participants
        self.participants[pd.guid_prefix] = RemoteParticipant(pd, time.monotonic(), addr[0])
        if new:
            for cb in self.on_discovery:
                cb("participant", pd)
            self._announce(pd.metatraffic_unicast)
            # Builtin SEDP endpoints are matched implicitly with every participant.
            for w, remote in (
                (self.sedp_pub_writer, rtps.ENTITYID_SEDP_PUBLICATIONS_READER),
                (self.sedp_sub_writer, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_READER),
            ):
                w.add_reader(Guid(pd.guid_prefix, remote), pd.metatraffic_unicast, True)
            for r, remote in (
                (self.sedp_pub_reader, rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER),
                (self.sedp_sub_reader, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_WRITER),
            ):
                r.add_writer(Guid(pd.guid_prefix, remote), pd.metatraffic_unicast, True)

    def _on_sedp(self, sample: Sample) -> None:
        writer = sample.writer.entity == rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER
        try:
            ep = EndpointData.decode(sample.payload, writer)
        except (ValueError, struct.error):
            return
        table = self.remote_writers if writer else self.remote_readers
        new = ep.guid not in table
        table[ep.guid] = ep
        if new:
            for cb in self.on_discovery:
                cb("writer" if writer else "reader", ep)
        self._match_remote(ep)

    def _endpoint_locators(self, ep: EndpointData) -> list[Locator]:
        if ep.unicast:
            return ep.unicast
        rp = self.participants.get(ep.guid.prefix)
        return rp.data.default_unicast if rp else []

    def _match_remote(self, ep: EndpointData) -> None:
        locs = self._endpoint_locators(ep)
        if ep.writer:
            for r in self.readers.values():
                if not r.builtin and r.topic == ep.topic and r.type_name == ep.type_name and compatible(ep.qos, r.qos):
                    r.add_writer(ep.guid, locs, ep.qos.reliability == "reliable")
        else:
            for w in self.writers.values():
                if not w.builtin and w.topic == ep.topic and w.type_name == ep.type_name and compatible(w.qos, ep.qos):
                    w.add_reader(ep.guid, locs, ep.qos.reliability == "reliable", ep.qos.durability != "volatile")

    def _match_local(self, endpoint: Reader | Writer) -> None:
        remote = self.remote_writers if isinstance(endpoint, Reader) else self.remote_readers
        for ep in list(remote.values()):
            self._match_remote(ep)

    def _dispatch(self, reader: Reader, sample: Sample) -> None:
        if not reader.builtin and reader.type_name:
            ros_type = getattr(reader, "ros_type", "")
            if ros_type:
                try:
                    sample.value = cdr.deserialize(ros_type, sample.payload, self.registry)
                except (cdr.CdrError, UnicodeDecodeError, struct.error):
                    sample.value = None
        reader.callback(sample)

    # ---------------------------------------------------------- the loop --

    def start(self) -> Participant:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"rtps-{self.name}", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        next_announce = 0.0
        next_heartbeat = time.monotonic() + 0.5
        started = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_announce:
                with self.lock:
                    self._announce()
                    self._expire(now)
                # announce fast while joining, then at the configured period
                next_announce = now + (0.25 if now - started < 2 else self.announce_period)
            if now >= next_heartbeat:
                with self.lock:
                    for w in self.writers.values():
                        for proxy in w.readers.values():
                            if proxy.reliable and proxy.acked < w.seq:
                                w.heartbeat(proxy)
                next_heartbeat = now + 0.5
            for key, _ in self.sel.select(timeout=0.05):
                sock = key.fileobj
                for _ in range(64):
                    try:
                        data, addr = sock.recvfrom(65536)
                    except (BlockingIOError, InterruptedError):
                        break
                    except OSError:
                        break
                    self._handle(data, addr)

    def _expire(self, now: float) -> None:
        for prefix, rp in list(self.participants.items()):
            if now - rp.last_seen > rp.data.lease_duration.to_seconds():
                del self.participants[prefix]
                for table in (self.remote_writers, self.remote_readers):
                    for g in [g for g in table if g.prefix == prefix]:
                        del table[g]
                for endpoint in list(self.writers.values()):
                    for g in [g for g in endpoint.readers if g.prefix == prefix]:
                        del endpoint.readers[g]
                for endpoint in list(self.readers.values()):
                    for g in [g for g in endpoint.writers if g.prefix == prefix]:
                        del endpoint.writers[g]

    def spin(self, seconds: float) -> None:
        """Let the background thread work for ``seconds`` (it runs whether or not you spin)."""
        self.start()
        time.sleep(seconds)

    def wait_until(self, predicate: Callable[[], bool], timeout: float = 10.0, poll: float = 0.02) -> bool:
        self.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if predicate():
                    return True
            time.sleep(poll)
        with self.lock:
            return predicate()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for s in (self.mc, self.meta, self.user, self.out):
            s.close()
        self.sel.close()

    def __enter__(self) -> Participant:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------- ROS 2 API --

    def _type_hash(self, ros_type: str) -> str:
        try:
            return self.registry.type_hash(ros_type)
        except (KeyError, ValueError):
            return ""

    def _reader(self, dds_topic: str, ros_type: str, profile: EndpointQos, callback) -> Reader:
        r = Reader(
            self,
            self._new_entity(rtps.KIND_READER_NO_KEY),
            dds_topic,
            rtps.ros_type_to_dds(ros_type),
            profile,
            callback,
            self._type_hash(ros_type),
        )
        r.ros_type = ros_type
        with self.lock:
            self.readers[r.guid.entity] = r
            r.sedp_seq = self._announce_endpoint(r.endpoint_data())
            self._match_local(r)
        return r

    def _writer(self, dds_topic: str, ros_type: str, profile: EndpointQos) -> Writer:
        w = Writer(
            self, self._new_entity(rtps.KIND_WRITER_NO_KEY), dds_topic, rtps.ros_type_to_dds(ros_type), profile, self._type_hash(ros_type)
        )
        w.ros_type = ros_type
        with self.lock:
            self.writers[w.guid.entity] = w
            w.sedp_seq = self._announce_endpoint(w.endpoint_data())
            self._match_local(w)
        return w

    def create_subscription(
        self, topic: str, ros_type: str, callback: Callable[[Sample], None], profile: EndpointQos | None = None
    ) -> Reader:
        """Subscribe like ``node.create_subscription``; the callback gets a :class:`Sample`."""
        self.registry.message(ros_type)
        return self._reader(rtps.ros_topic_to_dds(topic), ros_type, profile or qos(), callback)

    def create_publisher(self, topic: str, ros_type: str, profile: EndpointQos | None = None) -> Publisher:
        self.registry.message(ros_type)
        return Publisher(self, self._writer(rtps.ros_topic_to_dds(topic), ros_type, profile or qos()))

    def create_client(self, service: str, srv_type: str) -> Client:
        return Client(self, service, srv_type)

    def create_service(self, service: str, srv_type: str, handler: Callable[[dict], dict]) -> Service:
        return Service(self, service, srv_type, handler)

    # --------------------------------------------------------- the graph --

    def graph(self) -> dict:
        """Everything discovered so far, the way ``ros2 topic list -t`` and friends see it."""
        with self.lock:
            topics: dict[str, dict] = {}
            for table, role in ((self.remote_writers, "publishers"), (self.remote_readers, "subscribers")):
                for ep in table.values():
                    mapped = rtps.dds_to_ros(ep.topic)
                    if mapped is None:
                        continue
                    kind, name = mapped
                    entry = topics.setdefault(
                        f"{kind} {name}",
                        {
                            "kind": kind,
                            "name": name,
                            "type": rtps.dds_type_to_ros(ep.type_name),
                            "publishers": 0,
                            "subscribers": 0,
                            "qos": set(),
                            "type_hash": ep.type_hash,
                        },
                    )
                    entry[role] += 1
                    entry["qos"].add(f"{ep.qos.reliability}/{ep.qos.durability}")
            for t in topics.values():
                t["qos"] = sorted(t["qos"])
            return {
                "participants": [
                    {
                        "guid_prefix": rp.data.guid_prefix.hex(),
                        "vendor": rp.data.vendor_name,
                        "address": rp.address,
                        "enclave": rp.data.enclave,
                        "locators": [str(x) for x in rp.data.default_unicast],
                    }
                    for rp in self.participants.values()
                ],
                "topics": sorted(topics.values(), key=lambda t: (t["kind"], t["name"])),
            }


class Publisher:
    def __init__(self, participant: Participant, writer: Writer) -> None:
        self.p, self.writer = participant, writer

    @property
    def matched(self) -> int:
        with self.p.lock:
            return self.writer.matched

    def publish(self, value: dict) -> int:
        payload = cdr.serialize(self.writer.ros_type, value, self.p.registry, pad=True)
        return self.writer.write_payload(payload)


class Client:
    """A service client that interoperates with rmw_fastrtps servers."""

    def __init__(self, participant: Participant, service: str, srv_type: str) -> None:
        self.p = participant
        srv = participant.registry.service(srv_type)
        self.request_type, self.response_type = srv.request.name, srv.response.name
        rq, rr = rtps.ros_service_to_dds(service)
        self.pending: dict[int, threading.Event] = {}
        self.replies: dict[int, dict] = {}
        self.reader = participant._reader(rr, self.response_type, qos(depth=1000), self._on_reply)
        self.writer = participant._writer(rq, self.request_type, qos(depth=1000))

    def service_is_ready(self) -> bool:
        """Matched both ways, and the server has acknowledged our reply reader.

        Matching our side is not enough: rmw_fastrtps drops a reply when the
        server has not yet matched the client's reply reader, so wait until the
        server's participant has acknowledged that reader's SEDP announcement.
        """
        with self.p.lock:
            if not (self.writer.matched and self.reader.matched):
                return False
            servers = {g.prefix for g in self.reader.writers}
            return any(self.p.announced_to(prefix, self.reader) and self.p.announced_to(prefix, self.writer) for prefix in servers)

    def wait_for_service(self, timeout: float = 10.0) -> bool:
        return self.p.wait_until(self.service_is_ready, timeout)

    def _on_reply(self, sample: Sample) -> None:
        ident = sample.related_identity()
        if ident is None:
            return
        guid, seq = ident
        if guid not in (self.reader.guid, self.writer.guid) or seq not in self.pending:
            return
        self.replies[seq] = sample.value
        self.pending[seq].set()

    def call(self, request: dict, timeout: float = 10.0) -> dict:
        payload = cdr.serialize(self.request_type, request, self.p.registry, pad=True)
        # rmw_fastrtps puts the client's reply reader in the request's related sample identity
        ident = _identity(self.reader.guid, SEQUENCE_UNKNOWN)
        inline = ParameterList().add(rtps.PID_RELATED_SAMPLE_IDENTITY, ident).add(rtps.PID_FASTDDS_RELATED_SAMPLE_IDENTITY, ident)
        with self.p.lock:
            seq = self.writer.seq + 1
            event = self.pending[seq] = threading.Event()
            self.writer.write_payload(payload, inline)
        if not event.wait(timeout):
            del self.pending[seq]
            raise TimeoutError(f"no reply from {self.writer.topic} within {timeout} s")
        del self.pending[seq]
        return self.replies.pop(seq)


class Service:
    """A service server that answers rmw_fastrtps clients."""

    def __init__(self, participant: Participant, service: str, srv_type: str, handler: Callable[[dict], dict]) -> None:
        self.p, self.handler = participant, handler
        srv = participant.registry.service(srv_type)
        self.request_type, self.response_type = srv.request.name, srv.response.name
        rq, rr = rtps.ros_service_to_dds(service)
        self.requests = 0
        self.writer = participant._writer(rr, self.response_type, qos(depth=1000))
        self.reader = participant._reader(rq, self.request_type, qos(depth=1000), self._on_request)

    def _on_request(self, sample: Sample) -> None:
        if sample.value is None:
            return
        self.requests += 1
        response = self.handler(sample.value)
        ident = sample.related_identity()
        target = ident[0] if ident and ident[0].entity != 0 else sample.writer
        reply_id = _identity(target, sample.seq)
        inline = ParameterList().add(rtps.PID_RELATED_SAMPLE_IDENTITY, reply_id).add(rtps.PID_FASTDDS_RELATED_SAMPLE_IDENTITY, reply_id)
        self.writer.write_payload(cdr.serialize(self.response_type, response, self.p.registry, pad=True), inline)
