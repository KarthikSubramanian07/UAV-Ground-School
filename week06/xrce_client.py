"""A DDS-XRCE client in pure Python: PX4's half of the link, without PX4 or ROS 2.

It logs in to a Micro XRCE-DDS Agent over UDP, asks it for a participant, then
for topics, publishers, subscribers, data writers and data readers, exactly as
PX4's ``uxrce_dds_client`` does. The agent turns them into real DDS entities, so
this client can read PX4's ``/fmu/out`` topics and write ``/fmu/in`` ones, which
is all the takeoff server needs: ``python -m week06 xrce fly`` runs the same
takeoff logic as the ROS 2 action server, with nothing but sockets in between.

Samples are plain CDR without the 4 byte encapsulation header (the agent adds
it), encoded and decoded with week05's CDR from the px4_msgs definitions
bundled in ``week06/interfaces``.

Everything goes over best effort stream 1. Replies to requests (STATUS) and
reliable traffic the agent may send are acknowledged so the agent never stalls.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from week05 import cdr
from week05.idl import Registry

from . import xrce

INTERFACES = Path(__file__).parent / "interfaces"
BEST_EFFORT = 0x01
ENCAPSULATION = b"\x00\x01\x00\x00"  # CDR little endian


def registry() -> Registry:
    from week05.idl import default_roots

    return Registry([INTERFACES, *default_roots()])


def dds_type(type_name: str) -> str:
    """px4_msgs/msg/VehicleStatus -> px4_msgs::msg::dds_::VehicleStatus_, ROS 2's DDS name."""
    pkg, kind, name = type_name.split("/")
    return f"{pkg}::{kind}::dds_::{name}_"


@dataclass
class Endpoint:
    topic: str
    type_name: str
    number: int
    callback: Callable | None = None
    received: int = 0


class XrceClient:
    """Log in to an agent and create PX4 style entities on it."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888, domain: int = 0, key: bytes | None = None, timeout: float = 2.0):
        self.addr = (host, port)
        self.domain = domain
        self.key = key or os.urandom(4)
        self.session_id = 0x81  # 0x80 and up: no client key in each header
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        self.registry = registry()
        self.sequence = 0
        self.request_id = 0x10
        self.next_number = 1
        self.participant = None
        self.readers: dict[bytes, Endpoint] = {}
        self.writers: dict[str, tuple[Endpoint, bytes]] = {}
        self.status: dict[int, dict] = {}
        self.agent: dict | None = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = False
        self.thread = None
        self.sent = 0
        self.received = 0

    # ------------------------------------------------------------ transport

    def _send(self, *subs: bytes, stream: int = BEST_EFFORT) -> None:
        with self.lock:
            seq = self.sequence if stream else 0
            if stream:
                self.sequence = (self.sequence + 1) & 0xFFFF
        head = xrce.header(self.session_id, stream, seq, self.key)
        self.sock.sendto(xrce.message(head, *subs), self.addr)
        self.sent += 1

    def _receive_loop(self) -> None:
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                msg = xrce.parse(data)
            except xrce.XrceError:
                continue
            self.received += 1
            self._handle(msg)

    def _handle(self, msg: xrce.Message) -> None:
        for sub in msg.submessages:
            body = sub.body
            if sub.name == "STATUS_AGENT":
                with self.cond:
                    self.agent = body
                    self.cond.notify_all()
            elif sub.name == "STATUS":
                with self.cond:
                    self.status[body["request_id"]] = body
                    self.cond.notify_all()
            elif sub.name == "DATA":
                reader = self.readers.get(bytes.fromhex(body["object_id"]))
                if reader is not None:
                    reader.received += 1
                    if reader.callback is not None:
                        reader.callback(self.decode(reader.type_name, body["data"]))
            elif sub.name == "HEARTBEAT":
                self._send(
                    xrce.submessage(xrce.SUB["ACKNACK"], 0, xrce.acknack_payload(body["last_unacked"] + 1, body["stream"])), stream=0
                )
        if msg.reliable:  # acknowledge everything up to this one
            self._send(xrce.submessage(xrce.SUB["ACKNACK"], 0, xrce.acknack_payload((msg.sequence + 1) & 0xFFFF, msg.stream_id)), stream=0)

    def decode(self, type_name: str, data: bytes) -> dict:
        return cdr.deserialize(type_name, ENCAPSULATION + data, self.registry)

    def encode(self, type_name: str, value: dict) -> bytes:
        return cdr.serialize(type_name, value, self.registry)[4:]

    # -------------------------------------------------------------- session

    def connect(self) -> dict:
        """CREATE_CLIENT, then wait for the agent's STATUS_AGENT."""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        payload = xrce.create_client_payload(self.key, self.session_id)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            # login goes on stream 0, with the client key in the header (session id & 0x80)
            head = xrce.header(self.session_id & 0x80, 0, 0, self.key)
            self.sock.sendto(xrce.message(head, xrce.submessage(xrce.SUB["CREATE_CLIENT"], 0, payload)), self.addr)
            with self.cond:
                self.cond.wait_for(lambda: self.agent is not None, timeout=0.3)
                if self.agent is not None:
                    break
        if self.agent is None:
            raise TimeoutError(f"no XRCE agent answered at {self.addr[0]}:{self.addr[1]}")
        if not self.agent["status"].startswith("OK"):
            raise ConnectionError(f"agent refused the session: {self.agent['status']}")
        self.participant = self._create("PARTICIPANT", 1, xrce.encode_binary("PARTICIPANT"))
        return self.agent

    def _create(self, kind: str, number: int, blob: bytes, parent: bytes | None = None) -> bytes:
        raw = xrce.object_id(number, kind)
        with self.lock:
            self.request_id = (self.request_id + 1) & 0xFFFF
            request = self.request_id
        # the payload starts 4 (header) + 4 (submessage header) bytes into the message
        payload = xrce.create_payload(request, raw, kind, blob, parent, self.domain, origin=8)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self._send(xrce.submessage(xrce.SUB["CREATE"], xrce.CREATE_REUSE | xrce.CREATE_REPLACE, payload))
            with self.cond:
                self.cond.wait_for(lambda: request in self.status, timeout=0.3)
                reply = self.status.get(request)
            if reply is not None:
                if not reply["status"].startswith("OK"):
                    raise ConnectionError(f"agent refused {kind.lower()} {number}: {reply['status']}")
                return raw
        raise TimeoutError(f"no STATUS for {kind.lower()} {number}")

    def _topic(self, topic: str, type_name: str) -> tuple[int, bytes]:
        number = self.next_number
        self.next_number += 1
        name = "rt" + topic if topic.startswith("/") else topic
        topic_id = self._create("TOPIC", number, xrce.encode_binary("TOPIC", name=name, type_name=dds_type(type_name)), self.participant)
        return number, topic_id

    def subscribe(self, topic: str, type_name: str, callback: Callable[[dict], None], reliable: bool = False, depth: int = 5) -> Endpoint:
        """Create topic, subscriber and data reader, then ask the agent to stream samples."""
        number, topic_id = self._topic(topic, type_name)
        sub = self._create("SUBSCRIBER", number, xrce.encode_binary("SUBSCRIBER"), self.participant)
        qos = {"reliable": reliable, "history": "keep_last", "depth": depth}
        reader = self._create("DATAREADER", number, xrce.encode_binary("DATAREADER", topic_id=topic_id, qos=qos), sub)
        endpoint = Endpoint(topic, type_name, number, callback)
        self.readers[reader] = endpoint
        with self.lock:
            self.request_id = (self.request_id + 1) & 0xFFFF
            request = self.request_id
        self._send(xrce.submessage(xrce.SUB["READ_DATA"], 0, xrce.read_data_payload(request, reader, BEST_EFFORT)))
        return endpoint

    def publisher(self, topic: str, type_name: str, reliable: bool = False) -> Endpoint:
        number, topic_id = self._topic(topic, type_name)
        pub = self._create("PUBLISHER", number, xrce.encode_binary("PUBLISHER"), self.participant)
        qos = {"reliable": reliable, "history": "keep_last", "depth": 10}
        writer = self._create("DATAWRITER", number, xrce.encode_binary("DATAWRITER", topic_id=topic_id, qos=qos), pub)
        endpoint = Endpoint(topic, type_name, number)
        self.writers[topic] = (endpoint, writer)
        return endpoint

    def write(self, topic: str, value: dict) -> None:
        endpoint, writer = self.writers[topic]
        with self.lock:
            self.request_id = (self.request_id + 1) & 0xFFFF
            request = self.request_id
        payload = request.to_bytes(2, "big") + writer + self.encode(endpoint.type_name, value)
        self._send(xrce.submessage(xrce.SUB["WRITE_DATA"], xrce.FORMAT_DATA, payload))

    def close(self) -> None:
        if self.running and self.agent is not None:
            # DELETE the client object: the agent drops the session and every entity in it
            payload = (0).to_bytes(2, "big") + b"\xff\xfe"
            with contextlib.suppress(OSError):
                self._send(xrce.submessage(xrce.SUB["DELETE"], 0, payload))
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.sock.close()

    def __enter__(self) -> XrceClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
