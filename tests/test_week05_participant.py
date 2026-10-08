"""Two pure Python participants talking over real UDP sockets on this machine."""

from __future__ import annotations

import os
import random
import time

import pytest

from week05 import participant as part
from week05.participant import Participant, qos

DOMAIN = 100 + os.getpid() % 100  # away from ROS 2's default domain 0


@pytest.fixture
def pair():
    a = Participant(domain=DOMAIN, name="a").start()
    b = Participant(domain=DOMAIN, name="b").start()
    yield a, b
    a.close()
    b.close()


def test_discovery(pair):
    a, b = pair
    assert a.wait_until(lambda: b.prefix in a.participants, 10)
    assert b.wait_until(lambda: a.prefix in b.participants, 10)


def test_reliable_topic_in_order(pair):
    a, b = pair
    got = []
    b.create_subscription("/chat", "std_msgs/msg/String", lambda s: got.append(s.value["data"]))
    pub = a.create_publisher("/chat", "std_msgs/msg/String")
    assert a.wait_until(lambda: pub.matched > 0, 10)
    for i in range(20):
        pub.publish({"data": f"m{i}"})
    assert b.wait_until(lambda: len(got) == 20, 10)
    assert got == [f"m{i}" for i in range(20)]
    graph = b.graph()
    assert any(t["name"] == "/chat" and t["publishers"] == 1 for t in graph["topics"])


def test_reliability_survives_packet_loss(pair, monkeypatch):
    a, b = pair
    got = []
    b.create_subscription("/lossy", "std_msgs/msg/String", lambda s: got.append(s.value["data"]))
    pub = a.create_publisher("/lossy", "std_msgs/msg/String", qos(depth=100))
    assert a.wait_until(lambda: pub.matched > 0, 10)
    rng = random.Random(1)
    real_send = a._send
    monkeypatch.setattr(a, "_send", lambda data, locs: None if rng.random() < 0.3 else real_send(data, locs))
    for i in range(30):
        pub.publish({"data": f"m{i}"})
    assert b.wait_until(lambda: len(got) == 30, 20), got
    assert got == [f"m{i}" for i in range(30)]


def test_best_effort_reader_takes_reliable_writer_but_not_the_reverse(pair):
    a, b = pair
    got = []
    b.create_subscription("/be", "std_msgs/msg/String", lambda s: got.append(1), qos("best_effort"))
    reliable = a.create_publisher("/be", "std_msgs/msg/String")
    assert a.wait_until(lambda: reliable.matched > 0, 10)
    strict = []
    b.create_subscription("/be2", "std_msgs/msg/String", lambda s: strict.append(1))
    best_effort = a.create_publisher("/be2", "std_msgs/msg/String", qos("best_effort"))
    time.sleep(1.5)
    assert best_effort.matched == 0  # requested reliable, offered best effort: no match


def test_transient_local_late_joiner(pair):
    a, b = pair
    pub = a.create_publisher("/mission", "std_msgs/msg/String", qos(durability="transient_local", depth=1))
    pub.publish({"data": "old"})
    pub.publish({"data": "latest"})
    late, volatile = [], []
    b.create_subscription(
        "/mission", "std_msgs/msg/String", lambda s: late.append(s.value["data"]), qos(durability="transient_local", depth=1)
    )
    b.create_subscription("/mission", "std_msgs/msg/String", lambda s: volatile.append(s.value["data"]))
    assert b.wait_until(lambda: late == ["latest"], 10)
    time.sleep(0.5)
    assert volatile == []


def test_large_samples_are_fragmented(pair):
    a, b = pair
    got = []
    b.create_subscription("/image", "sensor_msgs/msg/Image", lambda s: got.append(s.value))
    pub = a.create_publisher("/image", "sensor_msgs/msg/Image")
    assert a.wait_until(lambda: pub.matched > 0, 10)
    pixels = bytes(random.Random(3).getrandbits(8) for _ in range(64 * 48 * 3))
    pub.publish({"height": 48, "width": 64, "encoding": "rgb8", "step": 192, "data": pixels})
    assert b.wait_until(lambda: len(got) == 1, 10)
    assert got[0]["data"] == pixels and len(pixels) > 4 * part.FRAGMENT_SIZE


def test_service_round_trip(pair):
    a, b = pair
    service = b.create_service("/add_two_ints", "example_interfaces/srv/AddTwoInts", lambda r: {"sum": r["a"] + r["b"]})
    client = a.create_client("/add_two_ints", "example_interfaces/srv/AddTwoInts")
    assert client.wait_for_service(10)
    assert client.call({"a": 2, "b": 3}) == {"sum": 5}
    assert client.call({"a": -40, "b": 82}) == {"sum": 42}
    assert service.requests == 2


def test_qos_matching_rules():
    assert part.compatible(qos("reliable"), qos("best_effort"))
    assert not part.compatible(qos("best_effort"), qos("reliable"))
    assert part.compatible(qos(durability="transient_local"), qos())
    assert not part.compatible(qos(), qos(durability="transient_local"))
