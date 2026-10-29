"""week05 against a live ROS 2 Jazzy install. Runs in the CI's ROS container; skipped elsewhere.

    source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
    pytest tests/test_week05_ros.py

These are the claims the docs make, checked against the real thing: the pure
Python participant talks to the tutorial nodes in both directions over Fast DDS,
receives from Cyclone DDS, CDR agrees with rclpy, every installed interface's
type hash agrees with rosidl, and the QoS rules agree with rclpy.
"""

from __future__ import annotations

import glob
import importlib.util
import itertools
import json
import random
import shutil
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ros2") is None or importlib.util.find_spec("rclpy") is None or importlib.util.find_spec("ugs_interfaces") is None,
    reason="needs ROS 2 Jazzy and the built ros2_ws sourced",
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def experiments():
    from week05 import experiments

    return experiments


def test_type_hashes_match_rosidl_for_every_installed_interface():
    from week05.idl import Registry

    reg = Registry([ROOT / "ros2_ws" / "src", "/opt/ros/jazzy/share"])
    total = 0
    for path in glob.glob("/opt/ros/jazzy/share/*/*/*.json"):
        if Path(path).parent.name not in ("msg", "srv", "action"):
            continue
        for entry in json.loads(Path(path).read_text())["type_hashes"]:
            assert reg.type_hash(entry["type_name"]) == entry["hash_string"], entry["type_name"]
            total += 1
    assert total > 1000


def test_cdr_matches_rclpy(experiments):
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.set_message import set_message_fields
    from ugs_interfaces.msg import AllTypes

    from week05 import cdr
    from week05.idl import Registry

    reg = Registry([ROOT / "ros2_ws" / "src", "/opt/ros/jazzy/share"])
    rnd = random.Random(11)
    for _ in range(100):
        value = experiments.random_alltypes(rnd)
        ros = AllTypes()
        set_message_fields(ros, experiments._to_ros(value))
        theirs = serialize_message(ros)
        ours, padding = cdr.serialize_with_padding("ugs_interfaces/msg/AllTypes", value, reg)
        assert cdr.same_except_padding(ours, theirs[: len(ours)], padding)
        assert deserialize_message(ours, AllTypes) == ros


def test_qos_matches_rclpy():
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, LivelinessPolicy, QoSCompatibility, QoSProfile, ReliabilityPolicy, qos_check_compatible

    from week05 import qos

    rel = {n: getattr(ReliabilityPolicy, n.upper()) for n in qos.RELIABILITY}
    dur = {n: getattr(DurabilityPolicy, n.upper()) for n in qos.DURABILITY}
    liv = {n: getattr(LivelinessPolicy, n.upper()) for n in qos.LIVELINESS}
    levels = {QoSCompatibility.OK: "ok", QoSCompatibility.WARNING: "warning", QoSCompatibility.ERROR: "error"}
    profiles = list(itertools.product(rel, dur, liv, (0, 10**9), (0, 2 * 10**9)))

    def ros(p):
        q = QoSProfile(depth=10, reliability=rel[p[0]], durability=dur[p[1]], liveliness=liv[p[2]])
        q.deadline, q.liveliness_lease_duration = Duration(nanoseconds=p[3]), Duration(nanoseconds=p[4])
        return q

    for a in profiles:
        ra, ua = ros(a), qos.QoSProfile(reliability=a[0], durability=a[1], liveliness=a[2], deadline_ns=a[3], lease_ns=a[4])
        for b in profiles:
            level, reason = qos_check_compatible(ra, ros(b))
            assert qos.check_compatible(
                ua, qos.QoSProfile(reliability=b[0], durability=b[1], liveliness=b[2], deadline_ns=b[3], lease_ns=b[4])
            ) == (levels[level], reason)


def test_presets_match_rclpy():
    import rclpy.qos as rq

    from week05 import qos

    presets = {
        "default": rq.QoSPresetProfiles.DEFAULT.value,
        "sensor_data": rq.qos_profile_sensor_data,
        "services_default": rq.qos_profile_services_default,
        "parameters": rq.qos_profile_parameters,
        "parameter_events": rq.qos_profile_parameter_events,
    }
    for name, ros in presets.items():
        mine = qos.PRESETS[name]
        assert mine.reliability == ros.reliability.name.lower(), name
        assert mine.durability == ros.durability.name.lower(), name
        assert mine.depth == ros.depth, name


def test_interop_with_the_tutorial_nodes(experiments):
    results = experiments.interop()
    assert len(results["ros_talker_to_python_listener"]["received"]) == 5
    assert len(results["python_talker_to_ros_listener"]["listener_output"]) >= 4
    assert results["python_client_to_ros_service"]["reply"] == {"sum": 5}
    assert results["ros_clients_to_python_service"]["client_output"]["py_srvcli"]
    assert any("Sum: 42" in line for line in results["ros_clients_to_python_service"]["client_output"]["cpp_srvcli"])
    assert len(results["cyclone_talker_to_python_listener"]["received"]) == 3


def test_graph_sees_a_ros_node(experiments):
    talker = experiments.Proc("ros2 run py_pubsub talker")
    try:
        assert talker.wait_for("Publishing", 30)
        time.sleep(1.0)
        graph = experiments._participant_run("with Participant(name='week05_graph') as p:\n    p.spin(3.0)\n    result = p.graph()\n")
    finally:
        talker.stop()
    topics = {(t["name"], t["type"]) for t in graph["topics"]}
    assert ("/topic", "std_msgs/msg/String") in topics
    assert ("/rosout", "rcl_interfaces/msg/Log") in topics
