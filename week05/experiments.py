"""The live experiments behind docs/week05, run inside the ROS 2 container.

    python -m week05 docs --ros            # from the host: runs this in Docker
    python3 -m week05.experiments OUT_DIR   # inside the container, ROS sourced

Each step drives real ROS 2 Jazzy processes and records what happened:

1. ``transcripts.json``: the four tutorials (Python and C++ publisher and
   subscriber, service and client), exactly as the tutorial runs them;
2. ``captures/*.pcap``: the RTPS packets of the Python tutorials, with Fast
   DDS's shared memory transport off so they cross UDP where tcpdump sees them;
3. ``interop.json``: this repository's pure Python participant against the
   tutorial nodes, in all four directions, and against Cyclone DDS;
4. ``mission.json``: the drone mission, recorded by the pure Python participant
   (odometry, detections, action feedback and status), plus the launch log;
5. ``lab.json``: ``ros2 run ugs_drone lab``;
6. ``checks.json``: CDR against ``rclpy.serialization``, type hashes against
   every installed interface, QoS against ``rclpy.qos.qos_check_compatible``.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "ros2_ws"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Proc:
    """A ROS 2 process whose output is collected line by line."""

    def __init__(self, command: str, env: dict | None = None) -> None:
        full_env = dict(os.environ, RCUTILS_COLORIZED_OUTPUT="0", PYTHONUNBUFFERED="1", **(env or {}))
        self.lines: list[tuple[float, str]] = []
        self.start = time.monotonic()
        self.p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=full_env,
            start_new_session=True,
            executable="/bin/bash",
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.p.stdout:
            self.lines.append((time.monotonic() - self.start, ANSI.sub("", line.rstrip("\n"))))

    def wait_for(self, pattern: str, timeout: float = 30.0) -> bool:
        regex = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(regex.search(line) for _, line in self.lines):
                return True
            if self.p.poll() is not None and not self.thread.is_alive():
                break
            time.sleep(0.05)
        return any(regex.search(line) for _, line in self.lines)

    def stop(self, timeout: float = 5.0) -> list[str]:
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGINT)
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.thread.join(timeout=2)
        return self.text()

    def text(self) -> list[str]:
        return [line for _, line in self.lines]


def _clean(lines: list[str]) -> list[str]:
    """Keep the tutorial's own output; drop Ctrl+C tracebacks."""
    out = []
    for line in lines:
        if line.startswith(("Traceback", "  ", "KeyboardInterrupt", "[ros2run]", "rclpy.")) or not line.strip():
            continue
        out.append(line)
    return out


def transcripts() -> list[dict]:
    runs = []
    for lang, pkg, talker_msg in (
        ("Python", "py_pubsub", 'Publishing: "Hello World: 4"'),
        ("C++", "cpp_pubsub", "Publishing: 'Hello, world! 4'"),
    ):
        talker = Proc(f"ros2 run {pkg} talker")
        talker.wait_for("Publishing", 30)
        time.sleep(2.0)  # the listener joins partway, as in the tutorial
        listener = Proc(f"ros2 run {pkg} listener")
        listener.wait_for("I heard", 30)
        talker.wait_for(re.escape(talker_msg), 10)
        time.sleep(1.2)
        runs.append(
            {
                "title": f"{lang} publisher and subscriber",
                "terminals": [
                    {"command": f"ros2 run {pkg} talker", "output": _clean(talker.stop())},
                    {"command": f"ros2 run {pkg} listener", "output": _clean(listener.stop())},
                ],
            }
        )
    for lang, pkg, server_exe, client_exe in (("Python", "py_srvcli", "service", "client"), ("C++", "cpp_srvcli", "server", "client")):
        server = Proc(f"ros2 run {pkg} {server_exe}")
        time.sleep(1.5)
        client = Proc(f"ros2 run {pkg} {client_exe} 2 3")
        client.p.wait(30)
        client.thread.join(2)
        time.sleep(0.5)
        runs.append(
            {
                "title": f"{lang} service and client",
                "terminals": [
                    {"command": f"ros2 run {pkg} {server_exe}", "output": _clean(server.stop())},
                    {"command": f"ros2 run {pkg} {client_exe} 2 3", "output": _clean(client.text())},
                ],
            }
        )
    return runs


def capture(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    udp = {"FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"}
    for name, steps in (
        ("pubsub", [("ros2 run py_pubsub talker", "Publishing"), ("ros2 run py_pubsub listener", "I heard")]),
        ("srvcli", [("ros2 run py_srvcli service", None), ("ros2 run py_srvcli client 2 3", "Result of add_two_ints")]),
    ):
        dump = subprocess.Popen(
            ["tcpdump", "-i", "any", "-w", str(out / f"{name}.pcap"), "-U", "udp", "portrange", "7400-7700"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        procs = []
        for command, ready in steps:
            procs.append(Proc(command, udp))
            if ready:
                procs[-1].wait_for(ready, 30)
            else:
                time.sleep(2.0)
        time.sleep(2.0)  # a few more messages, then stop
        for p in reversed(procs):
            p.stop()
        time.sleep(0.5)
        dump.send_signal(signal.SIGINT)
        dump.wait(5)


def _participant_run(code: str, timeout: float = 40.0) -> dict:
    """Run a snippet against week05.participant in a clean interpreter with no ROS on its path."""
    script = (
        f"import json, sys, time\nsys.path.insert(0, {str(ROOT)!r})\nfrom week05.participant import Participant\nresult = {{}}\n"
        + code
        + "\nprint('RESULT ' + json.dumps(result), flush=True)\n"
    )
    env = {"PATH": "/usr/bin:/bin"}
    p = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=timeout, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"participant run failed:\n{p.stdout}\n{p.stderr}")


def interop() -> dict:
    results = {}
    talker = Proc("ros2 run py_pubsub talker")
    talker.wait_for("Publishing", 30)
    results["ros_talker_to_python_listener"] = _participant_run(
        """
got = []
with Participant(name='week05_listener') as p:
    p.create_subscription('/topic', 'std_msgs/msg/String', lambda s: got.append(s.value['data']))
    p.wait_until(lambda: len(got) >= 5, 20)
    result = {'received': got[:5], 'participants': len(p.participants), 'remote_writers': len(p.remote_writers)}
"""
    )
    talker.stop()

    listener = Proc("ros2 run py_pubsub listener")
    listener.wait_for(".", 3)
    time.sleep(2.0)
    results["python_talker_to_ros_listener"] = _participant_run(
        """
with Participant(name='week05_talker') as p:
    pub = p.create_publisher('/topic', 'std_msgs/msg/String')
    matched = p.wait_until(lambda: pub.matched > 0, 20)
    for i in range(5):
        pub.publish({'data': f'Hello from pure Python: {i}'})
        time.sleep(0.5)
    p.spin(1.0)
    result = {'matched': matched}
"""
    )
    listener.wait_for("Hello from pure Python: 4", 10)
    results["python_talker_to_ros_listener"]["listener_output"] = [line for line in listener.stop() if "I heard" in line]

    service = Proc("ros2 run py_srvcli service")
    time.sleep(2.0)
    results["python_client_to_ros_service"] = _participant_run(
        """
with Participant(name='week05_client') as p:
    c = p.create_client('/add_two_ints', 'example_interfaces/srv/AddTwoInts')
    ready = c.wait_for_service(20)
    t = time.perf_counter()
    reply = c.call({'a': 2, 'b': 3})
    ms = 1000 * (time.perf_counter() - t)
    result = {'ready': ready, 'request': {'a': 2, 'b': 3}, 'reply': reply, 'round_trip_ms': round(ms, 2)}
"""
    )
    results["python_client_to_ros_service"]["service_output"] = [
        line for line in service.stop() if "a: 2 b: 3" in line or "Incoming" in line
    ]

    server = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(ROOT)!r})\nfrom week05.cli import main\nmain(['serve-add-two-ints', '--seconds', '25'])",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    time.sleep(2.0)
    outputs = {}
    for pkg in ("py_srvcli", "cpp_srvcli"):
        c = Proc(f"ros2 run {pkg} client 7 35")
        c.p.wait(30)
        c.thread.join(2)
        outputs[pkg] = [line for line in c.text() if "Result" in line or "Sum" in line]
    server.terminate()
    served = server.communicate(timeout=10)[0]
    results["ros_clients_to_python_service"] = {
        "client_output": outputs,
        "service_output": [line for line in served.splitlines() if line.strip()],
    }

    cyclone = Proc("ros2 run py_pubsub talker", {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"})
    cyclone.wait_for("Publishing", 30)
    results["cyclone_talker_to_python_listener"] = _participant_run(
        """
got, vendors = [], []
with Participant(name='week05_listener') as p:
    p.create_subscription('/topic', 'std_msgs/msg/String', lambda s: got.append(s.value['data']))
    p.wait_until(lambda: len(got) >= 3, 20)
    vendors = sorted({rp.data.vendor_name for rp in p.participants.values()})
    result = {'received': got[:3], 'vendors': vendors}
"""
    )
    cyclone.stop()
    return results


def mission(out: Path) -> dict:
    launch = Proc("ros2 launch ugs_drone drone.launch.py")
    recorder = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, {!r})\nfrom week05.record import main\nmain(['--seconds', '90', '--out', {!r}])".format(
                str(ROOT), str(out / "mission_raw.json")
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    launch.wait_for("MISSION COMPLETE", 90)
    time.sleep(1.0)
    log = launch.stop()
    recorder.send_signal(signal.SIGINT)
    recorder.communicate(timeout=20)
    raw = json.loads((out / "mission_raw.json").read_text())
    (out / "mission_raw.json").unlink()
    events = []
    for line in log:
        m = re.match(r"\[(\w+)-\d+\] \[(\w+)\] \[([\d.]+)\] \[(\w+)\]: (.*)", line)
        if m and m.group(1) in ("planner", "px4_bridge") and "feedback:" not in m.group(5):
            events.append({"t": float(m.group(3)), "node": m.group(4), "level": m.group(2), "text": m.group(5)})
    t0 = events[0]["t"] if events else 0.0
    for e in events:
        e["t"] = round(e["t"] - t0, 3)
    raw["t0_wall"] = t0
    raw["events"] = events
    return raw


def lab(out: Path) -> dict:
    path = out / "lab_raw.json"
    subprocess.run(f"ros2 run ugs_drone lab --json {path}", shell=True, check=True, executable="/bin/bash", stdout=subprocess.DEVNULL)
    data = json.loads(path.read_text())
    path.unlink()
    return data


def checks() -> dict:
    from rclpy.qos import QoSProfile  # noqa: F401  (ROS must be sourced)
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.set_message import set_message_fields
    from ugs_interfaces.msg import AllTypes

    from . import cdr
    from .idl import Registry

    reg = Registry([WS / "src", "/opt/ros/jazzy/share"])
    rnd = random.Random(2026)
    same = exact = dirty = overshoot = back = 0
    trials = 300
    for _ in range(trials):
        value = random_alltypes(rnd)
        ros = AllTypes()
        set_message_fields(ros, _to_ros(value))
        theirs = serialize_message(ros)
        ours, padding = cdr.serialize_with_padding("ugs_interfaces/msg/AllTypes", value, reg)
        # rclpy's length is rmw_fastrtps's estimate, which can overshoot the CDR it wrote
        same += cdr.same_except_padding(ours, theirs[: len(ours)], padding)
        exact += ours == theirs
        dirty += any(theirs[i] for i in padding)
        overshoot += len(theirs) - len(ours)
        back += deserialize_message(ours, AllTypes) == ros
    hashes = total = 0
    for path in sorted(glob.glob("/opt/ros/jazzy/share/*/*/*.json")) + sorted(glob.glob(str(WS / "install/*/share/*/*/*.json"))):
        if Path(path).parent.name not in ("msg", "srv", "action"):
            continue
        for entry in json.loads(Path(path).read_text())["type_hashes"]:
            total += 1
            hashes += reg.type_hash(entry["type_name"]) == entry["hash_string"]
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, LivelinessPolicy, QoSCompatibility, ReliabilityPolicy, qos_check_compatible

    from . import qos as ugs_qos

    rel = {n: getattr(ReliabilityPolicy, n.upper()) for n in ugs_qos.RELIABILITY}
    dur = {n: getattr(DurabilityPolicy, n.upper()) for n in ugs_qos.DURABILITY}
    liv = {n: getattr(LivelinessPolicy, n.upper()) for n in ugs_qos.LIVELINESS}
    levels = {QoSCompatibility.OK: "ok", QoSCompatibility.WARNING: "warning", QoSCompatibility.ERROR: "error"}
    times = (0, 10**9, 2 * 10**9)
    profiles = list(itertools.product(rel, dur, liv, times, times))
    ros_profiles, our_profiles = [], []
    for p in profiles:
        q = QoSProfile(depth=10, reliability=rel[p[0]], durability=dur[p[1]], liveliness=liv[p[2]])
        q.deadline, q.liveliness_lease_duration = Duration(nanoseconds=p[3]), Duration(nanoseconds=p[4])
        ros_profiles.append(q)
        our_profiles.append(ugs_qos.QoSProfile(reliability=p[0], durability=p[1], liveliness=p[2], deadline_ns=p[3], lease_ns=p[4]))
    qos_pairs = qos_same = 0
    for ra, ua in zip(ros_profiles, our_profiles):
        for rb, ub in zip(ros_profiles, our_profiles):
            level, reason = qos_check_compatible(ra, rb)
            qos_pairs += 1
            qos_same += ugs_qos.check_compatible(ua, ub) == (levels[level], reason)
    return {
        "qos_pairs_identical": qos_same,
        "qos_pairs_total": qos_pairs,
        "cdr_alltypes_identical_except_padding": same,
        "cdr_alltypes_byte_identical": exact,
        "cdr_alltypes_rclpy_padding_not_zero": dirty,
        "cdr_alltypes_rclpy_deserializes_ours": back,
        "cdr_rclpy_mean_overshoot_bytes": round(overshoot / trials, 1),
        "cdr_alltypes_trials": trials,
        "type_hashes_identical": hashes,
        "type_hashes_total": total,
    }


def fixtures(out: Path) -> None:
    """Reference outputs from ROS 2 itself, for the tests that run without ROS."""
    import itertools

    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, LivelinessPolicy, QoSCompatibility, QoSProfile, ReliabilityPolicy, qos_check_compatible
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.set_message import set_message_fields
    from ugs_interfaces.msg import AllTypes

    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)
    cases = []
    for _ in range(60):
        value = random_alltypes(rnd)
        ros = AllTypes()
        set_message_fields(ros, _to_ros(value))
        cases.append({"type": "ugs_interfaces/msg/AllTypes", "value": _jsonable(value), "cdr": serialize_message(ros).hex()})
    from example_interfaces.srv import AddTwoInts
    from geometry_msgs.msg import Point
    from std_msgs.msg import String

    for msg, name, value in (
        (String(data="Hello World: 0"), "std_msgs/msg/String", {"data": "Hello World: 0"}),
        (String(data=""), "std_msgs/msg/String", {"data": ""}),
        (AddTwoInts.Request(a=2, b=3), "example_interfaces/srv/AddTwoInts_Request", {"a": 2, "b": 3}),
        (AddTwoInts.Response(sum=5), "example_interfaces/srv/AddTwoInts_Response", {"sum": 5}),
        (Point(x=1.5, y=-2.0, z=5.0), "geometry_msgs/msg/Point", {"x": 1.5, "y": -2.0, "z": 5.0}),
    ):
        cases.append({"type": name, "value": value, "cdr": serialize_message(msg).hex()})
    (out / "cdr_cases.json").write_text(json.dumps({"source": "rclpy.serialization.serialize_message, ROS 2 Jazzy", "cases": cases}) + "\n")

    vendored = {p.name for p in (ROOT / "week05" / "interfaces").iterdir()} | {"ugs_interfaces", "tutorial_interfaces"}
    hashes = {}
    for path in sorted(glob.glob("/opt/ros/jazzy/share/*/*/*.json")) + sorted(glob.glob(str(WS / "install/*/share/*/*/*.json"))):
        if Path(path).parent.name in ("msg", "srv", "action") and Path(path).parts[-3] in vendored:
            for entry in json.loads(Path(path).read_text())["type_hashes"]:
                hashes[entry["type_name"]] = entry["hash_string"]
    (out / "type_hashes.json").write_text(
        json.dumps({"source": "the .json files rosidl generated, ROS 2 Jazzy", "hashes": hashes}, indent=0, sort_keys=True) + "\n"
    )

    rel = {
        "system_default": ReliabilityPolicy.SYSTEM_DEFAULT,
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
        "unknown": ReliabilityPolicy.UNKNOWN,
        "best_available": ReliabilityPolicy.BEST_AVAILABLE,
    }
    dur = {
        "system_default": DurabilityPolicy.SYSTEM_DEFAULT,
        "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
        "volatile": DurabilityPolicy.VOLATILE,
        "unknown": DurabilityPolicy.UNKNOWN,
        "best_available": DurabilityPolicy.BEST_AVAILABLE,
    }
    liv = {
        "system_default": LivelinessPolicy.SYSTEM_DEFAULT,
        "automatic": LivelinessPolicy.AUTOMATIC,
        "manual_by_topic": LivelinessPolicy.MANUAL_BY_TOPIC,
        "unknown": LivelinessPolicy.UNKNOWN,
        "best_available": LivelinessPolicy.BEST_AVAILABLE,
    }
    times = [0, 10**9, 2 * 10**9]
    profiles = list(itertools.product(rel, dur, liv, times, times))
    levels = {QoSCompatibility.OK: "ok", QoSCompatibility.WARNING: "warning", QoSCompatibility.ERROR: "error"}

    def ros_profile(p):
        q = QoSProfile(depth=10, reliability=rel[p[0]], durability=dur[p[1]], liveliness=liv[p[2]])
        q.deadline = Duration(nanoseconds=p[3])
        q.liveliness_lease_duration = Duration(nanoseconds=p[4])
        return q

    qos_cases = []
    for _ in range(1500):
        a, b = rnd.choice(profiles), rnd.choice(profiles)
        level, reason = qos_check_compatible(ros_profile(a), ros_profile(b))
        qos_cases.append([list(a), list(b), levels[level], reason])
    (out / "qos_cases.json").write_text(json.dumps({"source": "rclpy.qos.qos_check_compatible, ROS 2 Jazzy", "cases": qos_cases}) + "\n")


def _jsonable(value):
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def random_alltypes(rnd: random.Random) -> dict:
    def s(n=12):
        return "".join(rnd.choice("abcxyz Ωλ🚁0123") for _ in range(rnd.randint(0, n)))

    def point():
        return {"x": rnd.uniform(-1e3, 1e3), "y": rnd.uniform(-1e3, 1e3), "z": rnd.uniform(-1e3, 1e3)}

    return {
        "b": rnd.random() < 0.5,
        "o": rnd.randint(0, 255),
        "c": rnd.randint(0, 255),
        "i8": rnd.randint(-128, 127),
        "u8": rnd.randint(0, 255),
        "i16": rnd.randint(-(2**15), 2**15 - 1),
        "u16": rnd.randint(0, 2**16 - 1),
        "i32": rnd.randint(-(2**31), 2**31 - 1),
        "u32": rnd.randint(0, 2**32 - 1),
        "i64": rnd.randint(-(2**63), 2**63 - 1),
        "u64": rnd.randint(0, 2**64 - 1),
        "f32": float(__import__("struct").unpack("f", __import__("struct").pack("f", rnd.uniform(-1e6, 1e6)))[0]),
        "f64": rnd.uniform(-1e12, 1e12),
        "s": s(),
        "w": s(),
        "bounded_s": s(8)[:8].encode()[:8].decode("utf-8", "ignore"),
        "fixed_i32": [rnd.randint(-5, 5) for _ in range(3)],
        "fixed_f64": [rnd.random(), -rnd.random()],
        "fixed_u8": bytes(rnd.randint(0, 255) for _ in range(4)),
        "fixed_s": [s(), s()],
        "seq_i16": [rnd.randint(-300, 300) for _ in range(rnd.randint(0, 5))],
        "seq_u8": bytes(rnd.randint(0, 255) for _ in range(rnd.randint(0, 9))),
        "seq_s": [s() for _ in range(rnd.randint(0, 3))],
        "seq_w": [s() for _ in range(rnd.randint(0, 3))],
        "bounded_f32": [0.5 * rnd.randint(-8, 8) for _ in range(rnd.randint(0, 4))],
        "p": point(),
        "points": [point() for _ in range(rnd.randint(0, 3))],
        "fixed_points": [point(), point()],
        "times": [{"sec": rnd.randint(0, 2**31 - 1), "nanosec": rnd.randint(0, 10**9 - 1)} for _ in range(rnd.randint(0, 2))],
        "with_default": rnd.randint(-9, 9),
        "seq_default": [rnd.random() for _ in range(rnd.randint(0, 3))],
    }


def _to_ros(value: dict) -> dict:
    import copy

    out = copy.deepcopy(value)  # set_message_fields replaces nested dicts with messages in place
    out["o"] = bytes([value["o"]])  # rclpy wants a byte as bytes
    out["fixed_u8"] = list(value["fixed_u8"])
    out["seq_u8"] = list(value["seq_u8"])
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    out = Path(args[0] if args else ROOT / "docs" / "week05")
    only = set(args[1:])
    out.mkdir(parents=True, exist_ok=True)

    def step(name: str) -> bool:
        if only and name not in only:
            return False
        print(f"== {name}", flush=True)
        return True

    if step("transcripts"):
        (out / "transcripts.json").write_text(json.dumps(transcripts(), indent=1, ensure_ascii=False) + "\n")
    if step("captures"):
        capture(out / "captures")
    if step("interop"):
        (out / "interop.json").write_text(json.dumps(interop(), indent=1, ensure_ascii=False) + "\n")
    if step("mission"):
        (out / "mission.json").write_text(json.dumps(mission(out), separators=(",", ":")) + "\n")
    if step("lab"):
        (out / "lab.json").write_text(json.dumps(lab(out), indent=1) + "\n")
    if step("fixtures"):
        fixtures(ROOT / "tests" / "data" / "week05")
    if step("checks"):
        (out / "checks.json").write_text(json.dumps(checks(), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
