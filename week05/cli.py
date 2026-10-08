"""Command line interface: ``python -m week05 <command>``.

ROS 2 in Docker (ROS 2 does not run natively on macOS or Windows):
  docker build     Build the ugs-ros:jazzy image from ros2_ws/Dockerfile
  docker shell     An interactive shell with ROS 2 sourced and the repo at /ugs
  docker test      colcon build and colcon test the whole workspace
  docker run CMD   Run one command in the container, e.g. "ros2 launch ugs_drone drone.launch.py"

ROS 2 without ROS 2 (pure Python, talks to real nodes on the same network):
  graph            Discover participants, topics and services, like ros2 topic list -t
  echo TOPIC TYPE  Print messages, like ros2 topic echo
  pub TOPIC TYPE VALUE    Publish a JSON value, like ros2 topic pub
  call SERVICE TYPE VALUE Call a service, like ros2 service call
  serve-add-two-ints      Answer /add_two_ints, the tutorial's service

The wire:
  decode PCAP      Dissect an RTPS capture, packet by packet
  cdr TYPE VALUE   Serialize a JSON value as CDR and show the bytes
  hash TYPE        The REP 2011 type hash (RIHS01) of an interface
  show TYPE        An interface definition with nested types expanded
  qos PUB SUB      Will a publisher and a subscription with these presets talk?

  docs [--ros]     Regenerate docs/week05 (--ros reruns the live experiments in Docker)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ugs-ros:jazzy"


def _docker(args: list[str], interactive: bool = False) -> int:
    base = ["docker", "run", "--rm", "-v", f"{ROOT}:/ugs", "-w", "/ugs/ros2_ws"]
    if interactive:
        base.append("-it")
    return subprocess.call(base + [IMAGE] + args)


def cmd_docker(args) -> int:
    if args.action == "build":
        return subprocess.call(["docker", "build", "-t", IMAGE, str(ROOT / "ros2_ws")])
    if args.action == "shell":
        return _docker(["bash"], interactive=True)
    if args.action == "test":
        script = "source /opt/ros/jazzy/setup.bash && colcon build && colcon test && colcon test-result --verbose"
        return _docker(["bash", "-c", script])
    command = " ".join(shlex.quote(c) for c in args.command)
    script = "source /opt/ros/jazzy/setup.bash && source install/setup.bash && " + command
    return _docker(["bash", "-c", script], interactive=sys.stdin.isatty())


def _value(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise SystemExit(f'VALUE must be JSON, like \'{{"data": "hi"}}\': {error}') from None
    if not isinstance(value, dict):
        raise SystemExit("VALUE must be a JSON object")
    return value


def _jsonable(value):
    if isinstance(value, bytes):
        return value.hex() if len(value) <= 64 else f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def cmd_graph(args) -> int:
    from .participant import Participant

    with Participant(domain=args.domain, name="week05_graph") as p:
        p.spin(args.seconds)
        g = p.graph()
    if args.json:
        print(json.dumps(g, indent=1))
        return 0
    print(f"{len(g['participants'])} participants")
    for part in g["participants"]:
        print(f"  {part['guid_prefix']}  {part['vendor']:<20} {part['address']}")
    for t in g["topics"]:
        print(f"{t['kind']:<16} {t['name']:<44} {t['type']:<44} pub {t['publishers']} sub {t['subscribers']}  {', '.join(t['qos'])}")
    return 0


def cmd_echo(args) -> int:
    from .participant import Participant, qos

    profile = qos("best_effort" if args.best_effort else "reliable", depth=10)
    count = 0
    with Participant(domain=args.domain, name="week05_echo") as p:

        def show(sample):
            nonlocal count
            count += 1
            print(json.dumps(_jsonable(sample.value)), flush=True)

        p.create_subscription(args.topic, args.type, show, profile)
        deadline = time.monotonic() + args.seconds if args.seconds else None
        try:
            while deadline is None or time.monotonic() < deadline:
                if args.count and count >= args.count:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_pub(args) -> int:
    from .participant import Participant

    value = _value(args.value)
    with Participant(domain=args.domain, name="week05_pub") as p:
        pub = p.create_publisher(args.topic, args.type)
        if not p.wait_until(lambda: pub.matched > 0, args.wait):
            print(f"no subscription matched {args.topic} within {args.wait} s; publishing anyway", file=sys.stderr)
        for i in range(args.times):
            pub.publish(value)
            print(f"publishing #{i + 1}: {json.dumps(value)}", flush=True)
            time.sleep(1.0 / args.rate)
        p.spin(0.5)  # let reliable retransmissions finish
    return 0


def cmd_call(args) -> int:
    from .participant import Participant

    with Participant(domain=args.domain, name="week05_call") as p:
        client = p.create_client(args.service, args.type)
        if not client.wait_for_service(args.wait):
            print(f"{args.service} is not available", file=sys.stderr)
            return 1
        start = time.perf_counter()
        reply = client.call(_value(args.value), timeout=args.wait)
        print(json.dumps(_jsonable(reply)))
        print(f"round trip {1000 * (time.perf_counter() - start):.2f} ms", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    from .participant import Participant

    with Participant(domain=args.domain, name="week05_service") as p:

        def add(request):
            print(f"Incoming request\na: {request['a']} b: {request['b']}", flush=True)
            return {"sum": request["a"] + request["b"]}

        p.create_service("/add_two_ints", "example_interfaces/srv/AddTwoInts", add)
        with contextlib.suppress(KeyboardInterrupt):
            p.spin(args.seconds or 10**9)
    return 0


def cmd_decode(args) -> int:
    from .dissect import dissect_pcap

    for line in dissect_pcap(args.pcap, verbose=args.verbose, limit=args.limit):
        print(line)
    return 0


def cmd_cdr(args) -> int:
    from . import cdr

    payload = cdr.serialize(args.type, _value(args.value))
    print(f"{len(payload)} bytes")
    for i in range(0, len(payload), 16):
        chunk = payload[i : i + 16]
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{i:04x}  {chunk.hex(' '):<47}  {text}")
    print(json.dumps(_jsonable(cdr.deserialize(args.type, payload))))
    return 0


def cmd_hash(args) -> int:
    from .idl import Registry

    print(Registry().type_hash(args.type))
    return 0


def cmd_show(args) -> int:
    from .idl import Registry

    reg = Registry()
    pkg, kind, name = args.type.split("/")
    if kind == "srv":
        srv = reg.service(args.type)
        print(reg.show(srv.request.name) + "\n---\n" + reg.show(srv.response.name))
    elif kind == "action":
        act = reg.action(args.type)
        print("\n---\n".join(reg.show(m.name) for m in (act.goal, act.result, act.feedback)))
    else:
        print(reg.show(args.type))
    return 0


def cmd_qos(args) -> int:
    from .qos import PRESETS, check_compatible

    for name in (args.pub, args.sub):
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    level, reason = check_compatible(PRESETS[args.pub], PRESETS[args.sub])
    print(f"publisher    {args.pub}: {PRESETS[args.pub].describe()}")
    print(f"subscription {args.sub}: {PRESETS[args.sub].describe()}")
    print(reason.replace(";", ";\n").strip() if reason else "OK: compatible")
    return 1 if level == "error" else 0


def cmd_docs(args) -> int:
    from .docs import generate

    generate(Path(args.out), ros=args.ros)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m week05", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("docker", help="build, enter or test the ROS 2 workspace in Docker")
    p.add_argument("action", choices=["build", "shell", "test", "run"])
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_docker)

    def live(p):
        p.add_argument("--domain", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", 0)))
        return p

    p = live(sub.add_parser("graph", help="discover the ROS 2 graph"))
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_graph)

    p = live(sub.add_parser("echo", help="print a topic"))
    p.add_argument("topic")
    p.add_argument("type")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--best-effort", action="store_true", help="subscribe best effort (sensor data)")
    p.set_defaults(func=cmd_echo)

    p = live(sub.add_parser("pub", help="publish to a topic"))
    p.add_argument("topic")
    p.add_argument("type")
    p.add_argument("value")
    p.add_argument("--times", type=int, default=5)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--wait", type=float, default=5.0)
    p.set_defaults(func=cmd_pub)

    p = live(sub.add_parser("call", help="call a service"))
    p.add_argument("service")
    p.add_argument("type")
    p.add_argument("value")
    p.add_argument("--wait", type=float, default=10.0)
    p.set_defaults(func=cmd_call)

    p = live(sub.add_parser("serve-add-two-ints", help="serve /add_two_ints"))
    p.add_argument("--seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("decode", help="dissect an RTPS pcap")
    p.add_argument("pcap")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_decode)

    for name, func, helptext in (("cdr", cmd_cdr, "serialize a value as CDR"),):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("type")
        p.add_argument("value")
        p.set_defaults(func=func)
    for name, func, helptext in (("hash", cmd_hash, "type hash"), ("show", cmd_show, "show an interface")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("type")
        p.set_defaults(func=func)

    p = sub.add_parser("qos", help="check QoS compatibility of two presets")
    p.add_argument("pub")
    p.add_argument("sub")
    p.set_defaults(func=cmd_qos)

    p = sub.add_parser("docs", help="regenerate docs/week05")
    p.add_argument("--out", default=str(ROOT / "docs" / "week05"))
    p.add_argument("--ros", action="store_true", help="rerun the live ROS 2 experiments in Docker first")
    p.set_defaults(func=cmd_docs)

    args = parser.parse_args(argv)
    return args.func(args)
