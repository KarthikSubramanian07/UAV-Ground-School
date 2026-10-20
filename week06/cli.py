"""Command line interface: ``python -m week06 <command>``.

PX4 SITL, Gazebo and ROS 2 in Docker (none of them run natively on macOS):
  docker build     Build ugs-ros:jazzy (week 5) then ugs-px4:jazzy (PX4 v1.17, Gazebo Harmonic, agent)
  docker shell     An interactive shell with ROS 2, px4_msgs and the agent sourced, the repo at /ugs
  docker test      colcon build and colcon test the workspace, ugs_px4 included
  docker run CMD   Run one command in the container

The assignment:
  fly              PX4 SITL + Gazebo + agent + takeoff action server + client: take off to 5 m,
                   hover, land, and record it (--altitude, --method, --climb-rate, --cancel-after)
  analyze FLIGHT   Rise time, overshoot, settling, hover error and drift of a recorded flight

DDS-XRCE, the PX4 to agent protocol, in pure Python:
  xrce decode PCAP          Dissect a capture of UDP port 8888, message by message
  xrce session PCAP         What PX4 asked the agent to create: every topic, type and QoS
  xrce echo TOPIC TYPE      Read a PX4 topic straight from the agent, like ros2 topic echo
  xrce fly                  Take off to 5 m with no ROS 2 at all: the same logic, over XRCE

  docs [--sitl]    Regenerate docs/week06 (--sitl reflies every flight in Docker first)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = "ugs-ros:jazzy"
IMAGE = "ugs-px4:jazzy"
FLIGHTS = ROOT / "docs" / "week06" / "flights"
SOURCE = "source /opt/ros/jazzy/setup.bash && source /opt/px4_ws/install/setup.bash"
BUILD = "colcon build --packages-up-to ugs_px4 --event-handlers console_cohesion- >/dev/null"


def _docker(args: list[str], interactive: bool = False, name: str | None = None) -> int:
    base = ["docker", "run", "--rm", "-v", f"{ROOT}:/ugs", "-w", "/ugs/ros2_ws", "--shm-size=512m"]
    if name:
        base += ["--name", name]
    if interactive:
        base.append("-it")
    return subprocess.call(base + [IMAGE] + args)


def _image_exists(image: str) -> bool:
    return subprocess.call(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def cmd_docker(args) -> int:
    if args.action == "build":
        if not _image_exists(BASE_IMAGE):
            rc = subprocess.call(["docker", "build", "-t", BASE_IMAGE, str(ROOT / "ros2_ws")])
            if rc:
                return rc
        return subprocess.call(["docker", "build", "--build-arg", f"JOBS={args.jobs}", "-t", IMAGE, str(ROOT / "ros2_ws" / "px4")])
    if args.action == "shell":
        return _docker(["bash"], interactive=True)
    if args.action == "test":
        script = f"{SOURCE} && colcon build && colcon test && colcon test-result --verbose"
        return _docker(["bash", "-c", script])
    command = " ".join(shlex.quote(c) for c in args.command)
    script = f"{SOURCE} && ({BUILD}) && source install/setup.bash && {command}"
    return _docker(["bash", "-c", script], interactive=sys.stdin.isatty())


def fly_script(
    name: str,
    altitude: float,
    method: str,
    climb_rate: float,
    cancel_after: float,
    land: bool,
    hover: float,
    capture: bool,
    truth: bool,
    xrce: bool = False,
    max_lead: float = 0.4,
) -> str:
    """The shell script `fly` runs in the container."""
    out = f"/ugs/docs/week06/flights/{name}"
    launch = (
        f"ros2 launch ugs_px4 sitl.launch.py altitude:={altitude} method:={method} "
        f"climb_rate:={climb_rate} cancel_after:={cancel_after} land:={'true' if land else 'false'} "
        f"hover:={hover} max_lead:={max_lead} record:={out}.jsonl"
    )
    lines = [
        "set -e",
        SOURCE,
        f"({BUILD}) && source install/setup.bash",
        f"mkdir -p {Path(out).parent}",
        "set +e",
    ]
    if capture:
        lines.append(f"tcpdump -i lo -U -s 0 -w {out}.pcap udp port 8888 >/dev/null 2>&1 & DUMP=$!")
    if truth:
        lines.append(f"python3 /ugs/week06/truth.py {out}.truth.jsonl & TRUTH=$!")
    if xrce:
        # PX4, Gazebo and the agent from the launch file; the flight from pure Python
        lines.append(f"ros2 launch ugs_px4 sitl.launch.py flight:=false > {out}.sitl.log 2>&1 & SITL=$!")
        lines.append("sleep 3")
        fly = f"cd /ugs && python3 -m week06 xrce fly --altitude {altitude} --record {out}.jsonl"
        lines.append(f"({fly}) 2>&1 | tee {out}.log; RC=${{PIPESTATUS[0]}}")
        # background jobs of a non-interactive bash ignore SIGINT, so stop the launch with SIGTERM
        lines.append("kill -TERM $SITL; wait $SITL")
    else:
        lines.append(f"{launch} 2>&1 | tee {out}.log; RC=${{PIPESTATUS[0]}}")
    if capture:
        lines.append("kill -TERM $DUMP 2>/dev/null; wait $DUMP 2>/dev/null")
    if truth:
        lines.append("kill -TERM $TRUTH 2>/dev/null; wait $TRUTH 2>/dev/null")
    lines.append("exit $RC")
    return "\n".join(lines)


def cmd_fly(args) -> int:
    name = args.name or ("xrce" if args.xrce else args.method) + f"_{args.altitude:g}m"
    script = fly_script(
        name,
        args.altitude,
        args.method,
        args.climb_rate,
        args.cancel_after,
        not args.no_land,
        args.hover,
        args.capture,
        args.truth,
        args.xrce,
        args.max_lead,
    )
    rc = _docker(["bash", "-c", script], name=f"ugs-px4-{name.replace('.', '_')}")
    path = FLIGHTS / f"{name}.jsonl"
    if path.is_file():
        from .flight import Flight, analyze

        try:
            print(json.dumps({k: v for k, v in analyze(Flight.load(path)).items() if k not in ("commands", "acks")}, indent=1))
        except ValueError as error:
            print(f"could not analyze {path}: {error}", file=sys.stderr)
    return rc


def cmd_analyze(args) -> int:
    from .flight import Flight, analyze

    result = analyze(Flight.load(args.flight))
    if not args.all:
        result = {k: v for k, v in result.items() if k not in ("commands", "acks")}
    print(json.dumps(result, indent=1))
    return 0


def cmd_xrce(args) -> int:
    if args.action in ("decode", "session"):
        from .dissect import dissect, session

        if args.action == "decode":
            for line in dissect(args.pcap, limit=args.limit, decode_samples=not args.no_samples):
                print(line)
        else:
            print(json.dumps(session(args.pcap), indent=1))
        return 0
    if args.action == "echo":
        import time

        from .xrce_client import XrceClient

        count = 0
        with XrceClient(args.host, args.port, args.domain) as client:

            def show(sample):
                nonlocal count
                count += 1
                print(json.dumps(sample), flush=True)

            client.subscribe(args.topic, args.type, show)
            end = time.monotonic() + args.seconds
            while time.monotonic() < end and (not args.count or count < args.count):
                time.sleep(0.05)
        return 0 if count else 1
    if args.action == "fly":
        from .xrce_fly import fly

        return fly(args.host, args.port, args.domain, args.altitude, land=not args.no_land, record=args.record)
    raise SystemExit(f"unknown xrce action {args.action}")


def cmd_docs(args) -> int:
    from .docs import regenerate

    return regenerate(sitl=args.sitl)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m week06", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("docker", help="build or use the PX4 SITL image")
    d.add_argument("action", choices=("build", "shell", "test", "run"))
    d.add_argument("command", nargs=argparse.REMAINDER)
    d.add_argument("--jobs", type=int, default=2, help="parallel compile jobs for the PX4 build (2)")
    d.set_defaults(func=cmd_docker)

    f = sub.add_parser("fly", help="fly the takeoff action against PX4 SITL in Gazebo")
    f.add_argument("--altitude", type=float, default=5.0)
    f.add_argument("--method", default="offboard", choices=("offboard", "takeoff_mode"))
    f.add_argument("--climb-rate", type=float, default=0.0)
    f.add_argument("--cancel-after", type=float, default=0.0)
    f.add_argument("--hover", type=float, default=5.0)
    f.add_argument("--no-land", action="store_true")
    f.add_argument("--capture", action="store_true", help="also capture the XRCE traffic to a pcap")
    f.add_argument("--truth", action="store_true", help="also record Gazebo's true pose")
    f.add_argument(
        "--max-lead", type=float, default=0.4, help="metres the climb reference may lead the vehicle; 0 runs it on a fixed clock"
    )
    f.add_argument("--xrce", action="store_true", help="fly with week06's pure Python XRCE client instead of ROS 2")
    f.add_argument("--name", help="file name under docs/week06/flights")
    f.set_defaults(func=cmd_fly)

    a = sub.add_parser("analyze", help="measure a recorded flight")
    a.add_argument("flight")
    a.add_argument("--all", action="store_true", help="include every command and ack")
    a.set_defaults(func=cmd_analyze)

    x = sub.add_parser("xrce", help="DDS-XRCE in pure Python")
    x.add_argument("action", choices=("decode", "session", "echo", "fly"))
    x.add_argument("pcap", nargs="?")
    x.add_argument("--topic", default="/fmu/out/vehicle_local_position_v1")
    x.add_argument("--type", default="px4_msgs/msg/VehicleLocalPosition")
    x.add_argument("--host", default="127.0.0.1")
    x.add_argument("--port", type=int, default=8888)
    x.add_argument("--domain", type=int, default=0)
    x.add_argument("--seconds", type=float, default=5.0)
    x.add_argument("--count", type=int, default=0)
    x.add_argument("--limit", type=int, default=0)
    x.add_argument("--no-samples", action="store_true")
    x.add_argument("--altitude", type=float, default=5.0)
    x.add_argument("--no-land", action="store_true")
    x.add_argument("--record", help="write the flight as JSON lines")
    x.set_defaults(func=cmd_xrce)

    o = sub.add_parser("docs", help="regenerate docs/week06")
    o.add_argument("--sitl", action="store_true")
    o.set_defaults(func=cmd_docs)

    args = parser.parse_args(argv)
    if args.command == "xrce" and args.action in ("decode", "session") and not args.pcap:
        parser.error(f"xrce {args.action} needs a PCAP")
    return args.func(args)
