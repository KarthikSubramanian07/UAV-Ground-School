"""Take off to 5 m with no ROS 2: the takeoff logic, straight over DDS-XRCE.

The ROS 2 action server and this function run the same state machine
(ros2_ws/src/ugs_px4/ugs_px4/takeoff.py); only the plumbing differs. Here the
"node" is week06's XrceClient, which asks the Micro XRCE-DDS Agent for the
same DDS readers and writers rclpy would create, so PX4 cannot tell them apart.

    python -m week06 xrce fly --altitude 5      # inside the container, beside PX4 and the agent
"""

from __future__ import annotations

import json
import math
import threading
import time

from ugs_px4.px4 import Snapshot, ack_name, command_name, nav_state_name  # noqa: E402
from ugs_px4.takeoff import Land, Takeoff  # noqa: E402

from . import _ugs_px4  # noqa: F401, E402  (puts ugs_px4 on the path)
from .xrce_client import XrceClient

RATE_HZ = 20.0


def _px4_time(t0: float) -> int:
    return int((time.monotonic() - t0) * 1e6)


def fly(host: str, port: int, domain: int, altitude: float, land: bool = True, hover: float = 3.0, record: str | None = None) -> int:
    lock = threading.Lock()
    snap = Snapshot()
    ground = {"z": None}
    log = open(record, "w") if record else None  # noqa: SIM115 - closed below
    t_start = time.monotonic()

    def write(kind, **fields):
        if log:
            log.write(json.dumps({"t": round(time.monotonic() - t_start, 4), "kind": kind, **fields}) + "\n")

    def say(text):
        print(f"[xrce fly] {text}", flush=True)
        write("log", node="takeoff_server", level=20, msg=text)

    def on_status(m):
        with lock:
            if snap.status_time is None or m["nav_state"] != snap.nav_state or m["arming_state"] != snap.arming_state:
                say(f"PX4 now {nav_state_name(m['nav_state'])}, {'armed' if m['arming_state'] == 2 else 'disarmed'}")
            snap.status_time = time.monotonic()
            snap.arming_state, snap.nav_state = m["arming_state"], m["nav_state"]
            snap.preflight_ok = m["pre_flight_checks_pass"]
        write(
            "status",
            px4_us=m["timestamp"],
            nav_state=m["nav_state"],
            arming_state=m["arming_state"],
            preflight_ok=m["pre_flight_checks_pass"],
            failsafe=m["failsafe"],
        )

    def on_position(m):
        with lock:
            snap.position_time = time.monotonic()
            snap.xy_valid, snap.z_valid = m["xy_valid"], m["z_valid"]
            snap.x, snap.y, snap.z, snap.vz = m["x"], m["y"], m["z"], m["vz"]
            snap.heading, snap.ref_alt = m["heading"], m["ref_alt"]
            if snap.landed and snap.z_valid and not snap.armed:
                ground["z"] = snap.z
        write(
            "position",
            px4_us=m["timestamp"],
            x=m["x"],
            y=m["y"],
            z=m["z"],
            vx=m["vx"],
            vy=m["vy"],
            vz=m["vz"],
            xy_valid=m["xy_valid"],
            z_valid=m["z_valid"],
            ref_alt=m["ref_alt"],
            heading=m["heading"],
        )

    def on_land(m):
        with lock:
            snap.landed = m["landed"]

    def on_ack(m):
        with lock:
            snap.acks.append((time.monotonic(), m["command"], m["result"]))
        say(f"<- PX4 ack {command_name(m['command'])}: {ack_name(m['result'])}")

    with XrceClient(host, port, domain) as client:
        say(f"logged in to the agent at {host}:{port}: {client.agent['status']}")
        client.subscribe("/fmu/out/vehicle_status_v1", "px4_msgs/msg/VehicleStatus", on_status)
        client.subscribe("/fmu/out/vehicle_local_position_v1", "px4_msgs/msg/VehicleLocalPosition", on_position)
        client.subscribe("/fmu/out/vehicle_land_detected", "px4_msgs/msg/VehicleLandDetected", on_land)
        client.subscribe("/fmu/out/vehicle_command_ack", "px4_msgs/msg/VehicleCommandAck", on_ack)
        client.publisher("/fmu/in/offboard_control_mode", "px4_msgs/msg/OffboardControlMode")
        client.publisher("/fmu/in/trajectory_setpoint", "px4_msgs/msg/TrajectorySetpoint")
        client.publisher("/fmu/in/vehicle_command", "px4_msgs/msg/VehicleCommand")
        say(f"created {client.next_number - 1} topics with their readers and writers")

        def send(command):
            p = [float(x) for x in command.params]
            client.write(
                "/fmu/in/vehicle_command",
                {
                    "timestamp": _px4_time(t_start),
                    "param1": p[0],
                    "param2": p[1],
                    "param3": p[2],
                    "param4": p[3],
                    "param5": p[4],
                    "param6": p[5],
                    "param7": p[6],
                    "command": command.command,
                    "target_system": 1,
                    "target_component": 1,
                    "source_system": 1,
                    "source_component": 1,
                    "confirmation": 0,
                    "from_external": True,
                },
            )
            write("command", command=command.command, params=[None if math.isnan(x) else x for x in p])
            say(f"-> PX4 {command}")

        def stream(sp):
            stamp = _px4_time(t_start)
            ff = not math.isnan(sp.vz)
            client.write(
                "/fmu/in/offboard_control_mode",
                {
                    "timestamp": stamp,
                    "position": True,
                    "velocity": ff,
                    "acceleration": False,
                    "attitude": False,
                    "body_rate": False,
                    "thrust_and_torque": False,
                    "direct_actuator": False,
                },
            )
            nan = math.nan
            client.write(
                "/fmu/in/trajectory_setpoint",
                {
                    "timestamp": stamp,
                    "position": [sp.x, sp.y, sp.z],
                    "velocity": [nan, nan, sp.vz],
                    "acceleration": [nan] * 3,
                    "jerk": [nan] * 3,
                    "yaw": sp.yaw,
                    "yawspeed": nan,
                },
            )
            write("setpoint", x=sp.x, y=sp.y, z=sp.z, vz=None if math.isnan(sp.vz) else sp.vz)

        def run(logic, until_hover=0.0):
            held = None
            last_phase = None
            while True:
                with lock:
                    out = logic.update(time.monotonic(), snap)
                    snap.acks = []
                for c in out.commands:
                    send(c)
                if out.setpoint is not None:
                    stream(out.setpoint)
                    held = out.setpoint
                if out.phase != last_phase:
                    say(f"[{out.phase}]")
                    last_phase = out.phase
                    write(
                        "feedback",
                        action="land" if isinstance(logic, Land) else "takeoff",
                        phase=out.phase,
                        altitude=None,
                        vertical_speed=None,
                        remaining=None,
                    )
                if out.done is not None:
                    return out, held
                time.sleep(1.0 / RATE_HZ)

        say(f"/takeoff to {altitude:.1f} m at 1.0 m/s (offboard)")
        out, held = run(Takeoff(altitude, time.monotonic(), ground["z"]))
        say(("succeeded: " if out.done else "aborted: ") + out.message)
        ok = bool(out.done)
        if ok and land:
            end = time.monotonic() + hover
            while time.monotonic() < end:  # keep the offboard heartbeat while hovering
                if held is not None:
                    stream(held)
                time.sleep(1.0 / RATE_HZ)
            say("/land")
            out, _ = run(Land(time.monotonic(), ground["z"]))
            say(("LAND succeeded: " if out.done else "LAND failed: ") + out.message)
            ok = bool(out.done)
        say(f"{client.sent} XRCE messages sent, {client.received} received")
    if log:
        log.close()
    return 0 if ok else 1
