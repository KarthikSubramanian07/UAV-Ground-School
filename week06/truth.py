"""Record Gazebo's true pose of the drone, to measure PX4's estimate against.

Runs in the PX4 container next to the flight (``python -m week06 fly --truth``):
subscribes to Gazebo's ``/world/<world>/dynamic_pose/info`` through gz-transport
and writes the model's position as JSON lines, stamped with CLOCK_MONOTONIC so
week06.flight can line it up with the flight recorder's samples.

Gazebo's world frame is ENU (z up); the drone's z is its base link's height.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

MODEL = "x500_0"


# PX4 starts Gazebo with GZ_IP=127.0.0.1; gz-transport only discovers peers on the same interface
os.environ.setdefault("GZ_IP", "127.0.0.1")


def record_with_bindings(path: str, world: str) -> bool:
    try:
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
    except ImportError:
        return False
    out = open(path, "w")  # noqa: SIM115 - closed at exit
    last = [0.0]

    def on_poses(msg: Pose_V) -> None:
        now = time.monotonic()
        if now - last[0] < 0.02:  # 50 Hz is plenty
            return
        for pose in msg.pose:
            if pose.name == MODEL:
                p = pose.position
                out.write(json.dumps({"monotonic": now, "kind": "truth", "x": p.x, "y": p.y, "z": p.z}) + "\n")
                last[0] = now
                break

    node = Node()
    node.subscribe(Pose_V, f"/world/{world}/dynamic_pose/info", on_poses)
    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    while not stop:
        time.sleep(0.1)
        out.flush()
    out.close()
    return True


def record_with_cli(path: str, world: str) -> None:
    """Fallback: parse `gz topic -e --json-output`."""
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", f"/world/{world}/dynamic_pose/info", "--json-output"], stdout=subprocess.PIPE, text=True
    )
    stop = []
    signal.signal(signal.SIGINT, lambda *a: (stop.append(1), proc.terminate()))
    signal.signal(signal.SIGTERM, lambda *a: (stop.append(1), proc.terminate()))
    with open(path, "w") as out:
        for line in proc.stdout:
            if stop:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            for pose in msg.get("pose", []):
                if pose.get("name") == MODEL:
                    p = pose.get("position", {})
                    out.write(
                        json.dumps(
                            {
                                "monotonic": time.monotonic(),
                                "kind": "truth",
                                "x": p.get("x", 0.0),
                                "y": p.get("y", 0.0),
                                "z": p.get("z", 0.0),
                            }
                        )
                        + "\n"
                    )
                    out.flush()


def main() -> int:
    path = sys.argv[1]
    world = sys.argv[2] if len(sys.argv) > 2 else "default"
    if not record_with_bindings(path, world):
        record_with_cli(path, world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
