"""Record a live ROS 2 mission with the pure Python participant (no ROS needed).

    python -m week05.record --seconds 60 --out mission.json

Subscribes to the drone's topics, including the hidden ones an action is made
of (``/fly_to/_action/feedback`` and ``/fly_to/_action/status``), decodes
every sample from CDR and writes a compact timeline for the showcase page.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time

from .participant import Participant, qos

STATUS = {0: "unknown", 1: "accepted", 2: "executing", 3: "canceling", 4: "succeeded", 5: "canceled", 6: "aborted"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    t0 = time.time()
    data = {"odometry": [], "detections": [], "feedback": [], "status": [], "battery": [], "frames": 0, "frame_bytes": 0}
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    last_odom = [0.0]

    def odometry(s):
        now = time.time() - t0
        if s.value is None or now - last_odom[0] < 0.1:  # keep 10 Hz of the 50 Hz stream
            return
        last_odom[0] = now
        p = s.value["pose"]["pose"]["position"]
        v = s.value["twist"]["twist"]["linear"]
        data["odometry"].append(
            [
                round(now, 3),
                round(p["x"], 3),
                round(p["y"], 3),
                round(p["z"], 3),
                round((v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2) ** 0.5, 2),
            ]
        )

    def detections(s):
        if s.value and s.value["detections"]:
            d = s.value["detections"][0]
            c = d["bbox"]["center"]["position"]
            data["detections"].append(
                [round(time.time() - t0, 3), round(c["x"], 1), round(c["y"], 1), round(d["results"][0]["hypothesis"]["score"], 3)]
            )

    def feedback(s):
        if s.value:
            data["feedback"].append(
                [round(time.time() - t0, 3), s.value["goal_id"]["uuid"].hex()[:8], round(s.value["feedback"]["distance_remaining"], 2)]
            )

    def status(s):
        if s.value:
            for g in s.value["status_list"]:
                entry = [round(time.time() - t0, 3), g["goal_info"]["goal_id"]["uuid"].hex()[:8], STATUS.get(g["status"], str(g["status"]))]
                if not any(e[1:] == entry[1:] for e in data["status"]):
                    data["status"].append(entry)

    def battery(s):
        if s.value:
            data["battery"].append([round(time.time() - t0, 3), round(s.value["percentage"], 4)])

    def frame(s):
        data["frames"] += 1
        data["frame_bytes"] = len(s.payload)

    sensor = qos("best_effort", depth=5)
    with Participant(name="week05_recorder") as p:
        p.create_subscription("/odometry", "nav_msgs/msg/Odometry", odometry, sensor)
        p.create_subscription("/detections", "vision_msgs/msg/Detection2DArray", detections)
        p.create_subscription("/battery", "sensor_msgs/msg/BatteryState", battery)
        p.create_subscription("/camera/image", "sensor_msgs/msg/Image", frame, sensor)
        p.create_subscription("/fly_to/_action/feedback", "ugs_interfaces/action/FlyToWaypoint_FeedbackMessage", feedback)
        p.create_subscription(
            "/fly_to/_action/status", "action_msgs/msg/GoalStatusArray", status, qos(durability="transient_local", depth=1)
        )
        deadline = time.monotonic() + args.seconds
        while not stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
        data["graph"] = p.graph()
        data["stats"] = dict(p.stats)
    data["t0_wall"] = t0
    with open(args.out, "w") as f:
        json.dump(data, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
