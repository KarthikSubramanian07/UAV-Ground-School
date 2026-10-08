"""Regenerate docs/week05 from the recorded experiments.

    python -m week05 docs          # offline: rebuild the derived files below
    python -m week05 docs --ros    # first rerun the live experiments in Docker

The live data (transcripts, captures, interop, mission, lab, checks) comes from
:mod:`week05.experiments`. This module derives, with no ROS needed:

* ``wire.json``: a guided tour of real packets, every byte labelled, for the page;
* ``summary.json``: the headline numbers;
* ``RESULTS.md``: everything above as readable tables.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import pcap, rtps
from .dissect import dissect
from .idl import Registry
from .qos import matrix

ROOT = Path(__file__).resolve().parents[1]
LIVE = ("transcripts.json", "interop.json", "mission.json", "lab.json", "checks.json", "captures/pubsub.pcap", "captures/srvcli.pcap")

TOUR = (
    (
        "spdp",
        "pubsub",
        "A participant announces itself",
        "SPDP, sent to the multicast group 239.255.0.1 on port 7400 (domain 0). It says who this participant is (its GUID prefix and vendor), "
        "where to reach it (unicast locators), which builtin endpoints it has, and how long to consider it alive (the lease). ROS 2 adds the enclave in the user data.",
    ),
    (
        "sedp_pub",
        "pubsub",
        "The talker announces its publisher",
        "SEDP, sent reliably to each participant that SPDP found. The topic is rt/topic (ROS 2 prefixes rt/ to every topic), the type is std_msgs::msg::dds_::String_, "
        "the QoS is reliable and volatile, and Jazzy adds the REP 2011 type hash in the user data.",
    ),
    (
        "hello",
        "pubsub",
        "Hello World on the wire",
        "A DATA submessage carrying one sample. INFO_DST says which participant it is for, INFO_TS when it was written. The payload is CDR: 00 01 00 00 "
        "(little endian, plain CDR), a uint32 length that counts the terminating NUL, then the characters.",
    ),
    (
        "heartbeat",
        "pubsub",
        "The writer: 'I have samples first to last'",
        "Reliable delivery is a conversation. The writer's HEARTBEAT lists the sequence numbers it still holds; a reader that is missing any asks for them.",
    ),
    (
        "acknack",
        "pubsub",
        "The reader: 'I have everything below base'",
        "The ACKNACK answers with a sequence number set: everything below the base has arrived, and the bitmap lists what is missing. An empty set is a pure acknowledgement.",
    ),
    (
        "request",
        "srvcli",
        "A service request",
        "Services are two topics: rq/add_two_intsRequest and rr/add_two_intsReply. The request carries a RELATED_SAMPLE_IDENTITY naming the client's reply reader, "
        "twice: in the standard parameter (0x0083) and in Fast DDS's own (0x800f).",
    ),
    (
        "reply",
        "srvcli",
        "Its reply",
        "The reply quotes the same identity with the request's sequence number filled in, which is how the client matches it to the request it sent.",
    ),
)


def _pick(datagrams, kind):
    types = {}
    for dg in datagrams:
        msg = rtps.parse(dg.payload)
        if msg is None:
            continue
        for rec in rtps.interpret(msg):
            s = rec.submessage
            if (
                isinstance(s, rtps.Data)
                and s.payload is not None
                and s.writer in (rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_WRITER)
            ):
                ep = rtps.EndpointData.decode(s.payload, s.writer == rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER)
                types[rtps.Guid(rec.source_prefix, ep.guid.entity)] = (ep.topic, ep.writer)
            topic = types.get(rtps.Guid(rec.source_prefix, getattr(s, "writer", -1)), ("", False))[0]
            if kind == "spdp" and isinstance(s, rtps.Data) and s.writer == rtps.ENTITYID_SPDP_WRITER and s.payload:
                return dg
            sedp_pub = isinstance(s, rtps.Data) and s.writer == rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER and s.payload
            if kind == "sedp_pub" and sedp_pub and rtps.EndpointData.decode(s.payload, True).topic == "rt/topic":
                return dg
            if kind == "hello" and isinstance(s, rtps.Data) and topic == "rt/topic" and s.payload:
                return dg
            if kind == "heartbeat" and isinstance(s, rtps.Heartbeat) and topic == "rt/topic":
                return dg
            if (
                kind == "acknack"
                and isinstance(s, rtps.AckNack)
                and types.get(rtps.Guid(rec.dest_prefix or b"", s.writer), ("",))[0] == "rt/topic"
            ):
                return dg
            if kind == "request" and isinstance(s, rtps.Data) and topic == "rq/add_two_intsRequest" and s.payload:
                return dg
            if kind == "reply" and isinstance(s, rtps.Data) and topic == "rr/add_two_intsReply" and s.payload:
                return dg
    raise LookupError(f"no {kind} packet in the capture")


def wire(out: Path) -> list[dict]:
    registry = Registry()
    captures = {name: list(pcap.read(out / "captures" / f"{name}.pcap")) for name in ("pubsub", "srvcli")}
    tour = []
    for kind, capture, title, text in TOUR:
        dg = _pick(captures[capture], kind)
        types = {}
        for prior in captures[capture]:  # learn writer types from SEDP, as a sniffer would
            dissect(prior.payload, registry, types)
            if prior is dg:
                break
        d = dissect(dg.payload, registry, types)
        tour.append(
            {
                "id": kind,
                "title": title,
                "text": text,
                "from": f"{dg.src}:{dg.sport}",
                "to": f"{dg.dst}:{dg.dport}",
                "port": rtps.describe_port(dg.dport),
                "summary": [s for s in d.summary],
                "hex": dg.payload.hex(),
                "spans": [s.as_dict() for s in d.spans],
            }
        )
    return tour


def summary(out: Path) -> dict:
    checks = json.loads((out / "checks.json").read_text())
    interop = json.loads((out / "interop.json").read_text())
    lab = json.loads((out / "lab.json").read_text())
    mission = json.loads((out / "mission.json").read_text())
    captures = {name: list(pcap.read(out / "captures" / f"{name}.pcap")) for name in ("pubsub", "srvcli")}
    complete = next(e for e in mission["events"] if "MISSION COMPLETE" in e["text"])
    target = next(e for e in mission["events"] if "target confirmed" in e["text"])
    blocking = {r["executor"]: r for r in lab["blocking"]}
    return {
        "type_hashes": [checks["type_hashes_identical"], checks["type_hashes_total"]],
        "qos_pairs": [checks["qos_pairs_identical"], checks["qos_pairs_total"]],
        "cdr": [checks["cdr_alltypes_identical_except_padding"], checks["cdr_alltypes_trials"]],
        "cdr_rclpy_reads_ours": checks["cdr_alltypes_rclpy_deserializes_ours"],
        "cdr_rclpy_dirty_padding": checks["cdr_alltypes_rclpy_padding_not_zero"],
        "cdr_rclpy_overshoot": checks["cdr_rclpy_mean_overshoot_bytes"],
        "interop_directions": 4,
        "client_round_trip_ms": interop["python_client_to_ros_service"]["round_trip_ms"],
        "cyclone_vendors": interop["cyclone_talker_to_python_listener"]["vendors"],
        "packets": {name: len(grams) for name, grams in captures.items()},
        "mission_seconds": complete["t"],
        "mission_outcome": complete["text"],
        "target_confirmed": target["text"],
        "recorded_frames": mission["frames"],
        "recorded_megabytes": round(mission["stats"]["bytes_in"] / 1e6, 1),
        "recorded_topics": len(mission["graph"]["topics"]),
        "blocking_single_max_ms": blocking["SingleThreadedExecutor"]["max_gap_ms"],
        "blocking_groups_max_ms": blocking["MultiThreadedExecutor, separate groups"]["max_gap_ms"],
        "deadlock": [r["outcome"] for r in lab["deadlock"]],
    }


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(title for _, title in columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    body = ""
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key, "")
            cells.append(", ".join(v) if isinstance(v, list) else ("" if v is None else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def results_md(out: Path, s: dict, tour: list[dict]) -> str:
    lab = json.loads((out / "lab.json").read_text())
    interop = json.loads((out / "interop.json").read_text())
    transcripts = json.loads((out / "transcripts.json").read_text())
    parts = [
        "# Week 5 results\n",
        "Generated by `python -m week05 docs` from the live runs in this folder (rerun them with `python -m week05 docs --ros`). Every number below was measured in the ROS 2 Jazzy container.\n",
    ]
    parts.append("## The tutorials\n")
    for run in transcripts:
        parts.append(f"### {run['title']}\n")
        for term in run["terminals"]:
            lines = [line for line in term["output"] if line.strip()][:6]
            parts.append("```console\n$ " + term["command"] + "\n" + "\n".join(lines) + "\n```\n")
    parts.append("## Talking to ROS 2 without ROS 2\n")
    parts.append(
        f"The pure Python participant (`week05/participant.py`) against the tutorial nodes. Fast DDS in all {s['interop_directions']} directions, and Cyclone DDS:\n\n"
        f"* ROS talker to Python listener: received {', '.join(repr(x) for x in interop['ros_talker_to_python_listener']['received'][:3])} ...\n"
        f"* Python talker to ROS listener: `{interop['python_talker_to_ros_listener']['listener_output'][0].split(']: ', 1)[-1]}`\n"
        f"* Python client to ROS service: {interop['python_client_to_ros_service']['request']} gave {interop['python_client_to_ros_service']['reply']} in {s['client_round_trip_ms']} ms\n"
        f"* ROS clients (Python and C++) to the Python service: `{interop['ros_clients_to_python_service']['client_output']['cpp_srvcli'][0].split(']: ', 1)[-1]}`\n"
        f"* Cyclone DDS talker to Python listener: received {len(interop['cyclone_talker_to_python_listener']['received'])} messages; vendors seen: {', '.join(s['cyclone_vendors'])}\n"
    )
    parts.append("## Checked against ROS 2 itself\n")
    parts.append(
        f"| Check | Result |\n| --- | --- |\n"
        f"| REP 2011 type hashes of every installed interface | {s['type_hashes'][0]} of {s['type_hashes'][1]} identical |\n"
        f"| QoS compatibility against `rclpy.qos.qos_check_compatible` | {s['qos_pairs'][0]:,} of {s['qos_pairs'][1]:,} pairs identical |\n"
        f"| CDR against `rclpy.serialization`, random `AllTypes` messages | {s['cdr'][0]} of {s['cdr'][1]} identical except alignment padding |\n"
        f"| rclpy deserializes this repo's CDR | {s['cdr_rclpy_reads_ours']} of {s['cdr'][1]} |\n\n"
        f"Two findings: Fast CDR skips alignment padding without writing it, so rclpy's bytes carry leftover memory there ({s['cdr_rclpy_dirty_padding']} of {s['cdr'][1]} messages had non-zero padding); and `serialize_message` returns rmw_fastrtps's size estimate as the length, which ran {s['cdr_rclpy_overshoot']} bytes past the real CDR on average.\n"
    )
    parts.append("## QoS, measured\n")
    parts.append(
        _table(
            lab["reliability"],
            [
                ("publisher", "Publisher"),
                ("subscription", "Subscription"),
                ("matched", "Matched"),
                ("received", "Received of 20"),
                ("subscription_event", "Incompatible QoS event"),
                ("rclpy_check", "rclpy says"),
            ],
        )
    )
    parts.append(
        "\nA message published once, then late subscriptions join:\n\n"
        + _table(lab["durability"], [("publisher", "Publisher"), ("late_subscription", "Late subscription"), ("received", "Received")])
    )
    parts.append(
        "\n300 messages at 1 kHz into a subscription that takes 10 ms per message:\n\n"
        + _table(
            lab["depth"],
            [("history", "History"), ("reliability", "Reliability"), ("received", "Received of 300"), ("in_order", "In order")],
        )
    )
    parts.append(
        "\n## Callbacks, measured\n\nA 50 ms timer next to a subscription whose callback takes 300 ms:\n\n"
        + _table(
            lab["blocking"],
            [("executor", "Executor"), ("ticks", "Ticks in 3 s"), ("mean_gap_ms", "Mean gap (ms)"), ("max_gap_ms", "Worst gap (ms)")],
        )
    )
    parts.append(
        "\nCalling a service from inside a callback:\n\n"
        + _table(lab["deadlock"], [("setup", "Setup"), ("outcome", "Outcome"), ("ms", "Round trip (ms)")])
    )
    parts.append(
        f"\n## The mission\n\n`ros2 launch ugs_drone drone.launch.py`: {s['target_confirmed']}; {s['mission_outcome']} after {s['mission_seconds']:.1f} s. "
        f"The pure Python recorder saw {s['recorded_topics']} topics and services, decoded {s['recorded_frames']} camera frames ({s['recorded_megabytes']} MB, reassembled from DATA_FRAG), odometry, detections and the action's feedback and status.\n"
    )
    parts.append("\n## A tour of the packets\n")
    for t in tour:
        parts.append(
            f"### {t['title']}\n\n{t['text']}\n\n`{t['from']}` to `{t['to']}` ({t['port']}): {'; '.join(t['summary']) or 'no data submessages'}\n"
        )
    parts.append(
        "\n## QoS presets against each other\n\n"
        + _table(
            matrix(), [("publisher", "Publisher"), ("subscription", "Subscription"), ("compatibility", "Result"), ("reason", "Reason")]
        )
    )
    return "\n".join(parts).replace(chr(0x2013), "-").replace(chr(0x2014), "-")


def generate(out: Path, ros: bool = False) -> None:
    out = Path(out)
    if ros:
        script = (
            "source /opt/ros/jazzy/setup.bash && cd /ugs/ros2_ws && colcon build && source install/setup.bash && "
            f"cd /ugs && python3 -m week05.experiments /ugs/{out.resolve().relative_to(ROOT)}"
        )
        subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/ugs", "ugs-ros:jazzy", "bash", "-c", script], check=True)
    missing = [name for name in LIVE if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"missing live results {missing}: run python -m week05 docs --ros")
    tour = wire(out)
    s = summary(out)
    (out / "wire.json").write_text(json.dumps(tour, indent=0) + "\n")
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (out / "RESULTS.md").write_text(results_md(out, s, tour))
    print(f"wrote {out}/wire.json, summary.json and RESULTS.md")
