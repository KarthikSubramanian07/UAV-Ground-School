"""Regenerate docs/week06 from the recorded flights: ``python -m week06 docs [--sitl]``.

With ``--sitl`` every flight in FLIGHTS is flown again in Docker first (PX4
SITL in Gazebo, about two minutes each). Then, from docs/week06/flights:

* summary.json   every flight's measurements (week06.flight.analyze), the XRCE
                 session PX4 set up, and the versions flown
* replay.json    the flights downsampled to 10 Hz for the site's replay
* xrce.json      the session's topics and a tour of real packets, byte spans
                 labelled for the site's packet inspector
* RESULTS.md     all of it as tables

The raw capture of the 5 m flight is trimmed to the session's first seconds
(log in, every CREATE and STATUS, then a little telemetry) as flights/*.xrce.pcap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from week05.pcap import read, write

from . import dissect, xrce
from .flight import Flight, analyze, timeline

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "week06"
FLIGHTS_DIR = DOCS / "flights"

# name -> (title, python -m week06 fly arguments)
FLIGHTS = {
    "offboard_5m": ("Offboard, 5 m", ["--altitude", "5", "--capture", "--truth"]),
    "offboard_5m_fixed_clock": ("Offboard, 5 m, reference on a fixed clock", ["--altitude", "5", "--max-lead", "0"]),
    "takeoff_mode_5m": ("Takeoff mode, 5 m", ["--altitude", "5", "--method", "takeoff_mode", "--truth"]),
    "offboard_10m_2mps": ("Offboard, 10 m at 2 m/s", ["--altitude", "10", "--climb-rate", "2"]),
    "offboard_cancel": ("Canceled 4 s after lift off", ["--altitude", "10", "--cancel-after", "4"]),
    "xrce_5m": ("Pure Python over XRCE, 5 m", ["--altitude", "5", "--xrce"]),
}
VERSIONS = {"px4": "v1.17.0", "px4_msgs": "v1.17.0", "agent": "v2.4.3", "gazebo": "Harmonic", "ros2": "Jazzy", "vehicle": "x500 (gz_x500)"}
KEEP_SECONDS = 2.0  # of telemetry after the last CREATE, in the trimmed capture


def fly_all() -> int:
    for name, (_, args) in FLIGHTS.items():
        print(f"== {name}", flush=True)
        rc = subprocess.call([sys.executable, "-m", "week06", "fly", "--name", name, *args], cwd=ROOT)
        if rc:
            print(f"{name} exited {rc}", file=sys.stderr)
    return 0


def trim_capture(src: Path, dst: Path) -> int:
    """Keep the log in, every CREATE and STATUS, and KEEP_SECONDS of traffic after them."""
    grams = [d for d in read(src) if dissect.PORT in (d.sport, d.dport)]
    last_setup = 0.0
    for d in grams:
        try:
            m = xrce.parse(d.payload, decode=False)
        except xrce.XrceError:
            continue
        if any(s.name in ("CREATE", "STATUS", "CREATE_CLIENT", "STATUS_AGENT", "READ_DATA") for s in m.submessages):
            last_setup = d.time
    kept = [d for d in grams if d.time <= last_setup + KEEP_SECONDS]
    write(dst, kept)
    return len(kept)


PACKET_TOUR = [
    # (id, title, text, predicate on (direction, names)); the smallest matching datagram is shown
    (
        "login",
        "PX4 logs in",
        "CREATE_CLIENT: the XRCE cookie and version, eProsima's vendor id, PX4's client key (1), the session id and the MTU. Sent on stream 0, before any session exists; PX4 found the agent with GET_INFO first.",
        lambda d, n: d == "px4>agent" and "CREATE_CLIENT" in n,
    ),
    (
        "welcome",
        "The agent answers",
        "STATUS_AGENT: OK, and the agent's own XRCE version and vendor. From here on every message carries the session id, 0x81.",
        lambda d, n: d == "agent>px4" and "STATUS_AGENT" in n,
    ),
    (
        "participant",
        "A DDS participant, please",
        "CREATE of a participant in binary representation, on reliable stream 128. The agent makes a real DDS DomainParticipant on domain 0 for PX4.",
        lambda d, n: d == "px4>agent" and "CREATE:PARTICIPANT" in n,
    ),
    (
        "topic",
        "vehicle_status_v1: topic, publisher, writer",
        "One message, three CREATEs. The topic carries ROS 2's name mangling, done by PX4: rt/fmu/out/vehicle_status_v1 and px4_msgs::msg::dds_::VehicleStatus_, which is why ros2 topic list shows it, _v1 and all. The writer's QoS is two bytes of flags: best effort, transient local, so a default reliable ROS 2 subscription never matches it.",
        lambda d, n: d == "px4>agent" and "CREATE:TOPIC" in n and "rt/fmu/out/vehicle_status_v1" in n,
    ),
    (
        "status",
        "Created",
        "STATUS for each request: OK. The agent answers every CREATE on the reliable stream, so PX4 knows when its topics exist.",
        lambda d, n: d == "agent>px4" and "STATUS" in n,
    ),
    (
        "read",
        "Send me what you get",
        "READ_DATA turns a data reader on: from then on the agent streams every sample DDS delivers to it to PX4, on best effort stream 1. PX4 sends one for each /fmu/in reader, batched with the next topic's CREATEs.",
        lambda d, n: d == "px4>agent" and "READ_DATA" in n and any(x.startswith("datareader") for x in n),
    ),
    (
        "write",
        "Telemetry out",
        "WRITE_DATA: one sample of /fmu/out/vehicle_status_v1, plain CDR with no encapsulation header. The agent adds the header and publishes it to DDS, where the takeoff server reads it.",
        lambda d, n: d == "px4>agent" and "WRITE_DATA" in n and "rt/fmu/out/vehicle_status_v1" in n,
    ),
    (
        "setpoint",
        "A setpoint in",
        "DATA: a TrajectorySetpoint the takeoff server published in ROS 2, delivered to PX4's reader. NaN velocities and accelerations mean position control only.",
        lambda d, n: d == "agent>px4" and "DATA" in n and "rt/fmu/in/trajectory_setpoint" in n,
    ),
    (
        "heartbeat",
        "Reliability, XRCE style",
        "HEARTBEAT on reliable stream 128: the sequence numbers not yet acknowledged. The other side answers with an ACKNACK bitmap of what is missing.",
        lambda d, n: "HEARTBEAT" in n,
    ),
]


def _names(raw: bytes, s: dissect.Session) -> list[str]:
    try:
        m = xrce.parse(raw)
    except xrce.XrceError:
        return []
    names = []
    for sub in m.submessages:
        names.append(sub.name)
        if sub.name == "CREATE":
            rep = sub.body["representation"]
            names.append(f"CREATE:{rep['kind']}")
            names.append(rep.get("name", ""))
        if sub.name == "READ_DATA":
            names.append(sub.body["object"])
        if sub.name in ("WRITE_DATA", "DATA"):
            topic = s.topic_of(sub.body["object_id"])
            names.append(topic["name"] if topic else "")
    return names


def packet_tour(pcap: Path) -> list[dict]:
    s, _ = dissect.build_session(pcap)
    registry = dissect._registry()
    grams = sorted((d for d in read(pcap) if dissect.PORT in (d.sport, d.dport)), key=lambda d: (len(d.payload), d.time))
    tour = []
    for pid, title, text, want in PACKET_TOUR:
        for d in grams:
            direction = "px4>agent" if d.dport == dissect.PORT else "agent>px4"
            names = _names(d.payload, s)
            if names and want(direction, names) and len(d.payload) <= 600:
                m = xrce.parse(d.payload)
                summary = [dissect.describe(dissect.Event(d.time, direction, m, sub, d.payload), s, registry) for sub in m.submessages]
                tour.append(
                    {
                        "id": pid,
                        "title": title,
                        "text": text,
                        "from": "PX4 uxrce_dds_client" if direction == "px4>agent" else "MicroXRCEAgent",
                        "to": "MicroXRCEAgent" if direction == "px4>agent" else "PX4 uxrce_dds_client",
                        "port": f"UDP {d.dport}",
                        "hex": d.payload.hex(),
                        "spans": dissect.spans(d.payload),
                        "summary": summary,
                    }
                )
                break
    return tour


def _fmt(v, digits=2, unit=""):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}{unit}"
    return f"{v}{unit}"


def results_md(summary: dict) -> str:
    out = [
        "# Week 6 results",
        "",
        "Regenerated by `python -m week06 docs` from the flights in [`flights/`](flights). Every flight is PX4 "
        f"{VERSIONS['px4']} SITL flying the {VERSIONS['vehicle']} in Gazebo {VERSIONS['gazebo']}, through the Micro XRCE-DDS Agent "
        f"{VERSIONS['agent']}, on an 8 GB Apple M1 in Docker (arm64, headless). Altitudes are metres above the takeoff point; "
        "the settling band is 0.25 m.",
        "",
        "## The takeoffs",
        "",
        "| Flight | Goal | Result | Peak | Rise 10 to 90 % | Max climb | Overshoot | Settled after | Hover error (RMS) | Tracking error (RMS) | Drift | PX4 estimate vs Gazebo (RMS) | True hover altitude | Landed |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, r in summary["flights"].items():
        title = summary["titles"].get(name, name)
        out.append(
            f"| {title} | {_fmt(r['target'], 1, ' m')} | {r['result'] or 'n/a'} | {_fmt(r.get('peak_altitude'), 2, ' m')} | "
            f"{_fmt(r.get('rise_time'), 2, ' s')} | {_fmt(r.get('max_climb_rate'), 2, ' m/s')} | {_fmt(r.get('overshoot'), 2, ' m')} | "
            f"{_fmt(r.get('settling_time'), 1, ' s')} | {_fmt(r.get('hover_rms_error'), 3, ' m')} | {_fmt(r.get('tracking_rms'), 3, ' m')} | "
            f"{_fmt(r.get('max_drift'), 2, ' m')} | {_fmt(r.get('estimate_vs_truth_rms'), 3, ' m')} | {_fmt(r.get('hover_truth_mean'), 2, ' m')} | {_fmt(r.get('landed'))} |"
        )
    out += [
        "",
        "## Timing",
        "",
        "| Flight | PX4 first heard | Preflight checks pass | Offboard | Armed | Lift off | Result | Land took |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in summary["flights"].items():
        out.append(
            f"| {summary['titles'].get(name, name)} | {_fmt(r.get('px4_first_heard'), 1, ' s')} | {_fmt(r.get('preflight_ok_at'), 1, ' s')} | "
            f"{_fmt(r.get('offboard_at'), 1, ' s')} | {_fmt(r.get('arm_time'), 1, ' s')} | {_fmt(r.get('liftoff'), 1, ' s')} | "
            f"{_fmt(r.get('result_time'), 1, ' s')} | {_fmt(r.get('land_time'), 1, ' s')} |"
        )
    out += ["", "Times are seconds since the flight recorder started, which is when the launch started PX4."]
    x = summary.get("xrce")
    if x:
        out += [
            "",
            "## The XRCE session",
            "",
            f"Dissected by `python -m week06 xrce session` from the capture of the 5 m flight ({x['duration_s']:.0f} s): PX4 logged in with client key "
            f"`{x['client']['client_key']}` (session 0x{x['client']['session_id']:02x}, MTU {x['client']['mtu']}) and created {x['created']} objects "
            f"in {x['participants']} participant. Traffic: {x['kbps']['px4>agent']} kbit/s from PX4, {x['kbps']['agent>px4']} kbit/s to it.",
            "",
            "| Topic | Type | Direction | Reliability | Durability | Depth | Samples | Rate | Bytes per sample |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for t in sorted(x["topics"], key=lambda t: (t["direction"], t["topic"])):
            q = t["qos"]
            out.append(
                f"| `{t['topic']}` | `{t['type']}` | {'PX4 publishes' if t['direction'] == 'out' else 'PX4 subscribes'} | "
                f"{'reliable' if q.get('reliable') else 'best effort'} | {q.get('durability', 'n/a')} | {q.get('depth', 'n/a')} | {t['samples']} | "
                f"{_fmt(t['rate_hz'], 1, ' Hz')} | {t['sample_bytes'] if t['sample_bytes'] is not None else 'n/a'} |"
            )
        out += ["", "| Submessage | Count |", "| --- | ---: |"]
        for k, v in sorted(x["submessages"].items(), key=lambda kv: -kv[1]):
            out.append(f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def regenerate(sitl: bool = False) -> int:
    if sitl:
        fly_all()
    summary: dict = {"versions": VERSIONS, "titles": {k: v[0] for k, v in FLIGHTS.items()}, "flights": {}}
    replay = []
    for name, (title, _) in FLIGHTS.items():
        path = FLIGHTS_DIR / f"{name}.jsonl"
        if not path.is_file():
            print(f"missing {path}; fly it with python -m week06 docs --sitl", file=sys.stderr)
            continue
        flight = Flight.load(path)
        r = analyze(flight)
        summary["flights"][name] = {k: v for k, v in r.items() if k not in ("commands", "acks")}
        tl = timeline(flight)
        replay.append({"id": name, "title": title, "target": r["target"], "method": r["method"], "metrics": summary["flights"][name], **tl})
    raw = FLIGHTS_DIR / "offboard_5m.pcap"  # the whole flight; too big to commit
    trimmed = FLIGHTS_DIR / "offboard_5m.xrce.pcap"
    previous = DOCS / "summary.json"
    if raw.is_file():
        summary["xrce"] = dissect.session(raw)
        trim_capture(raw, trimmed)
    elif previous.is_file() and "xrce" in json.loads(previous.read_text()):
        summary["xrce"] = json.loads(previous.read_text())["xrce"]  # measured on the full capture
    if trimmed.is_file():
        summary.setdefault("xrce", dissect.session(trimmed))
        (DOCS / "xrce.json").write_text(json.dumps({"session": summary["xrce"], "packets": packet_tour(trimmed)}, indent=1) + "\n")
    (DOCS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (DOCS / "replay.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    (DOCS / "RESULTS.md").write_text(results_md(summary))
    print(f"wrote {DOCS.relative_to(ROOT)}: {len(summary['flights'])} flights")
    return 0
