"""Drive a real ArduCopter (software in the loop) with this week's own MAVLink code.

ArduPilot's SITL build is the same flight code that runs on the Cube,
compiled for a laptop with a physics model in place of the sensors. This
module starts it, connects over TCP (SITL's stand in for a serial port),
and talks to it with :mod:`week04.mavlink` only: no pymavlink, no MAVProxy.

It is used for three checks that no amount of unit testing can replace:

1. **Parameters.** Every parameter in the generated ``.param`` file is sent
   with PARAM_SET and must come back in a PARAM_VALUE with the same value.
   A name the firmware does not know gets no reply at all.
2. **Link budget.** Stream rates are set on the link and the bytes per
   message are counted, to compare with :func:`week04.links.mavlink_budget`.
3. **Flight.** Arm, take off, command a pitch step, land, and record the
   response next to the :mod:`week04.pid` simulator's.

Build SITL once (about 5 minutes on an M1)::

    git clone --depth 1 --branch Copter-4.7.1 --recurse-submodules https://github.com/ArduPilot/ardupilot
    cd ardupilot && ./waf configure --board sitl && ./waf copter

then point ``ARDUPILOT_DIR`` at the checkout.
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import mavlink

MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_SET_MESSAGE_INTERVAL = 511
COPTER_MODES = {"STABILIZE": 0, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5, "RTL": 6, "LAND": 9}
PARAM_TYPE_REAL32 = 9


def ardupilot_dir() -> Path | None:
    env = os.environ.get("ARDUPILOT_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(Path.home() / "src" / "ardupilot")
    for c in candidates:
        if (c / "build" / "sitl" / "bin" / "arducopter").exists():
            return c
    return None


@dataclass
class Received:
    t: float
    name: str
    fields: dict
    size: int
    sysid: int
    compid: int
    signed: bool = False
    signature_ok: bool | None = None


@dataclass
class Sitl:
    """A running ArduCopter SITL instance and a MAVLink 2 connection to its SERIAL0."""

    root: Path
    home: str = "37.8719,-122.2585,72,0"  # Memorial Glade, UC Berkeley
    speedup: int = 1
    model: str = "quad"
    frame: dict | None = None  # a custom airframe (week04.performance.sitl_frame); SITL only reads it from its working directory
    instance: int = 0
    defaults: list[str] = field(default_factory=lambda: ["copter.parm"])
    boot_params: dict[str, float] = field(default_factory=dict)  # applied at boot, e.g. MAVn_* stream rates
    extra_args: list[str] = field(default_factory=list)
    workdir: Path | None = None
    capture: bytearray = field(default_factory=bytearray)  # tlog: 8 byte big endian usec + frame
    log: list[Received] = field(default_factory=list)
    bytes_by_msg: Counter = field(default_factory=Counter)
    count_by_msg: Counter = field(default_factory=Counter)
    statustext: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.dialect = mavlink.Dialect.default()
        self.parser = mavlink.Parser(self.dialect)
        self.seq = 0
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.t0 = time.monotonic()
        self.key: bytes | None = None
        self.link_id = 0
        self.last_heartbeat = -1.0

    # ------------------------------------------------------- lifecycle ----

    def start(self, wipe: bool = True, timeout: float = 30.0) -> Sitl:
        self.workdir = Path(self.workdir or tempfile.mkdtemp(prefix="sitl-"))
        binary = self.root / "build" / "sitl" / "bin" / "arducopter"
        if self.frame is not None:
            (self.workdir / "frame.json").write_text(json.dumps(self.frame))
            self.model = "quad:frame.json"
        files = [str(self.root / "Tools" / "autotest" / "default_params" / d) for d in self.defaults]
        if self.boot_params:
            extra = self.workdir / "boot.parm"
            extra.write_text("".join(f"{k} {v}\n" for k, v in self.boot_params.items()))
            files.append(str(extra))
        defaults = ",".join(files)
        args = [
            str(binary),
            "--model",
            self.model,
            "--speedup",
            str(self.speedup),
            "--home",
            self.home,
            "-I",
            str(self.instance),
            "--defaults",
            defaults,
        ]
        if wipe:
            args.append("--wipe")
        args += self.extra_args
        self.proc = subprocess.Popen(args, cwd=self.workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port = 5760 + 10 * self.instance
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                break
            except OSError:
                if time.monotonic() > deadline or self.proc.poll() is not None:
                    self.stop()
                    raise RuntimeError("SITL did not open its MAVLink port") from None
                time.sleep(0.2)
        self.sock.settimeout(0.05)
        self.t0 = time.monotonic()
        self.wait("HEARTBEAT", timeout=timeout)
        return self

    def stop(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -------------------------------------------------------------- I/O ----

    def now(self) -> float:
        return time.monotonic() - self.t0

    def send(self, name: str, **values):
        frame = mavlink.encode(
            self.dialect[name], values, seq=self.seq, sysid=255, compid=190, key=self.key, link_id=self.link_id, timestamp=self._timestamp()
        )
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendall(frame)

    def _timestamp(self) -> int:
        # MAVLink signing time: 10 us units since 2015-01-01
        return int((time.time() - 1420070400) * 1e5)

    def pump(self, duration: float = 0.0) -> list[Received]:
        """Read everything that arrives within ``duration`` seconds (at least one poll)."""
        out: list[Received] = []
        end = time.monotonic() + duration
        while True:
            if self.now() - self.last_heartbeat >= 1.0:  # a GCS must say hello once a second
                self.last_heartbeat = self.now()
                self.send("HEARTBEAT", type=6, autopilot=8, base_mode=0, custom_mode=0, system_status=4, mavlink_version=3)
            try:
                data = self.sock.recv(65536)
            except TimeoutError:
                data = b""
            if data:
                for packet in self.parser.feed(data):
                    name, values = self.parser.decode(packet)
                    now = self.now()
                    usec = int(time.time() * 1e6)
                    self.capture += struct.pack(">Q", usec) + packet.raw
                    self.bytes_by_msg[name] += len(packet.raw)
                    self.count_by_msg[name] += 1
                    ok = mavlink.verify_signature(packet, self.key) if (packet.signed and self.key) else None
                    r = Received(now, name, values, len(packet.raw), packet.sysid, packet.compid, packet.signed, ok)
                    if name == "STATUSTEXT":
                        self.statustext.append(values["text"])
                    self.log.append(r)
                    out.append(r)
            if time.monotonic() >= end:
                return out

    def wait(self, name: str, timeout: float = 10.0, where=None) -> Received:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for r in self.pump(0.05):
                if r.name == name and r.sysid == 1 and (where is None or where(r.fields)):
                    return r
        raise TimeoutError(f"no {name} within {timeout} s")

    # ---------------------------------------------------------- commands ----

    def command(self, command: int, *params: float, timeout: float = 5.0) -> int:
        p = list(params) + [0.0] * (7 - len(params))
        self.send(
            "COMMAND_LONG",
            target_system=1,
            target_component=1,
            command=command,
            confirmation=0,
            **{f"param{i + 1}": p[i] for i in range(7)},
        )
        ack = self.wait("COMMAND_ACK", timeout, where=lambda f: f["command"] == command)
        return ack.fields["result"]

    def set_mode(self, mode: str) -> int:
        return self.command(MAV_CMD_DO_SET_MODE, 1, COPTER_MODES[mode])

    def set_param(self, name: str, value: float, timeout: float = 2.0) -> float | None:
        """PARAM_SET, then wait for the echo. Returns the value the vehicle now holds, None if it never answered."""
        self.send("PARAM_SET", target_system=1, target_component=1, param_id=name, param_value=float(value), param_type=PARAM_TYPE_REAL32)
        try:
            r = self.wait("PARAM_VALUE", timeout, where=lambda f: f["param_id"] == name)
        except TimeoutError:
            return None
        return r.fields["param_value"]

    def get_param(self, name: str, timeout: float = 2.0) -> float | None:
        self.send("PARAM_REQUEST_READ", target_system=1, target_component=1, param_id=name, param_index=-1)
        try:
            return self.wait("PARAM_VALUE", timeout, where=lambda f: f["param_id"] == name).fields["param_value"]
        except TimeoutError:
            return None

    def message_interval(self, name: str, hz: float) -> int:
        interval = -1 if hz <= 0 else 1e6 / hz
        return self.command(MAV_CMD_SET_MESSAGE_INTERVAL, self.dialect[name].id, interval)

    def setup_signing(self, key: bytes):
        """Hand the vehicle a 32 byte key (SETUP_SIGNING), then sign everything we send from now on."""
        self.send("SETUP_SIGNING", target_system=1, target_component=1, secret_key=list(key), initial_timestamp=self._timestamp())
        self.key = key

    def attitude_target(self, roll_deg: float, pitch_deg: float, yaw_deg: float, thrust: float):
        """SET_ATTITUDE_TARGET in GUIDED: hold this attitude with this collective thrust (0 to 1)."""
        r, p, y = (math.radians(v) / 2 for v in (roll_deg, pitch_deg, yaw_deg))
        q = [
            math.cos(r) * math.cos(p) * math.cos(y) + math.sin(r) * math.sin(p) * math.sin(y),
            math.sin(r) * math.cos(p) * math.cos(y) - math.cos(r) * math.sin(p) * math.sin(y),
            math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.cos(p) * math.sin(y),
            math.cos(r) * math.cos(p) * math.sin(y) - math.sin(r) * math.sin(p) * math.cos(y),
        ]
        self.send(
            "SET_ATTITUDE_TARGET",
            time_boot_ms=int(self.now() * 1000),
            target_system=1,
            target_component=1,
            type_mask=0b00000111,  # ignore body rates, use the quaternion and thrust
            q=q,
            body_roll_rate=0.0,
            body_pitch_rate=0.0,
            body_yaw_rate=0.0,
            thrust=thrust,
        )

    def wait_ready(self, timeout: float = 90.0):
        """Wait until both EKF cores report they are using GPS (the prerequisite for arming in GUIDED)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.5)
            if any("is using GPS" in s for s in self.statustext):
                return
        raise TimeoutError("vehicle never became ready (EKF not using GPS)")


# ------------------------------------------------------------ analysis ----


@dataclass
class FlightResult:
    hover_throttle: float
    step_t: list[float]
    step_pitch: list[float]
    target_deg: float
    altitude: list[tuple[float, float]]
    statustext: list[str]


def pitch_step_flight(sim: Sitl, target_deg: float = 10.0, altitude: float = 10.0) -> FlightResult:
    """Arm in GUIDED, climb, measure hover throttle, then hold a pitch step and record the response."""
    sim.wait_ready()
    if sim.set_mode("GUIDED") != 0 or sim.command(MAV_CMD_COMPONENT_ARM_DISARM, 1) != 0:
        raise RuntimeError("could not arm: " + "; ".join(sim.statustext[-5:]))
    sim.command(MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, altitude)
    t0 = sim.now()
    while sim.now() - t0 < 25:
        sim.pump(0.5)
        pos = [r for r in sim.log if r.name == "GLOBAL_POSITION_INT"]
        if pos and pos[-1].fields["relative_alt"] / 1000 > altitude * 0.97:
            break
    sim.pump(6.0)  # settle
    hud = [r.fields["throttle"] for r in sim.log if r.name == "VFR_HUD" and r.t > sim.now() - 4]
    hover = sum(hud) / len(hud) / 100.0 if hud else float("nan")
    thrust = 0.5  # in attitude mode GUIDED treats 0.5 as "hold altitude" only with GUID_OPTIONS; we use climb rate 0
    # hold level for a second, then the step, then level again
    phases = [(1.0, 0.0), (2.5, target_deg), (2.0, 0.0)]
    start = sim.now()
    for duration, pitch in phases:
        end = sim.now() + duration
        while sim.now() < end:
            sim.attitude_target(0.0, pitch, 0.0, thrust)
            sim.pump(0.02)
    t, series = euler_step_series(sim.log, start + 1.0)
    keep = [i for i, ti in enumerate(t) if ti <= 2.5]
    alt = [(r.t, r.fields["relative_alt"] / 1000) for r in sim.log if r.name == "GLOBAL_POSITION_INT"]
    sim.set_mode("LAND")
    sim.pump(1.0)
    return FlightResult(hover, [t[i] for i in keep], [series[i] for i in keep], target_deg, alt, list(sim.statustext))


def measured_rates(log: list[Received], start: float, end: float) -> dict[str, tuple[float, float]]:
    """Messages per second and bytes per second for each message name within [start, end)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in log:
        if start <= r.t < end and r.sysid == 1:
            counts[r.name][0] += 1
            counts[r.name][1] += r.size
    span = end - start
    return {name: (n / span, b / span) for name, (n, b) in counts.items()}


def euler_step_series(log: list[Received], start: float) -> tuple[list[float], list[float]]:
    t, pitch = [], []
    for r in log:
        if r.name == "ATTITUDE" and r.t >= start:
            t.append(r.t - start)
            pitch.append(math.degrees(r.fields["pitch"]))
    return t, pitch


# --------------------------------------------------------- experiments ----


class _Port:
    """A second MAVLink connection (SITL's SERIAL1 on TCP 5762 + 10 * instance)."""

    def __init__(self, port: int, dialect: mavlink.Dialect):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        self.sock.settimeout(0.05)
        self.parser = mavlink.Parser(dialect)
        self.dialect = dialect
        self.seq = 0
        self.log: list[tuple[float, str, dict, int, bool]] = []

    def send(self, name: str, key: bytes | None = None, **values):
        frame = mavlink.encode(
            self.dialect[name],
            values,
            seq=self.seq,
            sysid=255,
            compid=191,
            key=key,
            link_id=1,
            timestamp=int((time.time() - 1420070400) * 1e5),
        )
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendall(frame)

    def pump(self, t0: float):
        try:
            data = self.sock.recv(65536)
        except (TimeoutError, OSError):
            return
        for p in self.parser.feed(data):
            name, values = self.parser.decode(p)
            self.log.append((time.monotonic() - t0, name, values, len(p.raw), p.signed))

    def close(self):
        self.sock.close()


def _param_echo(sim: Sitl, port: _Port, name: str, key: bytes | None, timeout: float = 2.0) -> bool:
    port.send("PARAM_REQUEST_READ", key=key, target_system=1, target_component=1, param_id=name, param_index=-1)
    start = len(port.log)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        sim.pump(0.02)
        port.pump(sim.t0)
        if any(entry[1] == "PARAM_VALUE" and entry[2]["param_id"] == name for entry in port.log[start:]):
            return True
    return False


def experiments(root: Path, design=None, window: float = 20.0) -> dict:
    """Everything the docs report from a real ArduCopter: parameters, link budget, channel mapping, signing, flight."""
    import os as _os

    from . import links, params, performance
    from .design import Design
    from .uart import UartConfig

    design = design or Design.load()
    result: dict = {"firmware": "ArduCopter 4.7.1 SITL (built from the Copter-4.7.1 tag)"}

    # 1. every generated parameter, sent and echoed
    plist = params.generate(design)
    with Sitl(root) as sim:
        echoed = {p.name: sim.set_param(p.name, p.value) for p in plist}
    rejected = sorted(n for n, v in echoed.items() if v is None)
    mismatched = sorted(
        n for n, v in echoed.items() if v is not None and abs(v - next(p.value for p in plist if p.name == n)) > 1e-3 * max(1.0, abs(v))
    )
    result["params"] = {
        "sent": len(plist),
        "accepted": len(plist) - len(rejected) - len(mismatched),
        "rejected": rejected,
        "mismatched": mismatched,
    }

    # 2 to 4. telemetry plan on TELEM1 (SERIAL1 is MAVLink channel 2), signing on that link
    link = design.link("telemetry")
    boot = {"SERIAL1_PROTOCOL": 2, "SERIAL1_BAUD": 57}
    boot.update({f"MAV2_{k}": v for k, v in link.spec["streams"].items()})
    boot.update({f"MAV1_{k}": 0 for k in link.spec["streams"]})
    boot["MAV1_EXTRA1"] = 1
    sim = Sitl(root, boot_params=boot, extra_args=["--serial1", f"tcp:{5762 + 10 * 0}"])
    sim.start()
    port = _Port(5762, sim.dialect)
    try:
        sim.wait_ready()
        drain = time.monotonic() + 1.0  # the TCP buffer holds everything sent while we waited: read it and throw it away
        while time.monotonic() < drain:
            sim.pump(0.02)
            port.pump(sim.t0)
        port.log.clear()
        t_start = time.monotonic() - sim.t0
        end = time.monotonic() + window
        while time.monotonic() < end:
            sim.pump(0.02)
            port.pump(sim.t0)
        t_end = time.monotonic() - sim.t0
        measured: dict[str, list[float]] = {}
        for t, name, _, size, _ in port.log:
            if t_start <= t < t_end:
                m = measured.setdefault(name, [0, 0])
                m[0] += 1
                m[1] += size
        span = t_end - t_start
        observed = set(measured)
        budget = links.mavlink_budget("telemetry", UartConfig(57600), link.spec["streams"], include=observed)
        rows = []
        for stream, msg, hz, size in budget.rows:
            got = measured.get(msg, [0, 0])
            rows.append(
                {
                    "stream": stream,
                    "message": msg,
                    "predicted_hz": hz,
                    "measured_hz": round(got[0] / span, 2),
                    "max_frame_bytes": size,
                    "measured_bytes_per_s": round(got[1] / span, 1),
                }
            )
        predicted_messages = {r["message"] for r in rows}
        unexpected = {n: round(c / span, 2) for n, (c, _) in measured.items() if n not in predicted_messages}
        result["link_budget"] = {
            "window_s": round(span, 1),
            "predicted_bytes_per_s": round(budget.bytes_per_second, 1),
            "measured_bytes_per_s": round(sum(b for _, b in measured.values()) / span, 1),
            "rows": rows,
            "unexpected_messages_hz": unexpected,
        }
        usb_attitude = sum(1 for r in sim.log if r.name == "ATTITUDE" and t_start <= r.t < t_end) / span
        telem_attitude = measured.get("ATTITUDE", [0])[0] / span
        result["channel_mapping"] = {
            "MAV1_EXTRA1": 1,
            "MAV2_EXTRA1": link.spec["streams"]["EXTRA1"],
            "serial0_attitude_hz": round(usb_attitude, 2),
            "serial1_attitude_hz": round(telem_attitude, 2),
            "conclusion": "MAV2_* set the rates on SERIAL1 (TELEM1) because it is the second MAVLink port",
        }

        # signing: the key goes in over USB, then TELEM1 must reject unsigned requests
        key = _os.urandom(32)
        unsigned_before = _param_echo(sim, port, "MAV2_EXTRA1", None)
        sim.setup_signing(key)
        sim.pump(1.0)
        t_sign = time.monotonic() - sim.t0
        sim.pump(5.0)
        unsigned_after = _param_echo(sim, port, "MAV2_EXTRA1", None)
        signed_after = _param_echo(sim, port, "MAV2_EXTRA1", key)
        after = [r for r in sim.log if r.t > t_sign + 0.2 and r.sysid == 1]
        result["signing"] = {
            "telem1_unsigned_request_before_key": unsigned_before,
            "telem1_unsigned_request_after_key": unsigned_after,
            "telem1_signed_request_after_key": signed_after,
            "usb_packets_after_key": len(after),
            "usb_packets_signed": sum(r.signed for r in after),
            "usb_signatures_verified": sum(bool(r.signature_ok) for r in after),
        }
        sim.send("SETUP_SIGNING", target_system=1, target_component=1, secret_key=[0] * 32, initial_timestamp=0)  # all zero disables it
    finally:
        port.close()
        sim.stop()

    # 5. fly this airframe
    perf = performance.analyse(design)
    frame = performance.sitl_frame(design, perf)
    rates = {"MAV1_EXT_STAT": 2, "MAV1_POSITION": 10, "MAV1_EXTRA1": 50, "MAV1_EXTRA2": 10, "MAV1_EXTRA3": 2, "MAV1_RC_CHAN": 10}
    sim = Sitl(root, boot_params=dict(rates, MOT_THST_HOVER=round(perf.hover_throttle, 3)), frame=frame)
    sim.start()
    try:
        flight = pitch_step_flight(sim)
        capture = bytes(sim.capture)
    finally:
        sim.stop()
    result["flight"] = {
        "frame": frame,
        "hover_throttle_measured": round(flight.hover_throttle, 3),
        "hover_throttle_model": round(perf.hover_throttle, 3),
        "step_target_deg": flight.target_deg,
        "step_t": [round(t, 3) for t in flight.step_t],
        "step_pitch_deg": [round(p, 2) for p in flight.step_pitch],
    }
    result["_capture"] = capture
    return result
