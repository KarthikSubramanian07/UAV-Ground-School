"""The flight controller half of the design: an ArduPilot ``.param`` file generated from the wiring.

"How does firmware influence the capabilities of the flight controller?"
Mostly through parameters: the same Cube is a racing quad, a survey drone or
a boat depending on a few hundred numbers. Here they follow from the design:

* each UART's ``SERIALn_PROTOCOL`` and ``SERIALn_BAUD`` from the link on it,
* CAN, GPS, rangefinder, gimbal and battery monitor setup from the parts,
* motor outputs and DShot from the motor link,
* telemetry rates from the stream plan, using the 4.7 ``MAVn_*`` names where
  ``n`` counts MAVLink ports in serial order (not the serial number),
* PID gains for pitch and roll from the :mod:`week04.pid` tuner on this
  airframe's inertia.

:func:`validate` then checks every name and value against ArduCopter 4.7.1's
own parameter metadata (generated from the source by ArduPilot's
``param_parse.py``; the relevant subset is in ``data/ardupilot_params.json``),
and :mod:`week04.sitl` loads the file into a running ArduCopter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .check import mavlink_ports
from .design import PROTOCOLS, Design

METADATA = Path(__file__).with_name("data") / "ardupilot_params.json"

BAUD_CODES = {
    1200: 1,
    2400: 2,
    4800: 4,
    9600: 9,
    19200: 19,
    38400: 38,
    57600: 57,
    111100: 111,
    115200: 115,
    230400: 230,
    256000: 256,
    460800: 460,
    500000: 500,
    921600: 921,
    1500000: 1500,
}


@dataclass
class Param:
    name: str
    value: float
    why: str


def generate(design: Design, gains: dict | None = None) -> list[Param]:
    out: list[Param] = []

    def put(name: str, value: float, why: str):
        out.append(Param(name, value, why))

    fc = design.fc
    put("FRAME_CLASS", 1, "quad")
    put("FRAME_TYPE", 1, "X")

    # serial ports
    for link in design.links:
        port = link.a[0]
        if port.part != fc.name or link.layer != "uart" or port.kind == "usb":
            continue
        n = port.spec["serial_index"]
        proto = PROTOCOLS[link.protocol]
        put(f"SERIAL{n}_PROTOCOL", proto["serial_protocol"], f"{port.id}: {proto['label']} ({link.spec['purpose']})")
        if link.protocol != "crsf":  # ArduPilot sets the CRSF baud itself
            put(f"SERIAL{n}_BAUD", BAUD_CODES[link.spec["baud"]], f"{port.id}: {link.spec['baud']} baud")
        if link.spec.get("flow_control") and port.spec.get("flow_control"):
            put(f"BRD_SER{n}_RTSCTS", 1, f"{port.id}: hardware flow control on")
        if link.protocol == "dds_xrce":
            put("DDS_ENABLE", 1, "start the DDS client (needs a build with AP_DDS)")
        if link.protocol == "crsf":
            put("RSSI_TYPE", 3, "RSSI from the receiver protocol")
            put("RC_OPTIONS", 8704, "bit 9: suppress CRSF mode messages for ELRS, bit 13: 420 kbaud for ELRS")
        if link.protocol == "siyi":
            put("MNT1_TYPE", 8, "SIYI gimbal")
            put("CAM1_TYPE", 4, "camera triggered through the mount driver")
            put("MNT1_PITCH_MIN", -90, "gimbal can look straight down")
            put("MNT1_PITCH_MAX", 25, "")
            put("MNT1_YAW_MIN", -135, "")
            put("MNT1_YAW_MAX", 135, "")

    # telemetry stream rates on each MAVLink port
    channels = {serial: i + 1 for i, (serial, _) in enumerate(mavlink_ports(design))}
    for link in design.links:
        if link.protocol not in ("mavlink1", "mavlink2") or not link.spec.get("streams"):
            continue
        n = channels[link.a[0].spec["serial_index"]]
        for stream, hz in link.spec["streams"].items():
            put(f"MAV{n}_{stream}", hz, f"{link.a[0].id} is MAVLink channel {n}")

    # CAN and GNSS
    for link in design.links:
        if link.protocol == "dronecan":
            bus = int(link.a[0].id[-1])
            put(f"CAN_P{bus}_DRIVER", 1, f"{link.a[0].id} uses the first CAN driver")
            put("CAN_D1_PROTOCOL", 1, "DroneCAN")
            put(f"CAN_P{bus}_BITRATE", link.spec.get("bitrate", 1000000), "1 Mbit/s")
            if design.parts[link.b[0].part].part["role"] == "gnss":
                put("GPS1_TYPE", 9, "Here4 on DroneCAN")
                put("GPS_AUTO_CONFIG", 1, "let ArduPilot configure the receiver")
                put("BRD_SAFETY_DEFLT", 0, "the Here4 has no safety switch")

    # rangefinder
    for link in design.links:
        if link.protocol == "i2c" and design.parts[link.b[0].part].part["role"] == "rangefinder":
            put("RNGFND1_TYPE", 25, "Benewake TFmini on I2C")
            put("RNGFND1_ADDR", link.spec["address"], f"0x{link.spec['address']:02X}")
            put("RNGFND1_ORIENT", 25, "pointing down")
            put("RNGFND1_MIN", 0.1, "metres (4.7 uses metres)")
            put("RNGFND1_MAX", 6, "metres, outdoor figure from the ArduPilot wiki")

    # battery monitor
    pm = design.parts["pm"].part
    ap = pm.get("ardupilot", {}).get("params", {})
    put("BATT_MONITOR", 4, "analog voltage and current")
    put("BATT_VOLT_MULT", ap.get("BATT_VOLT_MULT", 18.182), "PM02 V3 divider (the board default is for the Power Brick Mini)")
    put("BATT_AMP_PERVLT", ap.get("BATT_AMP_PERVLT", 36.364), "PM02 V3 current scale")
    battery = design.parts["battery"].part["specs"]
    put("BATT_CAPACITY", battery["capacity_mah"], "mAh")
    put("BATT_LOW_VOLT", round(3.5 * battery["cells"], 1), "3.5 V per cell under load")
    put("BATT_CRT_VOLT", round(3.3 * battery["cells"], 1), "3.3 V per cell under load")
    put("BATT_FS_LOW_ACT", 2, "RTL on low battery")
    put("BATT_FS_CRT_ACT", 1, "land on critical battery")

    # motors
    for link in design.links:
        rate = PROTOCOLS[link.protocol].get("dshot")
        if not rate:
            continue
        put("MOT_PWM_TYPE", {150: 4, 300: 5, 600: 6, 1200: 7}[rate], f"DShot{rate}")
        put("SERVO_DSHOT_ESC", 2, "BLHeli_S")
        for i, port in enumerate(link.a):
            servo = 8 + int(port.id[3:]) if port.id.startswith("AUX") else int(port.id[4:])
            put(f"SERVO{servo}_FUNCTION", 33 + i, f"{port.id} drives motor {i + 1}")
        for main in range(1, 5):
            put(f"SERVO{main}_FUNCTION", 0, "MAIN outputs unused (motors are on AUX)")

    # propulsion tuning from the model
    from . import performance

    perf = performance.analyse(design)
    put("MOT_THST_HOVER", round(perf.hover_throttle, 3), "hover throttle predicted by week04.performance")
    put("MOT_BAT_VOLT_MAX", battery["cells"] * 4.2, "voltage scaling of thrust")
    put("MOT_BAT_VOLT_MIN", battery["cells"] * 3.3, "")

    # attitude gains
    if gains:
        for name, value in gains.items():
            put(name, round(value, 4), "from week04.pid, tuned on this airframe's pitch inertia")

    # failsafes
    put("FS_THR_ENABLE", 1, "RC loss: RTL")
    put("FS_GCS_ENABLE", 1, "ground station loss: RTL")
    put("FENCE_ENABLE", 1, "geofence on")
    put("FENCE_RADIUS", 300, "metres")
    put("FENCE_ALT_MAX", 120, "metres, the US Part 107 ceiling")
    return out


def write(params: list[Param], path: str | Path, header: str = "") -> Path:
    path = Path(path)
    lines = [f"# {line}" for line in header.splitlines()] if header else []
    for p in params:
        value = int(p.value) if float(p.value).is_integer() else p.value
        lines.append(f"{p.name},{value}" + (f"  # {p.why}" if p.why else ""))
    path.write_text("\n".join(lines) + "\n")
    return path


def read(path: str | Path) -> dict[str, float]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, value = line.replace("\t", ",").replace(" ", ",").split(",", 1)
        out[name] = float(value.strip(","))
    return out


# ---------------------------------------------------------- validation ----


def metadata() -> dict[str, dict]:
    return json.loads(METADATA.read_text())


@dataclass
class Problem:
    name: str
    value: float
    message: str


def validate(params: list[Param], meta: dict[str, dict] | None = None) -> list[Problem]:
    """Every name must exist in ArduCopter 4.7.1, every value must be an allowed enum value, bitmask or in range."""
    meta = meta or metadata()
    problems: list[Problem] = []
    seen: set[str] = set()
    for p in params:
        if p.name in seen:
            problems.append(Problem(p.name, p.value, "set twice"))
        seen.add(p.name)
        m = meta.get(p.name)
        if m is None:
            problems.append(Problem(p.name, p.value, "no such parameter in ArduCopter 4.7.1"))
            continue
        if "Values" in m and float(p.value).is_integer() and str(int(p.value)) not in m["Values"] and "Range" not in m:
            problems.append(Problem(p.name, p.value, f"not one of {sorted(m['Values'], key=float)}"))
        if "Bitmask" in m:
            allowed = sum(1 << int(b) for b in m["Bitmask"])
            if int(p.value) & ~allowed:
                problems.append(Problem(p.name, p.value, "sets bits that have no meaning"))
        if "Range" in m and "Values" not in m:
            lo, hi = float(m["Range"]["low"]), float(m["Range"]["high"])
            if not lo <= p.value <= hi:
                problems.append(Problem(p.name, p.value, f"outside {lo} to {hi}"))
    return problems


def extract_metadata(pdef: dict, names: set[str]) -> dict[str, dict]:
    """Pull the entries for ``names`` out of ArduPilot's apm.pdef.json (grouped by prefix)."""
    flat = {name: entry for group in pdef.values() for name, entry in group.items()}
    keep = ("DisplayName", "Values", "Bitmask", "Range", "Units", "RebootRequired")
    return {n: {k: flat[n][k] for k in keep if k in flat[n]} for n in sorted(names) if n in flat}
