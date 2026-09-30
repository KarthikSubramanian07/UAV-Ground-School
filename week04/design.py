"""The drone as data: parts from the catalog, and every wire between them.

``data/catalog.json`` holds the researched parts (each figure with its
source, anything unpublished marked as an estimate). ``data/design.json``
says which parts this drone uses and how they connect: data links with a
protocol, power feeds, and radio links. Everything else this week (the
compatibility checker, the harness list, the parameter file, the wiring
diagram, the performance model) is computed from these two files, so a
change in the design shows up everywhere at once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).with_name("data")

# protocol -> physical layer and how ArduPilot selects it
PROTOCOLS: dict[str, dict] = {
    "mavlink1": {"layer": "uart", "serial_protocol": 1, "label": "MAVLink 1"},
    "mavlink2": {"layer": "uart", "serial_protocol": 2, "label": "MAVLink 2"},
    "dds_xrce": {"layer": "uart", "serial_protocol": 45, "label": "DDS XRCE (ROS 2)", "feature": "AP_DDS"},
    "siyi": {"layer": "uart", "serial_protocol": 8, "label": "SIYI gimbal serial", "fixed_baud": 115200},
    "crsf": {"layer": "uart", "serial_protocol": 23, "label": "CRSF (ExpressLRS)", "fixed_baud": 420000, "needs_rx_dma": True},
    "benewake_uart": {"layer": "uart", "serial_protocol": 9, "label": "Benewake serial"},
    "dronecan": {"layer": "can", "label": "DroneCAN"},
    "i2c": {"layer": "i2c", "label": "I2C"},
    "ethernet": {"layer": "ethernet", "label": "Ethernet"},
    "dshot300": {"layer": "pwm", "label": "DShot300", "dshot": 300},
    "dshot600": {"layer": "pwm", "label": "DShot600", "dshot": 600},
    "pwm": {"layer": "pwm", "label": "PWM"},
}

LAYER_OF_KIND = {"uart": "uart", "usb": "uart", "can": "can", "i2c": "i2c", "ethernet": "ethernet", "pwm": "pwm"}


@dataclass
class Port:
    part: str  # instance name, e.g. "fc"
    spec: dict

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def kind(self) -> str:
        return self.spec.get("kind", "")

    @property
    def ref(self) -> str:
        return f"{self.part}.{self.id}"

    def pins(self) -> list[tuple[int, str, set[str]]]:
        """(pin number, raw signal name, canonical roles) for every pin."""
        out = []
        for i, raw in enumerate(self.spec.get("signals", [])):
            m = re.search(r"\(pin (\d+)\)", raw)
            out.append((int(m.group(1)) if m else i + 1, raw, roles(raw)))
        return out


def roles(signal: str) -> set[str]:
    """Canonical roles of a vendor's pin name: "UART1_TXD (pin 8)" is TX, "RXD/SDA" is both RX and SDA."""
    s = re.sub(r"\(pin \d+\)", "", signal).upper()
    tokens = set(re.split(r"[^A-Z0-9]+", s)) - {""}
    out = set()
    if "GND" in tokens:
        out.add("GND")
    if tokens & {"5V", "VCC", "VDD", "VBUS"} or "5V" in s:
        out.add("5V")
    if "3V3" in tokens:
        out.add("3V3")
    for role, names in {
        "TX": {"TX", "TXD", "UART1_TXD"},
        "RX": {"RX", "RXD", "UART1_RXD"},
        "CTS": {"CTS"},
        "RTS": {"RTS"},
        "SCL": {"SCL"},
        "SDA": {"SDA"},
    }.items():
        if tokens & names or any(t.endswith("_" + n) or t.startswith(n) and len(t) <= len(n) + 1 for t in tokens for n in names):
            out.add(role)
    if "CAN_H" in s.replace(" ", "_"):
        out.add("CAN_H")
    if "CAN_L" in s.replace(" ", "_"):
        out.add("CAN_L")
    for token in tokens:
        if token.endswith("TXD") or token.endswith("TX"):
            out.add("TX")
        if token.endswith("RXD") or token.endswith("RX"):
            out.add("RX")
    if "DSM" in tokens or "SBUS" in tokens:
        out.discard("RX")
    return out


@dataclass
class Instance:
    name: str
    part: dict
    info: dict

    @property
    def count(self) -> int:
        return self.info.get("count", 1)

    @property
    def label(self) -> str:
        return self.info.get("label", self.part["name"])

    def port(self, port_id: str) -> Port:
        for p in self.part.get("ports", []):
            if p["id"] == port_id:
                return Port(self.name, p)
        raise KeyError(f"{self.name} ({self.part['id']}) has no port {port_id}")


@dataclass
class Link:
    spec: dict
    a: list[Port]
    b: list[Port]

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def protocol(self) -> str:
        return self.spec["protocol"]

    @property
    def layer(self) -> str:
        return PROTOCOLS[self.protocol]["layer"]


@dataclass
class Design:
    spec: dict
    catalog: dict
    parts: dict[str, Instance] = field(default_factory=dict)
    ground: dict[str, dict] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)

    @classmethod
    def load(cls, design: str | Path | dict | None = None, catalog: str | Path | dict | None = None) -> Design:
        spec = design if isinstance(design, dict) else json.loads(Path(design or DATA / "design.json").read_text())
        cat = catalog if isinstance(catalog, dict) else json.loads(Path(catalog or DATA / "catalog.json").read_text())
        by_id = {c["id"]: c for c in cat["components"]}
        d = cls(spec, cat)
        for name, info in spec["parts"].items():
            if info["part"] not in by_id:
                raise KeyError(f"part {name}: {info['part']} is not in the catalog")
            d.parts[name] = Instance(name, by_id[info["part"]], info)
        for name, info in spec.get("ground", {}).items():
            d.ground[name] = dict(info, part_spec=by_id.get(info.get("part", "")))
        for link in spec["links"]:
            d.links.append(Link(link, d.ports(link["a"]), d.ports(link["b"])))
        return d

    def ports(self, ref: str) -> list[Port]:
        """ "fc.TELEM1" is one port; "fc.AUX1-4" expands to AUX1..AUX4."""
        name, port = ref.split(".", 1)
        inst = self.parts[name]
        m = re.fullmatch(r"([A-Z]+)(\d+)-(\d+)", port)
        if m:
            return [inst.port(f"{m.group(1)}{i}") for i in range(int(m.group(2)), int(m.group(3)) + 1)]
        return [inst.port(port)]

    def part(self, catalog_id: str) -> dict:
        return next(c for c in self.catalog["components"] if c["id"] == catalog_id)

    @property
    def fc(self) -> Instance:
        return next(i for i in self.parts.values() if i.part["role"] == "flight_controller")

    def link(self, link_id: str) -> Link:
        return next(link for link in self.links if link.id == link_id)


# ------------------------------------------------------------ harness ----


@dataclass
class Wire:
    a_pin: int | None
    a_signal: str
    b_pin: int | None
    b_signal: str
    role: str
    note: str = ""


@dataclass
class Cable:
    link: str
    a: str
    b: str
    a_connector: str
    b_connector: str
    wires: list[Wire]
    missing: list[str]
    notes: list[str]

    @property
    def custom(self) -> bool:
        return _family(self.a_connector) != _family(self.b_connector)


def _family(connector: str | None) -> str:
    c = (connector or "").upper()
    for fam in ("JST-GH", "MOLEX", "HEADER", "SOLDER", "RJ45", "USB", "SERVO"):
        if fam in c or (fam == "SERVO" and "0.1IN" in c) or (fam == "HEADER" and "2.54" in c):
            return fam
    return c or "unspecified"


def _find(port: Port, role: str) -> tuple[int, str] | None:
    for pin, raw, r in port.pins():
        if role in r:
            return pin, raw
    return None


# (role on side a, role on side b, required) per layer; the UART lines cross over
_PAIRS = {
    "uart": [("TX", "RX", True), ("RX", "TX", True), ("RTS", "CTS", False), ("CTS", "RTS", False), ("GND", "GND", True)],
    "can": [("CAN_H", "CAN_H", True), ("CAN_L", "CAN_L", True), ("GND", "GND", True)],
    "i2c": [("SCL", "SCL", True), ("SDA", "SDA", True), ("GND", "GND", True)],
}


def cable(link: Link) -> Cable | None:
    """Pin by pin wiring for a point to point link, crossing TX and RX, honouring who powers the device."""
    if link.layer not in _PAIRS or len(link.a) != 1 or len(link.b) != 1:
        return None
    a, b = link.a[0], link.b[0]
    if "usb" in (a.kind, b.kind):
        return None  # an off the shelf USB cable
    wires, missing, notes = [], [], []
    for ra, rb, required in _PAIRS[link.layer]:
        if ra in ("RTS", "CTS") and not link.spec.get("flow_control"):
            continue
        fa, fb = _find(a, ra), _find(b, rb)
        if fa and fb:
            wires.append(Wire(fa[0], fa[1], fb[0], fb[1], f"{ra} to {rb}"))
        elif required:
            missing.append(f"{a.ref if not fa else b.ref} has no {ra if not fa else rb} pin")
        elif fa or fb:
            notes.append(f"flow control: only one side has {ra if fa else rb}, leave it unconnected")
    powered = link.spec.get("powered_by", "fc")
    fa, fb = _find(a, "5V"), _find(b, "5V")
    if powered == "fc" and fa and fb:
        wires.append(Wire(fa[0], fa[1], fb[0], fb[1], "5V", "device powered by the flight controller"))
    elif fa and fb:
        notes.append(f"leave the 5V pins unconnected: {b.part} is powered by {powered}, sharing ground only")
    return Cable(link.id, a.ref, b.ref, a.spec.get("connector", ""), b.spec.get("connector", ""), wires, missing, notes)


def harness(design: Design) -> list[Cable]:
    return [c for c in (cable(link) for link in design.links) if c is not None]
