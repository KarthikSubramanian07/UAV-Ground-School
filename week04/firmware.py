"""Would it fly on PX4 or Betaflight? The firmware question, applied to this design.

The lecture asks how firmware (Betaflight, PX4) shapes what a flight
controller can do. The honest answer for one drone is a list: every link and
feature the Protocol Pro design relies on, and whether each firmware supports
it, with the parameter or driver that does it and a source for every cell.
The table lives in ``data/firmware.json`` (researched from each firmware's
own documentation and source); this module checks it covers the design and
turns it into a verdict per firmware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data" / "firmware.json"
FIRMWARES = ("ardupilot", "px4", "betaflight")
NAMES = {"ardupilot": "ArduPilot", "px4": "PX4", "betaflight": "Betaflight"}
LEVELS = ("yes", "partial", "no", "unconfirmed")


def load(path: str | Path | None = None) -> dict:
    data = json.loads(Path(path or DATA).read_text())
    problems = validate(data)
    if problems:
        raise ValueError("firmware table: " + "; ".join(problems))
    return data


def validate(data: dict) -> list[str]:
    """Every cell needs a support level, a one sentence explanation and an https source."""
    problems = []
    ids = [f["id"] for f in data["features"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate feature ids")
    for f in data["features"]:
        for fw in FIRMWARES:
            cell = f.get(fw)
            if not cell:
                problems.append(f"{f['id']}: no {fw} cell")
                continue
            if cell.get("support") not in LEVELS:
                problems.append(f"{f['id']}.{fw}: support must be one of {LEVELS}")
            if not cell.get("how"):
                problems.append(f"{f['id']}.{fw}: missing explanation")
            if not str(cell.get("source", "")).startswith("https://"):
                problems.append(f"{f['id']}.{fw}: source must be an https URL")
    return problems


def required_features(design) -> dict[str, str]:
    """Feature id -> why this design needs it, read from the design itself."""
    need = {"board": f"the flight controller is a {design.fc.part['name']}", "control": "every multirotor", "config": "every build"}
    by_link = {
        "telemetry": "telemetry",
        "companion_dds": "companion_dds",
        "companion_mavlink": "companion_mavlink",
        "gimbal": "gimbal",
        "rc": "rc",
        "can": "can",
        "lidar": "lidar",
        "motors": "motors",
    }
    for link in design.links:
        feature = by_link.get(link.id)
        if feature:
            need[feature] = f"link {link.id}: {link.spec.get('purpose', link.protocol)}"
    if "can" in need and "telemetry" in need:
        need["rtk"] = "RTK corrections go up the telemetry link and on to the DroneCAN GPS"
    if any(p["id"] == "ADSB_INTERNAL" for p in design.fc.part["ports"]):
        need["adsb"] = "the ADS-B receiver built into the carrier board"
    if any(f.get("from", "").startswith("pm.") and f.get("to", "").startswith("fc.POWER") for f in design.spec.get("power", [])):
        need["battery"] = "the PM02 power module on POWER1"
    need["autonomy"] = "a survey drone flies missions, a geofence and return to launch"
    return need


@dataclass
class Verdict:
    firmware: str
    supported: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)

    @property
    def flies(self) -> bool:
        """Without the board, the motors or the control loop nothing else matters."""
        return not {"board", "motors", "control"} & set(self.lost)

    def as_dict(self) -> dict:
        return {
            "firmware": NAMES[self.firmware],
            "flies": self.flies,
            "supported": self.supported,
            "partial": self.partial,
            "lost": self.lost,
            "unconfirmed": self.unconfirmed,
        }


def verdicts(design, data: dict | None = None) -> dict[str, Verdict]:
    data = data or load()
    features = {f["id"]: f for f in data["features"]}
    need = required_features(design)
    missing = sorted(set(need) - set(features))
    if missing:
        raise ValueError(f"the firmware table does not cover {missing}")
    out = {}
    for fw in FIRMWARES:
        v = Verdict(fw)
        bucket = {"yes": v.supported, "partial": v.partial, "no": v.lost, "unconfirmed": v.unconfirmed}
        for fid in need:
            bucket[features[fid][fw]["support"]].append(fid)
        out[fw] = v
    return out


SYMBOL = {"yes": "yes", "partial": "partly", "no": "no", "unconfirmed": "unconfirmed"}


def markdown(design, data: dict | None = None) -> str:
    data = data or load()
    need = required_features(design)
    v = verdicts(design, data)
    versions = data["versions"]
    lines = [
        "# Would Protocol Pro fly on PX4 or Betaflight?",
        "",
        "Generated by `python -m week04 docs` from [`week04/data/firmware.json`](../../week04/data/firmware.json). "
        f"Versions: {versions['ardupilot']}, PX4 {versions['px4']}, Betaflight {versions['betaflight']}.",
        "",
    ]
    for fw in FIRMWARES:
        r = v[fw]
        lines.append(
            f"* **{NAMES[fw]}**: {len(r.supported)} of {len(need)} features supported, {len(r.partial)} with changes, "
            f"{len(r.lost)} lost"
            + (f", {len(r.unconfirmed)} unconfirmed" if r.unconfirmed else "")
            + ("." if r.flies else ". It cannot fly this drone.")
        )
    lines += [
        "",
        "| Feature | Why this design needs it | " + " | ".join(NAMES[f] for f in FIRMWARES) + " |",
        "| --- | --- |" + " --- |" * 3,
    ]
    for f in data["features"]:
        if f["id"] not in need:
            continue
        cells = []
        for fw in FIRMWARES:
            c = f[fw]
            cells.append(f"**{SYMBOL[c['support']]}**: {c['how']} ([source]({c['source']}))")
        lines.append(f"| {f['feature']} | {need[f['id']]} | " + " | ".join(cells) + " |")
    if data.get("notes"):
        lines += ["", "Notes:", ""] + [f"* {n}" for n in data["notes"]]
    return "\n".join(lines) + "\n"


def report(design, data: dict | None = None) -> dict:
    data = data or load()
    return {
        "versions": data["versions"],
        "needed": required_features(design),
        "verdicts": {fw: v.as_dict() for fw, v in verdicts(design, data).items()},
    }
