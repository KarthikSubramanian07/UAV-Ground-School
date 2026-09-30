"""The wiring diagram, drawn from the design (no hand placed wires).

Parts sit where ``design.json`` puts them. Every flight controller link
becomes an orthogonal wire from a pin on the controller's edge facing the
part. Wires on one side get their own vertical lane, the lowest pin taking
the outermost lane, so no two data wires cross or overlap. Power runs on a
separate trunk below the controller. Each wire and card carries
``data-link`` or ``data-part`` so the showcase page can highlight it.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape

from .design import Design

W, H = 1200, 820
FC_W, FC_H = 250, 230
CARD_W, CARD_H = 190, 74
LANE0, LANE_STEP = 64, 40

COLORS = {
    "mavlink2": "#f08a3c",
    "dds_xrce": "#b58cff",
    "siyi": "#4cc9d8",
    "crsf": "#6ad38a",
    "dronecan": "#f2c94c",
    "i2c": "#ff7eb6",
    "dshot300": "#eceff1",
    "ethernet": "#6ea8ff",
    "power": "#d9534f",
    "power_5v": "#d9a441",
}

SHORT = {
    "mavlink2": "MAVLink 2",
    "dds_xrce": "DDS",
    "siyi": "SIYI",
    "crsf": "CRSF",
    "dronecan": "DroneCAN",
    "i2c": "I2C",
    "dshot300": "DShot300",
    "ethernet": "Ethernet RTSP",
}


def wire_label(link) -> str:
    s = link.spec
    if s.get("baud"):
        return f"{SHORT[link.protocol]} {s['baud']}"
    if s.get("bitrate"):
        return f"{SHORT[link.protocol]} {s['bitrate'] // 1000}k"
    if s.get("address") is not None:
        return f"{SHORT[link.protocol]} 0x{s['address']:02X}"
    if link.a[0].kind == "usb":
        return f"{SHORT[link.protocol]} USB"
    return SHORT.get(link.protocol, link.protocol)


def _side(fc_xy, xy) -> str:
    dx, dy = xy[0] - fc_xy[0], xy[1] - fc_xy[1]
    if abs(dx) > FC_W / 2 + 40:
        return "left" if dx < 0 else "right"
    return "top" if dy < 0 else "bottom"


STYLE = (
    ".wiring{background:#12191d}.grid{stroke:#1b252a;stroke-width:1}"
    ".card rect{fill:#1a242a;stroke:#34444c;stroke-width:1.2}.card .t{fill:#eef2f3;font-size:14px;font-weight:600}"
    ".card .s{fill:#8fa3ab;font-size:10.5px}.fc rect{fill:#e86f2a;stroke:#ffb27a}.fc .t{fill:#1b1109;font-size:19px}"
    ".fc .s{fill:#3a1f0c}.pin{font-size:10px;fill:#2a1407;font-weight:600}"
    ".wire{fill:none;stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}.wire.power{stroke-width:4}"
    ".lbl{font-size:10.5px;fill:#dfe6e9;paint-order:stroke;stroke:#12191d;stroke-width:4px}"
    ".dot{stroke:#12191d;stroke-width:2}.ant{fill:none;stroke:#8fa3ab;stroke-width:1.3;stroke-dasharray:3 3}"
    ".legend text{font-size:11px;fill:#cfd8dc}"
)


def render(design: Design) -> str:
    fc = design.fc
    fx, fy = fc.info["x"], fc.info["y"]
    left, right, top, bottom = fx - FC_W / 2, fx + FC_W / 2, fy - FC_H / 2, fy + FC_H / 2
    parts = {n: i for n, i in design.parts.items() if "x" in i.info}
    xy = {n: (i.info["x"], i.info["y"]) for n, i in parts.items()}

    wires: list[str] = []
    labels: list[str] = []
    pins: list[str] = []

    def wire(link_id: str, color: str, d: str, end: tuple[float, float] | None = None, power: bool = False):
        cls = "wire power" if power else "wire"
        wires.append(f'<path class="{cls}" data-link="{link_id}" stroke="{color}" d="{d}"/>')
        if end:
            wires.append(f'<circle class="dot" data-link="{link_id}" cx="{end[0]:.0f}" cy="{end[1]:.0f}" r="4" fill="{color}"/>')

    def label(link_id: str, x: float, y: float, text: str, anchor: str = "middle"):
        labels.append(f'<text class="lbl" data-link="{link_id}" x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}">{escape(text)}</text>')

    # ------------------------------------------------ flight controller ----
    fc_links = [lk for lk in design.links if lk.a[0].part == fc.name and lk.b[0].part in parts]
    sides: dict[str, list] = defaultdict(list)
    for lk in fc_links:
        sides[_side((fx, fy), xy[lk.b[0].part])].append(lk)

    for side, links in sides.items():
        horizontal = side in ("left", "right")
        links.sort(key=lambda lk: xy[lk.b[0].part][1] if horizontal else xy[lk.b[0].part][0])
        n = len(links)
        per_card = defaultdict(list)
        for lk in links:
            per_card[lk.b[0].part].append(lk)
        for i, lk in enumerate(links):
            color = COLORS.get(lk.protocol, "#cccccc")
            f = (i + 1) / (n + 1)
            px, py = xy[lk.b[0].part]
            siblings = per_card[lk.b[0].part]
            k = siblings.index(lk)
            spread = (k - (len(siblings) - 1) / 2) * 22
            pid = lk.a[0].id if len(lk.a) == 1 else f"{lk.a[0].id}-{lk.a[-1].id[-1]}"
            if horizontal:
                ay = top + f * FC_H
                ax = left if side == "left" else right
                sign = -1 if side == "left" else 1
                lane = ax + sign * (LANE0 + LANE_STEP * i)
                ex = px - sign * CARD_W / 2
                ey = py + spread
                if abs(ey - ay) < 12 and len(siblings) == 1:
                    ey = ay
                d = f"M{ax:.0f},{ay:.0f} H{ex:.0f}" if abs(ey - ay) < 1 else f"M{ax:.0f},{ay:.0f} H{lane:.0f} V{ey:.0f} H{ex:.0f}"
                wire(lk.id, color, d, (ex, ey))
                label(lk.id, (lane + ex) / 2 if abs(ey - ay) >= 1 else (ax + ex) / 2, ey - 7, wire_label(lk))
                pins.append(
                    f'<text class="pin" x="{ax - sign * 8:.0f}" y="{ay + 3.5:.0f}" text-anchor="{"start" if side == "left" else "end"}">{escape(pid)}</text>'
                )
            else:
                ax = left + f * FC_W
                ay = top if side == "top" else bottom
                ey = py + (CARD_H / 2 if side == "top" else -CARD_H / 2)
                ex = px + spread
                if abs(ex - ax) < 1:
                    d = f"M{ax:.0f},{ay:.0f} V{ey:.0f}"
                else:
                    mid = (ay + ey) / 2
                    d = f"M{ax:.0f},{ay:.0f} V{mid:.0f} H{ex:.0f} V{ey:.0f}"
                wire(lk.id, color, d, (ex, ey))
                ly = ey - 22 if side == "bottom" else (ay + ey) / 2 + 4
                label(lk.id, ex + 10, ly, wire_label(lk), "start")
                pins.append(
                    f'<text class="pin" x="{ax:.0f}" y="{ay + (16 if side == "top" else -8):.0f}" text-anchor="middle">{escape(pid)}</text>'
                )

    # links between two peripherals (the camera's Ethernet to the Jetson)
    for lk in design.links:
        a, b = lk.a[0].part, lk.b[0].part
        if a == fc.name or a not in parts or b not in parts:
            continue
        (x1, y1), (x2, y2) = xy[a], xy[b]
        color = COLORS.get(lk.protocol, "#cccccc")
        if abs(x1 - x2) < 1:
            ya, yb = (y1 + CARD_H / 2, y2 - CARD_H / 2) if y1 < y2 else (y1 - CARD_H / 2, y2 + CARD_H / 2)
            wire(lk.id, color, f"M{x1 + 40:.0f},{ya:.0f} V{yb:.0f}", (x1 + 40, yb))
            label(lk.id, x1 + 50, (ya + yb) / 2 + 4, wire_label(lk), "start")

    # ------------------------------------------------------------ power ----
    pmx, pmy = xy["pm"]
    trunk_y = pmy + 20
    fc_power_y = pmy - 20
    for feed in design.spec.get("power", []):
        a, b = feed["from"].split(".")[0], feed["to"].split(".")[0]
        if a not in xy or b not in xy or (a == "pm" and b == "camera"):
            continue  # the camera feed is drawn up the left margin below
        (x1, y1), (x2, y2) = xy[a], xy[b]
        color = COLORS["power"]
        if b == fc.name:
            px = left + 30
            wire(
                f"power:{feed['id']}",
                COLORS["power_5v"],
                f"M{x1 + CARD_W / 2:.0f},{fc_power_y:.0f} H{px:.0f} V{bottom:.0f}",
                (px, bottom),
                True,
            )
            label(f"power:{feed['id']}", (x1 + CARD_W / 2 + px) / 2, fc_power_y - 7, "5.2 V + sensing")
            pins.append(f'<text class="pin" x="{px:.0f}" y="{bottom - 8:.0f}" text-anchor="middle">POWER1</text>')
        elif abs(x1 - x2) < 1:  # stacked cards in a column
            ya, yb = (y1 + CARD_H / 2, y2 - CARD_H / 2) if y1 < y2 else (y1 - CARD_H / 2, y2 + CARD_H / 2)
            wire(f"power:{feed['id']}", color, f"M{x1 - 40:.0f},{ya:.0f} V{yb:.0f}", None, True)
        elif a == "pm":  # along the trunk, then down to the part
            drop = x2 + 40
            wire(f"power:{feed['id']}", color, f"M{x1 + CARD_W / 2:.0f},{trunk_y:.0f} H{drop:.0f} V{y2 - CARD_H / 2:.0f}", None, True)
        else:
            wire(f"power:{feed['id']}", color, f"M{x1:.0f},{y1 - CARD_H / 2:.0f} V{y2 + CARD_H / 2:.0f}", None, True)
    # the camera takes the pack voltage up the left margin
    for feed in design.spec.get("power", []):
        if feed["from"] == "pm.BATT_OUT" and feed["to"].startswith("camera"):
            cx, cy = xy["camera"]
            x = cx - CARD_W / 2 - 18
            wire(
                f"power:{feed['id']}",
                COLORS["power"],
                f"M{pmx - CARD_W / 2:.0f},{pmy:.0f} H{x:.0f} V{cy:.0f} H{cx - CARD_W / 2:.0f}",
                None,
                True,
            )
    labels.append(f'<text class="lbl" x="{(pmx + CARD_W / 2 + 12):.0f}" y="{trunk_y + 16:.0f}">battery 13.2 to 16.8 V</text>')

    # ------------------------------------------------------------ draw ----
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" class="wiring" role="img" aria-labelledby="wiring-title wiring-desc" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">',
        '<title id="wiring-title">Protocol Pro wiring diagram</title>',
        f'<desc id="wiring-desc">{escape(design.spec["summary"])}</desc>',
        f"<style>{STYLE}</style>",
        f'<rect width="{W}" height="{H}" fill="#12191d"/>',
    ]
    out += [f'<line class="grid" x1="{x}" y1="0" x2="{x}" y2="{H}"/>' for x in range(0, W + 1, 40)]
    out += [f'<line class="grid" x1="0" y1="{y}" x2="{W}" y2="{y}"/>' for y in range(0, H + 1, 40)]
    out += [w for w in wires if "power" in w] + [w for w in wires if "power" not in w]

    for name, inst in parts.items():
        x, y = xy[name]
        if name == fc.name:
            out.append(f'<g class="card fc" data-part="{name}"><rect x="{left}" y="{top}" width="{FC_W}" height="{FC_H}" rx="14"/>')
            out.append(f'<text class="t" x="{x}" y="{y - 4}" text-anchor="middle">{escape(inst.label)}</text>')
            out.append(f'<text class="s" x="{x}" y="{y + 16}" text-anchor="middle">ArduCopter 4.7.1, ADS-B carrier</text>')
            out.append(f'<text class="s" x="{x}" y="{y + 31}" text-anchor="middle">ADS-B IN on SERIAL5</text></g>')
            continue
        count = f"{inst.count} x " if inst.count > 1 else ""
        out.append(
            f'<g class="card" data-part="{name}"><rect x="{x - CARD_W / 2}" y="{y - CARD_H / 2}" width="{CARD_W}" height="{CARD_H}" rx="10"/>'
        )
        out.append(f'<text class="t" x="{x}" y="{y - 5}" text-anchor="middle">{escape(count + inst.label)}</text>')
        out.append(f'<text class="s" x="{x}" y="{y + 14}" text-anchor="middle">{escape(inst.part.get("vendor", ""))}</text></g>')
    out += pins + labels

    for name, text in (("radio", "915 MHz to ground"), ("rx", "2.4 GHz from the Boxer"), ("gnss", "GNSS L1 and L5")):
        if name not in xy:
            continue
        x, y = xy[name]
        if name == "gnss":
            ax, ay = x + CARD_W / 2 + 6, y
            out += [f'<path class="ant" d="M{ax + 4},{ay - r} A{r},{r} 0 0 1 {ax + 4},{ay + r}"/>' for r in (10, 18, 26)]
            out.append(f'<text class="lbl" x="{ax + 36}" y="{ay + 4}">{text}</text>')
        else:
            ax, ay = x + CARD_W / 2 - 24, y - CARD_H / 2
            out += [f'<path class="ant" d="M{ax - r},{ay - 4} A{r},{r} 0 0 1 {ax + r},{ay - 4}"/>' for r in (10, 18, 26)]
            out.append(f'<text class="lbl" x="{ax - 34}" y="{ay - 12}" text-anchor="end">{text}</text>')

    out.append(f'<g class="legend" transform="translate(372,{H - 52})">')
    items = [
        ("MAVLink", "mavlink2"),
        ("DDS", "dds_xrce"),
        ("CRSF", "crsf"),
        ("SIYI", "siyi"),
        ("DroneCAN", "dronecan"),
        ("I2C", "i2c"),
        ("DShot", "dshot300"),
        ("Ethernet", "ethernet"),
        ("Battery", "power"),
        ("5 V", "power_5v"),
    ]
    for i, (name, key) in enumerate(items):
        x, y = (i % 5) * 96, (i // 5) * 20
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{COLORS[key]}" stroke-width="3.5" stroke-linecap="round"/>')
        out.append(f'<text x="{x + 28}" y="{y + 4}">{name}</text>')
    out.append("</g></svg>")
    return "\n".join(out)
