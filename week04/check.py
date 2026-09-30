"""Design rule check: "think about communication protocols and ensure compatibility", as code.

Every rule looks at the design and the catalog and returns findings:

* ``error``: will not work (wrong protocol, pins that do not exist, a rail
  over its limit, an I2C address clash).
* ``warning``: works on paper but is marginal or needs an action (a custom
  cable, a firmware feature missing from the stable build, a tight margin).
* ``ok``: the rule was checked and passed, so the report proves coverage.

``tests/test_week04_check.py`` breaks the design one way at a time and makes
sure the matching rule catches it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from . import links as linkmod
from . import performance
from .design import LAYER_OF_KIND, PROTOCOLS, Design, Link, cable, harness
from .uart import UartConfig

LEVELS = ("error", "warning", "ok")


@dataclass
class Finding:
    level: str
    rule: str
    where: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


class Report(list):
    def add(self, level: str, rule: str, where: str, message: str):
        self.append(Finding(level, rule, where, message))

    def count(self, level: str) -> int:  # type: ignore[override]
        return sum(1 for f in self if f.level == level)

    @property
    def passed(self) -> bool:
        return self.count("error") == 0


# ------------------------------------------------------------ helpers ----


def _supply_range(design: Design, source: str) -> tuple[float, float]:
    """Voltage range available at a power source reference like "pm.BATT_OUT" or "bec12.VOUT"."""
    name, port = source.split(".")
    inst = design.parts[name]
    role = inst.part["role"]
    battery = design.parts["battery"].part["specs"]
    cells = battery["cells"]
    pack = (3.3 * cells, 4.2 * cells)  # loaded low cutoff to full charge
    if role in ("battery",) or (role == "power_module" and port == "BATT_OUT"):
        return pack
    if role == "power_module":
        return (5.1, 5.3)
    if role == "regulator":
        setting = inst.info.get("setting_v", inst.part["specs"].get("default_output_v", 5.2))
        needs = inst.part["specs"].get("min_input_for_12v_out_v", 14) - 12
        low = min(setting, pack[0] - needs) if setting > 6 else setting
        return (round(low * 0.99, 2), round(setting * 1.01, 2))
    if role == "flight_controller":
        return tuple(design.fc.part["specs"].get("rail_5v_range", [4.9, 5.5]))
    raise ValueError(f"{source} is not a power source")


def _load_amps(inst_part: dict, volts: float) -> tuple[float, float]:
    """(typical, peak) current a part draws at ``volts``."""
    s = inst_part.get("supply") or {}
    if s.get("typical_w"):
        return s["typical_w"] / volts, s.get("peak_w", s["typical_w"]) / volts
    typical = s.get("typical_a") or 0.0
    return typical, s.get("peak_a") or typical


def _consumer(design: Design, ref: str):
    return design.parts[ref.split(".")[0]]


# -------------------------------------------------------------- rules ----


def rule_ports_exist_and_layers_match(design: Design, r: Report):
    for link in design.links:
        for side in (link.a, link.b):
            for port in side:
                if port.spec.get("available") is False:
                    r.add(
                        "error", "port available", link.id, f"{port.ref} is not usable on this board ({port.spec.get('notes', '')[:120]})"
                    )
                kind_layer = LAYER_OF_KIND.get(port.kind)
                if link.protocol not in PROTOCOLS:
                    r.add("error", "known protocol", link.id, f"unknown protocol {link.protocol}")
                    return
                if kind_layer != link.layer and not (port.kind == "usb" and link.layer == "uart"):
                    r.add("error", "physical layer", link.id, f"{port.ref} is a {port.kind} port but {link.protocol} runs on {link.layer}")
        r.add("ok", "physical layer", link.id, f"{PROTOCOLS[link.protocol]['label']} over {link.layer}")


def rule_point_to_point(design: Design, r: Report):
    """UART is "exactly 2 devices" (slide 12); CAN and I2C are buses."""
    used = defaultdict(list)
    for link in design.links:
        for port in link.a + link.b:
            used[port.ref].append(link)
    for ref, links in used.items():
        layers = {link.layer for link in links}
        if len(links) > 1 and layers & {"uart", "pwm", "ethernet"}:
            r.add(
                "error",
                "one device per UART",
                ref,
                f"{ref} is used by {', '.join(link.id for link in links)}; a UART connects exactly two devices",
            )
    r.add("ok", "one device per UART", "all", f"{len(used)} ports, none double booked")


def rule_device_speaks_protocol(design: Design, r: Report):
    for link in design.links:
        for port in link.b:
            speaks = port.spec.get("speaks")
            if speaks is None:
                r.add("warning", "device protocol", link.id, f"catalog does not say which protocols {port.ref} speaks")
            elif link.protocol not in speaks and not (link.layer == "uart" and "mavlink2" in speaks and link.protocol == "mavlink1"):
                r.add("error", "device protocol", link.id, f"{port.ref} speaks {', '.join(speaks)}, not {link.protocol}")
            else:
                r.add("ok", "device protocol", link.id, f"{port.ref} speaks {link.protocol}")


def rule_baud(design: Design, r: Report):
    for link in design.links:
        if link.layer != "uart" or link.a[0].kind == "usb":
            continue
        proto = PROTOCOLS[link.protocol]
        baud = link.spec.get("baud")
        fixed = proto.get("fixed_baud")
        if fixed and baud != fixed:
            r.add("error", "baud rate", link.id, f"{link.protocol} runs at {fixed} baud, the link says {baud}")
        elif not baud:
            r.add("error", "baud rate", link.id, "no baud rate set")
        else:
            r.add("ok", "baud rate", link.id, f"{baud} baud on both ends")
        default = link.b[0].spec.get("default_baud") or _part_of(design, link.b[0]).get("specs", {}).get("default_serial_baud")
        if default and default != baud:
            r.add("warning", "baud rate", link.id, f"{link.b[0].ref} ships at {default} baud: reconfigure it to {baud}")


def _part_of(design: Design, port) -> dict:
    return design.parts[port.part].part


def rule_rx_dma(design: Design, r: Report):
    for link in design.links:
        if PROTOCOLS[link.protocol].get("needs_rx_dma"):
            port = link.a[0]
            if port.spec.get("rx_dma") is False:
                r.add(
                    "error",
                    "receive DMA",
                    link.id,
                    f"{port.ref} has no receive DMA; {link.protocol} at {link.spec.get('baud')} baud drops bytes there",
                )
            else:
                r.add("ok", "receive DMA", link.id, f"{port.ref} receives with DMA")


def rule_logic_levels(design: Design, r: Report):
    for link in design.links:
        if link.layer not in ("uart", "i2c", "can") or link.a[0].kind == "usb":
            continue
        if link.layer == "can":
            r.add("ok", "logic level", link.id, "CAN is differential; transceivers set the levels")
            continue
        a = link.a[0].spec.get("logic_v") or _part_of(design, link.a[0]).get("logic_v")
        b = link.b[0].spec.get("logic_v") or _part_of(design, link.b[0]).get("logic_v")
        if a is None or b is None:
            r.add("warning", "logic level", link.id, f"unknown signal level ({a} V and {b} V); measure before connecting")
        elif abs(a - b) > 0.5:
            r.add("error", "logic level", link.id, f"{a} V logic on {link.a[0].ref} but {b} V on {link.b[0].ref}: needs a level shifter")
        else:
            estimated = "logic_v" in _part_of(design, link.b[0]).get("estimates", {})
            r.add(
                "warning" if estimated else "ok",
                "logic level",
                link.id,
                f"{a} V to {b} V" + (" (device level is an estimate)" if estimated else ""),
            )


def rule_harness(design: Design, r: Report):
    for c in harness(design):
        for m in c.missing:
            r.add("error", "pins", c.link, m)
        if not c.missing:
            r.add(
                "ok",
                "pins",
                c.link,
                f"{len(c.wires)} wires, TX and RX crossed" if any(w.role == "TX to RX" for w in c.wires) else f"{len(c.wires)} wires",
            )
        if c.custom:
            r.add(
                "warning",
                "connectors",
                c.link,
                f"custom cable: {c.a_connector or 'unspecified'} to {c.b_connector or 'unspecified (vendor cable)'}",
            )
        pin_order = [w for w in c.wires if w.a_pin != w.b_pin]
        if pin_order and not c.custom:
            r.add("warning", "connectors", c.link, "same connector family but different pin order: do not use a straight through cable")


def rule_i2c(design: Design, r: Report):
    buses = defaultdict(list)
    for link in design.links:
        if link.layer == "i2c":
            buses[link.a[0].ref].append(link)
    internal = design.fc.part["specs"].get("internal_compass", {})
    for bus, links in buses.items():
        addresses = defaultdict(list)
        for link in links:
            addresses[link.spec.get("address")].append(link.b[0].ref)
            limit = _part_of(design, link.b[0]).get("specs", {}).get("i2c_max_hz")
            if limit and link.spec.get("clock_hz", 100000) > limit:
                r.add("error", "I2C clock", link.id, f"{link.spec['clock_hz']} Hz is above {link.b[0].ref}'s {limit} Hz")
        for addr, devices in addresses.items():
            if addr is None:
                r.add("error", "I2C address", bus, f"{devices} has no address")
            elif len(devices) > 1:
                r.add("error", "I2C address", bus, f"address 0x{addr:02X} used by {', '.join(devices)}")
            else:
                r.add(
                    "ok",
                    "I2C address",
                    bus,
                    f"0x{addr:02X} unique on {bus} (the internal compass at {internal.get('address', '0x0C')} is on a separate internal bus)",
                )


def rule_can(design: Design, r: Report):
    buses = defaultdict(list)
    for link in design.links:
        if link.layer == "can":
            buses[link.a[0].ref].append(link)
    for bus, links in buses.items():
        rates = {link.spec.get("bitrate", 1000000) for link in links}
        if len(rates) > 1:
            r.add("error", "CAN bitrate", bus, f"nodes disagree on bitrate: {sorted(rates)}")
        ends = [link for link in links if link.b[0].spec.get("terminated")]
        if not ends:
            r.add("warning", "CAN termination", bus, "no terminated node at the far end: add a 120 ohm terminator")
        else:
            r.add("ok", "CAN termination", bus, f"terminated at the autopilot and at {ends[-1].b[0].ref}")
        r.add("ok", "CAN node ids", bus, "the autopilot runs DroneCAN dynamic node allocation; no fixed ids to clash")
        # bus load from the messages a Here4 sends
        from . import dronecan

        load = 0.0
        for name, hz, sample in (
            ("uavcan.equipment.gnss.Fix2", 10, {"covariance": [0.0] * 6}),
            ("uavcan.equipment.ahrs.MagneticFieldStrength2", 25, {"magnetic_field_covariance": []}),
            ("uavcan.equipment.air_data.StaticPressure", 25, {}),
            ("uavcan.equipment.air_data.StaticTemperature", 2, {}),
            ("uavcan.protocol.NodeStatus", 2, {}),
            ("ardupilot.gnss.Status", 1, {}),
            ("uavcan.equipment.gnss.RTCMStream", 2, {"data": list(range(128))}),
        ):
            dtype = dronecan.load(name)
            load += linkmod_can_load(dronecan.bus_messages(dtype, dict(dtype.default(), **sample), hz), min(rates))
        level = "ok" if load < 0.3 else "warning"
        r.add(level, "CAN bus load", bus, f"{100 * load:.1f} percent at {min(rates) // 1000} kbit/s (worst case stuffing)")


def linkmod_can_load(messages, bitrate):
    from .can import bus_load

    return bus_load(messages, bitrate)


def rule_firmware_features(design: Design, r: Report):
    missing = set(design.fc.part.get("firmware", {}).get("missing_features", []))
    custom = "custom" in design.spec.get("firmware", "")
    for link in design.links:
        feature = PROTOCOLS[link.protocol].get("feature")
        if feature and feature in missing:
            level = "warning" if custom else "error"
            r.add(
                level,
                "firmware features",
                link.id,
                f"{link.protocol} needs {feature}, which the stable {design.fc.part['firmware']['stable']} build for this board leaves out: build custom firmware (custom.ardupilot.org) with it",
            )
    r.add("ok", "firmware features", "all", "every other protocol is in the stable build")


def rule_mavlink_channels(design: Design, r: Report):
    ports = mavlink_ports(design)
    limit = design.fc.part.get("firmware", {}).get("mavlink_channels", 8)
    if len(ports) > limit:
        r.add("error", "MAVLink channels", "fc", f"{len(ports)} MAVLink ports, the firmware has {limit} channels")
    r.add("ok", "MAVLink channels", "fc", "; ".join(f"SERIAL{s} ({p}) is MAV{i + 1}" for i, (s, p) in enumerate(ports)))


def mavlink_ports(design: Design) -> list[tuple[int, str]]:
    """Serial ports that speak MAVLink, in serial order: ArduPilot 4.7 numbers MAVn_* parameters this way."""
    out = {}
    for link in design.links:
        if link.protocol in ("mavlink1", "mavlink2"):
            port = link.a[0]
            out[port.spec.get("serial_index")] = port.id
    for p in design.fc.part["ports"]:  # built in MAVLink users such as the ADS-B receiver
        if p["id"] == "ADSB_INTERNAL":
            out[p["serial_index"]] = p["id"]
    return sorted(out.items())


def rule_bandwidth(design: Design, r: Report):
    for link in design.links:
        if not link.spec.get("streams") or not link.spec.get("baud"):
            continue
        budget = linkmod.design_budget(design, link)
        use = budget.utilisation
        level = "error" if use > 1.0 else "warning" if use > 0.7 else "ok"
        r.add(
            level,
            "bandwidth",
            link.id,
            f"{budget.bytes_per_second:.0f} B/s of {budget.config.bytes_per_second:.0f} B/s ({100 * use:.0f} percent at {budget.config.label()})",
        )
    crsf = UartConfig(420000)
    rc = 150 * 26 + 50 * 12  # 150 Hz RC frames down, a telemetry frame per 1:2 ratio back
    r.add(
        "ok",
        "bandwidth",
        "rc",
        f"CRSF at 150 Hz uses {rc} B/s of {crsf.bytes_per_second:.0f} B/s ({100 * rc / crsf.bytes_per_second:.0f} percent)",
    )


def rule_power(design: Design, r: Report):
    fc = design.fc
    groups = fc.part["specs"]["supply_groups"]
    typical, peak = defaultdict(float), defaultdict(float)
    for link in design.links:
        if link.spec.get("powered_by", "fc") != "fc" or link.layer not in ("uart", "can", "i2c"):
            continue
        port = link.a[0]
        group = port.spec.get("supply_group")
        dev = _part_of(design, link.b[0])
        t, p = _load_amps(dev, 5.0)
        typical[group] += t
        peak[group] += p
        lo, hi = _supply_range(design, "fc.POWER1")
        s = dev.get("supply") or {}
        if s.get("min_v") and s.get("max_v"):
            if lo < s["min_v"] or hi > s["max_v"]:
                level = "warning" if lo >= s["min_v"] - 0.2 and hi <= s["max_v"] + 0.5 else "error"
                r.add(
                    level,
                    "supply voltage",
                    link.id,
                    f"{link.b[0].ref} wants {s['min_v']} to {s['max_v']} V; the Cube's 5 V rail spans {lo} to {hi} V",
                )
            else:
                r.add("ok", "supply voltage", link.id, f"{lo} to {hi} V inside {s['min_v']} to {s['max_v']} V")
    for group, amps in typical.items():
        spec = groups.get(group, {})
        limit, peak_limit = spec.get("limit_a", 1.0), spec.get("peak_a", spec.get("limit_a", 1.0))
        if amps > limit or peak[group] > peak_limit:
            r.add("error", "5 V budget", f"fc {group}", f"{amps:.2f} A typical, {peak[group]:.2f} A peak on a {limit} A group")
        else:
            r.add("ok", "5 V budget", f"fc {group}", f"{amps:.2f} A typical, {peak[group]:.2f} A peak of {limit} A ({peak_limit} A peak)")
    total = sum(peak.values())
    if total > groups["total"]["limit_a"]:
        r.add("error", "5 V budget", "fc total", f"{total:.2f} A peak over the {groups['total']['limit_a']} A peripheral budget")
    # radios must never draw from the carrier
    for link in design.links:
        dev = _part_of(design, link.b[0])
        if dev["role"] == "telemetry_radio":
            if link.spec.get("powered_by", "fc") == "fc":
                r.add(
                    "error",
                    "radio power",
                    link.id,
                    "CubePilot: never power radio devices off the carrier board; use a separate 5 V BEC and share ground",
                )
            else:
                r.add("ok", "radio power", link.id, f"radio powered by {link.spec['powered_by']}, ground shared")
    # every power feed: voltage window and regulator load
    loads = defaultdict(float)
    for feed in design.spec.get("power", []):
        src, dst = feed["from"], feed["to"]
        dst_inst = _consumer(design, dst)
        if dst_inst.part["role"] in ("power_module",):
            continue
        lo, hi = _supply_range(design, src)
        s = dst_inst.part.get("supply") or {}
        if dst_inst.part["role"] == "flight_controller":
            s = {"min_v": fc.part["supply"]["min_v"], "max_v": fc.part["supply"]["max_v"]}
        if s.get("min_v") is not None and s.get("max_v") is not None:
            margin = min(lo - s["min_v"], s["max_v"] - hi)
            if margin < 0:
                r.add("error", "supply voltage", feed["id"], f"{dst} needs {s['min_v']} to {s['max_v']} V but {src} gives {lo} to {hi} V")
            elif margin < 0.5:
                r.add(
                    "warning",
                    "supply voltage",
                    feed["id"],
                    f"{dst}: {lo} to {hi} V against {s['min_v']} to {s['max_v']} V, only {margin:.2f} V margin",
                )
            else:
                r.add("ok", "supply voltage", feed["id"], f"{dst}: {lo} to {hi} V inside {s['min_v']} to {s['max_v']} V")
        if _consumer(design, src).part["role"] == "regulator" and dst_inst.part["role"] not in ("regulator",):
            loads[src] += _load_amps(dst_inst.part, lo)[1]
    for src, amps in loads.items():
        rating = _consumer(design, src).part["specs"].get("continuous_a", 0)
        level = "ok" if amps <= 0.8 * rating else "warning" if amps <= rating else "error"
        r.add(level, "regulator load", src, f"{amps:.2f} A peak of {rating} A continuous")
    # the brick that powers the Cube itself
    brick = design.parts["pm"].part["specs"].get("bec_output", "")
    fc_a = fc.part["specs"].get("fmu_plus_io_power_budget_a", 0.55) + sum(peak.values())
    r.add(
        "ok" if fc_a < 3.0 else "error",
        "regulator load",
        "pm.FC_POWER",
        f"{fc_a:.2f} A for the Cube and its peripherals from the PM02 {brick}",
    )


def rule_propulsion(design: Design, r: Report):
    perf = performance.analyse(design)
    tw = perf.thrust_to_weight
    r.add(
        "ok" if tw >= 2.0 else "warning" if tw >= 1.6 else "error",
        "thrust to weight",
        "airframe",
        f"{tw:.2f} at {perf.mass_kg:.2f} kg (2.0 or more leaves control authority in wind)",
    )
    hover = perf.hover_throttle
    r.add(
        "ok" if hover <= 0.65 else "warning" if hover <= 0.75 else "error",
        "hover throttle",
        "airframe",
        f"{100 * hover:.0f} percent throttle to hover",
    )
    esc_limit = design.parts["esc"].part["specs"]["current_rating_a"]
    motor_a = perf.motor_full_throttle_a
    r.add(
        "ok" if motor_a <= 0.9 * esc_limit else "warning" if motor_a <= esc_limit else "error",
        "ESC current",
        "esc",
        f"{motor_a:.1f} A per motor at full throttle on a {esc_limit} A ESC",
    )
    battery = design.parts["battery"].part
    cont = battery["specs"]["c_rating"] * battery["specs"]["capacity_mah"] / 1000
    r.add(
        "ok" if perf.full_throttle_current_a < cont else "error",
        "battery C rating",
        "battery",
        f"{perf.full_throttle_current_a:.0f} A at full throttle of {cont:.0f} A continuous",
    )
    pm = design.parts["pm"].part["specs"]
    r.add(
        "ok"
        if perf.full_throttle_current_a <= pm["continuous_a"]
        else "warning"
        if perf.full_throttle_current_a <= pm["burst_a"]
        else "error",
        "power module rating",
        "pm",
        f"{perf.full_throttle_current_a:.0f} A at full throttle, {perf.hover_current_a + perf.avionics_w / 14.8:.0f} A at hover; PM02 V3 is {pm['continuous_a']} A continuous, {pm['burst_a']} A burst",
    )
    xt60 = 30.0
    hover_a = perf.hover_current_a + perf.avionics_w / 14.8
    level = "ok" if perf.full_throttle_current_a <= xt60 else "warning" if hover_a <= xt60 else "error"
    r.add(
        level,
        "connector rating",
        "battery.MAIN",
        f"XT60 with 12 AWG is rated {xt60:.0f} A continuous (Holybro): {hover_a:.0f} A at hover, {perf.full_throttle_current_a:.0f} A in full throttle bursts",
    )
    r.add(
        "ok" if perf.hover_minutes >= 10 else "warning",
        "flight time",
        "battery",
        f"{perf.hover_minutes:.1f} min hover to 20 percent charge",
    )


def rule_dshot(design: Design, r: Report):
    for link in design.links:
        rate = PROTOCOLS[link.protocol].get("dshot")
        if not rate:
            continue
        for port in link.a:
            notes = port.spec.get("notes", "")
            if port.id.startswith("MAIN") and "BRD_IO_DSHOT" in notes:
                r.add("warning", "DShot outputs", link.id, f"{port.ref} is on the IOMCU: DShot needs BRD_IO_DSHOT=1 and costs PPM input")
        rates = _part_of(design, link.b[0]).get("specs", {}).get("dshot_rates", [])
        if rate not in rates:
            r.add("error", "DShot rate", link.id, f"DShot{rate} is not supported by {link.b[0].ref} (supports {rates})")
        else:
            r.add("ok", "DShot rate", link.id, f"DShot{rate} on {', '.join(p.id for p in link.a)} (FMU timers, no IOMCU firmware needed)")
        groups = {"AUX1-4": {"AUX1", "AUX2", "AUX3", "AUX4"}, "AUX5-6": {"AUX5", "AUX6"}}
        used = {p.id for p in link.a}
        for name, members in groups.items():
            if used & members and not members <= used:
                r.add("warning", "timer groups", link.id, f"{name} share a timer: every output in the group must use DShot{rate}")


def rule_radio(design: Design, r: Report):
    bands = []
    for w in design.spec.get("wireless", []):
        bands.append((w["id"], *w["band_mhz"]))
        if w.get("tx_dbm") is not None and w.get("sensitivity_dbm") is not None:
            f = (w["band_mhz"][0] + w["band_mhz"][1]) / 2 * 1e6
            rng = linkmod.max_range_m(w["tx_dbm"], w["gain_dbi"][0], w["gain_dbi"][1], w["sensitivity_dbm"], f)
            r.add("ok", "radio range", w["id"], f"{rng / 1000:.1f} km free space with a 10 dB fade margin")
    for i, (a, lo_a, hi_a) in enumerate(bands):
        for b, lo_b, hi_b in bands[i + 1 :]:
            if lo_a <= hi_b and lo_b <= hi_a:
                r.add("warning", "band overlap", f"{a} and {b}", "share a band: keep antennas apart")
    if not any(w["id"] == "wifi" and w["band_mhz"][0] < 2500 for w in design.spec.get("wireless", [])):
        r.add("ok", "band overlap", "wifi", "Jetson WiFi on 5 GHz keeps 2.4 GHz clear for ExpressLRS")


def rule_known_issues(design: Design, r: Report):
    for inst in design.parts.values():
        notes = inst.part.get("notes", "")
        if "JetPack 7" in notes or inst.part["id"].startswith("jetson"):
            r.add(
                "warning",
                "known issues",
                inst.name,
                "open NVIDIA report: UART1 TX does not drive on JetPack 7 (L4T R39.2); stay on JetPack 6 until fixed",
            )
            break
    r.add("warning", "known issues", "lidar", "TFmini-S ships in UART mode: send the I2C mode command once before wiring it to I2C")


RULES = [
    rule_ports_exist_and_layers_match,
    rule_point_to_point,
    rule_device_speaks_protocol,
    rule_baud,
    rule_rx_dma,
    rule_logic_levels,
    rule_harness,
    rule_i2c,
    rule_can,
    rule_firmware_features,
    rule_mavlink_channels,
    rule_bandwidth,
    rule_power,
    rule_dshot,
    rule_propulsion,
    rule_radio,
    rule_known_issues,
]


def check(design: Design) -> Report:
    report = Report()
    for rule in RULES:
        try:
            rule(design, report)
        except Exception as exc:  # a rule that crashes is itself a finding
            report.add("error", rule.__name__.removeprefix("rule_").replace("_", " "), "design", f"could not evaluate: {exc!r}")
    return report


def link_of(design: Design, link_id: str) -> Link:
    return design.link(link_id)


__all__ = ["Finding", "Report", "check", "cable", "mavlink_ports"]
