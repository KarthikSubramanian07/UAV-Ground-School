"""Regenerate docs/week04: every figure, table and data file the README and the showcase page use.

``python -m week04 docs`` rebuilds everything that does not need ArduPilot;
``python -m week04 docs --sitl`` also reruns the SITL experiments (a few
minutes) and refreshes ``sitl.json`` and the flight capture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import check, dshot, i2c, journey, links, params, performance, pid, rc, uart
from .can import CanFrame
from .design import Design, harness
from .diagram import render


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    else:
        path.write_text(data)
    print(f"wrote {path}")


# ------------------------------------------------------------ tables ----


def bom(design: Design) -> dict:
    rows = []
    for name, inst in list(design.parts.items()):
        p = inst.part
        rows.append(
            {
                "part": name,
                "name": p["name"] if "setting_v" not in inst.info else f"{p['name'].split(' (')[0]}, set to {inst.info['setting_v']:g} V",
                "role": p["role"],
                "count": inst.count,
                "unit_usd": 0.0 if inst.info.get("included_in") else p.get("price_usd"),
                "included_in": inst.info.get("included_in"),
                "mass_g": p.get("mass_g"),
                "estimated": sorted(p.get("estimates", {})),
                "url": p.get("url"),
            }
        )
    total = sum((r["unit_usd"] or 0) * r["count"] for r in rows)
    return {"rows": rows, "total_usd": round(total, 2), "unpriced": [r["part"] for r in rows if r["unit_usd"] is None]}


def bom_md(b: dict) -> str:
    lines = ["| Part | Role | Qty | Unit price | Mass | Source |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in b["rows"]:
        price = (
            f"in the {r['included_in']} kit"
            if r.get("included_in")
            else f"${r['unit_usd']:.2f}"
            if r["unit_usd"] is not None
            else "not found"
        )
        mass = f"{r['mass_g']:g} g" + (" (est.)" if "mass_g" in r["estimated"] else "") if r["mass_g"] is not None else ""
        lines.append(f"| {r['name']} | {r['role'].replace('_', ' ')} | {r['count']} | {price} | {mass} | [link]({r['url']}) |")
    lines.append(f"| **Total (airborne)** | | | **${b['total_usd']:.2f}** | | |")
    return "\n".join(lines) + "\n"


def check_md(report) -> str:
    lines = [
        f"{report.count('error')} errors, {report.count('warning')} warnings, {report.count('ok')} checks passed.",
        "",
        "| Level | Rule | Where | Finding |",
        "| --- | --- | --- | --- |",
    ]
    order = {"error": 0, "warning": 1, "ok": 2}
    for f in sorted(report, key=lambda f: order[f.level]):
        lines.append(f"| {f.level} | {f.rule} | `{f.where}` | {f.message} |")
    return "\n".join(lines) + "\n"


def harness_md(cables) -> str:
    out = []
    for c in cables:
        out.append(f"### `{c.link}`: {c.a} to {c.b}\n")
        out.append(f"{c.a_connector} to {c.b_connector or 'vendor cable'}" + (" (custom cable)" if c.custom else "") + "\n")
        out.append("| From pin | Signal | To pin | Signal | Note |\n| ---: | --- | ---: | --- | --- |")
        for w in c.wires:
            out.append(f"| {w.a_pin} | {w.a_signal} | {w.b_pin} | {w.b_signal} | {w.note} |")
        for n in c.notes:
            out.append(f"\n{n[0].upper() + n[1:]}.")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------- scope ----


def scope() -> dict:
    """Short, annotated captures of each protocol for the page's logic analyser."""
    out = {}
    cfg = uart.UartConfig(57600)
    frame = bytes([0xFD, 0x09])
    out["uart"] = {
        "title": "UART 57600 8N1: the first two bytes of a MAVLink 2 frame",
        "levels": uart.encode(frame, cfg, idle_bits=2),
        "fields": [[0, 1, "start"], [1, 9, "0xFD LSB first"], [9, 10, "stop"], [12, 13, "start"], [13, 21, "0x09"], [21, 22, "stop"]],
        "unit": f"{cfg.bit_time_us:.2f} us per bit",
    }
    sb = rc.SbusFrame([rc.us_to_raw(1500)] * 16).encode()[:2]
    lv = uart.encode(sb, uart.SBUS)
    out["sbus"] = {
        "title": "SBUS 100000 8E2 inverted: header 0x0F, then channel 1 begins",
        "levels": lv,
        "fields": [
            [0, 1, "start"],
            [1, 9, "0x0F"],
            [9, 10, "parity"],
            [10, 12, "stop"],
            [12, 13, "start"],
            [13, 21, "ch1 bits 0 to 7"],
            [21, 22, "parity"],
            [22, 24, "stop"],
        ],
        "unit": "10 us per bit",
    }
    crsf = rc.crsf_rc([rc.us_to_raw(1500)] * 16)
    out["crsf"] = {
        "title": "CRSF RC channels frame (26 bytes at 420000 baud)",
        "bytes": crsf.hex(),
        "fields": [
            [0, 1, "sync 0xC8"],
            [1, 2, "length 24"],
            [2, 3, "type 0x16"],
            [3, 25, "16 x 11 bit channels"],
            [25, 26, "CRC-8/DVB-S2"],
        ],
    }
    f = dshot.encode(dshot.throttle_value(0.25))
    out["dshot"] = {
        "title": f"DShot600, throttle 25 percent (value {f >> 5}, frame 0x{f:04X})",
        "levels": dshot.waveform(f, 600, 8),
        "samples_per_bit": 8,
        "fields": [[0, 88, "11 bit throttle"], [88, 96, "telemetry"], [96, 128, "CRC"]],
        "unit": "1.67 us per bit",
    }
    cf = CanFrame(0x1004260A, bytes.fromhex("4fea03d300133e80"), extended=True)
    header = cf.header_bits()
    stuffed = cf.stuffed_bits()
    # indices of stuff bits in the stuffed stream
    marks, run_bit, run, j = [], -1, 0, 0
    for b in header + [(cf.crc() >> (14 - k)) & 1 for k in range(15)]:
        j += 1
        if b == run_bit:
            run += 1
        else:
            run_bit, run = b, 1
        if run == 5:
            marks.append(j)
            j += 1
            run_bit, run = 1 - b, 1
    out["can"] = {
        "title": "CAN 2.0B: first frame of a DroneCAN RTCMStream transfer",
        "levels": cf.wire_bits(),
        "stuff_bits": marks,
        "fields": [
            [0, 1, "SOF"],
            [1, 12, "ID 28..18"],
            [12, 14, "SRR IDE"],
            [14, 32, "ID 17..0"],
            [32, 35, "RTR r1 r0"],
            [35, 39, "DLC 8"],
            [39, 103, "data"],
            [103, 118, "CRC-15"],
        ],
        "fields_are_unstuffed": True,
        "stuffed_length": len(stuffed),
        "unit": "1 us per bit",
    }
    t = i2c.transaction(0x10, write=bytes([0x00]), read=bytes([0x59, 0x59]))
    out["i2c"] = {"title": "I2C: read two bytes from the TFmini-S at 0x10", "sda": t.sda, "scl": t.scl, "segments": t.segments}
    return out


# --------------------------------------------------------------- PID ----


def _down(xs, step=2, nd=4):
    return [round(v, nd) for v in xs[::step]]


def pid_data(design: Design, sitl_result: dict | None) -> dict:
    from .cli import plant_for

    base = pid.Plant()
    sc = pid.Scenario()
    presets = {}
    for name, g in pid.PRESETS.items():
        tr = pid.simulate(g, base, sc)
        m = pid.metrics(tr, sc)
        presets[name] = {
            "gains": {"kp": g.kp, "ki": g.ki, "kd": g.kd},
            "angle_deg": _down([math.degrees(a) for a in tr.angle], 2, 3),
            "metrics": m.as_dict(),
            "damping": pid.damping_ratio(g, base),
        }
    ku, tu = pid.ultimate_gain(base)
    zn = pid.ziegler_nichols(base)
    tuned = pid.tune(base)
    airframe = plant_for(design)
    t = airframe.tau_max
    defaults = pid.Gains(0.135 * t, 0.135 * t, 0.0036 * t, mode="cascade", angle_p=4.5)
    cascade_tuned = pid.tune_cascade(airframe)
    step_sc = pid.Scenario(duration=2.5, disturbance_at=10.0)
    ardu = pid.simulate(defaults, pid.Plant(**{**airframe.__dict__, "disturbance": 0.0}), step_sc)
    out = {
        "plant": base.__dict__,
        "scenario": sc.__dict__,
        "t": _down(pid.simulate(pid.PRESETS["preeti"], base, sc).t, 2, 4),
        "presets": presets,
        "ziegler_nichols": {
            "ku": ku,
            "tu": tu,
            "gains": {"kp": zn.kp, "ki": zn.ki, "kd": zn.kd},
            "metrics": pid.metrics(pid.simulate(zn, base, sc), sc).as_dict(),
        },
        "tuned": {
            "gains": {"kp": tuned.kp, "ki": tuned.ki, "kd": tuned.kd},
            "metrics": pid.metrics(pid.simulate(tuned, base, sc), sc).as_dict(),
            "damping": pid.damping_ratio(tuned, base),
        },
        "critical_kd_at_kp_1_5": pid.critical_kd(1.5, base),
        "airframe": {
            "plant": airframe.__dict__,
            "ardupilot_defaults": {k: v for k, v in pid.ARDUPILOT_DEFAULTS.items()},
            "tuned_in_ardupilot_units": {"kp": cascade_tuned.kp / t, "ki": cascade_tuned.ki / t, "kd": cascade_tuned.kd / t},
            "sim_step_t": _down(ardu.t, 4, 3),
            "sim_step_deg": _down([math.degrees(a) for a in ardu.angle], 4, 2),
            "sim_metrics": pid.metrics(ardu, step_sc).as_dict(),
        },
    }
    if sitl_result and sitl_result.get("flight"):
        fl = sitl_result["flight"]
        out["airframe"]["sitl_step_t"] = fl["step_t"]
        out["airframe"]["sitl_step_deg"] = fl["step_pitch_deg"]
        out["airframe"]["sitl_metrics"] = step_metrics(fl["step_t"], fl["step_pitch_deg"], fl["step_target_deg"])
    return out


def step_metrics(t: list[float], y: list[float], target: float) -> dict:
    t10 = next((ti for ti, yi in zip(t, y) if yi >= 0.1 * target), None)
    t90 = next((ti for ti, yi in zip(t, y) if yi >= 0.9 * target), None)
    settle = None
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - target) > 0.05 * target:
            settle = t[i + 1] if i + 1 < len(t) else None
            break
    return {
        "rise_time": None if t10 is None or t90 is None else round(t90 - t10, 3),
        "overshoot": round(max(0.0, max(y) / target - 1), 4),
        "settling_time_5pct": settle,
    }


def pid_svg(data: dict) -> str:
    """The four slide characters on one chart, for the README."""
    w, h, pad = 760, 300, 42
    t = data["t"]
    colors = {"preeti": "#e0584f", "preeti_darren": "#4cc9d8", "isabelle_too_much": "#b58cff", "all_three": "#f08a3c"}
    names = {"preeti": "P only", "preeti_darren": "PD, critically damped", "isabelle_too_much": "PID, too much I", "all_three": "PID"}
    lo, hi = -5.0, 25.0

    def xy(ti, yi):
        yi = max(lo, min(hi, yi))
        return pad + (w - pad - 10) * ti / t[-1], h - pad - (h - 2 * pad) * (yi - lo) / (hi - lo)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="ui-sans-serif, system-ui" font-size="11">',
        f'<rect width="{w}" height="{h}" fill="#fbfbfa"/>',
    ]
    for v in range(-5, 26, 5):
        _, y = xy(0, v)
        out.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{w - 10}" y2="{y:.1f}" stroke="#e3e6e8"/><text x="{pad - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#667">{v}</text>'
        )
    for s in range(0, int(t[-1]) + 1):
        x, _ = xy(s, lo)
        out.append(f'<text x="{x:.1f}" y="{h - pad + 16}" text-anchor="middle" fill="#667">{s} s</text>')
    xd, _ = xy(data["scenario"]["disturbance_at"], lo)
    out.append(
        f'<line x1="{xd:.1f}" y1="{pad - 10}" x2="{xd:.1f}" y2="{h - pad}" stroke="#99a" stroke-dasharray="4 4"/><text x="{xd + 4:.1f}" y="{pad - 2}" fill="#667">disturbance on</text>'
    )
    _, yt = xy(0, 10)
    out.append(f'<line x1="{pad}" y1="{yt:.1f}" x2="{w - 10}" y2="{yt:.1f}" stroke="#222" stroke-dasharray="2 3"/>')
    for i, (name, p) in enumerate(data["presets"].items()):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(ti, yi) for ti, yi in zip(t, p["angle_deg"])))
        out.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" points="{pts}"/>')
        out.append(
            f'<rect x="{pad + 12 + i * 170}" y="12" width="14" height="4" fill="{colors[name]}"/><text x="{pad + 32 + i * 170}" y="18" fill="#223">{names[name]}</text>'
        )
    out.append(f'<text x="14" y="{h / 2}" transform="rotate(-90 14 {h / 2})" text-anchor="middle" fill="#667">pitch, degrees</text></svg>')
    return "\n".join(out)


# ------------------------------------------------------------- build ----


def build(out: Path, run_sitl: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    design = Design.load()
    report = check.check(design)
    cables = harness(design)
    perf = performance.analyse(design)
    _, calibration = performance.calibrate(design)
    plist = params.generate(design)
    problems = params.validate(plist)
    if problems:
        raise SystemExit(f"parameter problems: {problems}")

    _write(out / "wiring.svg", render(design))
    _write(out / "check.json", [f.as_dict() for f in report])
    _write(out / "check.md", check_md(report))
    _write(out / "harness.md", harness_md(cables))
    _write(
        out / "harness.json",
        [
            {
                "link": c.link,
                "a": c.a,
                "b": c.b,
                "a_connector": c.a_connector,
                "b_connector": c.b_connector,
                "custom": c.custom,
                "wires": [w.__dict__ for w in c.wires],
                "notes": c.notes,
            }
            for c in cables
        ],
    )
    params.write(
        plist,
        out / "protocol_pro.param",
        header=f"Protocol Pro: {design.spec['summary']}\nFirmware: {design.spec['firmware']}\nGenerated by python -m week04 docs; every value validated against ArduCopter 4.7.1 parameter metadata",
    )
    print(f"wrote {out / 'protocol_pro.param'}")
    b = bom(design)
    _write(out / "bom.json", b)
    _write(out / "bom.md", bom_md(b))
    pd = perf.as_dict()
    pd["calibration"] = calibration
    pd["sitl_frame"] = performance.sitl_frame(design, perf)
    _write(out / "performance.json", pd)

    budgets = {}
    for link in design.links:
        if link.spec.get("streams"):
            bud = links.design_budget(design, link)
            budgets[link.id] = {
                "config": bud.config.label(),
                "capacity_bytes_per_s": round(bud.config.bytes_per_second, 1),
                "bytes_per_s": round(bud.bytes_per_second, 1),
                "utilisation": round(bud.utilisation, 4),
                "by_stream": {k: round(v, 1) for k, v in bud.by_stream().items()},
                "extra": [[n, round(v, 1)] for n, v in bud.extra],
                "rows": [{"stream": s, "message": m, "hz": hz, "bytes": size} for s, m, hz, size in bud.rows],
            }
    budgets["rtcm"] = {
        "base_bytes_per_s": links.rtcm_base_bytes_per_second(),
        "over_mavlink_bytes_per_s": links.rtcm_over_mavlink(links.rtcm_base_bytes_per_second()),
    }
    _write(out / "budget.json", budgets)
    _write(out / "journey.json", journey.run())
    _write(out / "site.json", site_bundle(design, report, cables))
    _write(out / "scope.json", scope())

    sitl_path = out / "sitl.json"
    if run_sitl:
        from . import sitl

        root = sitl.ardupilot_dir()
        if root is None:
            raise SystemExit("no ArduPilot SITL build: set ARDUPILOT_DIR")
        result = sitl.experiments(root, design)
        capture = result.pop("_capture")
        _write(sitl_path, result)
        (out / "sitl_flight.tlog").write_bytes(trim_tlog(capture, 25.0))
        print(f"wrote {out / 'sitl_flight.tlog'}")
    sitl_result = json.loads(sitl_path.read_text()) if sitl_path.exists() else None
    pdata = pid_data(design, sitl_result)
    _write(out / "pid.json", pdata)
    _write(out / "pid.svg", pid_svg(pdata))
    _write(out / "summary.json", summary(design, report, perf, plist, budgets, sitl_result))


def trim_tlog(data: bytes, seconds: float) -> bytes:
    """Keep the last ``seconds`` of a tlog."""
    from . import mavlink

    entries, _ = mavlink.read_tlog(data)
    if not entries:
        return b""
    end = entries[-1][0]
    return b"".join(usec.to_bytes(8, "big") + p.raw for usec, p in entries if usec >= end - seconds * 1e6)


def site_bundle(design: Design, report, cables) -> dict:
    """Everything the wiring explorer shows when a wire or a part is clicked."""
    from .design import PROTOCOLS
    from .diagram import wire_label

    by_link = {c.link: c for c in cables}
    links_out = {}
    for link in design.links:
        c = by_link.get(link.id)
        links_out[link.id] = {
            "label": wire_label(link),
            "protocol": PROTOCOLS[link.protocol]["label"],
            "from": ", ".join(p.ref for p in link.a),
            "to": ", ".join(p.ref for p in link.b),
            "purpose": link.spec.get("purpose", ""),
            "powered_by": link.spec.get("powered_by"),
            "wires": [[w.a_pin, w.a_signal, w.b_pin, w.b_signal] for w in c.wires] if c else [],
            "connectors": [c.a_connector, c.b_connector or "vendor cable"] if c else [],
            "notes": c.notes if c else [],
            "findings": [[f.level, f.rule, f.message] for f in report if f.where == link.id],
        }
    for feed in design.spec.get("power", []):
        links_out["power:" + feed["id"]] = {
            "label": "power",
            "protocol": "Power",
            "from": feed["from"],
            "to": feed["to"],
            "purpose": "power feed",
            "findings": [[f.level, f.rule, f.message] for f in report if f.where in (feed["id"], feed["from"], feed["to"])],
            "wires": [],
        }
    parts_out = {}
    for name, inst in design.parts.items():
        p = inst.part
        parts_out[name] = {
            "name": p["name"],
            "vendor": p.get("vendor"),
            "role": p["role"].replace("_", " "),
            "count": inst.count,
            "price_usd": p.get("price_usd"),
            "mass_g": p.get("mass_g"),
            "supply": p.get("supply"),
            "logic_v": p.get("logic_v"),
            "url": p.get("url"),
            "estimates": p.get("estimates", {}),
            "notes": p.get("notes", ""),
            "ports": [q["id"] for q in p.get("ports", [])][:24],
            "findings": [[f.level, f.rule, f.message] for f in report if f.where == name or f.where.startswith(name + ".")],
        }
    return {"links": links_out, "parts": parts_out}


def summary(design, report, perf, plist, budgets, sitl_result) -> dict:
    from . import mavlink

    d = mavlink.Dialect.default()
    s = {
        "errors": report.count("error"),
        "warnings": report.count("warning"),
        "passed": report.count("ok"),
        "parameters": len(plist),
        "mavlink_messages": len(d.messages),
        "mass_kg": round(perf.mass_kg, 2),
        "hover_minutes": round(perf.hover_minutes, 1),
        "thrust_to_weight": round(perf.thrust_to_weight, 2),
        "telemetry_utilisation": budgets["telemetry"]["utilisation"],
        "links": len(design.links),
        "protocols": sorted({link.protocol for link in design.links}),
    }
    if sitl_result:
        s["sitl"] = {
            "params_accepted": sitl_result["params"]["accepted"],
            "params_sent": sitl_result["params"]["sent"],
            "link_predicted": sitl_result["link_budget"]["predicted_bytes_per_s"],
            "link_measured": sitl_result["link_budget"]["measured_bytes_per_s"],
            "signatures_verified": sitl_result["signing"]["usb_signatures_verified"],
        }
    return s
