"""Parity checks between the week 04 Python protocol codecs and the C++17 port in week04/cpp.

Skipped unless WEEK04_CPP_BUILD points at a build directory that contains the
``protocol_tool`` executable, for example::

    cmake -S week04/cpp -B build/cpp4 && cmake --build build/cpp4 -j
    WEEK04_CPP_BUILD=build/cpp4 python -m pytest tests/test_week04_cpp_parity.py -v

Every case is generated from a fixed seed, sent to ``protocol_tool`` as one
request line, and the JSON answer must equal what the Python function returns
(or the name of the exception class it raises) exactly: same bytes, same bits,
same counters.
"""

from __future__ import annotations

import json
import os
import random
import string
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from week04 import can, crc, dshot, mavlink, rc

BUILD = os.environ.get("WEEK04_CPP_BUILD")


def _binary(name: str) -> Path | None:
    if not BUILD:
        return None
    for candidate in (Path(BUILD) / name, Path(BUILD) / "Release" / name, Path(BUILD) / f"{name}.exe"):
        if candidate.is_file():
            return candidate.resolve()
    return None


TOOL = _binary("protocol_tool")

pytestmark = pytest.mark.skipif(TOOL is None, reason="set WEEK04_CPP_BUILD to a build directory containing protocol_tool")

Case = tuple[str, Callable[[], dict]]  # request line, Python oracle


def _hex(data: bytes) -> str:
    return data.hex() or "-"


def _bits(bits: list[int]) -> str:
    return "".join(map(str, bits)) or "-"


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _oracle(fn: Callable[[], dict]) -> dict:
    try:
        return fn()
    except Exception as exc:  # the C++ side reports the Python exception class name
        return {"error": type(exc).__name__}


def _check(cases: list[Case], setup: list[str] | None = None) -> None:
    """Run all requests through one protocol_tool process and compare line by line."""
    setup = setup or []
    lines = setup + [request for request, _ in cases]
    result = subprocess.run([str(TOOL)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    answers = [json.loads(line) for line in result.stdout.splitlines()][len(setup) :]
    assert len(answers) == len(cases), f"{len(answers)} answers for {len(cases)} requests:\n{result.stderr}"
    bad = []
    for (request, fn), got in zip(cases, answers):
        expected = _oracle(fn)
        if got != expected:
            bad.append(f"{request[:160]}\n  Python: {str(expected)[:300]}\n  C++:    {str(got)[:300]}")
    assert not bad, f"{len(bad)} of {len(cases)} cases differ, first ones:\n" + "\n".join(bad[:5])


# ---------------------------------------------------------------- CRCs ----


def crc_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    funcs = {"x25": crc.x25_accumulate, "x25_bitwise": crc.x25_bitwise, "ccitt_false": crc.ccitt_false, "dvb_s2": crc.dvb_s2}
    samples = [crc.CHECK_INPUT, b""] + [rng.randbytes(rng.randint(1, 80)) for _ in range(40)]
    for name, fn in funcs.items():
        for data in samples:
            cases.append((f"crc {name} {_hex(data)}", lambda fn=fn, data=data: {"crc": fn(data)}))
        for _ in range(15):
            data, init = rng.randbytes(rng.randint(0, 40)), rng.randrange(256 if name == "dvb_s2" else 0x10000)
            cases.append((f"crc {name} {_hex(data)} {init}", lambda fn=fn, data=data, init=init: {"crc": fn(data, init)}))
    for init in (255, 256, 1000, 0xFFFF):  # CRC-8 state must fit in a byte
        data = rng.randbytes(rng.randint(0, 10))
        cases.append((f"crc dvb_s2 {_hex(data)} {init}", lambda data=data, init=init: {"crc": crc.dvb_s2(data, init)}))
    for n in [0, 1, 19, 83, 200] + [rng.randint(0, 300) for _ in range(30)]:
        bits = [rng.randint(0, 1) for _ in range(n)]
        cases.append((f"can15 {_bits(bits)}", lambda bits=bits: {"crc": crc.can15(bits)}))
    check_bits = crc.bytes_to_bits(crc.CHECK_INPUT)
    cases.append((f"can15 {_bits(check_bits)}", lambda: {"crc": crc.can15(check_bits)}))
    for _ in range(40):
        v, inv = rng.randrange(1 << 16), rng.random() < 0.5
        cases.append((f"dshot_crc {v} {_flag(inv)}", lambda v=v, inv=inv: {"crc": crc.dshot_crc(v, inv)}))
    return cases


def sha_cases(rng: random.Random) -> list[Case]:
    import hashlib

    lengths = [0, 1, 3, 55, 56, 63, 64, 65, 119, 120, 128, 300] + [rng.randint(0, 200) for _ in range(20)]
    datas = [b"abc"] + [rng.randbytes(n) for n in lengths]
    return [(f"sha256 {_hex(d)}", lambda d=d: {"digest": hashlib.sha256(d).hexdigest()}) for d in datas]


def test_crcs_match_python() -> None:
    rng = random.Random(40)
    _check(crc_cases(rng) + sha_cases(rng))


# ------------------------------------------------------------ SBUS, CRSF ----


def _channels(rng: random.Random, bad: float = 0.0) -> list[int]:
    ch = [rng.choice([rc.RAW_MIN, rc.RAW_MID, rc.RAW_MAX, rng.randrange(2048)]) for _ in range(16)]
    if rng.random() < bad:
        ch[rng.randrange(16)] = rng.choice([-1, 2048, 5000])
    return ch


def _sbus_dict(frame: rc.SbusFrame) -> dict:
    return {"channels": frame.channels, "ch17": frame.ch17, "ch18": frame.ch18, "frame_lost": frame.frame_lost, "failsafe": frame.failsafe}


def sbus_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for _ in range(80):
        ch, flags = _channels(rng, bad=0.1), [rng.random() < 0.5 for _ in range(4)]
        request = "sbus_encode " + " ".join(map(str, ch)) + " " + " ".join(map(_flag, flags))
        cases.append((request, lambda ch=ch, flags=flags: {"hex": rc.SbusFrame(ch, *flags).encode().hex()}))
    for _ in range(80):
        data = bytearray(rc.SbusFrame(_channels(rng), *[rng.random() < 0.5 for _ in range(4)]).encode())
        data[23] = rng.randrange(256)  # any flag byte, including the reserved bits
        roll = rng.random()
        if roll < 0.1:
            data[0] ^= 0x01
        elif roll < 0.2:
            data[24] = 0x04
        elif roll < 0.3:
            data = data[: rng.randrange(25)]
        data = bytes(data)
        cases.append((f"sbus_decode {_hex(data)}", lambda data=data: _sbus_dict(rc.SbusFrame.decode(data))))
    for _ in range(30):
        ch = _channels(rng, bad=0.2)
        cases.append(("pack11 " + " ".join(map(str, ch)), lambda ch=ch: {"hex": rc.pack11(ch).hex()}))
        data = rng.randbytes(rng.randint(0, 30))
        cases.append((f"unpack11 {_hex(data)}", lambda data=data: {"channels": rc.unpack11(data)}))
    return cases


def _half(rng: random.Random, scale: int, lo: int, hi: int) -> float:
    """A value that lands on (or next to) an exact .5 after scaling, to exercise round half to even."""
    return rng.randint(lo * 2 * scale, hi * 2 * scale) / (2 * scale)


def _edge(rng: random.Random, usual, *edges, p: float = 0.08):
    """Usually ``usual``, sometimes one of the boundary values (some of which do not fit their field)."""
    return rng.choice(edges) if rng.random() < p else usual


def crsf_builder_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for _ in range(50):
        ch = _channels(rng, bad=0.1)
        cases.append(("crsf_rc " + " ".join(map(str, ch)), lambda ch=ch: {"hex": rc.crsf_rc(ch).hex()}))
    for _ in range(70):
        volts = _edge(rng, rng.choice([rng.uniform(0, 60), _half(rng, 10, 0, 60)]), -0.2, 6553.6, 6553.4, 6553.45)
        amps = rng.choice([rng.uniform(0, 200), _half(rng, 10, 0, 200)])
        used = _edge(rng, rng.randrange(1 << 24), 0, (1 << 24) - 1, 1 << 24, -1)
        percent = _edge(rng, rng.randint(0, 100), 255, 256)
        cases.append(
            (
                f"crsf_battery {volts!r} {amps!r} {used} {percent}",
                lambda v=volts, a=amps, u=used, p=percent: {"hex": rc.crsf_battery(v, a, u, p).hex()},
            )
        )
    for _ in range(60):
        angles = [_edge(rng, rng.choice([rng.uniform(-3.2, 3.2), _half(rng, 10000, -3, 3)]), 3.27675, -3.27685, 3.2767) for _ in range(3)]
        cases.append(("crsf_attitude " + " ".join(map(repr, angles)), lambda a=angles: {"hex": rc.crsf_attitude(*a).hex()}))
    for _ in range(70):
        lat = rng.choice([rng.uniform(-90, 90), _half(rng, 10**7, -90, 90)])
        lon = _edge(rng, rng.uniform(-180, 180), 214.7483647, 214.7483648, -214.7483649)
        speed = _edge(rng, rng.choice([rng.uniform(0, 400), _half(rng, 10, 0, 400)]), -1.0, 6553.5)
        heading = rng.choice([rng.uniform(0, 360), _half(rng, 100, 0, 359)])
        alt = _edge(rng, rng.choice([rng.uniform(-900, 9000), rng.randint(-1000, 60000) + 0.5]), 64535.4, 64536.0, -1000.6)
        sats = _edge(rng, rng.randint(0, 40), 255, 256)
        args = (lat, lon, speed, heading, alt, sats)
        request = "crsf_gps " + " ".join(repr(x) for x in args)
        cases.append((request, lambda args=args: {"hex": rc.crsf_gps(*args).hex()}))
    for _ in range(50):
        values = [rng.randrange(256) for _ in range(10)]
        values[3], values[9] = rng.randint(-128, 127), rng.randint(-128, 127)
        if rng.random() < 0.15:
            values[rng.randrange(10)] = rng.choice([-129, 256, 128])
        cases.append(("crsf_link_statistics " + " ".join(map(str, values)), lambda v=values: {"hex": rc.crsf_link_statistics(*v).hex()}))
    for n in [0, 1, 4, 58, 59, 60] + [rng.randint(0, 64) for _ in range(30)]:
        mode = "".join(rng.choice(string.printable) for _ in range(n))
        cases.append((f"crsf_flight_mode {_hex(mode.encode('ascii'))}", lambda m=mode: {"hex": rc.crsf_flight_mode(m).hex()}))
    for _ in range(50):
        ftype = _edge(rng, rng.randrange(256), 256)
        address = _edge(rng, rng.choice([rc.CRSF_SYNC, 0xEA, 0xEE, rng.randrange(256)]), 300)
        payload = rng.randbytes(_edge(rng, rng.randint(0, 61), 60, 61, 62, 64, p=0.2))
        cases.append(
            (
                f"crsf_frame {ftype} {address} {_hex(payload)}",
                lambda t=ftype, a=address, p=payload: {"hex": rc.crsf_frame(t, p, a).hex()},
            )
        )
    return cases


def _random_crsf_frame(rng: random.Random) -> bytes:
    kind = rng.randrange(6)
    if kind == 0:
        return rc.crsf_rc(_channels(rng))
    if kind == 1:
        return rc.crsf_battery(rng.uniform(0, 30), rng.uniform(0, 90), rng.randrange(10000), rng.randint(0, 100))
    if kind == 2:
        return rc.crsf_attitude(*(rng.uniform(-3, 3) for _ in range(3)))
    if kind == 3:
        return rc.crsf_gps(
            rng.uniform(-90, 90), rng.uniform(-180, 180), rng.uniform(0, 100), rng.uniform(0, 359), rng.uniform(0, 500), rng.randint(0, 20)
        )
    if kind == 4:
        return rc.crsf_link_statistics(*(rng.randrange(128) for _ in range(10)))
    return rc.crsf_flight_mode(rng.choice(["ACRO", "ANGLE", "STAB", "LOITER", "RTL", "!FS!"]))


def _crsf_parse(chunks: list[bytes]) -> dict:
    parser = rc.CrsfParser()
    frames = [{"address": p.address, "type": p.type, "payload": p.payload.hex()} for chunk in chunks for p in parser.feed(chunk)]
    return {"frames": frames, "crc_errors": parser.crc_errors, "dropped_bytes": parser.dropped_bytes, "buffered": len(parser.buffer)}


def _chunks(rng: random.Random, stream: bytes) -> list[bytes]:
    if rng.random() < 0.4:
        return [stream]
    cuts = sorted(rng.sample(range(1, max(2, len(stream))), k=min(rng.randint(1, 8), max(1, len(stream) - 1))))
    return [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)])]


def crsf_parse_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for _ in range(60):
        stream = bytearray()
        for _ in range(rng.randint(0, 12)):
            if rng.random() < 0.3:  # noise, often made of plausible sync bytes
                stream += bytes(rng.choice([rc.CRSF_SYNC, 0xEA, 0xEE, 0xEC, rng.randrange(256)]) for _ in range(rng.randint(1, 6)))
            frame = bytearray(_random_crsf_frame(rng))
            if rng.random() < 0.15:
                frame[rng.randrange(len(frame))] ^= 1 << rng.randrange(8)
            stream += frame
        if rng.random() < 0.3:
            stream += bytes([rc.CRSF_SYNC, rng.randint(2, 62)])  # a frame cut off by the end of the capture
        chunks = _chunks(rng, bytes(stream))
        cases.append(("crsf_parse " + " ".join(_hex(c) for c in chunks), lambda c=chunks: _crsf_parse(c)))
    return cases


def test_sbus_and_crsf_match_python() -> None:
    rng = random.Random(41)
    _check(sbus_cases(rng) + crsf_builder_cases(rng) + crsf_parse_cases(rng))


# --------------------------------------------------------------- DShot ----


def dshot_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for _ in range(120):
        value = rng.choice([0, 1, 47, 48, 2047, rng.randrange(2048), rng.randrange(2048), -1, 2048])
        tel, bidir = rng.random() < 0.5, rng.random() < 0.5
        request = f"dshot_encode {value} {_flag(tel)} {_flag(bidir)}"
        cases.append((request, lambda v=value, t=tel, b=bidir: {"frame": dshot.encode(v, t, b)}))
    for _ in range(120):
        frame, bidir = rng.randrange(1 << 16), rng.random() < 0.5
        if rng.random() < 0.5:
            frame = dshot.encode(rng.randrange(2048), rng.random() < 0.5, bidir)

        def dec(f=frame, b=bidir) -> dict:
            d = dshot.decode(f, b)
            return {"value": d.value, "telemetry": d.telemetry, "crc_ok": d.crc_ok}

        cases.append((f"dshot_decode {frame} {_flag(bidir)}", dec))
    periods = [1, 0x1FF, 0x200, 0xFFFF, 65408, 1 << 16, 1 << 20, 0, -5] + [rng.randint(1, 1 << rng.randint(1, 24)) for _ in range(60)]
    for p in periods:
        cases.append((f"erpm_word {p}", lambda p=p: {"word": dshot.erpm_word(p)}))
    for _ in range(80):
        word = rng.randrange(1 << 16)
        if rng.random() < 0.5:
            word = dshot.erpm_word(rng.randint(1, 1 << 20))
        cases.append((f"gcr_encode {word}", lambda w=word: {"line": dshot.gcr_encode(w)}))
    for _ in range(120):
        line = rng.randrange(1 << 21) if rng.random() < 0.5 else dshot.gcr_encode(dshot.erpm_word(rng.randint(1, 1 << 18)))
        if rng.random() < 0.2:
            line ^= 1 << rng.randrange(20)
        if rng.random() < 0.3:  # junk above the 20 line levels must be ignored
            line |= rng.randrange(1, 1 << 12) << 20

        def gdec(line=line) -> dict:
            word, ok = dshot.gcr_decode(line)
            return {"word": word, "ok": ok}

        cases.append((f"gcr_decode {line}", gdec))
    return cases


def test_dshot_matches_python() -> None:
    _check(dshot_cases(random.Random(42)))


# ----------------------------------------------------------------- CAN ----


def _random_can_args(rng: random.Random) -> tuple[bool, bool, int, bytes]:
    extended, remote = rng.random() < 0.5, rng.random() < 0.15
    can_id = rng.randrange(1 << (29 if extended else 11))
    if rng.random() < 0.15:  # runs of equal bits, so the frame needs stuffing
        can_id = 0 if rng.random() < 0.5 else (1 << (29 if extended else 11)) - 1
    data = b"" if remote else rng.choice([b"", bytes(8), b"\xff" * 8, rng.randbytes(rng.randint(0, 8))])
    return extended, remote, can_id, data


def _can_request(extended: bool, remote: bool, can_id: int, data: bytes) -> str:
    return f"{_flag(extended)} {_flag(remote)} {can_id} {_hex(data)}"


def can_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for k in range(150):
        ext, remote, can_id, data = _random_can_args(rng)
        if k % 25 == 0:
            can_id = rng.choice([-1, 1 << (29 if ext else 11)])
        if k % 30 == 1:
            data = rng.randbytes(9)
        if k % 20 == 3:  # a remote frame that carries data
            remote, data = True, rng.randbytes(rng.randint(1, 8))
        acked, interframe = rng.random() < 0.8, rng.random() < 0.7

        def frame(ext=ext, remote=remote, can_id=can_id, data=data, acked=acked, interframe=interframe) -> dict:
            f = can.CanFrame(can_id, data, ext, remote)
            return {
                "header_bits": _bits(f.header_bits()),
                "crc": f.crc(),
                "stuffed_bits": _bits(f.stuffed_bits()),
                "wire_bits": _bits(f.wire_bits(acked, interframe)),
            }

        request = f"can_frame {_can_request(ext, remote, can_id, data)} {_flag(acked)} {_flag(interframe)}"
        cases.append((request, frame))

    def decoded(wire: list[int]) -> dict:
        f = can.decode(wire)
        return {"id": f.can_id, "extended": f.extended, "remote": f.remote, "data": f.data.hex()}

    for _ in range(200):
        ext, remote, can_id, data = _random_can_args(rng)
        wire = can.CanFrame(can_id, data, ext, remote).wire_bits(acked=rng.random() < 0.8)
        roll = rng.random()
        if roll < 0.25:
            wire[rng.randrange(len(wire))] ^= 1  # a bit error anywhere
        elif roll < 0.35:
            wire = wire[: rng.randrange(len(wire))]  # capture cut short
        elif roll < 0.45:
            wire = [rng.randint(0, 1) for _ in range(rng.randint(0, 140))]
        cases.append((f"can_decode {_bits(wire)}", lambda w=wire: decoded(w)))
    # Every possible cut of a few stuffing heavy frames: ends inside a field, right where a stuff bit
    # is due, and right after the CRC (no delimiter).
    for ext, can_id, data in [(False, 0, bytes(8)), (True, (1 << 29) - 1, b"\xff" * 3), (False, 0x7F0, b"\x00\x1f")]:
        wire = can.CanFrame(can_id, data, ext).wire_bits()
        for cut in range(len(wire) + 1):
            cases.append((f"can_decode {_bits(wire[:cut])}", lambda w=wire[:cut]: decoded(w)))
    for _ in range(60):
        bits = [rng.choice([0, 0, 0, 1]) if rng.random() < 0.5 else rng.randint(0, 1) for _ in range(rng.randint(0, 120))]
        cases.append((f"can_stuff {_bits(bits)}", lambda b=bits: {"bits": _bits(can.stuff(b))}))
        stuffed = can.stuff(bits)
        if stuffed and rng.random() < 0.3:
            stuffed[rng.randrange(len(stuffed))] ^= 1
        cases.append((f"can_destuff {_bits(stuffed)}", lambda b=stuffed: {"bits": _bits(can.destuff(b))}))
    for _ in range(100):
        n = rng.randint(1, 6)
        specs = []
        base = rng.randrange(1 << 11)
        for _ in range(n):
            ext, remote, can_id, data = _random_can_args(rng)
            if rng.random() < 0.5:  # share the top 11 identifier bits so SRR, IDE and RTR decide
                can_id = (base << 18 | rng.randrange(1 << 18)) if ext else base
            specs.append((ext, remote, can_id, data))
        if rng.random() < 0.1:
            specs.append(specs[0])

        def arb(specs=specs) -> dict:
            r = can.arbitrate([can.CanFrame(i, d, e, rm) for e, rm, i, d in specs])
            return {"winner": r.winner, "lost_at": [[k, v] for k, v in sorted(r.lost_at.items())], "bus": _bits(r.bus)}

        request = f"can_arbitrate {len(specs)} " + " ".join(_can_request(*s) for s in specs)
        cases.append((request, arb))
    cases.append(("can_arbitrate 0", lambda: {"winner": can.arbitrate([]).winner}))
    return cases


def test_can_matches_python() -> None:
    _check(can_cases(random.Random(43)))


# ------------------------------------------------------------- MAVLink ----

_INT_RANGES = {
    "int8_t": (-(1 << 7), (1 << 7) - 1),
    "uint8_t": (0, 255),
    "uint8_t_mavlink_version": (0, 255),
    "int16_t": (-(1 << 15), (1 << 15) - 1),
    "uint16_t": (0, (1 << 16) - 1),
    "int32_t": (-(1 << 31), (1 << 31) - 1),
    "uint32_t": (0, (1 << 32) - 1),
    "int64_t": (-(1 << 63), (1 << 63) - 1),
    "uint64_t": (0, (1 << 64) - 1),
}


def _scalar(rng: random.Random, ctype: str):
    if ctype in ("float", "double"):
        return rng.choice([0.0, rng.uniform(-1e4, 1e4), rng.uniform(-1, 1), float(rng.randint(-100, 100))])
    lo, hi = _INT_RANGES[ctype]
    return rng.choice([0, lo, hi, rng.randint(lo, hi), rng.randint(0, min(hi, 100))])


def _values(rng: random.Random, msg: mavlink.Message) -> dict:
    """Random field values; many are left out so payloads end in zeros and v2 truncation kicks in."""
    zero_share = rng.choice([0.0, 0.3, 0.7, 1.0])
    values: dict = {}
    for f in msg.fields:
        if rng.random() < zero_share:
            continue
        if f.type == "char":
            n = max(f.array, 1)
            values[f.name] = "".join(rng.choice(string.ascii_letters + "_ ") for _ in range(rng.randint(0, n)))
        elif f.array:
            values[f.name] = [_scalar(rng, f.type) for _ in range(rng.randint(0, f.array))]
        else:
            values[f.name] = _scalar(rng, f.type)
    return values


def _random_encode_args(rng: random.Random, msg: mavlink.Message) -> dict:
    version = 1 if msg.id <= 255 and rng.random() < 0.35 else 2
    signed = version == 2 and rng.random() < 0.5
    return {
        "seq": rng.choice([rng.randrange(256), 255, 256, 1000]),
        "sysid": rng.choice([1, 255, rng.randrange(256)]),
        "compid": rng.choice([1, 190, rng.randrange(256)]),
        "version": version,
        "key": rng.choice([bytes(32), rng.randbytes(32), b"", rng.randbytes(rng.randint(1, 70))]) if signed else None,
        "link_id": rng.randrange(256),
        "timestamp": rng.choice([0, rng.randrange(1 << 48), (1 << 48) - 1]),
    }


def _encode_request(msg: mavlink.Message, payload: bytes, args: dict) -> str:
    key = "none" if args["key"] is None else _hex(args["key"])
    return (
        f"mav_encode {args['version']} {args['seq']} {args['sysid']} {args['compid']} {msg.id} {msg.crc_extra} "
        f"{msg.base_length} {_hex(payload)} {key} {args['link_id']} {args['timestamp']}"
    )


def mavlink_encode_cases(rng: random.Random) -> list[Case]:
    dialect = mavlink.Dialect.default()
    messages = list(dialect.messages.values())
    sample = rng.sample(messages, 120) + [m for m in messages if m.length != m.base_length][:30]
    cases: list[Case] = []
    for msg in sample:
        for _ in range(3):
            values, args = _values(rng, msg), _random_encode_args(rng, msg)
            payload = msg.pack(values)
            cases.append((_encode_request(msg, payload, args), lambda m=msg, v=values, a=args: {"hex": mavlink.encode(m, v, **a).hex()}))
    # The error paths: v1 with a 16 bit id, bad system ids, timestamps and link ids that overflow.
    big = next(m for m in messages if m.id > 255)
    small = dialect["HEARTBEAT"]
    bad_args = [
        (big, {"version": 1}),
        (small, {"sysid": 256}),
        (small, {"compid": -1, "version": 1}),
        (small, {"key": b"k", "timestamp": 1 << 48}),
        (small, {"key": b"k", "timestamp": -1}),
        (small, {"key": b"k", "link_id": 256}),
        (small, {"seq": -1}),
        (small, {"version": 1, "key": b"no signing in v1"}),
        (small, {"version": 1, "key": b""}),
        (big, {"version": 1, "key": bytes(32)}),
    ]
    for msg, override in bad_args:
        args = {**_random_encode_args(rng, msg), "version": 2, "key": None, **override}
        values = _values(rng, msg)
        payload = msg.pack(values)
        cases.append((_encode_request(msg, payload, args), lambda m=msg, v=values, a=args: {"hex": mavlink.encode(m, v, **a).hex()}))
    return cases


def _packet_dict(p: mavlink.Packet) -> dict:
    return {
        "version": p.version,
        "seq": p.seq,
        "sysid": p.sysid,
        "compid": p.compid,
        "msgid": p.msgid,
        "payload": p.payload.hex(),
        "incompat": p.incompat,
        "compat": p.compat,
        "signature": p.signature.hex() if p.signature is not None else None,
        "raw": p.raw.hex(),
    }


def _mav_parse(chunks: list[bytes]) -> dict:
    parser = mavlink.Parser()
    packets = [_packet_dict(p) for chunk in chunks for p in parser.feed(chunk)]
    return {
        "packets": packets,
        "crc_errors": parser.crc_errors,
        "unknown": parser.unknown,
        "duplicates": parser.duplicates,
        "bad_flags": parser.bad_flags,
        "dropped_bytes": parser.dropped_bytes,
        "lost": parser.lost,
        "buffered": len(parser.buffer),
    }


def _with_incompat(frame: bytearray, msg: mavlink.Message, flags: int) -> bytearray:
    """Set unknown incompatibility flags and fix the CRC, so the flags are the frame's only fault."""
    n = frame[1]
    frame[2] = flags
    frame[10 + n : 12 + n] = crc.x25_accumulate([*frame[1 : 10 + n], msg.crc_extra]).to_bytes(2, "little")
    return frame


def mavlink_parse_cases(rng: random.Random) -> list[Case]:
    dialect = mavlink.Dialect.default()
    messages = list(dialect.messages.values())
    stranger = mavlink.Message(60001, "NOT_IN_DIALECT", [mavlink.Field("x", "uint32_t")], crc_extra=77)
    cases: list[Case] = []
    for _ in range(60):
        stream = bytearray()
        seq = rng.randrange(256)
        for _ in range(rng.randint(0, 10)):
            if rng.random() < 0.3:  # noise, often start markers
                stream += bytes(rng.choice([mavlink.STX_V1, mavlink.STX_V2, rng.randrange(256)]) for _ in range(rng.randint(1, 5)))
            msg = stranger if rng.random() < 0.08 else rng.choice(messages)
            args = _random_encode_args(rng, msg)
            args["seq"] = seq
            args["sysid"], args["compid"] = rng.choice([(1, 1), (1, 1), (255, 190)])  # few senders, so seq gaps and repeats count
            seq = (seq + rng.choice([1, 1, 1, 0, 2, 5])) & 0xFF  # gaps count as lost packets, repeats as duplicates
            frame = bytearray(mavlink.encode(msg, _values(rng, msg), **args))
            if frame[0] == mavlink.STX_V2 and rng.random() < 0.12:
                frame = _with_incompat(frame, msg, frame[2] | rng.choice([0x02, 0x04, 0x80]))
            if rng.random() < 0.12:
                frame[rng.randrange(len(frame))] ^= 1 << rng.randrange(8)
            stream += frame
        if rng.random() < 0.3:
            stream += bytes([mavlink.STX_V2, rng.randrange(256), 0, 0])  # a frame cut off by the end of the capture
        chunks = _chunks(rng, bytes(stream))
        cases.append(("mav_parse " + " ".join(_hex(c) for c in chunks), lambda c=chunks: _mav_parse(c)))
    return cases


def _mav_table() -> str:
    return "mav_table " + " ".join(f"{m.id}:{m.crc_extra}" for m in mavlink.Dialect.default().messages.values())


def test_mavlink_encode_matches_python() -> None:
    _check(mavlink_encode_cases(random.Random(44)))


def test_mavlink_parse_matches_python() -> None:
    _check(mavlink_parse_cases(random.Random(45)), setup=[_mav_table()])
