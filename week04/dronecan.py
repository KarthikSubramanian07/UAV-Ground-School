"""DroneCAN (UAVCAN v0), the application protocol on top of CAN that the Here4 GPS speaks.

Three layers, all implemented here from the specification and checked
against the reference ``pydronecan`` implementation in the tests:

1. **DSDL.** Message layouts are defined in ``.uavcan`` files (a copy of the
   ones this design uses is in ``week04/dsdl``, MIT licensed). :func:`load`
   parses them, and :class:`DsdlType` serialises values bit by bit: fields
   are packed MSB first with no padding, multi byte integers are little
   endian, and the last dynamic array of a message drops its length prefix
   (tail array optimisation).
2. **Signatures.** Every type has a 64 bit signature: CRC-64-WE of its
   normalised definition, extended with the signatures of nested types. The
   sender seeds the transfer CRC with it, so two nodes that disagree on a
   layout reject each other's multi frame transfers instead of misreading
   them.
3. **Transfers.** A payload that fits in 7 bytes goes in one CAN frame plus a
   tail byte (start, end, toggle, 5 bit transfer id). Longer payloads are
   prefixed with a CRC-16/CCITT-FALSE and split across frames with the
   toggle bit alternating.

The 29 bit CAN identifier of a message is ``priority << 24 | type id << 8 |
source node id``, so arbitration (see :mod:`week04.can`) favours urgent
messages first, then lower type ids.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .can import CanFrame
from .crc import ccitt_false

DSDL_ROOT = Path(__file__).with_name("dsdl")

# --------------------------------------------------------- CRC-64-WE ----

_MASK64 = (1 << 64) - 1


class Signature:
    """CRC-64-WE (poly 0x42F0E1EBA9EA3693, init and xorout all ones), resumable."""

    def __init__(self, extend_from: int | None = None):
        self.crc = _MASK64 if extend_from is None else (extend_from & _MASK64) ^ _MASK64

    def add(self, data: bytes | str) -> Signature:
        for b in data.encode("ascii") if isinstance(data, str) else data:
            self.crc ^= (b << 56) & _MASK64
            for _ in range(8):
                self.crc = ((self.crc << 1) & _MASK64) ^ 0x42F0E1EBA9EA3693 if self.crc & (1 << 63) else (self.crc << 1) & _MASK64
        return self

    @property
    def value(self) -> int:
        return self.crc ^ _MASK64


# -------------------------------------------------------------- types ----


@dataclass
class Primitive:
    kind: str  # bool, uint, int, float
    bits: int
    saturated: bool = True

    @property
    def min_bits(self) -> int:
        return self.bits

    def normalized(self) -> str:
        name = "bool" if self.kind == "bool" else f"{self.kind}{self.bits}"
        return ("saturated " if self.saturated else "truncated ") + name

    def signature(self) -> int | None:
        return None

    def default(self):
        return 0.0 if self.kind == "float" else 0


@dataclass
class Void:
    bits: int

    @property
    def min_bits(self) -> int:
        return self.bits

    def normalized(self) -> str:
        return f"void{self.bits}"

    def signature(self) -> int | None:
        return None


@dataclass
class Array:
    element: Any
    size: int
    dynamic: bool

    @property
    def min_bits(self) -> int:
        return 0 if self.dynamic else self.size * self.element.min_bits

    @property
    def length_bits(self) -> int:
        return math.ceil(math.log2(self.size + 1))

    def normalized(self) -> str:
        return f"{self.element.normalized()}[{'<=' if self.dynamic else ''}{self.size}]"

    def signature(self) -> int | None:
        return self.element.signature()

    def default(self):
        return [] if self.dynamic else [_default(self.element) for _ in range(self.size)]


@dataclass
class DsdlType:
    full_name: str
    type_id: int | None
    fields: list[tuple[str | None, Any]] = field(default_factory=list)

    @property
    def min_bits(self) -> int:
        return sum(t.min_bits for _, t in self.fields)

    def normalized(self) -> str:
        return self.full_name

    def source_definition(self) -> str:
        lines = [self.full_name] + [t.normalized() if name is None else f"{t.normalized()} {name}" for name, t in self.fields]
        return "\n".join(lines)

    def dsdl_signature(self) -> int:
        return Signature().add(self.source_definition()).value

    def signature(self) -> int:
        sig = Signature(self.dsdl_signature())
        for _, t in self.fields:
            nested = t.signature()
            if nested is not None:
                before = sig.value
                sig.add(nested.to_bytes(8, "little"))
                sig.add(before.to_bytes(8, "little"))
        return sig.value

    def default(self) -> dict:
        return {name: _default(t) for name, t in self.fields if name is not None}

    # ---------------------------------------------------- serialisation ----

    def encode(self, value: dict, tao: bool = True) -> bytes:
        w = _BitWriter()
        _write_compound(w, self, value, tao)
        return w.to_bytes()

    def decode(self, data: bytes, tao: bool = True) -> dict:
        r = _BitReader(data)
        return _read_compound(r, self, tao)


def _default(t):
    return t.default()


# ------------------------------------------------------------ parsing ----

_PRIMITIVE = re.compile(r"^(bool|uint(\d+)|int(\d+)|float(16|32|64))$")
_ARRAY = re.compile(r"^(.+)\[(<=|<)?(\d+)\]$")
_CACHE: dict[tuple[str, str], DsdlType] = {}


def _find(full_name: str, root: Path) -> Path:
    *ns, short = full_name.split(".")
    directory = root.joinpath(*ns)
    for path in directory.glob("*.uavcan"):
        stem = path.stem.split(".")
        if stem[-1] == short:
            return path
    raise FileNotFoundError(f"no DSDL definition for {full_name} under {root}")


def _parse_type(text: str, namespace: str, root: Path):
    saturated = True
    if text.startswith("truncated "):
        saturated, text = False, text[len("truncated ") :]
    elif text.startswith("saturated "):
        text = text[len("saturated ") :]
    m = _ARRAY.match(text)
    if m:
        element = _parse_type(m.group(1), namespace, root)
        size = int(m.group(3)) - (1 if m.group(2) == "<" else 0)
        return Array(element, size, m.group(2) is not None)
    if text.startswith("void"):
        return Void(int(text[4:]))
    m = _PRIMITIVE.match(text)
    if m:
        if text == "bool":
            return Primitive("bool", 1, saturated)
        kind = "uint" if m.group(2) else "int" if m.group(3) else "float"
        return Primitive(kind, int(m.group(2) or m.group(3) or m.group(4)), saturated)
    full = text if "." in text else f"{namespace}.{text}"
    return load(full, root)


def load(full_name: str, root: Path | str = DSDL_ROOT) -> DsdlType:
    """Parse a message definition (constants and comments are skipped; services and unions are not supported)."""
    root = Path(root)
    key = (str(root), full_name)
    if key in _CACHE:
        return _CACHE[key]
    path = _find(full_name, root)
    parts = path.stem.split(".")
    type_id = int(parts[0]) if len(parts) == 2 else None
    namespace = full_name.rsplit(".", 1)[0]
    result = DsdlType(full_name, type_id)
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("@") or line == "---":
            raise NotImplementedError(f"{full_name}: {line} is not supported by this minimal parser")
        if "=" in re.sub(r"\[[^\]]*\]", "", line):  # a constant (array bounds like [<=9] do not count)
            continue
        tokens = line.rsplit(None, 1)
        if len(tokens) == 1 or tokens[0] in ("truncated", "saturated"):
            result.fields.append((None, _parse_type(line, namespace, root)))
        else:
            result.fields.append((tokens[1], _parse_type(tokens[0], namespace, root)))
    _CACHE[key] = result
    return result


# ------------------------------------------------------------ bit I/O ----


class _BitWriter:
    def __init__(self):
        self.bits: list[int] = []

    def write(self, value: int, n: int):
        """Write an n bit unsigned value the UAVCAN v0 way: little endian bytes, each MSB first."""
        chunks = []
        remaining = n
        while remaining > 0:
            take = min(8, remaining)
            chunks.append((value & ((1 << take) - 1), take))
            value >>= take
            remaining -= take
        for v, k in chunks:
            self.bits += [(v >> (k - 1 - i)) & 1 for i in range(k)]

    def to_bytes(self) -> bytes:
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))


class _BitReader:
    def __init__(self, data: bytes):
        self.bits = [(b >> (7 - i)) & 1 for b in data for i in range(8)]
        self.pos = 0

    @property
    def remaining(self) -> int:
        return len(self.bits) - self.pos

    def read(self, n: int) -> int:
        value, shift, remaining = 0, 0, n
        while remaining > 0:
            take = min(8, remaining)
            chunk = 0
            for _ in range(take):
                chunk = (chunk << 1) | self.bits[self.pos]
                self.pos += 1
            value |= chunk << shift
            shift += take
            remaining -= take
        return value


def _write_primitive(w: _BitWriter, t: Primitive, value):
    if t.kind == "float":
        fmt = {16: "<e", 32: "<f", 64: "<d"}[t.bits]
        v = float(value)
        if t.bits == 16 and t.saturated and math.isfinite(v):
            v = max(-65504.0, min(65504.0, v))
        w.write(int.from_bytes(struct.pack(fmt, v), "little"), t.bits)
        return
    v = int(value)
    lo, hi = (0, (1 << t.bits) - 1) if t.kind in ("uint", "bool") else (-(1 << (t.bits - 1)), (1 << (t.bits - 1)) - 1)
    v = max(lo, min(hi, v)) if t.saturated else v
    w.write(v & ((1 << t.bits) - 1), t.bits)


def _read_primitive(r: _BitReader, t: Primitive):
    raw = r.read(t.bits)
    if t.kind == "float":
        fmt = {16: "<e", 32: "<f", 64: "<d"}[t.bits]
        return struct.unpack(fmt, raw.to_bytes(t.bits // 8, "little"))[0]
    if t.kind == "int" and raw >> (t.bits - 1):
        raw -= 1 << t.bits
    return bool(raw) if t.kind == "bool" else raw


def _write(w: _BitWriter, t, value, tao: bool):
    if isinstance(t, Primitive):
        _write_primitive(w, t, value)
    elif isinstance(t, Void):
        w.write(0, t.bits)
    elif isinstance(t, Array):
        items = list(value)[: t.size]
        if not t.dynamic:
            items += [_default(t.element)] * (t.size - len(items))
        elif tao and t.element.min_bits >= 8:
            for item in items:  # tail array: no length prefix, and the elements themselves are not optimised
                _write(w, t.element, item, False)
            return
        else:
            w.write(len(items), t.length_bits)
        for i, item in enumerate(items):
            _write(w, t.element, item, tao and i == len(items) - 1)
    else:
        _write_compound(w, t, value, tao)


def _read(r: _BitReader, t, tao: bool):
    if isinstance(t, Primitive):
        return _read_primitive(r, t)
    if isinstance(t, Void):
        r.read(t.bits)
        return None
    if isinstance(t, Array):
        if not t.dynamic:
            return [_read(r, t.element, tao and i == t.size - 1) for i in range(t.size)]
        if tao and t.element.min_bits >= 8:
            items = []
            while r.remaining >= 8 and len(items) < t.size:
                items.append(_read(r, t.element, False))
            return items
        n = r.read(t.length_bits)
        return [_read(r, t.element, tao and i == n - 1) for i in range(n)]
    return _read_compound(r, t, tao)


def _write_compound(w: _BitWriter, t: DsdlType, value: dict, tao: bool):
    for i, (name, ft) in enumerate(t.fields):
        last = i == len(t.fields) - 1
        _write(w, ft, None if name is None else value.get(name, _default(ft)), tao and last)


def _read_compound(r: _BitReader, t: DsdlType, tao: bool) -> dict:
    out = {}
    for i, (name, ft) in enumerate(t.fields):
        v = _read(r, ft, tao and i == len(t.fields) - 1)
        if name is not None:
            out[name] = v
    return out


# ---------------------------------------------------------- transfers ----


def message_id(priority: int, type_id: int, source_node: int) -> int:
    if not 1 <= source_node <= 127:
        raise ValueError("DroneCAN node ids run from 1 to 127 (0 is anonymous)")
    return ((priority & 0x1F) << 24) | ((type_id & 0xFFFF) << 8) | source_node


def parse_id(can_id: int) -> dict:
    return {
        "priority": (can_id >> 24) & 0x1F,
        "service": bool(can_id & 0x80),
        "type_id": (can_id >> 8) & 0xFFFF,
        "source_node": can_id & 0x7F,
    }


def transfer_frames(payload: bytes, signature: int, can_id: int, transfer_id: int) -> list[CanFrame]:
    tid = transfer_id & 0x1F
    if len(payload) <= 7:
        return [CanFrame(can_id, payload + bytes([0xC0 | tid]), extended=True)]
    crc = ccitt_false(payload, ccitt_false(signature.to_bytes(8, "little")))
    data = crc.to_bytes(2, "little") + payload
    frames = []
    chunks = [data[i : i + 7] for i in range(0, len(data), 7)]
    for i, chunk in enumerate(chunks):
        tail = (0x80 if i == 0 else 0) | (0x40 if i == len(chunks) - 1 else 0) | ((i & 1) << 5) | tid
        frames.append(CanFrame(can_id, chunk + bytes([tail]), extended=True))
    return frames


def encode_message(dtype: DsdlType, value: dict, source_node: int, transfer_id: int, priority: int = 16) -> list[CanFrame]:
    if dtype.type_id is None:
        raise ValueError(f"{dtype.full_name} has no default type id, it cannot be broadcast")
    return transfer_frames(dtype.encode(value), dtype.signature(), message_id(priority, dtype.type_id, source_node), transfer_id)


class TransferError(ValueError):
    pass


def reassemble(frames: list[CanFrame], signature: int) -> bytes:
    """Undo :func:`transfer_frames`, checking tail bits and the transfer CRC like a receiver must."""
    if not frames:
        raise TransferError("no frames")
    tails = [f.data[-1] for f in frames]
    if not tails[0] & 0x80 or not tails[-1] & 0x40:
        raise TransferError("missing start or end of transfer")
    for i, t in enumerate(tails):
        if (t >> 5) & 1 != i & 1:
            raise TransferError(f"toggle bit wrong in frame {i}")
        if t & 0x1F != tails[0] & 0x1F:
            raise TransferError("transfer id changed inside a transfer")
    body = b"".join(f.data[:-1] for f in frames)
    if len(frames) == 1:
        return body
    crc, payload = int.from_bytes(body[:2], "little"), body[2:]
    if ccitt_false(payload, ccitt_false(signature.to_bytes(8, "little"))) != crc:
        raise TransferError("transfer CRC mismatch (corrupted, or the two nodes disagree on the DSDL layout)")
    return payload


def bus_messages(dtype: DsdlType, value: dict, rate_hz: float) -> list[tuple[int, bool, float]]:
    """(data length, extended, rate) per CAN frame of a periodic message, for :func:`week04.can.bus_load`."""
    frames = encode_message(dtype, value, source_node=1, transfer_id=0)
    return [(len(f.data), True, rate_hz) for f in frames]
