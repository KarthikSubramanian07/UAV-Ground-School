"""CDR, the byte layout of every ROS 2 message on the wire.

ROS 2 messages travel as OMG CDR (XCDR version 1, "plain CDR"), the format
``rmw_fastrtps`` and ``rmw_cyclonedds`` both use:

* a 4 byte encapsulation header: ``00 01`` for little endian CDR, ``00 00`` for
  big endian, then two option bytes;
* each primitive aligned to its own size (8 byte types to 8), counted from the
  end of the header;
* a string is a uint32 length that includes the terminating NUL, then the bytes;
* a sequence is a uint32 element count, then the elements; fixed arrays have no
  count; nested messages are laid out inline.

Values are plain Python: ``dict`` for messages, ``int``/``float``/``bool``/``str``,
``list`` for arrays, and ``bytes`` for ``uint8``/``byte``/``char`` arrays. Fields
that are missing take ROS 2's zero defaults. The tests hold :func:`serialize`
to ``rclpy.serialization.serialize_message`` byte for byte.
"""

from __future__ import annotations

import struct

from .idl import PRIMITIVES, Message, Registry, Type

ENCAPSULATION_LE = b"\x00\x01\x00\x00"
ENCAPSULATION_BE = b"\x00\x00\x00\x00"

_FORMAT = {
    "bool": "?",
    "byte": "B",
    "char": "B",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}
_OCTETS = ("uint8", "byte", "char")
_RANGES = {
    "int8": (-(2**7), 2**7 - 1),
    "uint8": (0, 2**8 - 1),
    "byte": (0, 2**8 - 1),
    "char": (0, 2**8 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "uint16": (0, 2**16 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "uint32": (0, 2**32 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint64": (0, 2**64 - 1),
}


class CdrError(ValueError):
    """A value does not fit its type, or bytes do not decode as the type."""


def default_value(t: Type, registry: Registry):
    """ROS 2's zero value for a field type (ignoring .msg defaults)."""
    if t.array == "array":
        if t.base in _OCTETS:
            return bytes(t.size)
        return [default_value(t.element(), registry) for _ in range(t.size)]
    if t.array:
        return b"" if t.base in _OCTETS else []
    if t.is_string:
        return ""
    if t.base == "bool":
        return False
    if t.base in ("float32", "float64"):
        return 0.0
    if t.is_primitive:
        return 0
    return message_defaults(registry.message(t.base), registry)


def _literal(text: str, t: Type):
    """Turn a .msg default literal into a Python value."""
    import ast

    if t.array:
        items = ast.literal_eval(text)
        return [_literal(repr(x), t.element()) for x in items]
    if t.is_string:
        return ast.literal_eval(text) if text[:1] in "\"'" else text
    if t.base == "bool":
        return text.strip().lower() in ("true", "1")
    if t.base in ("float32", "float64"):
        return float(text)
    return int(text, 0)


def message_defaults(msg: Message, registry: Registry, use_msg_defaults: bool = True) -> dict:
    out = {}
    for f in msg.fields:
        if use_msg_defaults and f.default is not None:
            value = _literal(f.default, f.type)
            if f.type.base in _OCTETS and f.type.array:
                value = bytes(value)
            out[f.name] = value
        else:
            out[f.name] = default_value(f.type, registry)
    return out


class _Writer:
    def __init__(self, e: str) -> None:
        self.e = e
        self.buf = bytearray()
        self.padding: list[int] = []  # offsets (in the body) of alignment padding

    def align(self, n: int) -> None:
        gap = -len(self.buf) % n
        self.padding.extend(range(len(self.buf), len(self.buf) + gap))
        self.buf += bytes(gap)

    def prim(self, base: str, value) -> None:
        size = PRIMITIVES[base]
        self.align(size)
        if base in _RANGES:
            if isinstance(value, bytes | bytearray) and len(value) == 1:
                value = value[0]
            if isinstance(value, bool) or not isinstance(value, int):
                raise CdrError(f"{base} needs an int, got {value!r}")
            lo, hi = _RANGES[base]
            if not lo <= value <= hi:
                raise CdrError(f"{value} is out of range for {base}")
        elif base == "bool":
            value = bool(value)
        else:
            value = float(value)
        self.buf += struct.pack(self.e + _FORMAT[base], value)

    def count(self, n: int) -> None:
        self.align(4)
        self.buf += struct.pack(self.e + "I", n)


def _write(w: _Writer, t: Type, value, registry: Registry) -> None:
    if t.array:
        if t.array == "array" and len(value) != t.size:
            raise CdrError(f"{t} needs exactly {t.size} elements, got {len(value)}")
        if t.array == "bounded" and len(value) > t.size:
            raise CdrError(f"{t} holds at most {t.size} elements, got {len(value)}")
        if t.array != "array":
            w.count(len(value))
        if t.base in _OCTETS and isinstance(value, bytes | bytearray):
            w.buf += value
            return
        elem = t.element()
        for v in value:
            _write(w, elem, v, registry)
        return
    if t.base == "string":
        raw = value.encode("utf-8")
        if t.string_bound and len(raw) > t.string_bound:
            raise CdrError(f"string longer than its bound {t.string_bound}")
        w.count(len(raw) + 1)
        w.buf += raw + b"\0"
        return
    if t.base == "wstring":
        units = value.encode("utf-16-le")
        n = len(units) // 2
        if t.string_bound and n > t.string_bound:
            raise CdrError(f"wstring longer than its bound {t.string_bound}")
        w.count(n)
        # Fast CDR writes each UTF-16 code unit as a 4 byte wchar_t
        w.buf += b"".join(struct.pack(w.e + "I", u) for (u,) in struct.iter_unpack("<H", units))
        return
    if t.is_primitive:
        w.prim(t.base, value)
        return
    _write_message(w, registry.message(t.base), value, registry)


def _write_message(w: _Writer, msg: Message, value: dict, registry: Registry) -> None:
    if not isinstance(value, dict):
        raise CdrError(f"{msg.name} needs a dict, got {type(value).__name__}")
    unknown = set(value) - {f.name for f in msg.fields}
    if unknown:
        raise CdrError(f"{msg.name} has no field(s) {sorted(unknown)}")
    for f in msg.fields:
        v = value[f.name] if f.name in value else default_value(f.type, registry)
        try:
            _write(w, f.type, v, registry)
        except CdrError as error:
            raise CdrError(f"{msg.name}.{f.name}: {error}") from None


def serialize(type_name: str, value: dict, registry: Registry | None = None, big_endian: bool = False, pad: bool = False) -> bytes:
    """Serialize ``value`` as ``type_name`` (e.g. ``std_msgs/msg/String``) with the header.

    ``pad=True`` pads the payload to a multiple of 4 bytes, as Fast DDS does on the
    wire; ``rclpy.serialization`` does not pad, so neither does the default.
    Alignment padding is written as zeros.
    """
    return serialize_with_padding(type_name, value, registry, big_endian, pad)[0]


def serialize_with_padding(
    type_name: str, value: dict, registry: Registry | None = None, big_endian: bool = False, pad: bool = False
) -> tuple[bytes, set[int]]:
    """Like :func:`serialize`, plus the offsets of every alignment padding byte.

    Fast CDR (behind rclpy and rmw_fastrtps) skips over padding without writing
    it, so those bytes hold whatever was in memory: two correct serializations
    agree on every byte except these.
    """
    registry = registry or _default_registry()
    w = _Writer(">" if big_endian else "<")
    _write_message(w, registry.message(type_name), value, registry)
    if pad:
        w.align(4)
    header = ENCAPSULATION_BE if big_endian else ENCAPSULATION_LE
    return header + bytes(w.buf), {len(header) + i for i in w.padding}


def same_except_padding(ours: bytes, theirs: bytes, padding: set[int]) -> bool:
    """Equal length and equal on every byte that is not alignment padding."""
    return len(ours) == len(theirs) and all(a == b for i, (a, b) in enumerate(zip(ours, theirs)) if i not in padding)


class _Reader:
    def __init__(self, data: bytes, e: str) -> None:
        self.data, self.e, self.pos = data, e, 0

    def align(self, n: int) -> None:
        self.pos += -self.pos % n

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise CdrError(f"needs {n} bytes at offset {self.pos}, only {len(self.data) - self.pos} left")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def prim(self, base: str):
        size = PRIMITIVES[base]
        self.align(size)
        return struct.unpack(self.e + _FORMAT[base], self.take(size))[0]

    def count(self) -> int:
        self.align(4)
        return struct.unpack(self.e + "I", self.take(4))[0]


def _read(r: _Reader, t: Type, registry: Registry):
    if t.array:
        n = t.size if t.array == "array" else r.count()
        if t.array == "bounded" and n > t.size:
            raise CdrError(f"{t}: {n} elements exceed the bound")
        if t.base in _OCTETS:
            return bytes(r.take(n))
        if n > len(r.data):
            raise CdrError(f"{t}: implausible length {n}")
        elem = t.element()
        return [_read(r, elem, registry) for _ in range(n)]
    if t.base == "string":
        n = r.count()
        raw = r.take(n)
        if n == 0 or raw[-1:] != b"\0":
            raise CdrError("string is not NUL terminated")
        return raw[:-1].decode("utf-8")
    if t.base == "wstring":
        n = r.count()
        units = struct.unpack(r.e + "I" * n, r.take(4 * n))
        return struct.pack("<" + "H" * n, *units).decode("utf-16-le")
    if t.is_primitive:
        return r.prim(t.base)
    return _read_message(r, registry.message(t.base), registry)


def _read_message(r: _Reader, msg: Message, registry: Registry) -> dict:
    return {f.name: _read(r, f.type, registry) for f in msg.fields}


def deserialize(type_name: str, data: bytes, registry: Registry | None = None) -> dict:
    """Decode a serialized payload (header included). Trailing padding is allowed."""
    registry = registry or _default_registry()
    if len(data) < 4 or data[0] != 0 or data[1] not in (0, 1):
        raise CdrError(f"not a plain CDR payload (header {data[:4].hex()})")
    r = _Reader(bytes(data[4:]), "<" if data[1] == 1 else ">")
    value = _read_message(r, registry.message(type_name), registry)
    if len(r.data) - r.pos > 3:
        raise CdrError(f"{len(r.data) - r.pos} unexpected bytes after {type_name}")
    return value


_REGISTRY: Registry | None = None


def _default_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY
