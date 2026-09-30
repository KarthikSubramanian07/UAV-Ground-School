"""Classic CAN (ISO 11898-1) at the bit level: framing, bit stuffing, CRC-15 and arbitration.

The slides call CAN a device "group chat". What makes the group chat work
without a moderator:

* **Wired AND.** A 0 (dominant) from any node overrides 1s (recessive), so
  every node can listen while it talks.
* **Arbitration.** Everyone who wants to speak starts at once and sends its
  identifier MSB first. A node that sends a 1 but reads a 0 has lost and goes
  quiet, so the lowest identifier wins without a single bit being wasted.
  DroneCAN puts a priority field at the top of the identifier for this.
* **Bit stuffing.** After five equal bits the sender inserts one opposite
  bit, so the receivers always see edges to resynchronise their clocks.
* **CRC-15** over the unstuffed frame, and an ACK slot that any receiver
  pulls dominant, so the sender knows at least one node heard it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .crc import bytes_to_bits, can15


def _bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _int(bits: list[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


@dataclass(frozen=True)
class CanFrame:
    can_id: int
    data: bytes = b""
    extended: bool = False
    remote: bool = False

    def __post_init__(self):
        limit = 1 << (29 if self.extended else 11)
        if not 0 <= self.can_id < limit:
            raise ValueError(f"identifier 0x{self.can_id:X} does not fit a {'29' if self.extended else '11'} bit frame")
        if len(self.data) > 8:
            raise ValueError("classic CAN carries at most 8 data bytes")
        if self.remote and self.data:
            raise ValueError("a remote frame requests data, it carries none")

    # --------------------------------------------------------- encoding ----

    def header_bits(self) -> list[int]:
        """Start of frame through data field: the part the CRC covers."""
        dlc = _bits(len(self.data), 4)
        if self.extended:
            bits = [0] + _bits(self.can_id >> 18, 11) + [1, 1] + _bits(self.can_id & 0x3FFFF, 18)
            bits += [int(self.remote), 0, 0] + dlc
        else:
            bits = [0] + _bits(self.can_id, 11) + [int(self.remote), 0, 0] + dlc
        if not self.remote:
            bits += bytes_to_bits(self.data)
        return bits

    @property
    def arbitration_length(self) -> int:
        """Bits (before stuffing) that take part in arbitration, SOF excluded."""
        return 32 if self.extended else 12

    def crc(self) -> int:
        return can15(self.header_bits())

    def stuffed_bits(self) -> list[int]:
        """What goes on the wire from SOF to the end of the CRC, stuff bits included."""
        return stuff(self.header_bits() + _bits(self.crc(), 15))

    def wire_bits(self, acked: bool = True, interframe: bool = True) -> list[int]:
        tail = [1, 0 if acked else 1, 1] + [1] * 7  # CRC delimiter, ACK slot, ACK delimiter, EOF
        return self.stuffed_bits() + tail + ([1] * 3 if interframe else [])

    def duration_us(self, bitrate: int = 1_000_000) -> float:
        return len(self.wire_bits()) * 1e6 / bitrate


def stuff(bits: list[int]) -> list[int]:
    out: list[int] = []
    run_bit, run = -1, 0
    for b in bits:
        out.append(b)
        if b == run_bit:
            run += 1
        else:
            run_bit, run = b, 1
        if run == 5:
            out.append(1 - b)
            run_bit, run = 1 - b, 1  # the stuff bit starts the next run
    return out


class StuffError(ValueError):
    pass


def destuff(bits: list[int]) -> list[int]:
    out: list[int] = []
    run_bit, run = -1, 0
    i = 0
    while i < len(bits):
        b = bits[i]
        out.append(b)
        if b == run_bit:
            run += 1
        else:
            run_bit, run = b, 1
        if run == 5:
            i += 1
            if i < len(bits):
                if bits[i] == b:
                    raise StuffError(f"six equal bits at position {i}")
                run_bit, run = bits[i], 1
        i += 1
    return out


class CrcError(ValueError):
    pass


class TruncatedError(ValueError):
    pass


class _Destuffer:
    """Reads a stuffed bit stream one logical bit at a time, the way a CAN controller does.

    Destuffing has to be incremental: the receiver only learns where the
    stuffed region ends (after the CRC) once it has read the DLC.
    """

    def __init__(self, wire: list[int]):
        self.wire, self.pos, self.run_bit, self.run = wire, 0, -1, 0

    def take(self, n: int) -> list[int]:
        out = []
        for _ in range(n):
            if self.pos >= len(self.wire):
                raise TruncatedError("frame ends early")
            b = self.wire[self.pos]
            self.pos += 1
            out.append(b)
            if b == self.run_bit:
                self.run += 1
            else:
                self.run_bit, self.run = b, 1
            if self.run == 5:
                if self.pos >= len(self.wire):
                    raise TruncatedError("frame ends inside a stuff bit")
                stuffed = self.wire[self.pos]
                if stuffed == b:
                    raise StuffError(f"six equal bits at wire position {self.pos}")
                self.pos += 1
                self.run_bit, self.run = stuffed, 1
        return out


def decode(wire: list[int]) -> CanFrame:
    """Parse a frame from wire bits (as produced by :meth:`CanFrame.wire_bits`)."""
    r = _Destuffer(wire)
    raw = r.take(14)
    if raw[0] != 0:
        raise ValueError("frame must start with a dominant SOF")
    ide = raw[13]
    if ide:
        raw += r.take(25)
        can_id, remote, dlc_at = (_int(raw[1:12]) << 18) | _int(raw[14:32]), bool(raw[32]), 35
    else:
        raw += r.take(5)
        can_id, remote, dlc_at = _int(raw[1:12]), bool(raw[12]), 15
    dlc = _int(raw[dlc_at : dlc_at + 4])
    n = 0 if remote else min(dlc, 8)
    raw += r.take(8 * n)
    crc = _int(r.take(15))
    if can15(raw) != crc:
        raise CrcError(f"CRC mismatch: computed 0x{can15(raw):04X}, frame says 0x{crc:04X}")
    if r.pos >= len(wire) or wire[r.pos] != 1:
        raise ValueError("CRC delimiter must be recessive")
    data = bytes(_int(raw[dlc_at + 4 + 8 * k : dlc_at + 12 + 8 * k]) for k in range(n))
    return CanFrame(can_id, data, bool(ide), remote)


@dataclass
class Arbitration:
    winner: int
    lost_at: dict[int, int]  # node index -> bit index where it backed off
    bus: list[int]  # bus levels over the whole winning frame


def arbitrate(frames: list[CanFrame]) -> Arbitration:
    """Everyone starts transmitting at the same instant; simulate the wired AND bus bit by bit."""
    if not frames:
        raise ValueError("arbitration needs at least one transmitter")
    if len({(f.can_id, f.extended, f.remote) for f in frames}) != len(frames):
        raise ValueError("two nodes sending the same identifier is a configuration error on a CAN bus")
    streams = [f.wire_bits(interframe=False) for f in frames]
    active = set(range(len(frames)))
    lost: dict[int, int] = {}
    bus: list[int] = []
    t = 0
    while len(active) > 1:
        level = min(streams[i][t] for i in active)
        bus.append(level)
        for i in list(active):
            if streams[i][t] == 1 and level == 0:
                active.discard(i)
                lost[i] = t
        t += 1
    (w,) = active
    bus += streams[w][t:]
    return Arbitration(w, lost, bus)


def worst_case_bits(data_len: int, extended: bool) -> int:
    """Longest possible frame including stuff bits and interframe space (Davis et al., 2007)."""
    g = 54 if extended else 34  # bits exposed to stuffing, excluding data
    return g + 8 * data_len + 13 + (g + 8 * data_len - 1) // 4


def bus_load(messages: list[tuple[int, bool, float]], bitrate: int = 1_000_000) -> float:
    """Fraction of the bus used by (data length, extended, rate in Hz) message streams, worst case stuffing."""
    return sum(worst_case_bits(n, ext) * hz for n, ext, hz in messages) / bitrate
