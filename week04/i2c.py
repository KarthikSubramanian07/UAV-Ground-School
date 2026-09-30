"""I2C at the level of the two wires: SDA (data) and SCL (clock).

Both lines are open drain with pull up resistors, so any device can pull a
line low and nobody drives it high. That is what lets several devices share
two wires, with one master clocking at a time:

* **START**: SDA falls while SCL is high. **STOP**: SDA rises while SCL is high.
  Everywhere else SDA only changes while SCL is low.
* A byte is 8 bits MSB first; the receiver then pulls SDA low for one clock
  (ACK) or leaves it high (NACK).
* The first byte is the 7 bit address plus a read (1) or write (0) bit. A
  device that does not recognise the address simply does not ACK, which is
  also how an address clash goes unnoticed until two devices answer at once.

:func:`transaction` returns SDA and SCL sampled four times per bit, plus a
list of labelled segments for drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trace:
    sda: list[int] = field(default_factory=list)
    scl: list[int] = field(default_factory=list)
    segments: list[tuple[int, int, str]] = field(default_factory=list)  # start sample, end sample, label

    def _bit(self, value: int):
        # SCL low, set SDA, SCL high (receiver samples), SCL low
        self.scl += [0, 1, 1, 0]
        self.sda += [value] * 4

    def byte(self, value: int, label: str, ack: bool = True):
        start = len(self.sda)
        for i in range(7, -1, -1):
            self._bit((value >> i) & 1)
        self.segments.append((start, len(self.sda), label))
        start = len(self.sda)
        self._bit(0 if ack else 1)
        self.segments.append((start, len(self.sda), "ACK" if ack else "NACK"))


def transaction(address: int, write: bytes = b"", read: bytes = b"", present: bool = True) -> Trace:
    """A write of ``write`` then (with a repeated START) a read of ``read``, as the master sees the bus."""
    if not 0 <= address < 0x80:
        raise ValueError("7 bit addresses only")
    t = Trace()
    t.scl += [1, 1]
    t.sda += [1, 0]  # START
    t.segments.append((0, 2, "START"))
    if write or not read:
        t.byte(address << 1, f"0x{address:02X} write", ack=present)
        if present:
            for b in write:
                t.byte(b, f"0x{b:02X}")
    if read and present:
        start = len(t.sda)
        t.scl += [0, 1, 1]
        t.sda += [1, 1, 0]  # repeated START
        t.segments.append((start, len(t.sda), "Sr"))
        t.byte(address << 1 | 1, f"0x{address:02X} read")
        for i, b in enumerate(read):
            t.byte(b, f"0x{b:02X}", ack=i < len(read) - 1)  # the master NACKs the last byte
    start = len(t.sda)
    t.scl += [0, 1, 1]
    t.sda += [0, 0, 1]  # STOP
    t.segments.append((start, len(t.sda), "STOP"))
    return t


def decode(sda: list[int], scl: list[int]) -> list[tuple[str, int]]:
    """Recover (event, value) from sampled lines: START, STOP, and bytes with their ACK bit."""
    events: list[tuple[str, int]] = []
    bits: list[int] = []
    for i in range(1, len(sda)):
        if scl[i] and scl[i - 1] and sda[i] != sda[i - 1]:
            events.append(("START" if sda[i] == 0 else "STOP", 0))
            bits = []
        elif scl[i] and not scl[i - 1]:  # rising clock edge: sample
            bits.append(sda[i])
            if len(bits) == 9:
                value = sum(b << (7 - k) for k, b in enumerate(bits[:8]))
                events.append(("ACK" if bits[8] == 0 else "NACK", value))
                bits = []
    return events
