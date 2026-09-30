"""UART at the level of the wire: start bit, data bits LSB first, parity, stop bits.

A UART line idles high. Each byte is a low start bit, the data bits least
significant first, an optional parity bit and one or two high stop bits.
Both ends must agree on all of it (baud rate, data bits, parity, stop bits and
polarity) because there is no clock wire: the receiver resynchronises on
every start bit and samples in the middle of each bit period.

SBUS is the classic trap: 100000 baud, 8 data bits, even parity, two stop
bits, and the signal is *inverted*, so a plain UART reads garbage unless the
flight controller has an inverter (the Cube's RCIN port does).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UartConfig:
    baud: int = 115200
    data_bits: int = 8
    parity: str = "N"  # N, E or O
    stop_bits: int = 1
    inverted: bool = False

    @property
    def bits_per_frame(self) -> int:
        return 1 + self.data_bits + (0 if self.parity == "N" else 1) + self.stop_bits

    @property
    def bytes_per_second(self) -> float:
        """Payload bytes per second at 100 percent line utilisation."""
        return self.baud / self.bits_per_frame

    @property
    def bit_time_us(self) -> float:
        return 1e6 / self.baud

    def label(self) -> str:
        text = f"{self.baud} {self.data_bits}{self.parity}{self.stop_bits}"
        return text + " inverted" if self.inverted else text


UART_8N1 = UartConfig()
SBUS = UartConfig(baud=100_000, data_bits=8, parity="E", stop_bits=2, inverted=True)
CRSF = UartConfig(baud=420_000)
MAVLINK_TELEM = UartConfig(baud=57_600)
MAVLINK_COMPANION = UartConfig(baud=921_600)


class FramingError(ValueError):
    pass


def _parity_bit(byte: int, data_bits: int, parity: str) -> int:
    ones = bin(byte & ((1 << data_bits) - 1)).count("1")
    return ones & 1 if parity == "E" else (ones & 1) ^ 1


def frame_byte(byte: int, config: UartConfig = UART_8N1) -> list[int]:
    """Logic levels for one character, one entry per bit period (1 = high)."""
    bits = [0]
    bits += [(byte >> i) & 1 for i in range(config.data_bits)]
    if config.parity != "N":
        bits.append(_parity_bit(byte, config.data_bits, config.parity))
    bits += [1] * config.stop_bits
    return [b ^ 1 for b in bits] if config.inverted else bits


def encode(data: bytes, config: UartConfig = UART_8N1, idle_bits: int = 0) -> list[int]:
    """Levels for a burst of bytes, with optional idle (mark) time between characters."""
    idle = 0 if config.inverted else 1
    levels: list[int] = []
    for b in data:
        levels += frame_byte(b, config)
        levels += [idle] * idle_bits
    return levels


def decode(levels: list[int], config: UartConfig = UART_8N1) -> bytes:
    """Recover bytes from bit period samples, checking parity and stop bits like a real UART."""
    line = [b ^ 1 for b in levels] if config.inverted else list(levels)
    out = bytearray()
    i = 0
    n = config.bits_per_frame
    while i < len(line):
        if line[i] == 1:  # idle
            i += 1
            continue
        if i + n > len(line):
            raise FramingError(f"truncated character at bit {i}")
        data = line[i + 1 : i + 1 + config.data_bits]
        value = sum(bit << k for k, bit in enumerate(data))
        j = i + 1 + config.data_bits
        if config.parity != "N":
            if line[j] != _parity_bit(value, config.data_bits, config.parity):
                raise FramingError(f"parity error in character starting at bit {i}")
            j += 1
        if any(line[j + k] != 1 for k in range(config.stop_bits)):
            raise FramingError(f"missing stop bit in character starting at bit {i}")
        out.append(value)
        i += n
    return bytes(out)


def baud_error(nominal: int, actual: int) -> float:
    """Relative clock mismatch. UARTs tolerate roughly 2 to 3 percent in total across both ends."""
    return abs(actual - nominal) / nominal


def utilisation(bytes_per_second: float, config: UartConfig) -> float:
    return bytes_per_second / config.bytes_per_second
