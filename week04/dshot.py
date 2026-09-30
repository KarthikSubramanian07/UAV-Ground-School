"""DShot: the digital ESC protocol, including bidirectional eRPM telemetry.

A DShot frame is 16 bits sent MSB first::

    11 bit value | telemetry request bit | 4 bit CRC

Values 1 to 47 are commands (beep, spin direction, save settings...), 48 to
2047 are throttle and 0 means disarmed. Each bit is a fixed length pulse: a
1 is high for three quarters of the bit period, a 0 for three eighths, so the
ESC decodes by pulse width and needs no calibration, unlike PWM.

In bidirectional DShot the signal is inverted (idle low becomes idle high),
the CRC is inverted, and after each frame the ESC replies on the same wire
with the electrical period of the motor: a 16 bit word (3 bit exponent,
9 bit mantissa, 4 bit CRC) GCR coded into 20 bits and sent as 21 transitions.
ArduPilot uses the resulting RPM for its harmonic notch filter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .crc import dshot_crc

RATES_KBIT = {150: 150, 300: 300, 600: 600, 1200: 1200}
THROTTLE_MIN, THROTTLE_MAX = 48, 2047

COMMANDS = {
    0: "MOTOR_STOP",
    1: "BEEP1",
    6: "ESC_INFO",
    7: "SPIN_DIRECTION_1",
    8: "SPIN_DIRECTION_2",
    9: "3D_MODE_OFF",
    10: "3D_MODE_ON",
    12: "SAVE_SETTINGS",
    13: "EXTENDED_TELEMETRY_ENABLE",
    20: "SPIN_DIRECTION_NORMAL",
    21: "SPIN_DIRECTION_REVERSED",
}


def encode(value: int, telemetry: bool = False, bidirectional: bool = False) -> int:
    if not 0 <= value <= 2047:
        raise ValueError("DShot values are 11 bits")
    payload = (value << 1) | int(telemetry)
    return (payload << 4) | dshot_crc(payload, inverted=bidirectional)


def throttle_value(fraction: float) -> int:
    """Map 0..1 throttle onto the 48..2047 range (0 stays disarmed)."""
    if fraction <= 0:
        return 0
    return round(THROTTLE_MIN + min(fraction, 1.0) * (THROTTLE_MAX - THROTTLE_MIN))


@dataclass(frozen=True)
class DshotFrame:
    value: int
    telemetry: bool
    crc_ok: bool

    @property
    def is_command(self) -> bool:
        return 0 < self.value < THROTTLE_MIN

    @property
    def throttle(self) -> float | None:
        return None if self.value < THROTTLE_MIN else (self.value - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)


def decode(frame: int, bidirectional: bool = False) -> DshotFrame:
    payload = frame >> 4
    return DshotFrame(payload >> 1, bool(payload & 1), dshot_crc(payload, inverted=bidirectional) == frame & 0xF)


@dataclass(frozen=True)
class Timing:
    bit_ns: float
    t1h_ns: float
    t0h_ns: float
    frame_us: float


def timing(rate_kbit: int) -> Timing:
    """Pulse widths for DShot150/300/600/1200. A 16 bit frame at DShot600 takes 26.7 us."""
    bit = 1e6 / rate_kbit  # ns
    return Timing(bit, bit * 0.75, bit * 0.375, 16 * bit / 1000)


def waveform(frame: int, rate_kbit: int = 600, samples_per_bit: int = 8, bidirectional: bool = False) -> list[int]:
    """Pulse train sampled ``samples_per_bit`` times per bit, for drawing the scope trace."""
    high0 = round(samples_per_bit * 0.375)
    high1 = round(samples_per_bit * 0.75)
    levels: list[int] = []
    for i in range(15, -1, -1):
        high = high1 if (frame >> i) & 1 else high0
        levels += [1] * high + [0] * (samples_per_bit - high)
    return [1 - v for v in levels] if bidirectional else levels


def decode_waveform(levels: list[int], samples_per_bit: int = 8, bidirectional: bool = False) -> int:
    if bidirectional:
        levels = [1 - v for v in levels]
    frame = 0
    for i in range(16):
        high = sum(levels[i * samples_per_bit : (i + 1) * samples_per_bit])
        frame = (frame << 1) | int(high > samples_per_bit / 2)
    return frame


# ------------------------------------------------------ eRPM telemetry ----

GCR = [0x19, 0x1B, 0x12, 0x13, 0x1D, 0x15, 0x16, 0x17, 0x1A, 0x09, 0x0A, 0x0B, 0x1E, 0x0D, 0x0E, 0x0F]
GCR_INV = {code: nibble for nibble, code in enumerate(GCR)}


def erpm_word(period_us: int) -> int:
    """Pack an electrical period into 3 bit exponent and 9 bit mantissa, plus CRC."""
    if period_us <= 0:
        raise ValueError("period must be positive")
    exponent = 0
    mantissa = period_us
    while mantissa > 0x1FF:
        mantissa >>= 1
        exponent += 1
    if exponent > 7:
        mantissa, exponent = 0x1FF, 7
    value = (exponent << 9) | mantissa
    return (value << 4) | ((~(value ^ (value >> 4) ^ (value >> 8))) & 0xF)


def gcr_encode(word: int) -> int:
    """Four nibbles to 20 GCR bits, then to 20 line levels where a GCR 1 means "the line toggles"."""
    gcr = 0
    for shift in (12, 8, 4, 0):
        gcr = (gcr << 5) | GCR[(word >> shift) & 0xF]
    line, level = 0, 0
    for i in range(19, -1, -1):
        level ^= (gcr >> i) & 1
        line = (line << 1) | level
    return line  # 20 line levels following an implicit low start level


def gcr_decode(line: int) -> tuple[int, bool]:
    line &= 0xFFFFF
    gcr = line ^ (line >> 1)  # a transition between neighbours is a GCR 1
    word = 0
    for shift in (15, 10, 5, 0):
        code = (gcr >> shift) & 0x1F
        if code not in GCR_INV:
            return 0, False
        word = (word << 4) | GCR_INV[code]
    value, crc = word >> 4, word & 0xF
    ok = ((~(value ^ (value >> 4) ^ (value >> 8))) & 0xF) == crc
    return word, ok


def erpm_to_rpm(period_word: int, motor_poles: int) -> float:
    value = period_word >> 4
    period_us = (value & 0x1FF) << (value >> 9)
    if period_us == 0 or period_us >= 65408:  # the ESC's "motor stopped" code
        return 0.0
    erpm = 60e6 / period_us
    return erpm / (motor_poles / 2)
