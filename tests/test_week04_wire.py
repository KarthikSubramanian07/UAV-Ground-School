"""Checksums, UART framing, SBUS, CRSF, DShot and I2C at the bit level."""

import random

import pytest

from week04 import crc, dshot, i2c, rc, uart

# ------------------------------------------------------------------ CRC ----


@pytest.mark.parametrize(
    "fn,expected",
    [
        (lambda d: crc.x25_accumulate(d), 0x6F91),
        (lambda d: crc.x25_bitwise(d), 0x6F91),
        (lambda d: crc.ccitt_false(d), 0x29B1),
        (lambda d: crc.dvb_s2(d), 0xBC),
        (lambda d: crc.can15(crc.bytes_to_bits(d)), 0x059E),
    ],
)
def test_crc_catalogue_check_values(fn, expected):
    assert fn(crc.CHECK_INPUT) == expected


def test_x25_table_and_bitwise_agree():
    rng = random.Random(4)
    for _ in range(200):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(40)))
        assert crc.x25_accumulate(data) == crc.x25_bitwise(data)


def test_dvb_s2_rejects_state_outside_a_byte():
    with pytest.raises(ValueError):
        crc.dvb_s2(b"x", 256)


def test_dshot_crc_is_nibble_xor_and_inverts_for_bidirectional():
    assert crc.dshot_crc(0x123) == (0x1 ^ 0x2 ^ 0x3)
    assert crc.dshot_crc(0x123, inverted=True) == (~(0x1 ^ 0x2 ^ 0x3)) & 0xF


# ----------------------------------------------------------------- UART ----


def test_uart_frame_layout_8n1():
    # 0x55 is 01010101: LSB first after a low start bit, then a high stop bit
    assert uart.frame_byte(0x55) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("config", [uart.UartConfig(), uart.SBUS, uart.UartConfig(9600, 7, "O", 1), uart.CRSF])
def test_uart_round_trip(config):
    data = bytes(range(0, 256, 7))
    assert uart.decode(uart.encode(data, config, idle_bits=3), config) == bytes(b & ((1 << config.data_bits) - 1) for b in data)


def test_uart_detects_parity_and_stop_errors():
    levels = uart.encode(b"\x0f", uart.SBUS)
    flipped = list(levels)
    flipped[9] ^= 1  # parity bit
    with pytest.raises(uart.FramingError, match="parity"):
        uart.decode(flipped, uart.SBUS)
    plain = uart.encode(b"A")
    plain[-1] = 0
    with pytest.raises(uart.FramingError):
        uart.decode(plain + [1] * 10)


def test_sbus_is_inverted_and_reads_as_garbage_on_a_plain_uart():
    levels = uart.encode(bytes([0x0F]), uart.SBUS)
    assert levels[0] == 1  # an inverted start bit is high
    with pytest.raises(uart.FramingError):
        uart.decode(levels + [0] * 12, uart.UartConfig(100000, 8, "E", 2))


def test_throughput_figures():
    assert uart.UartConfig(57600).bytes_per_second == 5760
    assert uart.SBUS.bits_per_frame == 12
    assert uart.UartConfig(57600).label() == "57600 8N1"


# ------------------------------------------------------------- SBUS/CRSF ----


def test_raw_channel_mapping_matches_ardupilot_scale():
    assert rc.raw_to_us(992) == 1500
    assert rc.raw_to_us(172) == 987.5
    assert abs(rc.raw_to_us(1811) - 2011.875) < 1e-9
    for us in (1000, 1234, 1500, 2000):
        assert abs(rc.raw_to_us(rc.us_to_raw(us)) - us) <= 0.32


def test_pack11_round_trip_and_bounds():
    rng = random.Random(1)
    for _ in range(100):
        ch = [rng.randrange(2048) for _ in range(16)]
        assert rc.unpack11(rc.pack11(ch)) == ch
    with pytest.raises(ValueError):
        rc.pack11([2048] + [0] * 15)
    with pytest.raises(ValueError):
        rc.pack11([0] * 15)


def test_sbus_frame_round_trip_with_flags():
    f = rc.SbusFrame([100 * i for i in range(16)], ch17=True, frame_lost=True, failsafe=True)
    data = f.encode()
    assert len(data) == 25 and data[0] == 0x0F and data[-1] == 0x00 and data[23] == 0b1101
    assert rc.SbusFrame.decode(data) == f
    with pytest.raises(ValueError):
        rc.SbusFrame.decode(data[:-1] + b"\x01")


def test_crsf_frames_decode_to_what_was_encoded():
    frames = [
        (rc.crsf_rc(list(range(172, 172 + 16 * 100, 100))), {"channels": list(range(172, 172 + 16 * 100, 100))}),
        (rc.crsf_battery(15.2, 21.4, 1234, 76), {"volts": 15.2, "amps": 21.4, "used_mah": 1234, "percent": 76}),
        (rc.crsf_attitude(0.1, -0.2, 3.0), {"pitch": 0.1, "roll": -0.2, "yaw": 3.0}),
        (rc.crsf_flight_mode("LOITER"), {"mode": "LOITER"}),
        (rc.crsf_link_statistics(70, 72, 100, -5, 1, 7, 3, 80, 99, 4), {"rssi1": 70, "lq": 100, "snr": -5, "d_snr": 4}),
    ]
    parser = rc.CrsfParser()
    stream = b"".join(f for f, _ in frames)
    packets = list(parser.feed(stream))
    assert len(packets) == len(frames)
    for packet, (_, expected) in zip(packets, frames):
        decoded = packet.decoded()
        for k, v in expected.items():
            assert decoded[k] == pytest.approx(v)
    gps = rc.crsf_gps(37.8719, -122.2585, 36.0, 270.5, 120.0, 17)
    assert list(rc.CrsfParser().feed(gps))[0].decoded()["sats"] == 17


def test_crsf_rc_frame_is_26_bytes_with_a_valid_crc():
    f = rc.crsf_rc([992] * 16)
    assert len(f) == 26 and f[:3] == bytes([0xC8, 24, 0x16])
    assert crc.dvb_s2(f[2:-1]) == f[-1]


def test_crsf_parser_resyncs_after_noise_and_bad_crc():
    good = rc.crsf_rc([992] * 16)
    bad = bytearray(good)
    bad[10] ^= 0xFF
    parser = rc.CrsfParser()
    stream = b"\x00\x13\x37" + bytes(bad) + good + b"\xc8"
    out = []
    for i in range(0, len(stream), 5):  # arrives in small chunks, like a real UART
        out += list(parser.feed(stream[i : i + 5]))
    assert len(out) == 1 and out[0].decoded()["channels"] == [992] * 16
    assert parser.crc_errors >= 1 and parser.dropped_bytes >= 3


def test_crsf_frames_are_limited_to_64_bytes():
    with pytest.raises(ValueError):
        rc.crsf_frame(0x7F, bytes(62))


# ---------------------------------------------------------------- DShot ----


def test_dshot_encode_decode_all_values():
    for value in range(0, 2048, 7):
        for telemetry in (False, True):
            for bidir in (False, True):
                f = dshot.decode(dshot.encode(value, telemetry, bidir), bidir)
                assert (f.value, f.telemetry, f.crc_ok) == (value, telemetry, True)
    assert not dshot.decode(dshot.encode(1000) ^ 1).crc_ok


def test_dshot_throttle_mapping_and_commands():
    assert dshot.throttle_value(0) == 0
    assert dshot.throttle_value(1) == 2047
    assert dshot.throttle_value(0.5) == 1048
    assert dshot.decode(dshot.encode(7)).is_command
    assert dshot.decode(dshot.encode(48)).throttle == 0.0


def test_dshot_timing_and_waveform_round_trip():
    t = dshot.timing(600)
    assert abs(t.frame_us - 26.667) < 0.01
    assert t.t1h_ns == 2 * t.t0h_ns
    for bidir in (False, True):
        f = dshot.encode(1234, True, bidir)
        assert dshot.decode_waveform(dshot.waveform(f, 600, 8, bidir), 8, bidir) == f


def test_gcr_round_trip_for_every_period():
    for period in list(range(1, 5000, 3)) + [65000, 100000]:
        word = dshot.erpm_word(period)
        decoded, ok = dshot.gcr_decode(dshot.gcr_encode(word))
        assert ok and decoded == word


def test_gcr_detects_corruption_and_converts_rpm():
    line = dshot.gcr_encode(dshot.erpm_word(1000))
    assert not dshot.gcr_decode(line ^ 0b1000)[1] or dshot.gcr_decode(line ^ 0b1000)[0] != dshot.erpm_word(1000)
    # a 1000 us electrical period is 60000 eRPM; a 14 pole motor turns at a seventh of that
    assert abs(dshot.erpm_to_rpm(dshot.erpm_word(1000), 14) - 60000 / 7) < 1e-6
    assert dshot.erpm_to_rpm(dshot.erpm_word(70000), 14) == 0.0


# ------------------------------------------------------------------ I2C ----


def test_i2c_transaction_decodes_back():
    t = i2c.transaction(0x10, write=b"\x00", read=b"\x59\x59")
    events = i2c.decode(t.sda, t.scl)
    assert events == [("START", 0), ("ACK", 0x20), ("ACK", 0x00), ("START", 0), ("ACK", 0x21), ("ACK", 0x59), ("NACK", 0x59), ("STOP", 0)]
    labels = [s[2] for s in t.segments]
    assert labels[0] == "START" and labels[-1] == "STOP" and "Sr" in labels


def test_i2c_absent_device_is_a_nack():
    t = i2c.transaction(0x42, present=False)
    assert i2c.decode(t.sda, t.scl)[1] == ("NACK", 0x84)
    with pytest.raises(ValueError):
        i2c.transaction(0x80)


def test_sda_only_changes_while_scl_is_low_except_start_and_stop():
    t = i2c.transaction(0x10, write=b"\xa5")
    changes = [i for i in range(1, len(t.sda)) if t.sda[i] != t.sda[i - 1]]
    for i in changes:
        if t.scl[i] and t.scl[i - 1]:
            assert any(s <= i < e and lab in ("START", "STOP", "Sr") for s, e, lab in t.segments)
