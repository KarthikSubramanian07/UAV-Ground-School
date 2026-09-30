# Week 04 in C++17

A C++17 port of the week 04 protocol codecs: the checksums, the two RC receiver protocols, DShot, classic CAN at the bit level and MAVLink framing. It builds one executable, `protocol_tool`, and uses nothing but the standard library (no OpenCV, no crypto library: the SHA-256 that MAVLink signing needs is written out in `protocols.cpp`).

| file | ports | what it does |
| --- | --- | --- |
| `protocols.hpp`, `protocols.cpp` | `crc.py`, `rc.py`, `dshot.py`, `can.py`, the framing half of `mavlink.py` | the codecs, as a small library |
| `protocol_tool.cpp` | | reads one request per line on stdin, writes one JSON object per request on stdout |

Every function follows its Python original step for step, including Python's `round` (half to even), the struct range checks and the byte by byte resynchronisation of the stream parsers, so the outputs are byte exact (see [parity](#check-parity-with-python)). Failures are reported with the name of the exception Python raises in the same place.

## Build

From the repository root (only CMake and a C++17 compiler are needed):

```sh
cmake -S week04/cpp -B build/cpp4
cmake --build build/cpp4 -j
```

The default is a Release build compiled with `-Wall -Wextra -ffp-contract=off` (no fused multiply add, so `lat * 1e7` rounds exactly like Python before `round`).

## Run

Requests are whitespace separated tokens: bytes as hex, bits as strings of `0` and `1`, flags as `0` or `1`, and `-` for an empty value.

```sh
printf '%s\n' \
  'crc x25 313233343536373839' \
  'crsf_attitude 0.1 -0.25 3.1' \
  'dshot_encode 1046 0 1' \
  'can_frame 0 0 291 1122' \
  | build/cpp4/protocol_tool
# {"crc": 28561}
# {"hex": "c8081e03e8f63c791873"}
# {"frame": 33481}
# {"header_bits": "0001...", "crc": 1207, "stuffed_bits": "...", "wire_bits": "..."}
```

| request | answer |
| --- | --- |
| `crc <x25\|x25_bitwise\|ccitt_false\|dvb_s2> <hex> [init]`, `can15 <bits>`, `dshot_crc <value> <inverted>`, `sha256 <hex>` | `crc` or `digest` |
| `sbus_encode <16 channels> <ch17> <ch18> <frame_lost> <failsafe>`, `pack11 <16 channels>` | `hex` |
| `sbus_decode <hex>`, `unpack11 <hex>` | `channels` and the four flags |
| `crsf_frame <type> <address> <payload>`, `crsf_rc <16 channels>`, `crsf_battery <volts> <amps> <used_mah> <percent>`, `crsf_attitude <pitch> <roll> <yaw>`, `crsf_gps <lat> <lon> <speed_kmh> <heading_deg> <alt_m> <sats>`, `crsf_link_statistics <10 values>`, `crsf_flight_mode <ascii as hex>` | `hex` |
| `crsf_parse <chunk> [chunk ...]` | `frames` (address, type, payload), `crc_errors`, `dropped_bytes`, `buffered` |
| `dshot_encode <value> <telemetry> <bidirectional>`, `dshot_decode <frame> <bidirectional>`, `erpm_word <period_us>`, `gcr_encode <word>`, `gcr_decode <line>` | `frame`, `value`/`telemetry`/`crc_ok`, `word`, `line`, `word`/`ok` |
| `can_frame <extended> <remote> <id> <data> [acked interframe]` | `header_bits`, `crc`, `stuffed_bits`, `wire_bits` |
| `can_decode <bits>`, `can_stuff <bits>`, `can_destuff <bits>` | `id`/`extended`/`remote`/`data`, or `bits` |
| `can_arbitrate <n> <n times: extended remote id data>` | `winner`, `lost_at` (pairs of node and bit index), `bus` |
| `mav_encode <version> <seq> <sysid> <compid> <msgid> <crc_extra> <base_length> <payload> <key\|none> <link_id> <timestamp>` | `hex` of the whole frame |
| `mav_table <msgid:crc_extra ...>` then `mav_parse <chunk> [chunk ...]` | `packets`, `crc_errors`, `unknown`, `duplicates`, `bad_flags`, `dropped_bytes`, `lost`, `buffered` |

A request that fails answers `{"error": "ValueError"}` (or `OverflowError`, `error` for `struct.error`, `StuffError`, `CrcError`, `TruncatedError`, `UnicodeEncodeError`), and a malformed request answers `{"error": "BadRequest"}` with the reason on stderr.

## What is ported

- **CRCs.** CRC-16/MCRF4XX (table and bitwise), CRC-16/CCITT-FALSE, CRC-8/DVB-S2, CRC-15/CAN over bits and the DShot nibble XOR.
- **SBUS and CRSF.** 11 bit channel packing, SBUS frames, every CRSF frame builder and the `CrsfParser` resync logic (drop one byte on a bad address, length or CRC).
- **DShot.** Frame encode and decode (normal and bidirectional), the eRPM period word and its GCR line coding.
- **CAN.** Header bits for standard and extended frames, CRC-15, bit stuffing, wire bits, the incremental destuffing decoder and wired AND arbitration.
- **MAVLink.** v1 and v2 framing with CRC_EXTRA, v2 payload truncation, v2 signing (SHA-256 over key, frame, link id and timestamp) and the stream parser with its loss and duplicate counters (frames with unknown incompatibility flags are dropped).

## What is not ported

- **The MAVLink dialect.** The XML and JSON message definitions and `Message.pack`/`unpack` stay in Python. The C++ side takes the message id, CRC_EXTRA, base length and the packed payload, and does the framing.
- **`CrsfPacket.decoded`, DShot waveforms and timing, `worst_case_bits` and `bus_load`.** These are display helpers with no framing logic.

## Check parity with Python

```sh
WEEK04_CPP_BUILD=build/cpp4 PYTHONPATH=. python -m pytest tests/test_week04_cpp_parity.py -v
```

The test generates 2908 cases from fixed seeds and sends each codec's batch through one `protocol_tool` process. Every answer must equal the Python result exactly (or name the same exception):

| codec | cases | includes |
| --- | ---: | --- |
| CRCs and SHA-256 | 341 | the catalogue check values, random seeds, CRC-8 seeds that do not fit a byte, SHA-256 across the padding boundaries |
| SBUS | 220 | channels out of range, bad header, footer and length |
| CRSF builders | 386 | values that round half to even, fields just inside and just outside their struct range, frames over 64 bytes |
| CRSF parser | 60 | noisy streams of all frame types with flipped bits, fed in random chunks |
| DShot | 509 | commands, throttle, both CRC variants, eRPM exponent clamping, GCR lines with bit errors and junk above the 20 line levels |
| CAN | 872 | stuffing heavy identifiers, remote frames with data, bit errors, random bits, every possible cut of three stuffing heavy frames (inside a field, where a stuff bit is due, right after the CRC), arbitration between standard and extended frames that share identifier bits, and an empty bus |
| MAVLink encode | 460 | 150 messages of the `ardupilotmega` dialect packed in Python, v1 and v2, signed and unsigned, and the error paths (including a key on a v1 frame) |
| MAVLink parser | 60 | streams with noise, corrupted frames, unknown message ids, unknown incompatibility flags, signed frames, sequence gaps and repeated sequence numbers |

All 2908 cases pass, and the whole file runs in about a quarter of a second.
