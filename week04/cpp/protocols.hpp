// Week 04 protocol codecs in C++17: CRCs, SBUS, CRSF, DShot, classic CAN and MAVLink framing.
//
// A line for line port of week04/crc.py, rc.py, dshot.py, can.py and the framing half of
// mavlink.py. Outputs are byte exact with the Python code (tests/test_week04_cpp_parity.py).
// No dependencies beyond the standard library; SHA-256 for MAVLink signing is built in.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace week04 {

using Bytes = std::vector<std::uint8_t>;
using Bits = std::vector<std::uint8_t>;  // one 0 or 1 per element, MSB first like the wire

// Every failure carries the name of the exception class Python raises in the same place
// (ValueError, OverflowError, struct's "error", StuffError, CrcError, TruncatedError...), so the
// parity test can compare failures as well as results.
struct ProtocolError : std::runtime_error {
  std::string kind;
  ProtocolError(std::string kind_, const std::string& what) : std::runtime_error(what), kind(std::move(kind_)) {}
};

// ------------------------------------------------------------------ crc.py ----
namespace crc {
std::uint32_t x25(const Bytes& data, std::uint32_t crc = 0xFFFF);  // CRC-16/MCRF4XX, table driven
std::uint32_t x25_bitwise(const Bytes& data, std::uint32_t crc = 0xFFFF);  // MAVLink's crc_accumulate
std::uint32_t ccitt_false(const Bytes& data, std::uint32_t crc = 0xFFFF);  // CRC-16/CCITT-FALSE
std::uint32_t dvb_s2(const Bytes& data, std::uint32_t crc = 0);  // CRC-8/DVB-S2 (CRSF)
std::uint32_t can15(const Bits& bits);  // CRC-15/CAN over bits
Bits bytes_to_bits(const Bytes& data);
std::uint32_t dshot_crc(std::uint64_t value12, bool inverted = false);
}  // namespace crc

// ------------------------------------------------------------------- rc.py ----
namespace rc {
constexpr std::uint8_t SBUS_HEADER = 0x0F, SBUS_FOOTER = 0x00;
constexpr std::size_t SBUS_LEN = 25;
constexpr std::uint8_t CRSF_SYNC = 0xC8;
constexpr std::size_t CRSF_MAX_FRAME = 64;

enum CrsfType : std::uint8_t {
  GPS = 0x02,
  BATTERY = 0x08,
  LINK_STATISTICS = 0x14,
  RC_CHANNELS = 0x16,
  ATTITUDE = 0x1E,
  FLIGHT_MODE = 0x21,
};

Bytes pack11(const std::vector<std::int64_t>& channels);
std::vector<int> unpack11(const Bytes& data);

struct SbusFrame {
  std::vector<std::int64_t> channels;
  bool ch17 = false, ch18 = false, frame_lost = false, failsafe = false;
};
Bytes sbus_encode(const SbusFrame& frame);
SbusFrame sbus_decode(const Bytes& data);

Bytes crsf_frame(std::int64_t frame_type, const Bytes& payload, std::int64_t address = CRSF_SYNC);
Bytes crsf_rc(const std::vector<std::int64_t>& channels);
Bytes crsf_battery(double volts, double amps, std::int64_t used_mah, std::int64_t percent);
Bytes crsf_attitude(double pitch_rad, double roll_rad, double yaw_rad);
Bytes crsf_gps(double lat, double lon, double speed_kmh, double heading_deg, double alt_m, std::int64_t sats);
Bytes crsf_link_statistics(const std::array<std::int64_t, 10>& v);  // rssi1 rssi2 lq snr antenna rf_mode tx_power d_rssi d_lq d_snr
Bytes crsf_flight_mode(const std::string& mode);

struct CrsfPacket {
  std::uint8_t address = 0, type = 0;
  Bytes payload;
};

// Byte stream parser that resynchronises after noise (CrsfParser.feed).
struct CrsfParser {
  Bytes buffer;
  long crc_errors = 0, dropped_bytes = 0;
  std::vector<CrsfPacket> feed(const Bytes& data);
};
}  // namespace rc

// ---------------------------------------------------------------- dshot.py ----
namespace dshot {
std::uint32_t encode(std::int64_t value, bool telemetry = false, bool bidirectional = false);
struct Frame {
  std::uint64_t value = 0;
  bool telemetry = false, crc_ok = false;
};
Frame decode(std::uint64_t frame, bool bidirectional = false);
std::uint32_t erpm_word(std::int64_t period_us);
std::uint64_t gcr_encode(std::uint64_t word);
std::pair<std::uint32_t, bool> gcr_decode(std::uint64_t line);
}  // namespace dshot

// ------------------------------------------------------------------ can.py ----
namespace can {
struct Frame {
  std::uint32_t can_id = 0;
  Bytes data;
  bool extended = false, remote = false;

  // Validates like CanFrame.__post_init__ (ValueError on a bad identifier or more than 8 bytes).
  static Frame make(std::int64_t can_id, Bytes data, bool extended, bool remote);
  Bits header_bits() const;
  std::uint32_t crc() const;
  Bits stuffed_bits() const;
  Bits wire_bits(bool acked = true, bool interframe = true) const;
};

Bits stuff(const Bits& bits);
Bits destuff(const Bits& bits);
Frame decode(const Bits& wire);

struct Arbitration {
  std::size_t winner = 0;
  std::map<std::size_t, std::size_t> lost_at;  // node index -> bit index where it backed off
  Bits bus;
};
Arbitration arbitrate(const std::vector<Frame>& frames);
}  // namespace can

// -------------------------------------------------------------- mavlink.py ----
namespace mavlink {
constexpr std::uint8_t STX_V1 = 0xFE, STX_V2 = 0xFD, INCOMPAT_SIGNED = 0x01;

std::array<std::uint8_t, 32> sha256(const Bytes& data);

// Frames an already packed payload (Message.pack). base_length is the payload size without
// extension fields, which is all a v1 frame carries. key enables v2 signing.
Bytes encode(const Bytes& payload, std::int64_t msgid, std::uint8_t crc_extra, std::size_t base_length,
             std::int64_t seq = 0, std::int64_t sysid = 1, std::int64_t compid = 1, int version = 2,
             const std::optional<Bytes>& key = std::nullopt, std::int64_t link_id = 0, std::int64_t timestamp = 0);

struct Packet {
  int version = 2;
  std::uint8_t seq = 0, sysid = 0, compid = 0;
  std::uint32_t msgid = 0;
  Bytes payload;
  std::uint8_t incompat = 0, compat = 0;
  std::optional<Bytes> signature;  // link id (1) + timestamp (6) + signature (6)
  Bytes raw;
};

// Stream parser (mavlink.Parser.feed) with the dialect reduced to msgid -> CRC_EXTRA. Frames with
// incompatibility flags other than signing are dropped (bad_flags); a repeated sequence number
// counts as a duplicate rather than 255 lost packets.
struct Parser {
  std::map<std::uint32_t, std::uint8_t> crc_extra;
  Bytes buffer;
  long crc_errors = 0, unknown = 0, duplicates = 0, bad_flags = 0, dropped_bytes = 0, lost = 0;
  std::map<std::pair<int, int>, int> last_seq;
  std::vector<Packet> feed(const Bytes& data);
};
}  // namespace mavlink

}  // namespace week04
