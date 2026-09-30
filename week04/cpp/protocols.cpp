// Implementation of protocols.hpp. Each function follows its Python original step for step;
// the comments name the Python function it mirrors.
#include "protocols.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

namespace week04 {

namespace {

[[noreturn]] void fail(const char* kind, const std::string& what) { throw ProtocolError(kind, what); }

// Python's round(float) -> int: half to even, and errors for inf and nan.
double py_round(double x) {
  if (std::isnan(x)) fail("ValueError", "cannot convert float NaN to integer");
  if (std::isinf(x)) fail("OverflowError", "cannot convert float infinity to integer");
  return std::nearbyint(x);  // default rounding mode is to nearest, ties to even
}

// struct.pack range checks, which raise struct.error (class name "error").
std::int64_t in_range(double v, double lo, double hi, const char* code) {
  if (!(v >= lo && v <= hi)) fail("error", std::string("'") + code + "' format out of range");
  return static_cast<std::int64_t>(v);
}
std::int64_t in_range(std::int64_t v, std::int64_t lo, std::int64_t hi, const char* code) {
  if (v < lo || v > hi) fail("error", std::string("'") + code + "' format out of range");
  return v;
}

// bytes([v]) raises ValueError outside 0..255.
std::uint8_t byte(std::int64_t v) {
  if (v < 0 || v > 255) fail("ValueError", "bytes must be in range(0, 256)");
  return static_cast<std::uint8_t>(v);
}

void put_be(Bytes& out, std::uint64_t v, int n) {
  for (int i = n - 1; i >= 0; --i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void put_le(Bytes& out, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

// int.to_bytes(n): OverflowError for negative values or values that do not fit.
std::uint64_t fits_unsigned(std::int64_t v, int nbytes) {
  if (v < 0) fail("OverflowError", "can't convert negative int to unsigned");
  if (nbytes < 8 && static_cast<std::uint64_t>(v) >> (8 * nbytes)) fail("OverflowError", "int too big to convert");
  return static_cast<std::uint64_t>(v);
}

}  // namespace

// ================================================================== crc.py ====
namespace crc {
namespace {
std::array<std::uint16_t, 256> make_x25_table() {
  std::array<std::uint16_t, 256> t{};
  for (unsigned byte = 0; byte < 256; ++byte) {
    unsigned c = byte;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x8408 : c >> 1;
    t[byte] = static_cast<std::uint16_t>(c);
  }
  return t;
}
std::array<std::uint8_t, 256> make_dvb_s2_table() {
  std::array<std::uint8_t, 256> t{};
  for (unsigned byte = 0; byte < 256; ++byte) {
    unsigned c = byte;
    for (int k = 0; k < 8; ++k) c = (c & 0x80) ? ((c << 1) ^ 0xD5) & 0xFF : (c << 1) & 0xFF;
    t[byte] = static_cast<std::uint8_t>(c);
  }
  return t;
}
const std::array<std::uint16_t, 256> X25 = make_x25_table();
const std::array<std::uint8_t, 256> DVB_S2 = make_dvb_s2_table();
}  // namespace

std::uint32_t x25(const Bytes& data, std::uint32_t crc) {  // x25_accumulate
  for (std::uint8_t b : data) crc = (crc >> 8) ^ X25[(crc ^ b) & 0xFF];
  return crc;
}

std::uint32_t x25_bitwise(const Bytes& data, std::uint32_t crc) {
  for (std::uint8_t b : data) {
    std::uint32_t tmp = (b ^ (crc & 0xFF)) & 0xFF;
    tmp = (tmp ^ (tmp << 4)) & 0xFF;
    crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF;
  }
  return crc;
}

std::uint32_t ccitt_false(const Bytes& data, std::uint32_t crc) {
  for (std::uint8_t b : data) {
    crc ^= static_cast<std::uint32_t>(b) << 8;
    for (int k = 0; k < 8; ++k) crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) & 0xFFFF : (crc << 1) & 0xFFFF;
  }
  return crc;
}

std::uint32_t dvb_s2(const Bytes& data, std::uint32_t crc) {
  if (crc > 0xFF) fail("ValueError", "CRC-8 state must fit in a byte");
  for (std::uint8_t b : data) crc = DVB_S2[crc ^ b];
  return crc;
}

std::uint32_t can15(const Bits& bits) {
  std::uint32_t crc = 0;
  for (std::uint8_t bit : bits) {
    std::uint32_t nxt = (bit & 1) ^ ((crc >> 14) & 1);
    crc = (crc << 1) & 0x7FFF;
    if (nxt) crc ^= 0x4599;
  }
  return crc;
}

Bits bytes_to_bits(const Bytes& data) {
  Bits out;
  out.reserve(data.size() * 8);
  for (std::uint8_t b : data)
    for (int i = 0; i < 8; ++i) out.push_back((b >> (7 - i)) & 1);
  return out;
}

std::uint32_t dshot_crc(std::uint64_t value12, bool inverted) {
  std::uint32_t c = static_cast<std::uint32_t>((value12 ^ (value12 >> 4) ^ (value12 >> 8)) & 0x0F);
  return inverted ? (~c) & 0x0F : c;
}
}  // namespace crc

// =================================================================== rc.py ====
namespace rc {

Bytes pack11(const std::vector<std::int64_t>& channels) {
  if (channels.size() != 16) fail("ValueError", "need exactly 16 channels");
  Bytes out(22, 0);
  for (std::size_t i = 0; i < 16; ++i) {
    std::int64_t v = channels[i];
    if (v < 0 || v >= 2048) fail("ValueError", "channel " + std::to_string(i + 1) + " does not fit in 11 bits");
    for (int k = 0; k < 11; ++k)
      if ((v >> k) & 1) {
        std::size_t bit = 11 * i + k;
        out[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
      }
  }
  return out;
}

std::vector<int> unpack11(const Bytes& data) {
  // int.from_bytes(data[:22], "little"): missing bytes read as zero.
  std::vector<int> out(16, 0);
  for (std::size_t i = 0; i < 16; ++i)
    for (int k = 0; k < 11; ++k) {
      std::size_t bit = 11 * i + k;
      if (bit / 8 < std::min<std::size_t>(data.size(), 22) && ((data[bit / 8] >> (bit % 8)) & 1)) out[i] |= 1 << k;
    }
  return out;
}

Bytes sbus_encode(const SbusFrame& f) {
  std::uint8_t flags = static_cast<std::uint8_t>(f.ch17 | f.ch18 << 1 | f.frame_lost << 2 | f.failsafe << 3);
  Bytes out{SBUS_HEADER};
  Bytes ch = pack11(f.channels);
  out.insert(out.end(), ch.begin(), ch.end());
  out.push_back(flags);
  out.push_back(SBUS_FOOTER);
  return out;
}

SbusFrame sbus_decode(const Bytes& data) {
  if (data.size() != SBUS_LEN || data[0] != SBUS_HEADER || data[24] != SBUS_FOOTER)
    fail("ValueError", "not an SBUS frame (bad length, header or footer)");
  SbusFrame f;
  for (int v : unpack11(Bytes(data.begin() + 1, data.begin() + 23))) f.channels.push_back(v);
  std::uint8_t flags = data[23];
  f.ch17 = flags & 1;
  f.ch18 = flags & 2;
  f.frame_lost = flags & 4;
  f.failsafe = flags & 8;
  return f;
}

Bytes crsf_frame(std::int64_t frame_type, const Bytes& payload, std::int64_t address) {
  Bytes body{byte(frame_type)};
  body.insert(body.end(), payload.begin(), payload.end());
  if (body.size() + 3 > CRSF_MAX_FRAME) fail("ValueError", "CRSF frames are at most 64 bytes");
  Bytes out{byte(address), byte(static_cast<std::int64_t>(body.size()) + 1)};
  out.insert(out.end(), body.begin(), body.end());
  out.push_back(static_cast<std::uint8_t>(crc::dvb_s2(body)));
  return out;
}

Bytes crsf_rc(const std::vector<std::int64_t>& channels) { return crsf_frame(RC_CHANNELS, pack11(channels)); }

Bytes crsf_battery(double volts, double amps, std::int64_t used_mah, std::int64_t percent) {
  double v = py_round(volts * 10), a = py_round(amps * 10);
  Bytes p;
  put_be(p, in_range(v, 0, 65535, "H"), 2);
  put_be(p, in_range(a, 0, 65535, "H"), 2);
  put_be(p, fits_unsigned(used_mah, 3), 3);
  p.push_back(byte(percent));
  return crsf_frame(BATTERY, p);
}

Bytes crsf_attitude(double pitch_rad, double roll_rad, double yaw_rad) {
  double r[3] = {py_round(pitch_rad * 10000), py_round(roll_rad * 10000), py_round(yaw_rad * 10000)};
  Bytes p;
  for (double x : r) put_be(p, static_cast<std::uint64_t>(in_range(x, -32768, 32767, "h")), 2);
  return crsf_frame(ATTITUDE, p);
}

Bytes crsf_gps(double lat, double lon, double speed_kmh, double heading_deg, double alt_m, std::int64_t sats) {
  double la = py_round(lat * 1e7), lo = py_round(lon * 1e7), sp = py_round(speed_kmh * 10);
  double hd = py_round(heading_deg * 100), al = py_round(alt_m + 1000);
  Bytes p;
  put_be(p, static_cast<std::uint64_t>(in_range(la, -2147483648.0, 2147483647.0, "i")), 4);
  put_be(p, static_cast<std::uint64_t>(in_range(lo, -2147483648.0, 2147483647.0, "i")), 4);
  put_be(p, in_range(sp, 0, 65535, "H"), 2);
  put_be(p, in_range(hd, 0, 65535, "H"), 2);
  put_be(p, in_range(al, 0, 65535, "H"), 2);
  put_be(p, in_range(sats, 0, 255, "B"), 1);
  return crsf_frame(GPS, p);
}

Bytes crsf_link_statistics(const std::array<std::int64_t, 10>& v) {
  // ">BBBbBBBBBb": fields 3 (snr) and 9 (d_snr) are signed.
  Bytes p;
  for (std::size_t i = 0; i < 10; ++i) {
    bool is_signed = i == 3 || i == 9;
    std::int64_t x = is_signed ? in_range(v[i], -128, 127, "b") : in_range(v[i], 0, 255, "B");
    p.push_back(static_cast<std::uint8_t>(x));
  }
  return crsf_frame(LINK_STATISTICS, p);
}

Bytes crsf_flight_mode(const std::string& mode) {
  Bytes p;
  for (char c : mode) {
    auto u = static_cast<unsigned char>(c);
    if (u >= 0x80) fail("UnicodeEncodeError", "'ascii' codec can't encode character");
    p.push_back(u);
  }
  p.push_back(0);
  return crsf_frame(FLIGHT_MODE, p);
}

std::vector<CrsfPacket> CrsfParser::feed(const Bytes& data) {
  std::vector<CrsfPacket> out;
  buffer.insert(buffer.end(), data.begin(), data.end());
  while (buffer.size() >= 2) {
    std::size_t length = buffer[1];
    std::uint8_t a = buffer[0];
    bool known = a == CRSF_SYNC || a == 0xEA || a == 0xEE || a == 0xEC;
    if (!known || !(2 <= length && length <= CRSF_MAX_FRAME - 2)) {
      buffer.erase(buffer.begin());
      ++dropped_bytes;
      continue;
    }
    if (buffer.size() < length + 2) break;
    Bytes body(buffer.begin() + 2, buffer.begin() + static_cast<std::ptrdiff_t>(length + 1));
    std::uint8_t crc = buffer[length + 1];
    if (crc::dvb_s2(body) != crc) {
      ++crc_errors;
      buffer.erase(buffer.begin());
      ++dropped_bytes;
      continue;
    }
    CrsfPacket pkt;
    pkt.address = buffer[0];
    pkt.type = body[0];
    pkt.payload.assign(body.begin() + 1, body.end());
    buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(length + 2));
    out.push_back(std::move(pkt));
  }
  return out;
}
}  // namespace rc

// ================================================================ dshot.py ====
namespace dshot {
namespace {
constexpr std::uint8_t GCR[16] = {0x19, 0x1B, 0x12, 0x13, 0x1D, 0x15, 0x16, 0x17,
                                  0x1A, 0x09, 0x0A, 0x0B, 0x1E, 0x0D, 0x0E, 0x0F};
int gcr_inv(unsigned code) {
  for (int n = 0; n < 16; ++n)
    if (GCR[n] == code) return n;
  return -1;
}
std::uint32_t erpm_crc(std::uint32_t value) { return (~(value ^ (value >> 4) ^ (value >> 8))) & 0xF; }
}  // namespace

std::uint32_t encode(std::int64_t value, bool telemetry, bool bidirectional) {
  if (value < 0 || value > 2047) fail("ValueError", "DShot values are 11 bits");
  std::uint32_t payload = (static_cast<std::uint32_t>(value) << 1) | (telemetry ? 1u : 0u);
  return (payload << 4) | crc::dshot_crc(payload, bidirectional);
}

Frame decode(std::uint64_t frame, bool bidirectional) {
  std::uint64_t payload = frame >> 4;
  return Frame{payload >> 1, (payload & 1) != 0, crc::dshot_crc(payload, bidirectional) == (frame & 0xF)};
}

std::uint32_t erpm_word(std::int64_t period_us) {
  if (period_us <= 0) fail("ValueError", "period must be positive");
  std::uint64_t mantissa = static_cast<std::uint64_t>(period_us);
  std::uint32_t exponent = 0;
  while (mantissa > 0x1FF) {
    mantissa >>= 1;
    ++exponent;
  }
  if (exponent > 7) mantissa = 0x1FF, exponent = 7;
  std::uint32_t value = (exponent << 9) | static_cast<std::uint32_t>(mantissa);
  return (value << 4) | erpm_crc(value);
}

std::uint64_t gcr_encode(std::uint64_t word) {
  std::uint64_t gcr = 0;
  for (int shift : {12, 8, 4, 0}) gcr = (gcr << 5) | GCR[(word >> shift) & 0xF];
  std::uint64_t line = 0, level = 0;
  for (int i = 19; i >= 0; --i) {
    level ^= (gcr >> i) & 1;
    line = (line << 1) | level;
  }
  return line;
}

std::pair<std::uint32_t, bool> gcr_decode(std::uint64_t line) {
  line &= 0xFFFFF;  // 20 line levels
  std::uint64_t gcr = line ^ (line >> 1);
  std::uint32_t word = 0;
  for (int shift : {15, 10, 5, 0}) {
    int nibble = gcr_inv(static_cast<unsigned>((gcr >> shift) & 0x1F));
    if (nibble < 0) return {0, false};
    word = (word << 4) | static_cast<std::uint32_t>(nibble);
  }
  return {word, erpm_crc(word >> 4) == (word & 0xF)};
}
}  // namespace dshot

// ================================================================== can.py ====
namespace can {
namespace {
void append_bits(Bits& out, std::uint64_t value, int width) {
  for (int i = 0; i < width; ++i) out.push_back((value >> (width - 1 - i)) & 1);
}
std::uint64_t to_int(const Bits& bits, std::size_t from, std::size_t to) {
  std::uint64_t v = 0;
  for (std::size_t i = from; i < to && i < bits.size(); ++i) v = (v << 1) | bits[i];
  return v;
}

// _Destuffer: reads one logical bit at a time, skipping (and checking) stuff bits.
struct Destuffer {
  const Bits& wire;
  std::size_t pos = 0;
  int run_bit = -1, run = 0;

  Bits take(std::size_t n) {
    Bits out;
    for (std::size_t k = 0; k < n; ++k) {
      if (pos >= wire.size()) fail("TruncatedError", "frame ends early");
      std::uint8_t b = wire[pos++];
      out.push_back(b);
      if (b == run_bit) {
        ++run;
      } else {
        run_bit = b;
        run = 1;
      }
      if (run == 5) {
        if (pos >= wire.size()) fail("TruncatedError", "frame ends inside a stuff bit");
        std::uint8_t stuffed = wire[pos];
        if (stuffed == b) fail("StuffError", "six equal bits at wire position " + std::to_string(pos));
        ++pos;
        run_bit = stuffed;
        run = 1;
      }
    }
    return out;
  }
};
}  // namespace

Frame Frame::make(std::int64_t can_id, Bytes data, bool extended, bool remote) {
  std::int64_t limit = std::int64_t{1} << (extended ? 29 : 11);
  if (can_id < 0 || can_id >= limit) fail("ValueError", "identifier does not fit the frame format");
  if (data.size() > 8) fail("ValueError", "classic CAN carries at most 8 data bytes");
  if (remote && !data.empty()) fail("ValueError", "a remote frame requests data, it carries none");
  Frame f;
  f.can_id = static_cast<std::uint32_t>(can_id);
  f.data = std::move(data);
  f.extended = extended;
  f.remote = remote;
  return f;
}

Bits Frame::header_bits() const {
  Bits bits{0};
  if (extended) {
    append_bits(bits, can_id >> 18, 11);
    bits.push_back(1);  // SRR
    bits.push_back(1);  // IDE
    append_bits(bits, can_id & 0x3FFFF, 18);
    bits.push_back(remote ? 1 : 0);
    bits.push_back(0);
    bits.push_back(0);
  } else {
    append_bits(bits, can_id, 11);
    bits.push_back(remote ? 1 : 0);
    bits.push_back(0);
    bits.push_back(0);
  }
  append_bits(bits, data.size(), 4);
  if (!remote) {
    Bits d = crc::bytes_to_bits(data);
    bits.insert(bits.end(), d.begin(), d.end());
  }
  return bits;
}

std::uint32_t Frame::crc() const { return crc::can15(header_bits()); }

Bits Frame::stuffed_bits() const {
  Bits bits = header_bits();
  append_bits(bits, crc(), 15);
  return stuff(bits);
}

Bits Frame::wire_bits(bool acked, bool interframe) const {
  Bits bits = stuffed_bits();
  bits.push_back(1);                 // CRC delimiter
  bits.push_back(acked ? 0 : 1);     // ACK slot
  bits.push_back(1);                 // ACK delimiter
  bits.insert(bits.end(), 7, 1);     // EOF
  if (interframe) bits.insert(bits.end(), 3, 1);
  return bits;
}

Bits stuff(const Bits& bits) {
  Bits out;
  int run_bit = -1, run = 0;
  for (std::uint8_t b : bits) {
    out.push_back(b);
    if (b == run_bit) {
      ++run;
    } else {
      run_bit = b;
      run = 1;
    }
    if (run == 5) {
      out.push_back(static_cast<std::uint8_t>(1 - b));
      run_bit = 1 - b;
      run = 1;
    }
  }
  return out;
}

Bits destuff(const Bits& bits) {
  Bits out;
  int run_bit = -1, run = 0;
  std::size_t i = 0;
  while (i < bits.size()) {
    std::uint8_t b = bits[i];
    out.push_back(b);
    if (b == run_bit) {
      ++run;
    } else {
      run_bit = b;
      run = 1;
    }
    if (run == 5) {
      ++i;
      if (i < bits.size()) {
        if (bits[i] == b) fail("StuffError", "six equal bits at position " + std::to_string(i));
        run_bit = bits[i];
        run = 1;
      }
    }
    ++i;
  }
  return out;
}

Frame decode(const Bits& wire) {
  Destuffer r{wire};
  Bits raw = r.take(14);
  if (raw[0] != 0) fail("ValueError", "frame must start with a dominant SOF");
  bool ide = raw[13];
  std::uint64_t can_id;
  bool remote;
  std::size_t dlc_at;
  if (ide) {
    Bits more = r.take(25);
    raw.insert(raw.end(), more.begin(), more.end());
    can_id = (to_int(raw, 1, 12) << 18) | to_int(raw, 14, 32);
    remote = raw[32];
    dlc_at = 35;
  } else {
    Bits more = r.take(5);
    raw.insert(raw.end(), more.begin(), more.end());
    can_id = to_int(raw, 1, 12);
    remote = raw[12];
    dlc_at = 15;
  }
  std::uint64_t dlc = to_int(raw, dlc_at, dlc_at + 4);
  std::size_t n = remote ? 0 : static_cast<std::size_t>(std::min<std::uint64_t>(dlc, 8));
  Bits data_bits = r.take(8 * n);
  raw.insert(raw.end(), data_bits.begin(), data_bits.end());
  std::uint64_t crc_field = to_int(r.take(15), 0, 15);
  std::uint32_t computed = crc::can15(raw);
  if (computed != crc_field) fail("CrcError", "CRC mismatch");
  if (r.pos >= wire.size() || wire[r.pos] != 1) fail("ValueError", "CRC delimiter must be recessive");
  Bytes data;
  for (std::size_t k = 0; k < n; ++k)
    data.push_back(static_cast<std::uint8_t>(to_int(raw, dlc_at + 4 + 8 * k, dlc_at + 12 + 8 * k)));
  return Frame::make(static_cast<std::int64_t>(can_id), data, ide, remote);
}

Arbitration arbitrate(const std::vector<Frame>& frames) {
  if (frames.empty()) fail("ValueError", "arbitration needs at least one transmitter");
  std::set<std::tuple<std::uint32_t, bool, bool>> keys;
  for (const Frame& f : frames) keys.insert({f.can_id, f.extended, f.remote});
  if (keys.size() != frames.size())
    fail("ValueError", "two nodes sending the same identifier is a configuration error on a CAN bus");
  std::vector<Bits> streams;
  for (const Frame& f : frames) streams.push_back(f.wire_bits(true, false));
  std::set<std::size_t> active;
  for (std::size_t i = 0; i < frames.size(); ++i) active.insert(i);
  Arbitration result;
  std::size_t t = 0;
  auto bit = [&](std::size_t i) -> std::uint8_t {
    if (t >= streams[i].size()) fail("IndexError", "list index out of range");
    return streams[i][t];
  };
  while (active.size() > 1) {
    std::uint8_t level = 1;
    for (std::size_t i : active) level = std::min(level, bit(i));
    result.bus.push_back(level);
    for (auto it = active.begin(); it != active.end();) {
      if (bit(*it) == 1 && level == 0) {
        result.lost_at[*it] = t;
        it = active.erase(it);
      } else {
        ++it;
      }
    }
    ++t;
  }
  result.winner = *active.begin();
  const Bits& w = streams[result.winner];
  if (t < w.size()) result.bus.insert(result.bus.end(), w.begin() + static_cast<std::ptrdiff_t>(t), w.end());
  return result;
}
}  // namespace can

// ============================================================== mavlink.py ====
namespace mavlink {

// FIPS 180-4 SHA-256, the only primitive MAVLink 2 signing needs.
std::array<std::uint8_t, 32> sha256(const Bytes& data) {
  static const std::uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };

  Bytes msg = data;
  std::uint64_t bit_len = static_cast<std::uint64_t>(data.size()) * 8;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  put_be(msg, bit_len, 8);

  for (std::size_t off = 0; off < msg.size(); off += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = static_cast<std::uint32_t>(msg[off + 4 * i]) << 24 | static_cast<std::uint32_t>(msg[off + 4 * i + 1]) << 16 |
             static_cast<std::uint32_t>(msg[off + 4 * i + 2]) << 8 | msg[off + 4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      std::uint32_t ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 4; ++k) out[4 * i + k] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * k));
  return out;
}

namespace {
// _signature: link id + 48 bit timestamp + first 6 bytes of SHA-256(key + frame + those 7 bytes).
Bytes signature(const Bytes& key, const Bytes& signed_part, std::int64_t link_id, std::int64_t timestamp) {
  Bytes tail{byte(link_id)};
  put_le(tail, fits_unsigned(timestamp, 6), 6);
  Bytes material = key;
  material.insert(material.end(), signed_part.begin(), signed_part.end());
  material.insert(material.end(), tail.begin(), tail.end());
  auto digest = sha256(material);
  tail.insert(tail.end(), digest.begin(), digest.begin() + 6);
  return tail;
}

std::uint32_t frame_crc(const Bytes& frame, std::size_t header_len, std::size_t n, std::uint8_t crc_extra) {
  // x25_accumulate([*header[1:], *payload, crc_extra])
  Bytes covered(frame.begin() + 1, frame.begin() + static_cast<std::ptrdiff_t>(header_len + n));
  covered.push_back(crc_extra);
  return crc::x25(covered);
}
}  // namespace

Bytes encode(const Bytes& payload_in, std::int64_t msgid, std::uint8_t crc_extra, std::size_t base_length,
             std::int64_t seq, std::int64_t sysid, std::int64_t compid, int version, const std::optional<Bytes>& key,
             std::int64_t link_id, std::int64_t timestamp) {
  Bytes payload = payload_in;
  if (version == 1) {
    if (key) fail("ValueError", "MAVLink 1 has no signing; use version 2");
    if (msgid > 255) fail("ValueError", "message needs MAVLink 2");
    if (payload.size() > base_length) payload.resize(base_length);  // v1 has no extension fields
    Bytes frame{STX_V1, byte(static_cast<std::int64_t>(payload.size())), static_cast<std::uint8_t>(seq & 0xFF),
                byte(sysid), byte(compid), byte(msgid)};
    frame.insert(frame.end(), payload.begin(), payload.end());
    put_le(frame, frame_crc(frame, 6, payload.size(), crc_extra), 2);
    return frame;
  }
  // payload.rstrip(b"\0") or payload[:1]
  std::size_t n = payload.size();
  while (n > 0 && payload[n - 1] == 0) --n;
  if (n == 0) n = std::min<std::size_t>(payload.size(), 1);
  payload.resize(n);
  std::uint8_t incompat = key ? INCOMPAT_SIGNED : 0;
  Bytes frame{STX_V2, byte(static_cast<std::int64_t>(payload.size())), incompat, 0,
              static_cast<std::uint8_t>(seq & 0xFF), byte(sysid), byte(compid)};
  put_le(frame, fits_unsigned(msgid, 3), 3);
  frame.insert(frame.end(), payload.begin(), payload.end());
  put_le(frame, frame_crc(frame, 10, payload.size(), crc_extra), 2);
  if (key) {
    Bytes sig = signature(*key, frame, link_id, timestamp);
    frame.insert(frame.end(), sig.begin(), sig.end());
  }
  return frame;
}

std::vector<Packet> Parser::feed(const Bytes& data) {
  std::vector<Packet> out;
  buffer.insert(buffer.end(), data.begin(), data.end());
  auto drop_one = [&] {
    buffer.erase(buffer.begin());
    ++dropped_bytes;
  };
  while (!buffer.empty()) {
    std::uint8_t stx = buffer[0];
    if (stx != STX_V1 && stx != STX_V2) {
      drop_one();
      continue;
    }
    std::size_t header_len = stx == STX_V2 ? 10 : 6;
    if (buffer.size() < header_len) break;
    std::size_t n = buffer[1];
    bool is_signed = stx == STX_V2 && (buffer[2] & INCOMPAT_SIGNED);
    std::size_t total = header_len + n + 2 + (is_signed ? 13 : 0);
    if (buffer.size() < total) break;
    Bytes frame(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(total));
    Packet p;
    if (stx == STX_V2) {
      p.version = 2;
      p.seq = frame[4], p.sysid = frame[5], p.compid = frame[6];
      p.msgid = frame[7] | frame[8] << 8 | frame[9] << 16;
      p.incompat = frame[2], p.compat = frame[3];
    } else {
      p.version = 1;
      p.seq = frame[2], p.sysid = frame[3], p.compid = frame[4], p.msgid = frame[5];
    }
    if (p.incompat & ~INCOMPAT_SIGNED & 0xFF) {  // unknown incompatibility flags: drop, like a CRC error
      ++bad_flags;
      drop_one();
      continue;
    }
    auto it = crc_extra.find(p.msgid);
    if (it == crc_extra.end()) {
      ++unknown;
      drop_one();
      continue;
    }
    std::uint32_t crc = frame[header_len + n] | frame[header_len + n + 1] << 8;
    if (frame_crc(frame, header_len, n, it->second) != crc) {
      ++crc_errors;
      drop_one();
      continue;
    }
    buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(total));
    auto key = std::make_pair<int, int>(p.sysid, p.compid);
    auto last = last_seq.find(key);
    if (last != last_seq.end()) {
      int gap = (p.seq - last->second) & 0xFF;
      if (gap == 0) ++duplicates;
      else lost += gap - 1;
    }
    last_seq[key] = p.seq;
    p.payload.assign(frame.begin() + static_cast<std::ptrdiff_t>(header_len),
                     frame.begin() + static_cast<std::ptrdiff_t>(header_len + n));
    if (is_signed) p.signature = Bytes(frame.end() - 13, frame.end());
    p.raw = std::move(frame);
    out.push_back(std::move(p));
  }
  return out;
}
}  // namespace mavlink

}  // namespace week04
