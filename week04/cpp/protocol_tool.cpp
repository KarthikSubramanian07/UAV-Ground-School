// protocol_tool: a line oriented front end to protocols.hpp.
//
// Reads one request per line on stdin (whitespace separated tokens, hex for bytes, strings of
// 0 and 1 for bits, "-" for an empty value) and writes one JSON object per request on stdout.
// A request that fails writes {"error": "<Python exception class>"}. See README.md for the list.
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "protocols.hpp"

using namespace week04;

namespace {

struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------ token parsing ----

struct Args {
  std::vector<std::string> tok;
  std::size_t i = 1;

  bool more() const { return i < tok.size(); }
  const std::string& next() {
    if (i >= tok.size()) throw BadRequest("missing argument");
    return tok[i++];
  }
  std::int64_t integer() {
    const std::string& s = next();
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(s.c_str(), &end, 0);
    if (errno || *end) throw BadRequest("bad integer " + s);
    return v;
  }
  double real() {
    const std::string& s = next();
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (*end) throw BadRequest("bad number " + s);
    return v;
  }
  bool flag() { return integer() != 0; }
  Bytes hex() {
    const std::string& s = next();
    Bytes out;
    if (s == "-") return out;
    if (s.size() % 2) throw BadRequest("odd length hex");
    auto nib = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      throw BadRequest("bad hex digit");
    };
    for (std::size_t k = 0; k < s.size(); k += 2) out.push_back(static_cast<std::uint8_t>(nib(s[k]) << 4 | nib(s[k + 1])));
    return out;
  }
  Bits bits() {
    const std::string& s = next();
    Bits out;
    if (s == "-") return out;
    for (char c : s) {
      if (c != '0' && c != '1') throw BadRequest("bits must be 0 or 1");
      out.push_back(static_cast<std::uint8_t>(c - '0'));
    }
    return out;
  }
  std::vector<std::int64_t> integers(std::size_t n) {
    std::vector<std::int64_t> v;
    for (std::size_t k = 0; k < n; ++k) v.push_back(integer());
    return v;
  }
};

// ---------------------------------------------------------------- JSON out ----

std::string hex(const Bytes& b) {
  static const char* digits = "0123456789abcdef";
  std::string s;
  for (std::uint8_t x : b) s += digits[x >> 4], s += digits[x & 15];
  return s;
}
std::string bitstr(const Bits& b) {
  std::string s;
  for (std::uint8_t x : b) s += static_cast<char>('0' + x);
  return s;
}
std::string quote(const std::string& s) { return "\"" + s + "\""; }  // only hex, bits and names: no escaping needed

struct Obj {
  std::string s = "{";
  Obj& raw(const std::string& key, const std::string& value) {
    if (s.size() > 1) s += ", ";
    s += quote(key) + ": " + value;
    return *this;
  }
  template <typename T>
  Obj& num(const std::string& key, T v) { return raw(key, std::to_string(v)); }
  Obj& boolean(const std::string& key, bool v) { return raw(key, v ? "true" : "false"); }
  Obj& str(const std::string& key, const std::string& v) { return raw(key, quote(v)); }
  std::string done() const { return s + "}"; }
};

template <typename T>
std::string list(const std::vector<T>& v) {
  std::string s = "[";
  for (std::size_t k = 0; k < v.size(); ++k) s += (k ? ", " : "") + std::to_string(v[k]);
  return s + "]";
}

std::string join(const std::vector<std::string>& items) {
  std::string s = "[";
  for (std::size_t k = 0; k < items.size(); ++k) s += (k ? ", " : "") + items[k];
  return s + "]";
}

// ---------------------------------------------------------------- requests ----

can::Frame read_can_frame(Args& a) {
  bool extended = a.flag(), remote = a.flag();
  std::int64_t id = a.integer();
  return can::Frame::make(id, a.hex(), extended, remote);
}

mavlink::Parser mav_parser_template;  // holds the msgid -> CRC_EXTRA table set by mav_table

std::string handle(Args& a) {
  const std::string& op = a.tok[0];

  // crc <x25|x25_bitwise|ccitt_false|dvb_s2> <hex> [init]
  if (op == "crc") {
    std::string name = a.next();
    Bytes data = a.hex();
    bool seeded = a.more();
    std::uint32_t init = seeded ? static_cast<std::uint32_t>(a.integer()) : 0;
    std::uint32_t v;
    if (name == "x25") v = crc::x25(data, seeded ? init : 0xFFFF);
    else if (name == "x25_bitwise") v = crc::x25_bitwise(data, seeded ? init : 0xFFFF);
    else if (name == "ccitt_false") v = crc::ccitt_false(data, seeded ? init : 0xFFFF);
    else if (name == "dvb_s2") v = crc::dvb_s2(data, init);
    else throw BadRequest("unknown crc " + name);
    return Obj().num("crc", v).done();
  }
  if (op == "can15") return Obj().num("crc", crc::can15(a.bits())).done();
  if (op == "dshot_crc") {
    auto v = static_cast<std::uint64_t>(a.integer());
    return Obj().num("crc", crc::dshot_crc(v, a.flag())).done();
  }
  if (op == "sha256") {
    auto d = mavlink::sha256(a.hex());
    return Obj().str("digest", hex(Bytes(d.begin(), d.end()))).done();
  }

  // ------------------------------------------------------------------ rc ----
  if (op == "sbus_encode") {  // 16 channels, then ch17 ch18 frame_lost failsafe as 0/1
    rc::SbusFrame f;
    f.channels = a.integers(16);
    f.ch17 = a.flag(), f.ch18 = a.flag(), f.frame_lost = a.flag(), f.failsafe = a.flag();
    return Obj().str("hex", hex(rc::sbus_encode(f))).done();
  }
  if (op == "sbus_decode") {
    rc::SbusFrame f = rc::sbus_decode(a.hex());
    return Obj()
        .raw("channels", list(f.channels))
        .boolean("ch17", f.ch17)
        .boolean("ch18", f.ch18)
        .boolean("frame_lost", f.frame_lost)
        .boolean("failsafe", f.failsafe)
        .done();
  }
  if (op == "pack11") return Obj().str("hex", hex(rc::pack11(a.integers(16)))).done();
  if (op == "unpack11") return Obj().raw("channels", list(rc::unpack11(a.hex()))).done();
  if (op == "crsf_frame") {  // type address payload
    std::int64_t type = a.integer(), address = a.integer();
    return Obj().str("hex", hex(rc::crsf_frame(type, a.hex(), address))).done();
  }
  if (op == "crsf_rc") return Obj().str("hex", hex(rc::crsf_rc(a.integers(16)))).done();
  if (op == "crsf_battery") {
    double volts = a.real(), amps = a.real();
    std::int64_t used = a.integer(), percent = a.integer();
    return Obj().str("hex", hex(rc::crsf_battery(volts, amps, used, percent))).done();
  }
  if (op == "crsf_attitude") {
    double p = a.real(), r = a.real(), y = a.real();
    return Obj().str("hex", hex(rc::crsf_attitude(p, r, y))).done();
  }
  if (op == "crsf_gps") {
    double lat = a.real(), lon = a.real(), spd = a.real(), hdg = a.real(), alt = a.real();
    return Obj().str("hex", hex(rc::crsf_gps(lat, lon, spd, hdg, alt, a.integer()))).done();
  }
  if (op == "crsf_link_statistics") {
    std::array<std::int64_t, 10> v{};
    for (auto& x : v) x = a.integer();
    return Obj().str("hex", hex(rc::crsf_link_statistics(v))).done();
  }
  if (op == "crsf_flight_mode") {  // mode as hex so any character survives tokenising
    Bytes m = a.hex();
    return Obj().str("hex", hex(rc::crsf_flight_mode(std::string(m.begin(), m.end())))).done();
  }
  if (op == "crsf_parse") {  // one or more chunks fed in order
    rc::CrsfParser parser;
    std::vector<std::string> frames;
    while (a.more())
      for (const auto& p : parser.feed(a.hex()))
        frames.push_back(Obj().num("address", p.address).num("type", p.type).str("payload", hex(p.payload)).done());
    return Obj()
        .raw("frames", join(frames))
        .num("crc_errors", parser.crc_errors)
        .num("dropped_bytes", parser.dropped_bytes)
        .num("buffered", parser.buffer.size())
        .done();
  }

  // --------------------------------------------------------------- dshot ----
  if (op == "dshot_encode") {  // value telemetry bidirectional
    std::int64_t value = a.integer();
    bool telemetry = a.flag();
    return Obj().num("frame", dshot::encode(value, telemetry, a.flag())).done();
  }
  if (op == "dshot_decode") {  // frame bidirectional
    auto frame = static_cast<std::uint64_t>(a.integer());
    dshot::Frame f = dshot::decode(frame, a.flag());
    return Obj().num("value", f.value).boolean("telemetry", f.telemetry).boolean("crc_ok", f.crc_ok).done();
  }
  if (op == "erpm_word") return Obj().num("word", dshot::erpm_word(a.integer())).done();
  if (op == "gcr_encode") return Obj().num("line", dshot::gcr_encode(static_cast<std::uint64_t>(a.integer()))).done();
  if (op == "gcr_decode") {
    auto r = dshot::gcr_decode(static_cast<std::uint64_t>(a.integer()));
    return Obj().num("word", r.first).boolean("ok", r.second).done();
  }

  // ----------------------------------------------------------------- can ----
  if (op == "can_frame") {  // extended remote id data [acked interframe]
    can::Frame f = read_can_frame(a);
    bool acked = a.more() ? a.flag() : true;
    bool interframe = a.more() ? a.flag() : true;
    return Obj()
        .str("header_bits", bitstr(f.header_bits()))
        .num("crc", f.crc())
        .str("stuffed_bits", bitstr(f.stuffed_bits()))
        .str("wire_bits", bitstr(f.wire_bits(acked, interframe)))
        .done();
  }
  if (op == "can_stuff") return Obj().str("bits", bitstr(can::stuff(a.bits()))).done();
  if (op == "can_destuff") return Obj().str("bits", bitstr(can::destuff(a.bits()))).done();
  if (op == "can_decode") {
    can::Frame f = can::decode(a.bits());
    return Obj()
        .num("id", f.can_id)
        .boolean("extended", f.extended)
        .boolean("remote", f.remote)
        .str("data", hex(f.data))
        .done();
  }
  if (op == "can_arbitrate") {  // n, then n frames of: extended remote id data
    std::int64_t n = a.integer();
    std::vector<can::Frame> frames;
    for (std::int64_t k = 0; k < n; ++k) frames.push_back(read_can_frame(a));
    can::Arbitration r = can::arbitrate(frames);
    std::vector<std::string> lost;
    for (const auto& kv : r.lost_at) lost.push_back("[" + std::to_string(kv.first) + ", " + std::to_string(kv.second) + "]");
    return Obj().num("winner", r.winner).raw("lost_at", join(lost)).str("bus", bitstr(r.bus)).done();
  }

  // ------------------------------------------------------------- mavlink ----
  if (op == "mav_encode") {
    // version seq sysid compid msgid crc_extra base_length payload key link_id timestamp
    // key is "none" for an unsigned frame ("-" is an empty key, which still signs).
    int version = static_cast<int>(a.integer());
    std::int64_t seq = a.integer(), sysid = a.integer(), compid = a.integer(), msgid = a.integer();
    auto crc_extra = static_cast<std::uint8_t>(a.integer());
    auto base_length = static_cast<std::size_t>(a.integer());
    Bytes payload = a.hex();
    std::optional<Bytes> key;
    if (a.tok.at(a.i) == "none") ++a.i;
    else key = a.hex();
    std::int64_t link_id = a.integer(), timestamp = a.integer();
    Bytes frame = mavlink::encode(payload, msgid, crc_extra, base_length, seq, sysid, compid, version, key, link_id, timestamp);
    return Obj().str("hex", hex(frame)).done();
  }
  if (op == "mav_table") {  // msgid:crc_extra pairs, replaces the table
    mav_parser_template.crc_extra.clear();
    while (a.more()) {
      const std::string& t = a.next();
      auto colon = t.find(':');
      if (colon == std::string::npos) throw BadRequest("expected msgid:crc_extra");
      mav_parser_template.crc_extra[static_cast<std::uint32_t>(std::stoul(t.substr(0, colon)))] =
          static_cast<std::uint8_t>(std::stoul(t.substr(colon + 1)));
    }
    return Obj().num("messages", mav_parser_template.crc_extra.size()).done();
  }
  if (op == "mav_parse") {  // one or more chunks fed in order to a fresh parser
    mavlink::Parser parser;
    parser.crc_extra = mav_parser_template.crc_extra;
    std::vector<std::string> packets;
    while (a.more())
      for (const auto& p : parser.feed(a.hex()))
        packets.push_back(Obj()
                              .num("version", p.version)
                              .num("seq", p.seq)
                              .num("sysid", p.sysid)
                              .num("compid", p.compid)
                              .num("msgid", p.msgid)
                              .str("payload", hex(p.payload))
                              .num("incompat", p.incompat)
                              .num("compat", p.compat)
                              .raw("signature", p.signature ? quote(hex(*p.signature)) : "null")
                              .str("raw", hex(p.raw))
                              .done());
    return Obj()
        .raw("packets", join(packets))
        .num("crc_errors", parser.crc_errors)
        .num("unknown", parser.unknown)
        .num("duplicates", parser.duplicates)
        .num("bad_flags", parser.bad_flags)
        .num("dropped_bytes", parser.dropped_bytes)
        .num("lost", parser.lost)
        .num("buffered", parser.buffer.size())
        .done();
  }
  throw BadRequest("unknown request " + op);
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    Args a;
    std::istringstream in(line);
    for (std::string t; in >> t;) a.tok.push_back(t);
    if (a.tok.empty()) continue;
    std::string out;
    try {
      out = handle(a);
    } catch (const ProtocolError& e) {
      out = Obj().str("error", e.kind).done();
    } catch (const std::exception& e) {
      std::cerr << "protocol_tool: " << e.what() << " in: " << line << "\n";
      out = Obj().str("error", "BadRequest").done();
    }
    std::cout << out << "\n";
  }
  return 0;
}
