"""The drone design: harness, compatibility rules (each broken on purpose), parameters and performance."""

import copy
import json
from pathlib import Path

import pytest

from week04 import check, links, params, performance
from week04.design import DATA, Design, cable, harness, roles

BASE = json.loads((DATA / "design.json").read_text())
DOCS = Path(__file__).resolve().parents[1] / "docs" / "week04"


def variant(**changes) -> Design:
    spec = copy.deepcopy(BASE)
    for link_id, fields in changes.get("links", {}).items():
        link = next(lk for lk in spec["links"] if lk["id"] == link_id)
        link.update(fields)
    for link in changes.get("add_links", []):
        spec["links"].append(link)
    for name, info in changes.get("parts", {}).items():
        spec["parts"][name] = info
    spec.update(changes.get("top", {}))
    return Design.load(spec)


def findings(design: Design, rule: str, level: str = "error"):
    return [f for f in check.check(design) if f.rule == rule and f.level == level]


# -------------------------------------------------------------- design ----


def test_design_loads_every_part_and_link():
    d = Design.load()
    assert d.fc.part["id"] == "cube_orange_plus_adsb"
    assert len(d.links) == 9
    assert [p.id for p in d.link("motors").a] == ["AUX1", "AUX2", "AUX3", "AUX4"]
    with pytest.raises(KeyError):
        d.ports("fc.TELEM9")


@pytest.mark.parametrize(
    "signal,expected",
    [
        ("UART1_TXD (pin 8)", {"TX"}),
        ("RXD/SDA", {"RX", "SDA"}),
        ("TXD/SCL", {"TX", "SCL"}),
        ("VCC_5V", {"5V"}),
        ("GND (pin 6)", {"GND"}),
        ("CAN_H", {"CAN_H"}),
        ("DSM_RX", set()),
        ("SAFETY_LED", set()),
    ],
)
def test_pin_roles(signal, expected):
    assert roles(signal) == expected


def test_harness_crosses_tx_rx_and_follows_the_lidar_pin_order():
    d = Design.load()
    cables = {c.link: c for c in harness(d)}
    telem = cables["telemetry"]
    tx = next(w for w in telem.wires if w.role == "TX to RX")
    assert (tx.a_signal, tx.b_signal, tx.b_pin) == ("TX", "RX", 7)
    assert not any(w.role == "5V" for w in telem.wires)  # the radio has its own supply
    lidar = cables["lidar"]
    assert [(w.a_pin, w.b_pin) for w in lidar.wires] == [(2, 4), (3, 3), (4, 1), (1, 2)]  # reversed order on the sensor
    assert lidar.custom and not cables["can"].custom
    assert "companion_mavlink" not in cables  # plain USB cable
    assert cable(d.link("motors")) is None


# --------------------------------------------------------------- check ----


def test_the_design_passes_with_warnings_only():
    report = check.check(Design.load())
    assert report.count("error") == 0, [f for f in report if f.level == "error"]
    assert report.count("ok") > 50
    rules = {f.rule for f in report}
    for expected in (
        "baud rate",
        "logic level",
        "I2C address",
        "CAN bus load",
        "5 V budget",
        "bandwidth",
        "thrust to weight",
        "MAVLink channels",
    ):
        assert expected in rules


@pytest.mark.parametrize(
    "changes,rule",
    [
        ({"links": {"rc": {"a": "fc.GPS1"}, "gimbal": {"a": "fc.GPS2"}}}, "receive DMA"),
        ({"links": {"gimbal": {"a": "fc.TELEM2"}}}, "one device per UART"),
        ({"links": {"gimbal": {"protocol": "mavlink2"}}}, "device protocol"),
        ({"links": {"rc": {"baud": 115200}}}, "baud rate"),
        ({"links": {"radio_power_test": {}}} if False else {"links": {"telemetry": {"powered_by": "fc"}}}, "radio power"),
        ({"links": {"lidar": {"clock_hz": 1000000}}}, "I2C clock"),
        ({"links": {"gimbal": {"a": "fc.CONS"}}}, "port available"),
        ({"links": {"motors": {"protocol": "dshot600"}}}, "DShot rate"),
        ({"links": {"can": {"a": "fc.TELEM2"}}}, "physical layer"),
        ({"top": {"firmware": "ArduCopter 4.7.1 stable"}}, "firmware features"),
        ({"links": {"video": {"protocol": "unknown_proto"}}}, "known protocol"),
    ],
)
def test_each_rule_catches_its_mistake(changes, rule):
    d = variant(**changes)
    errors = findings(d, rule)
    assert errors, f"{rule} did not fire; errors were {[f for f in check.check(d) if f.level == 'error']}"


def test_i2c_address_clash_is_caught():
    second = {
        "id": "lidar2",
        "a": "fc.I2C2",
        "b": "lidar.MAIN",
        "protocol": "i2c",
        "clock_hz": 100000,
        "address": 16,
        "powered_by": "fc",
        "purpose": "second",
    }
    d = variant(add_links=[second])
    assert findings(d, "I2C address")


def test_logic_level_mismatch_is_caught():
    d = Design.load()
    rx = next(c for c in d.catalog["components"] if c["id"] == "radiomaster_rp3_v2")
    rx["logic_v"] = 5.0
    for p in rx["ports"]:
        p["logic_v"] = 5.0
    d = Design.load(BASE, d.catalog)
    assert findings(d, "logic level")


def test_payload_is_limited_by_thrust():
    d = variant(top={"payload_mass_g": {"sandbag": 3500}})
    assert findings(d, "thrust to weight") or findings(d, "hover throttle")


def test_5v_budget_overload_is_caught():
    cat = json.loads((DATA / "catalog.json").read_text())
    here4 = next(c for c in cat["components"] if c["id"] == "cubepilot_here4")
    here4["supply"]["typical_a"] = 1.2
    here4["supply"]["peak_a"] = 1.6
    assert findings(Design.load(BASE, cat), "5 V budget")


def test_mavlink_channel_order():
    ports = check.mavlink_ports(Design.load())
    assert ports == [(0, "USB_GH"), (1, "TELEM1"), (5, "ADSB_INTERNAL")]


# -------------------------------------------------------------- params ----


def test_generated_parameters_are_valid_for_4_7_1():
    d = Design.load()
    plist = params.generate(d)
    assert params.validate(plist) == []
    values = {p.name: p.value for p in plist}
    assert values["SERIAL1_PROTOCOL"] == 2 and values["SERIAL1_BAUD"] == 57
    assert values["SERIAL2_PROTOCOL"] == 45 and values["SERIAL2_BAUD"] == 921
    assert values["SERIAL4_PROTOCOL"] == 23 and "SERIAL4_BAUD" not in values
    assert values["SERIAL3_PROTOCOL"] == 8 and values["MNT1_TYPE"] == 8
    assert values["GPS1_TYPE"] == 9 and values["CAN_P1_DRIVER"] == 1
    assert values["MAV2_EXTRA1"] == 4 and values["MAV1_EXTRA1"] == 20  # TELEM1 is the second MAVLink port
    assert values["MOT_PWM_TYPE"] == 5 and [values[f"SERVO{n}_FUNCTION"] for n in range(9, 13)] == [33, 34, 35, 36]
    assert values["RNGFND1_ADDR"] == 16 and values["BATT_VOLT_MULT"] == 18.182


def test_validator_catches_old_names_and_bad_values():
    bad = [
        params.Param("SR1_EXTRA1", 4, ""),
        params.Param("GPS1_TYPE", 99, ""),
        params.Param("ATC_RAT_PIT_I", 2.0, ""),
        params.Param("RC_OPTIONS", 1 << 30, ""),
        params.Param("FRAME_CLASS", 1, ""),
        params.Param("FRAME_CLASS", 1, ""),
    ]
    messages = [p.message for p in params.validate(bad)]
    assert any("no such parameter" in m for m in messages)
    assert any("not one of" in m for m in messages)
    assert any("outside" in m for m in messages)
    assert any("bits" in m for m in messages)
    assert any("twice" in m for m in messages)


def test_param_file_round_trip(tmp_path):
    plist = params.generate(Design.load())
    path = params.write(plist, tmp_path / "x.param", header="test")
    back = params.read(path)
    assert back == {p.name: float(p.value) for p in plist}


def test_committed_param_file_matches_the_generator():
    assert params.read(DOCS / "protocol_pro.param") == {p.name: float(p.value) for p in params.generate(Design.load())}


# --------------------------------------------------------- performance ----


def test_calibration_reproduces_holybro_claims():
    d = Design.load()
    _, cal = performance.calibrate(d)
    assert abs(cal["reference_minutes"] - performance.HOLYBRO_HOVER_MINUTES) < 0.1
    assert abs(cal["thrust_at_70_n"] - cal["target_thrust_at_70_n"]) < 0.05
    assert 0.02 <= cal["cp_fitted"] <= 0.08 and cal["resistance_fitted_ohm"] > cal["resistance_estimate_ohm"]


def test_heavier_means_more_throttle_and_less_time():
    light = performance.analyse(Design.load())
    heavy = performance.analyse(variant(top={"payload_mass_g": {"extra": 600}}))
    assert heavy.hover_throttle > light.hover_throttle and heavy.hover_minutes < light.hover_minutes
    assert 1.8 < light.mass_kg < 2.2 and 2.0 < light.thrust_to_weight < 3.2


def test_propeller_model_is_monotonic():
    prop, _ = performance.calibrate(Design.load())
    thrust = [prop.at_throttle(t / 10, 14.8)[0] for t in range(1, 11)]
    assert thrust == sorted(thrust)


def test_inertia_and_sitl_frame():
    d = Design.load()
    perf = performance.analyse(d)
    ixx, iyy, izz = perf.inertia
    assert ixx == pytest.approx(iyy) and izz > ixx > 0.005
    frame = performance.sitl_frame(d, perf)
    assert set(frame) >= {"mass", "diagonal_size", "hoverThrOut", "disc_area", "moment_inertia"}
    assert frame["diagonal_size"] == 0.5


# --------------------------------------------------------------- links ----


def test_link_budget_arithmetic():
    budget = links.mavlink_budget("x", links.UartConfig(57600), {"EXTRA1": 10})
    attitude = next(r for r in budget.rows if r[1] == "ATTITUDE")
    assert attitude[3] == 12 + 28  # header, CRC and the 28 byte payload
    assert budget.bytes_per_second == pytest.approx(sum(hz * size for _, _, hz, size in budget.rows))
    assert "SIMSTATE" not in [r[1] for r in budget.rows]


def test_rtcm_sizes_and_radio_physics():
    assert links.rtcm_msm4_bytes(10, 2) == 173
    assert 500 < links.rtcm_base_bytes_per_second() < 800
    assert links.free_space_path_loss_db(1000, 915e6) == pytest.approx(91.67, abs=0.02)
    r = links.max_range_m(30, 3, 3, -105, 915e6, fade_margin_db=0)
    assert links.free_space_path_loss_db(r, 915e6) == pytest.approx(30 + 3 + 3 + 105, abs=1e-6)


def test_sitl_measurements_agree_with_the_prediction():
    data = json.loads((DOCS / "sitl.json").read_text())
    rows = data["link_budget"]["rows"]
    for r in rows:
        if r["message"] in links.CONDITIONAL:
            continue
        assert abs(r["measured_hz"] - r["predicted_hz"]) <= 0.1 * r["predicted_hz"] + 0.1, r
        assert r["measured_bytes_per_s"] <= r["predicted_hz"] * r["max_frame_bytes"] * 1.1, r
    assert data["link_budget"]["measured_bytes_per_s"] <= data["link_budget"]["predicted_bytes_per_s"]
    assert data["params"]["rejected"] == ["BRD_SER1_RTSCTS", "BRD_SER2_RTSCTS", "DDS_ENABLE"]
    s = data["signing"]
    assert s["telem1_unsigned_request_before_key"] and not s["telem1_unsigned_request_after_key"] and s["telem1_signed_request_after_key"]
    assert s["usb_packets_signed"] == s["usb_signatures_verified"] > 0
    assert data["channel_mapping"]["serial1_attitude_hz"] == pytest.approx(4, abs=0.3)
