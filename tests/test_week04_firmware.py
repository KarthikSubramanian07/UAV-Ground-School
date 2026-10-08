"""The firmware question applied to the design: is the table complete, sourced and turned into the right verdicts."""

import copy
import json

import pytest

from week04 import cli, firmware
from week04.design import Design


@pytest.fixture(scope="module")
def design():
    return Design.load()


@pytest.fixture(scope="module")
def data():
    return firmware.load()


def test_every_cell_is_sourced(data):
    assert firmware.validate(data) == []
    for f in data["features"]:
        for fw in firmware.FIRMWARES:
            assert f[fw]["how"].endswith("."), f"{f['id']}.{fw}"
    text = firmware.DATA.read_text()
    assert chr(0x2014) not in text and chr(0x2013) not in text  # no em or en dashes


def test_validate_catches_bad_cells(data):
    broken = copy.deepcopy(data)
    broken["features"][0]["px4"]["source"] = "docs.px4.io"
    broken["features"][1]["betaflight"]["support"] = "maybe"
    del broken["features"][2]["ardupilot"]
    problems = firmware.validate(broken)
    assert len(problems) == 3


def test_the_table_covers_every_link_in_the_design(design, data):
    need = firmware.required_features(design)
    for link in ("telemetry", "companion_dds", "companion_mavlink", "gimbal", "rc", "can", "lidar", "motors"):
        assert link in need
    assert {"board", "rtk", "adsb", "battery", "autonomy", "control", "config"} <= set(need)
    assert set(need) <= {f["id"] for f in data["features"]}


def test_needs_follow_the_design(design):
    """Drop the rangefinder and the CAN bus, and the firmware question changes with it."""
    spec = copy.deepcopy(design.spec)
    spec["links"] = [link for link in spec["links"] if link["id"] not in ("lidar", "can")]
    smaller = Design.load(spec, design.catalog)
    need = firmware.required_features(smaller)
    assert "lidar" not in need and "can" not in need and "rtk" not in need


def test_verdicts(design, data):
    v = firmware.verdicts(design, data)
    assert v["ardupilot"].lost == [] and v["ardupilot"].flies
    assert v["ardupilot"].partial == ["companion_dds"]  # AP_DDS needs a custom build, which the design already plans
    assert v["px4"].lost == [] and v["px4"].flies
    assert {"telemetry", "rc", "gimbal", "lidar"} <= set(v["px4"].partial)
    assert "board" in v["betaflight"].lost and not v["betaflight"].flies


def test_missing_rows_are_an_error(design, data):
    thin = copy.deepcopy(data)
    thin["features"] = [f for f in thin["features"] if f["id"] != "gimbal"]
    with pytest.raises(ValueError, match="gimbal"):
        firmware.verdicts(design, thin)


def test_markdown_and_report(design, data):
    md = firmware.markdown(design, data)
    assert md.count("([source](https://") == 3 * len(firmware.required_features(design))
    assert "It cannot fly this drone." in md and md.count("It cannot fly") == 1
    report = firmware.report(design, data)
    assert json.loads(json.dumps(report))["verdicts"]["betaflight"]["flies"] is False


def test_cli(capsys):
    assert cli.main(["firmware"]) == 0
    out = capsys.readouterr().out
    assert "ArduPilot: 14 of 15 supported" in out and "Betaflight" in out
