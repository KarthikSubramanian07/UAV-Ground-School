"""The week05 command line."""

from __future__ import annotations

from pathlib import Path

import pytest

from week05.cli import main

CAPTURES = Path(__file__).resolve().parents[1] / "docs" / "week05" / "captures"


def test_cdr(capsys):
    assert main(["cdr", "std_msgs/msg/String", '{"data": "Hello World: 0"}']) == 0
    out = capsys.readouterr().out
    assert "23 bytes" in out and "00 01 00 00 0f 00 00 00" in out


def test_hash_and_show(capsys):
    main(["hash", "std_msgs/msg/String"])
    assert capsys.readouterr().out.strip().startswith("RIHS01_df668c74")
    main(["show", "ugs_interfaces/action/FlyToWaypoint"])
    assert capsys.readouterr().out.split("---")[2].strip() == "float32 distance_remaining"
    main(["show", "example_interfaces/srv/AddTwoInts"])
    assert "int64 sum" in capsys.readouterr().out


def test_qos(capsys):
    assert main(["qos", "sensor_data", "default"]) == 1
    assert "Best effort publisher and reliable subscription" in capsys.readouterr().out
    assert main(["qos", "default", "default"]) == 0


def test_bad_value():
    with pytest.raises(SystemExit):
        main(["cdr", "std_msgs/msg/String", "not json"])


@pytest.mark.skipif(not (CAPTURES / "pubsub.pcap").is_file(), reason="no capture")
def test_decode(capsys):
    main(["decode", str(CAPTURES / "pubsub.pcap"), "--limit", "50"])
    assert "discovery multicast" in capsys.readouterr().out
