import json

import cv2

from week02 import cli, synth


def test_colors_command(tmp_path, capsys):
    image = tmp_path / "targets.jpg"
    cv2.imwrite(str(image), synth.suas_targets()[0])
    report = tmp_path / "report.json"
    code = cli.main(["colors", str(image), "--out", str(tmp_path / "layers"), "--json", str(report), "--no-show"])
    assert code == 0
    output = capsys.readouterr().out
    assert "octagon" in output and "star" in output
    data = json.loads(report.read_text())
    assert {layer["name"] for layer in data} >= {"red", "blue", "yellow", "purple", "orange"}
    assert (tmp_path / "layers" / "contact_sheet.png").exists()


def test_colors_command_missing_file(tmp_path, capsys):
    assert cli.main(["colors", str(tmp_path / "missing.jpg"), "--no-show"]) == 2
    assert "No image found" in capsys.readouterr().err


def test_stitch_command_reports_accuracy(small_flight, tmp_path, capsys):
    out = tmp_path / "pano.jpg"
    code = cli.main([
        "stitch", str(small_flight.video), "--out", str(out), "--every", "4", "--no-show",
        "--path-overlay", str(tmp_path / "path.jpg"), "--transforms", str(tmp_path / "t.csv"),
        "--truth", str(small_flight.truth_csv), "--world", str(small_flight.world_png),
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "pose RMSE" in output and "ZNCC" in output
    assert cv2.imread(str(out)) is not None and (tmp_path / "path.jpg").exists() and (tmp_path / "t.csv").exists()


def test_frames_command(small_flight, tmp_path, capsys):
    assert cli.main(["frames", str(small_flight.video), str(tmp_path / "f"), "--every", "100"]) == 0
    assert "wrote 3 frames" in capsys.readouterr().out


def test_simulate_quick(tmp_path, capsys):
    assert cli.main(["simulate", str(tmp_path), "--quick"]) == 0
    for name in ["stop_sign.jpg", "apple.jpg", "targets.jpg", "targets.json", "world.png", "flight.mp4", "flight_truth.csv"]:
        assert (tmp_path / name).exists(), name
