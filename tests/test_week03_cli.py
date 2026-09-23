import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from week03 import cli, evaluate

DATA = Path(__file__).resolve().parents[1] / "docs" / "week03" / "photos"
PHOTOS = [("polka_dots_1.png", 0.99), ("polka_dots_2.jpg", 0.97), ("polka_dots_3.jpg", 0.95)]


def test_synth_dots_and_compare(tmp_path, capsys):
    assert cli.main(["synth", str(tmp_path / "scenes"), "--preset", "flat", "--seed", "1"]) == 0
    truth = json.loads((tmp_path / "scenes" / "flat.json").read_text())
    assert len(truth["dots"]) == 40
    image = str(tmp_path / "scenes" / "flat.png")
    assert (
        cli.main(
            ["dots", image, "--method", "doh", "--out", str(tmp_path / "dots.jpg"), "--json", str(tmp_path / "dots.json"), "--no-show"]
        )
        == 0
    )
    report = json.loads((tmp_path / "dots.json").read_text())
    assert report["method"] == "doh" and report["count"] >= 35
    assert cv2.imread(str(tmp_path / "dots.jpg")) is not None
    assert cli.main(["dots", image, "--colors", "red", "--min-radius", "10", "--no-show"]) == 0
    assert cli.main(["compare", image, "--out", str(tmp_path / "sheet.jpg"), "--no-show"]) == 0
    assert cv2.imread(str(tmp_path / "sheet.jpg")).shape[1] == 1600
    out = capsys.readouterr().out
    assert "doh" in out and "contrast" in out


def test_objects_and_targets_commands(tmp_path):
    image = np.full((300, 400, 3), 180, np.uint8)
    cv2.circle(image, (100, 150), 70, (40, 60, 235), 25)
    cv2.rectangle(image, (230, 90), (350, 210), (190, 60, 110), -1)
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), image)
    assert cli.main(["objects", str(path), "--json", str(tmp_path / "o.json"), "--out", str(tmp_path / "o.jpg"), "--no-show"]) == 0
    labels = sorted(p["label"] for p in json.loads((tmp_path / "o.json").read_text())["pieces"])
    assert labels == ["cube", "ring"]
    assert cli.main(["targets", str(path), "--json", str(tmp_path / "t.json"), "--no-show"]) == 0


def test_benchmark_command(tmp_path):
    assert (
        cli.main(["benchmark", str(tmp_path), "--data", str(tmp_path / "missing"), "--seeds", "1", "--presets", "flat", "--no-ablation"])
        == 0
    )
    report = json.loads((tmp_path / "dots_benchmark.json").read_text())
    assert set(report["synthetic"]) == {"flat"} and report["synthetic_overall"]["log"]["f1"] > 0.95
    assert "Synthetic scenes" in (tmp_path / "dots_benchmark.md").read_text()


def test_errors_are_reported(capsys):
    assert cli.main(["dots", "does-not-exist.jpg", "--no-show"]) == 2
    assert "No image found" in capsys.readouterr().err
    assert cli.main(["yolo", "train", "not-an-experiment"]) == 2


def test_yolo_report(tmp_path):
    classes = {n: {"instances": 1, "AP50": 0.5, "AP50-95": 0.3, "P": 1, "R": 1} for n in ("ball", "goalkeeper", "player", "referee")}
    results = {
        "baseline": {
            "experiment": {"split": "clip", "imgsz": 640, "description": "test"},
            "full_frame": {"mAP50": 0.5, "mAP50-95": 0.3, "classes": classes, "ms_per_frame": 40},
            "train_seconds": 600,
        }
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    assert cli.main(["yolo", "report", str(tmp_path / "results.json"), "--out", str(tmp_path / "r.md")]) == 0
    assert "| `baseline` |" in (tmp_path / "r.md").read_text()


@pytest.mark.parametrize("name,minimum", PHOTOS)
def test_ground_school_photos(name, minimum):
    path = DATA / name
    if not path.is_file():
        pytest.skip("the ground school photos are not in docs/week03/photos")
    from week03 import dots

    report = dots.find_dots(cv2.imread(str(path)), method="log")
    score = evaluate.score_against(report.dots, evaluate.truth_for(name))
    assert score.fp == 0
    assert score.f1 >= minimum
