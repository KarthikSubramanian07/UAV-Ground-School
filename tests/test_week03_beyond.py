"""Challenge 1 beyond the polka dots: noise estimate, denoising, truth lookup and the objects and shapes study."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from week03 import beyond, cli, dots, evaluate, objects

DATA = Path(__file__).resolve().parents[1] / "docs" / "week03" / "photos"
PHOTOS = pytest.mark.skipif(not (DATA / "shapes.png").is_file(), reason="the ground school photos are not in docs/week03/photos")


def test_noise_estimate_recovers_added_noise():
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(60, 190, 400, dtype=np.float32), (300, 1))
    clean = np.dstack([ramp, ramp[:, ::-1], np.full_like(ramp, 128)]).astype(np.uint8)
    assert max(dots.estimate_noise(clean)) < 0.3  # a smooth image has no noise, whatever its gradient
    noisy = np.clip(clean + rng.normal(0, 6, clean.shape), 0, 255).astype(np.uint8)
    lightness_noise = dots.estimate_noise(noisy)[0]
    assert 3.0 < lightness_noise < 9.0


def test_denoise_is_off_by_default_and_cleans_when_asked():
    rng = np.random.default_rng(1)
    image = np.full((120, 160, 3), 150, np.uint8)
    cv2.circle(image, (80, 60), 25, (60, 40, 160), -1)
    noisy = np.clip(image + rng.normal(0, 12, image.shape), 0, 255).astype(np.uint8)
    assert dots.denoise(noisy, 0) is noisy
    assert dots.DotSettings().denoise == 0.0
    cleaned = dots.denoise(noisy, 10)
    assert np.abs(cleaned.astype(int) - image).mean() < np.abs(noisy.astype(int) - image).mean() / 2


def test_dot_truth_path_skips_mask_annotations():
    assert evaluate.dot_truth_path("polka_dots_1.png").name == "polka_dots_1.json"
    assert evaluate.dot_truth_path("shapes.png").name == "shapes.json"
    assert evaluate.dot_truth_path("objects.jpg") is None  # masks, not dots
    assert evaluate.dot_truth_path("nothing_here.png") is None


def test_shapes_truth_is_ten_purple_dots():
    truth = evaluate.truth_for("shapes.png")
    assert len(truth.dots) == truth.required == 10
    assert all(14 < d.radius < 17 and d.color == "purple" for d in truth.dots)


def test_compare_on_a_photo_with_mask_truth_does_not_crash(tmp_path):
    """``objects.json`` holds masks; compare used to read it as dots and raise KeyError."""
    image = np.full((160, 200, 3), 200, np.uint8)
    for x in (50, 100, 150):
        cv2.circle(image, (x, 80), 14, (40, 40, 200), -1)
    path = tmp_path / "objects.jpg"
    cv2.imwrite(str(path), image)
    assert cli.main(["compare", str(path), "--min-radius", "8", "--out", str(tmp_path / "sheet.jpg"), "--no-show"]) == 0
    assert cli.main(["dots", str(path), "--denoise", "5", "--min-roundness", "0.9", "--no-show"]) == 0


def _truth_object(name, label, box, shape):
    mask = np.zeros(shape, np.uint8)
    x, y, w, h = box
    mask[y : y + h, x : x + w] = 1
    return objects.TruthObject(label, name, mask, box)


def test_box_matching_is_one_to_one_and_ranks_the_strongest():
    truth = [_truth_object("a", "cube", (10, 10, 50, 50), (200, 200)), _truth_object("b", "cone", (120, 120, 40, 40), (200, 200))]
    found = [
        beyond.Detection((12, 12, 48, 48), 9.0),  # a
        beyond.Detection((11, 11, 50, 50), 8.0),  # a again: a duplicate is a false positive
        beyond.Detection((150, 10, 20, 20), 7.0),  # nothing
        beyond.Detection((120, 121, 40, 40), 1.0),  # b, but weakest
    ]
    score = beyond.score_objects(found, truth)
    assert score["matched"] == 2 and score["false_positives"] == 2 and score["missed"] == []
    assert score["strongest_matched"] == 1  # only the top two count, and one of them is the duplicate
    assert 0.9 < score["mean_box_iou"] <= 1.0


def test_scale_space_finds_solid_pieces_but_not_a_ring():
    image = np.full((360, 480, 3), 170, np.uint8)
    cv2.circle(image, (110, 180), 60, (200, 60, 60), -1)  # a solid piece
    cv2.circle(image, (340, 180), 75, (40, 80, 230), 24)  # a ring: its center is background
    found = beyond.scale_space_objects(image, "log")
    strongest = found[0].circle
    assert abs(strongest[0] - 110) < 6 and abs(strongest[1] - 180) < 6 and 45 < strongest[2] < 80
    assert not any(abs(d.circle[0] - 340) < 10 and abs(d.circle[1] - 180) < 10 and d.circle[2] > 60 for d in found)


@PHOTOS
def test_contours_find_every_dot_in_the_shapes_photo():
    image = cv2.imread(str(DATA / "shapes.png"))
    score = evaluate.score_against(beyond.contour_dots(image), evaluate.truth_for("shapes.png"))
    assert score.tp == 10 and score.fp == 0


def test_committed_study_matches_the_readme_claims():
    result = json.loads((Path(__file__).resolve().parents[1] / "docs" / "week03" / "beyond.json").read_text())
    obj, shp = result["objects"], result["shapes"]
    assert obj["objects.py"]["matched"] == 6 and obj["objects.py"]["false_positives"] == 0
    for kind in ("log", "dog", "doh"):
        assert obj[kind]["strongest_matched"] == 4 and set(obj[kind]["missed"]) == {"ring_top", "ring_bottom"}
    assert all(shp[m]["tp"] == 0 for m in dots.METHODS)  # no dot detector survives the raw noise
    assert shp["log+nlm"]["tp"] == 8 and shp["log+nlm+0.90"]["tp"] == 10 and shp["contours"]["f1"] == 1.0
    noise = result["noise_lab"]
    assert noise["shapes.png"][1] > 4 * max(v[1] for k, v in noise.items() if k != "shapes.png")
