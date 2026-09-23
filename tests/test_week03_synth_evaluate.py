import json
import math

import cv2
import numpy as np
import pytest

from week03 import blobs, evaluate, synth


def test_truth_follows_the_pixel_center_convention():
    # One dot, no nuisances: the intensity centroid must equal the truth center.
    for seed in range(50):
        scene = synth.polka_scene(synth.SceneConfig(width=200, height=160, dots=1, radius=(20, 20), seed=seed))
        dot = scene.dots[0]
        if not dot.truncated:
            break
    gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    weight = np.abs(gray - np.median(gray))
    ys, xs = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    cx = (weight * xs).sum() / weight.sum()
    cy = (weight * ys).sum() / weight.sum()
    assert math.hypot(cx - dot.x, cy - dot.y) < 0.05
    assert math.sqrt(weight.sum() / weight.max() / math.pi) == pytest.approx(dot.radius, rel=0.03)


def test_scenes_are_reproducible_and_presets_cover_the_nuisances():
    a = synth.polka_scene(synth.SceneConfig.preset("fabric", seed=2))
    b = synth.polka_scene(synth.SceneConfig.preset("fabric", seed=2))
    assert np.array_equal(a.image, b.image)
    for name in synth.PRESETS:
        scene = synth.polka_scene(synth.SceneConfig.preset(name, seed=0))
        assert scene.image.shape == (480, 640, 3) and len(scene.dots) > 10
    assert len(synth.polka_scene(synth.SceneConfig.preset("distractors")).distractors) == 25
    with pytest.raises(ValueError):
        synth.SceneConfig.preset("nope")


def test_dots_never_overlap():
    scene = synth.polka_scene(synth.SceneConfig.preset("crowded", seed=4))
    for i, a in enumerate(scene.dots):
        for b in scene.dots[i + 1 :]:
            assert math.hypot(a.x - b.x, a.y - b.y) > a.radius + b.radius


def test_tilted_truth_stays_on_the_dots():
    scene = synth.polka_scene(synth.SceneConfig.preset("tilted", seed=1))
    lab = cv2.cvtColor(scene.image, cv2.COLOR_BGR2Lab).astype(np.float64)
    wall = np.median(lab.reshape(-1, 3), axis=0)
    inside = [d for d in scene.dots if 0 <= d.x < 640 and 0 <= d.y < 480 and d.radius > 6]
    off = [np.linalg.norm(lab[int(d.y), int(d.x)] - wall) for d in inside]
    assert np.median(off) > 20  # dot centers land on dot colored pixels


def test_match_dots_counts_and_tolerances():
    truth = [blobs.Blob(10, 10, 10), blobs.Blob(50, 50, 10), blobs.Blob(90, 90, 10)]
    det = [blobs.Blob(11, 10, 10), blobs.Blob(50, 50, 30), blobs.Blob(200, 200, 5)]
    s = evaluate.match_dots(det, truth)
    assert (s.tp, s.fp, s.fn) == (1, 2, 2)  # the second detection is 3x too big
    assert s.precision == pytest.approx(1 / 3) and s.recall == pytest.approx(1 / 3)
    assert s.center_error == pytest.approx(1.0)


def test_difficult_truth_and_ignored_detections_are_not_scored():
    truth = [blobs.Blob(10, 10, 10), blobs.Blob(50, 50, 10)]
    det = [blobs.Blob(10, 10, 10), blobs.Blob(300, 300, 10)]
    s = evaluate.match_dots(det, truth, difficult=[False, True], ignored=[False, True])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.f1 == 1.0


def test_one_to_one_matching():
    truth = [blobs.Blob(10, 10, 10)]
    det = [blobs.Blob(10, 10, 10), blobs.Blob(11, 10, 10)]
    s = evaluate.match_dots(det, truth)
    assert (s.tp, s.fp) == (1, 1)


def test_annotation_files_are_consistent():
    for name in ("polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg"):
        t = evaluate.truth_for(name)
        assert t.image == name and t.required > 20
        assert len(t.dots) == len(t.difficult)
        w, h = t.size
        for d in t.dots:
            assert -d.radius <= d.x <= w + d.radius and -d.radius <= d.y <= h + d.radius
    three = evaluate.truth_for("polka_dots_3.jpg")
    assert three.ignored(300, 50) and not three.ignored(250, 450)
    objects = json.loads((evaluate.ANNOTATIONS / "objects.json").read_text())
    assert sorted(o["label"] for o in objects["objects"]) == ["cone", "cone", "cube", "cube", "ring", "ring"]


def test_iou_helpers():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[:, :5] = 1
    b[:, :] = 1
    assert evaluate.mask_iou(a, b) == pytest.approx(0.5)
    assert evaluate.box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert evaluate.box_iou((0, 0, 1, 1), (5, 5, 1, 1)) == 0.0
