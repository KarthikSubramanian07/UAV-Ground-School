from pathlib import Path

import cv2
import numpy as np
import pytest

from week03 import objects, targets

DATA = Path(__file__).resolve().parents[1] / "docs" / "week03" / "photos"


def cone(image, x, y, s, color, angle=0.0):
    """A traffic cone: a tapered body on a square flange, optionally rotated about (x, y)."""
    body = np.array([[-0.18, -1.0], [0.18, -1.0], [0.42, 0.55], [-0.42, 0.55]])
    base = np.array([[-0.62, 0.5], [0.62, 0.5], [0.62, 0.75], [-0.62, 0.75]])
    c, si = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -si], [si, c]])
    for poly in (body, base):
        pts = (poly @ rot.T) * s + [x, y]
        cv2.fillPoly(image, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)


def cube(image, x, y, s, color):
    """A cube seen from above a corner: a hexagon."""
    a = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    pts = np.stack([x + s * np.cos(a), y + s * np.sin(a)], axis=1)
    cv2.fillPoly(image, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)


def ring(image, x, y, r, color, thickness):
    cv2.ellipse(image, (x, y), (r, int(r * 0.8)), 10, 0, 360, color, thickness, cv2.LINE_AA)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    image = (np.full((600, 900, 3), 185, np.float32) + rng.normal(0, 6, (600, 900, 1))).clip(0, 255).astype(np.uint8)
    ring(image, 140, 140, 90, (40, 60, 235), 30)
    cube(image, 420, 130, 80, (190, 60, 110))
    cube(image, 440, 265, 80, (190, 60, 110))  # touching the first cube
    cone(image, 700, 170, 120, (20, 200, 245))
    cone(image, 200, 430, 120, (20, 200, 245), angle=1.3)  # lying on its side
    ring(image, 650, 460, 100, (40, 60, 235), 34)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_synthetic_scene_is_read_correctly(scene):
    report = objects.detect_objects(scene)
    assert report.counts() == {"cone": 2, "cube": 2, "ring": 2}
    assert len(report.hues) == 3


def test_classification_ignores_color_and_orientation(scene):
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 180
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for variant in (shifted, cv2.rotate(scene, cv2.ROTATE_90_CLOCKWISE), cv2.flip(scene, 0)):
        assert objects.detect_objects(variant).counts() == {"cone": 2, "cube": 2, "ring": 2}


def test_touching_cubes_are_split(scene):
    report = objects.detect_objects(scene, objects.ObjectSettings(split_touching=False))
    assert report.counts().get("cube", 0) < 2


def test_features_of_simple_shapes():
    square = np.zeros((200, 200), np.uint8)
    cv2.rectangle(square, (50, 50), (150, 150), 1, -1)
    f, _ = objects.describe(square)
    assert f.solidity == pytest.approx(1, abs=0.01) and f.rect_fill == pytest.approx(1, abs=0.02) and f.hole_ratio == 0
    donut = np.zeros((200, 200), np.uint8)
    cv2.circle(donut, (100, 100), 80, 1, 25)
    f, _ = objects.describe(donut)
    assert f.hole_ratio > 0.3 and f.hole_centered < 0.05
    assert objects.classify(f)[0] == "ring"
    triangle = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(triangle, [np.array([[100, 10], [190, 190], [10, 190]])], 1)
    f, _ = objects.describe(triangle)
    assert f.triangularity == pytest.approx(1, abs=0.03)


def test_hue_clusters_wrap_around_red():
    hues = np.concatenate([np.full(500, 178), np.full(500, 2), np.full(500, 60)])
    centers = objects.hue_clusters(hues)
    assert len(centers) == 2


def test_annotation_and_summary(scene):
    report = objects.detect_objects(scene)
    out = objects.annotate(scene, report.pieces)
    assert out.shape == scene.shape
    summary = report.summary()
    assert summary["counts"] == report.counts() and len(summary["pieces"]) == 6


def test_scoring_against_masks(scene):
    report = objects.detect_objects(scene)
    truth = [objects.TruthObject(p.label, p.label, p.mask, p.bbox) for p in report.pieces]
    score = objects.score_objects(report.pieces, truth)
    assert score["matched"] == 6 and score["correct_class"] == 6 and score["mean_mask_iou"] == pytest.approx(1.0)


@pytest.mark.skipif(not (DATA / "objects.jpg").is_file(), reason="the ground school photos are not in docs/week03/photos")
def test_ground_school_objects_photo():
    image = cv2.imread(str(DATA / "objects.jpg"))
    report = objects.detect_objects(image)
    score = objects.score_objects(report.pieces, objects.load_objects())
    assert score["correct_class"] == 6 and score["false_positives"] == 0
    assert score["mean_mask_iou"] > 0.92 and score["mean_box_iou"] > 0.93


def test_targets_on_a_synthetic_card():
    rng = np.random.default_rng(1)
    image = (np.full((400, 700, 3), 128, np.float32) + rng.normal(0, 5, (400, 700, 1))).clip(0, 255).astype(np.uint8)
    a = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
    cv2.fillPoly(image, [np.round(np.stack([150 + 60 * np.cos(a), 150 + 60 * np.sin(a)], 1)).astype(np.int32)], (230, 200, 30))
    cv2.rectangle(image, (300, 100), (480, 200), (40, 40, 210), -1)
    for _i, (x, y) in enumerate([(560, 120), (600, 120), (580, 155)]):
        cv2.circle(image, (x, y), 13, (160, 60, 150), -1)
    found = targets.find_targets(image)
    shapes = sorted(t.shape for t in found)
    assert "pentagon" in shapes and "rectangle" in shapes and "group of 3 dots" in shapes


@pytest.mark.skipif(not (DATA / "shapes.png").is_file(), reason="the ground school photos are not in docs/week03/photos")
def test_ground_school_shapes_photo():
    found = targets.find_targets(cv2.imread(str(DATA / "shapes.png")))
    shapes = {t.shape for t in found}
    assert {"rectangle", "pentagon", "ellipse", "group of 10 dots"} <= shapes
