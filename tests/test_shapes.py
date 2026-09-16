import math

import cv2
import numpy as np
import pytest

from week02 import shapes, synth


def render(name: str, radius: float, rotation: float, size: int = 600) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    synth.fill_polygon(mask, synth.SHAPE_BUILDERS[name]((size / 2, size / 2), radius, rotation), 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 0.8)
    return (mask > 127).astype(np.uint8) * 255


@pytest.mark.parametrize("name", sorted(synth.SHAPE_BUILDERS))
@pytest.mark.parametrize("radius", [22, 60, 200])
@pytest.mark.parametrize("rotation", [0.13, 1.9, 4.4])
def test_classifies_every_suas_shape(name, radius, rotation):
    match = shapes.classify(render(name, radius, rotation))
    assert match is not None
    assert match.shape == name, match
    assert match.confidence > 0.85


def test_letter_holes_do_not_change_the_shape():
    mask = render("pentagon", 120, 0.4)
    cv2.putText(mask, "Q", (240, 360), cv2.FONT_HERSHEY_DUPLEX, 4, 0, 18)
    assert shapes.classify(mask).shape == "pentagon"


def test_long_rectangles_use_their_own_aspect_ratio():
    mask = np.zeros((400, 400), np.uint8)
    box = cv2.boxPoints(((200, 200), (300, 60), 25)).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    match = shapes.classify(mask)
    assert match.shape == "rectangle" and match.confidence > 0.95


def test_random_blob_is_irregular():
    rng = np.random.default_rng(3)
    angles = np.sort(rng.uniform(0, 2 * math.pi, 14))
    radii = rng.uniform(40, 160, 14)
    points = np.stack([200 + radii * np.cos(angles), 200 + radii * np.sin(angles)], axis=1).astype(np.int32)
    mask = np.zeros((400, 400), np.uint8)
    cv2.fillPoly(mask, [points], 255)
    assert shapes.classify(mask).shape == "irregular"


def test_tiny_or_empty_blobs_are_skipped():
    assert shapes.classify(np.zeros((50, 50), np.uint8)) is None
    tiny = np.zeros((50, 50), np.uint8)
    tiny[20:25, 20:25] = 255
    assert shapes.classify(tiny) is None


def test_fill_holes():
    ring = np.zeros((100, 100), np.uint8)
    cv2.circle(ring, (50, 50), 40, 255, -1)
    cv2.circle(ring, (50, 50), 20, 0, -1)
    assert shapes.fill_holes(ring)[50, 50] == 255
