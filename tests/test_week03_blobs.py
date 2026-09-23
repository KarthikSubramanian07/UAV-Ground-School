import math

import cv2
import numpy as np
import pytest

from week03 import blobs


def disk_image(size=160, cx=80.3, cy=79.6, r=20.0, value=1.0, background=0.0, ss=8):
    """An anti aliased disk rendered by supersampling."""
    big = np.full((size * ss, size * ss), background, np.float32)
    cv2.circle(
        big,
        (int(round(((cx + 0.5) * ss - 0.5) * 16)), int(round(((cy + 0.5) * ss - 0.5) * 16))),
        int(round(r * ss * 16)),
        value,
        -1,
        cv2.LINE_AA,
        shift=4,
    )
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


def test_sigma_ladder_is_geometric_and_covers_the_range():
    s = blobs.sigma_ladder(2.0, 32.0, per_octave=4)
    assert s[0] == pytest.approx(2.0)
    assert s[-1] == pytest.approx(32.0)
    ratios = s[1:] / s[:-1]
    assert np.allclose(ratios, ratios[0])
    assert len(s) == 17
    with pytest.raises(ValueError):
        blobs.sigma_ladder(0, 3)


def test_log_peaks_at_the_matched_scale_with_the_predicted_height():
    r = 20.0
    image = disk_image(r=r)
    sigmas = blobs.sigma_ladder(5, 30, 12)
    stack, eff = blobs.scale_space(image, sigmas, "log", polarity="bright", pyramid=False)
    center = stack[:, 80, 80]
    best = eff[int(np.argmax(center))]
    assert best == pytest.approx(r / math.sqrt(2), rel=0.08)
    assert center.max() == pytest.approx(blobs.DISK_PEAK, rel=0.05)


@pytest.mark.parametrize("kind", ["log", "dog", "doh"])
def test_detectors_find_a_disk_center_and_radius(kind):
    image = disk_image(cx=80.3, cy=79.6, r=18.0)
    found = blobs.detect_scale_space(image, kind, min_radius=4, max_radius=40, threshold=0.2, polarity="bright")
    assert len(found) == 1
    b = found[0]
    assert math.hypot(b.x - 80.3, b.y - 79.6) < 0.35
    assert b.radius == pytest.approx(18.0, rel=0.12)


@pytest.mark.parametrize("kind", ["log", "dog", "doh"])
def test_polarity(kind):
    dark = 1.0 - disk_image(r=15)
    assert blobs.detect_scale_space(dark, kind, min_radius=4, max_radius=30, threshold=0.2, polarity="bright") == []
    assert len(blobs.detect_scale_space(dark, kind, min_radius=4, max_radius=30, threshold=0.2, polarity="dark")) == 1
    assert len(blobs.detect_scale_space(dark, kind, min_radius=4, max_radius=30, threshold=0.2, polarity="both")) == 1


def test_color_blob_with_no_brightness_change_is_found_in_lab():
    # A blue disk on a gray of the same lightness: invisible in grayscale, obvious in CIELAB.
    lab = np.zeros((140, 140, 3), np.float32)
    lab[..., 0] = 60
    disk = disk_image(size=140, cx=70, cy=70, r=16)
    lab[..., 2] = -40 * disk
    found = blobs.detect_scale_space(lab, "log", min_radius=4, max_radius=30, threshold=5)
    assert len(found) == 1 and math.hypot(found[0].x - 70, found[0].y - 70) < 0.5
    gray = lab[..., 0]
    assert blobs.detect_scale_space(gray, "log", min_radius=4, max_radius=30, threshold=5) == []


def test_pyramid_matches_full_resolution():
    image = disk_image(size=240, cx=120, cy=118, r=50)
    sigmas = blobs.sigma_ladder(20, 60, 6)
    full, _ = blobs.scale_space(image, sigmas, "log", "bright", pyramid=False)
    fast, _ = blobs.scale_space(image, sigmas, "log", "bright", pyramid=True)
    assert np.abs(full[:, 118, 120] - fast[:, 118, 120]).max() < 0.02


def test_edges_do_not_trigger_doh():
    step = np.zeros((120, 120), np.float32)
    step[:, 60:] = 1.0
    assert blobs.detect_scale_space(step, "doh", min_radius=4, max_radius=30, threshold=0.15, polarity="both") == []


def test_sidelobes_are_suppressed():
    strong = blobs.Blob(50, 50, 20, response=30)
    necklace = [blobs.Blob(50 + 28 * math.cos(a), 50 + 28 * math.sin(a), 8, response=6) for a in np.linspace(0, 6, 8)]
    far = blobs.Blob(150, 50, 8, response=6)
    kept = blobs.suppress_sidelobes([strong, *necklace, far])
    assert strong in kept and far in kept and len(kept) == 2
    # With signatures, a faint neighbor of the same polarity survives; an opposite one does not.
    strong._signature = np.array([-30.0, 0.0, 0.0])
    same = blobs.Blob(50, 78, 8, response=6)
    same._signature = np.array([-5.0, 3.0, 0.0])
    ring = blobs.Blob(22, 50, 8, response=6)
    ring._signature = np.array([6.0, 0.0, 0.0])
    kept = blobs.suppress_sidelobes([strong, same, ring])
    assert same in kept and ring not in kept


def test_circle_overlap_values():
    assert blobs.circle_overlap(5, 5, 20) == 0
    assert blobs.circle_overlap(10, 3, 2) == 1  # small one inside
    half = blobs.circle_overlap(1.0, 1.0, 0.0)
    assert half == pytest.approx(1.0)
    # Two unit circles one radius apart overlap by 39.1% of either.
    assert blobs.circle_overlap(1.0, 1.0, 1.0) == pytest.approx(0.391, abs=0.001)


def test_prune_keeps_the_strongest():
    a = blobs.Blob(10, 10, 5, response=2)
    b = blobs.Blob(11, 10, 5, response=3)
    c = blobs.Blob(40, 40, 5, response=1)
    kept = blobs.prune_overlaps([a, b, c], 0.5)
    assert b in kept and c in kept and a not in kept


def test_simple_blob_detector_wrapper_and_filters():
    image = np.full((200, 300), 255, np.uint8)
    cv2.circle(image, (60, 100), 25, 0, -1)
    cv2.rectangle(image, (150, 75), (200, 125), 0, -1)
    both = blobs.detect_simple(image, blobs.BlobFilter(min_circularity=None, min_convexity=None, min_inertia=None))
    assert len(both) == 2
    round_only = blobs.detect_simple(image, blobs.BlobFilter(min_circularity=0.85))
    assert len(round_only) == 1
    assert round_only[0].x == pytest.approx(60, abs=1) and round_only[0].radius == pytest.approx(25, rel=0.1)


def test_contour_detector_filters_like_simple_blob_detector():
    mask = np.zeros((200, 300), np.uint8)
    cv2.circle(mask, (60, 100), 25, 255, -1)
    cv2.rectangle(mask, (150, 75), (200, 125), 255, -1)
    cv2.ellipse(mask, (250, 100), (40, 8), 0, 0, 360, 255, -1)
    found = blobs.detect_contours(mask, blobs.BlobFilter(blob_color=None, min_circularity=0.85, min_inertia=0.5))
    assert len(found) == 1
    assert found[0].radius == pytest.approx(25, rel=0.05)
    nested = np.full((100, 100), 255, np.uint8)
    cv2.circle(nested, (50, 50), 20, 0, 2)
    # With nested contours a drawn ring yields its inner disk and the hole it cuts in the background.
    rings = blobs.detect_contours(nested, blobs.BlobFilter(blob_color=None, min_area=100, max_area=5000), nested=True)
    assert len(rings) == 2
    assert len(blobs.prune_overlaps(rings, 0.5)) == 1


def test_matches_scikit_image_blob_log():
    feature = pytest.importorskip("skimage.feature")
    image = np.zeros((200, 200), np.float32)
    truth = [(50, 60, 10), (140, 70, 16), (100, 150, 6)]
    for x, y, r in truth:
        image = np.maximum(image, disk_image(size=200, cx=x, cy=y, r=r))
    theirs = feature.blob_log(image, min_sigma=3, max_sigma=15, num_sigma=25, threshold=0.1)
    ours = blobs.detect_scale_space(
        image, "log", min_radius=3 * math.sqrt(2), max_radius=15 * math.sqrt(2), per_octave=10, threshold=0.1, polarity="bright"
    )
    assert len(theirs) == len(ours) == 3
    for y, x, s in theirs:
        nearest = min(ours, key=lambda b: math.hypot(b.x - x, b.y - y))
        assert math.hypot(nearest.x - x, nearest.y - y) <= 1.0
        assert nearest.radius == pytest.approx(s * math.sqrt(2), rel=0.15)
