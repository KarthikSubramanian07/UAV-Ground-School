import math

import cv2
import numpy as np
import pytest

from week02 import features, mosaic, synth


@pytest.fixture(scope="module")
def world():
    return synth.voxel_world(width_blocks=120, height_blocks=90, seed=11)


def view(world, x, y, angle, scale=1.0, size=(320, 240)):
    pose = np.array([x, y, angle, scale])
    matrix = synth.pose_matrix(pose, *size)
    image = cv2.warpAffine(world, matrix[:2], size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return image, matrix


@pytest.mark.parametrize("detector", ["orb", "sift"])
def test_recovers_known_similarity(world, detector):
    a, ma = view(world, 330, 260, 0.0)
    b, mb = view(world, 390, 290, math.radians(20), 1.06)
    extract = features.FeatureExtractor(detector, 1500)
    matcher = features.Matcher(extract.norm)
    alignment, reason = features.align(extract(b), extract(a), matcher)
    assert alignment is not None, reason
    truth = np.linalg.inv(ma) @ mb  # frame b pixels to frame a pixels
    grid = mosaic.frame_corners(320, 240)
    error = np.linalg.norm(mosaic.apply(alignment.matrix, grid) - mosaic.apply(truth, grid), axis=1)
    assert error.max() < 1.5
    assert alignment.inliers >= 30


def test_unrelated_images_do_not_align(world):
    a, _ = view(world, 200, 200, 0)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, a.shape, dtype=np.uint8)
    extract = features.FeatureExtractor("orb", 1000)
    alignment, reason = features.align(extract(a), extract(noise), features.Matcher(extract.norm))
    assert alignment is None and reason


def test_blank_images_are_handled():
    extract = features.FeatureExtractor("orb", 500)
    blank = extract(np.full((120, 160, 3), 128, np.uint8))
    assert len(blank) == 0
    assert features.Matcher(extract.norm)(blank, blank).shape == (0, 2)


def test_bucketing_spreads_keypoints():
    keypoints = [cv2.KeyPoint(float(x % 50), float(y % 50), 5, response=float(x)) for x in range(200) for y in range(0, 200, 10)]
    keypoints += [cv2.KeyPoint(300.0, 300.0, 5, response=1.0)]
    kept = features.bucket_keypoints(keypoints, 400, 400, (4, 4), 64)
    assert len(kept) <= 64
    assert any(kp.pt == (300.0, 300.0) for kp in kept)


def test_unknown_detector():
    with pytest.raises(ValueError):
        features.FeatureExtractor("surf")


def test_estimate_similarity_needs_points():
    matrix, mask = features.estimate_similarity(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    assert matrix is None and not mask.any()
