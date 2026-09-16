import math

import cv2
import numpy as np
import pytest

from week02 import mosaic, synth


def similarity(angle, scale, tx, ty):
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def test_param_round_trip():
    m = similarity(0.3, 1.1, 5, -7)
    assert np.allclose(mosaic.from_params(mosaic.to_params(m)), m)


def make_constraints(truth, pairs, rng, noise=0.0, points=40):
    constraints = []
    for i, j, loop in pairs:
        src = rng.uniform(0, 300, (points, 2))
        world = mosaic.apply(truth[i], src)
        dst = mosaic.apply(np.linalg.inv(truth[j]), world) + rng.normal(0, noise, (points, 2))
        constraints.append(mosaic.Constraint(i, j, src, dst, loop))
    return constraints


def test_bundle_adjustment_removes_drift_with_loop_closures():
    rng = np.random.default_rng(1)
    n = 12
    truth = [similarity(0.02 * k, 1.0, 120.0 * k, 15.0 * math.sin(k)) for k in range(n)]
    # Odometry that drifts: each step has a small rotation and scale error.
    drifted = [truth[0]]
    for k in range(1, n):
        step = np.linalg.inv(truth[k - 1]) @ truth[k]
        drifted.append(drifted[-1] @ similarity(0.004, 1.003, 0.8, -0.5) @ step)
    pairs = [(k, k - 1, False) for k in range(1, n)] + [(11, 0, True), (8, 2, True), (10, 4, True)]
    constraints = make_constraints(truth, pairs, rng, noise=0.3)

    before = max(np.abs(d - t)[:2, 2].max() for d, t in zip(drifted, truth))
    adjusted, report = mosaic.bundle_adjust(drifted, constraints)
    after = max(np.abs(a - t)[:2, 2].max() for a, t in zip(adjusted, truth))
    assert before > 10
    assert after < 1.0
    assert report.rms_after < report.rms_before
    assert report.loop_closures == 3
    assert np.allclose(adjusted[0], drifted[0])  # gauge frame is fixed


def test_bundle_adjustment_is_robust_to_an_outlier_constraint():
    rng = np.random.default_rng(2)
    truth = [similarity(0, 1, 100.0 * k, 0) for k in range(5)]
    constraints = make_constraints(truth, [(k, k - 1, False) for k in range(1, 5)] + [(4, 0, True)], rng)
    constraints[1].dst[:10] += 60  # a quarter of one constraint is garbage
    adjusted, _ = mosaic.bundle_adjust(truth, constraints)
    assert max(np.abs(a - t)[:2, 2].max() for a, t in zip(adjusted, truth)) < 0.5


def test_bundle_adjust_without_constraints_is_identity():
    transforms = [np.eye(3), similarity(0, 1, 10, 0)]
    adjusted, report = mosaic.bundle_adjust(transforms, [])
    assert np.allclose(adjusted[1], transforms[1]) and report.constraints == 0


@pytest.fixture(scope="module")
def tiles():
    world = synth.voxel_world(width_blocks=90, height_blocks=50, block=6, seed=4)
    transforms, frames = [], []
    for k, (x, y) in enumerate([(0, 0), (140, 10), (280, 0), (140, 120)]):
        m = similarity(0.05 * (k - 1), 1.0, x, y)
        frames.append(cv2.warpAffine(world, m[:2], (240, 180), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        transforms.append(m)
    return world, frames, transforms


def test_gain_compensation_recovers_exposure(tiles):
    _, frames, transforms = tiles
    true_gains = np.array([1.0, 0.8, 1.15, 0.9])
    exposed = [np.clip(f.astype(np.float32) * g, 0, 255).astype(np.uint8) for f, g in zip(frames, true_gains)]
    gains = mosaic.gain_compensation(exposed, transforms, sigma_g=1.0)
    corrected = gains * true_gains
    assert np.ptp(corrected) < 0.05


@pytest.mark.parametrize("blend", ["overwrite", "feather", "multiband"])
def test_blends_reconstruct_the_world(tiles, blend):
    world, frames, transforms = tiles
    image, canvas_transforms, coverage = mosaic.render(frames, transforms, blend)
    offset = canvas_transforms[0] @ np.linalg.inv(transforms[0])
    expected = cv2.warpAffine(world, offset[:2], (image.shape[1], image.shape[0]))
    inner = cv2.erode(coverage.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # The voxel texture is pure high frequency noise, so compare after a light
    # blur to measure alignment and blending rather than resampling noise.
    blur = lambda img: cv2.GaussianBlur(img, (5, 5), 1.2).astype(int)  # noqa: E731
    error = np.abs(blur(image) - blur(expected))[inner].mean()
    assert error < 4, error
    assert coverage.mean() > 0.5


def test_render_rejects_unknown_blend(tiles):
    _, frames, transforms = tiles
    with pytest.raises(ValueError):
        mosaic.render(frames, transforms, "poisson")


def test_canvas_limit():
    with pytest.raises(MemoryError):
        mosaic.plan_canvas([(100, 100)] * 2, [np.eye(3), similarity(0, 1, 1e6, 1e6)], max_pixels=10**6)


def test_footprint_overlap():
    assert mosaic.footprint_overlap(np.eye(3), np.eye(3), 100, 50) == pytest.approx(1.0)
    assert mosaic.footprint_overlap(np.eye(3), similarity(0, 1, 50, 0), 100, 50) == pytest.approx(0.5)
    assert mosaic.footprint_overlap(np.eye(3), similarity(0, 1, 500, 0), 100, 50) == 0
