"""Measure stitching quality against simulator ground truth.

A mosaic is only defined up to a similarity transform (where the canvas
origin is, and how it is rotated and scaled), so the estimated keyframe
footprints are first aligned to the true footprints with the closed form
Umeyama least squares solution. What remains is real error:

* ``rmse_px`` and ``max_px``: how far keyframe corners land from where they
  truly are, in world pixels.
* ``zncc``: zero normalized cross correlation between the mosaic and the true
  world rendered into the mosaic's frame, over covered pixels. 1.0 is perfect;
  it ignores global brightness, so exposure differences are not penalized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import mosaic
from .synth import FlightTruth


@dataclass
class Evaluation:
    keyframes: int
    rmse_px: float
    max_px: float
    zncc: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least squares similarity (3x3) mapping ``src`` points onto ``dst``."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign[1, 1] = -1
    R = U @ sign @ Vt
    scale = np.trace(np.diag(S) @ sign) / (s**2).sum(axis=1).mean()
    t = mu_d - scale * R @ mu_s
    matrix = np.eye(3)
    matrix[:2, :2] = scale * R
    matrix[:2, 2] = t
    return matrix


def evaluate(
    transforms: list[np.ndarray],
    frame_indices: list[int],
    truth: FlightTruth,
    panorama: np.ndarray | None = None,
    world: np.ndarray | None = None,
    frame_scale: float = 1.0,
) -> Evaluation:
    """Compare estimated keyframe transforms (frame to panorama) with the truth."""
    w, h = truth.plan.frame_width, truth.plan.frame_height
    grid = np.array([[x, y] for x in np.linspace(0, w, 5) for y in np.linspace(0, h, 5)])
    to_video = np.diag([frame_scale, frame_scale, 1.0])  # full size pixels to stitched frame pixels
    estimated, actual = [], []
    for matrix, index in zip(transforms, frame_indices):
        estimated.append(mosaic.apply(matrix @ to_video, grid))
        actual.append(mosaic.apply(truth.frame_to_world(index), grid))
    estimated = np.vstack(estimated)
    actual = np.vstack(actual)
    panorama_to_world = umeyama_similarity(estimated, actual)
    errors = np.linalg.norm(mosaic.apply(panorama_to_world, estimated) - actual, axis=1)

    zncc = None
    if panorama is not None and world is not None:
        world_in_panorama = cv2.warpAffine(
            world, panorama_to_world[:2], (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        covered = (panorama.max(axis=2) > 0) & (world_in_panorama.max(axis=2) > 0)
        covered = cv2.erode(covered.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if covered.sum() > 100:
            a = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)[covered].astype(np.float64)
            b = cv2.cvtColor(world_in_panorama, cv2.COLOR_BGR2GRAY)[covered].astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            zncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return Evaluation(len(frame_indices), float(np.sqrt(np.mean(errors**2))), float(errors.max()), zncc)


def load_transforms(path) -> tuple[list[np.ndarray], list[int]]:
    """Read a transforms CSV written by Python or the C++ port."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    transforms = [np.vstack([row[1:7].reshape(2, 3), [0, 0, 1]]) for row in data]
    return transforms, [int(row[0]) for row in data]
