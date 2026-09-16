"""Keypoints, descriptors and robust two view alignment.

Details that matter on aerial footage:

* CLAHE (adaptive histogram equalization) before detection, so low contrast
  terrain such as water or snow still produces keypoints.
* Grid bucketing: detectors love a single high contrast patch, but a
  transform estimated from points clumped in one corner is poorly
  conditioned. Keeping the strongest responses per grid cell spreads
  keypoints across the whole frame.
* Symmetric matching: a match is kept only if it passes Lowe's ratio test in
  both directions and each point picks the other as its best match.
* RANSAC for a 4 degree of freedom similarity (rotation, uniform scale,
  translation), which is the right motion model for a nadir camera at
  roughly constant altitude, followed by Levenberg Marquardt refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    points: np.ndarray
    """(N, 2) float32 keypoint locations in pixels."""
    descriptors: np.ndarray | None
    size: tuple[int, int]
    """(width, height) of the image the features came from."""

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Alignment:
    matrix: np.ndarray
    """3x3 similarity mapping source pixels onto destination pixels."""
    src_points: np.ndarray
    dst_points: np.ndarray
    """Inlier correspondences, (M, 2) each."""
    matches: int

    @property
    def inliers(self) -> int:
        return len(self.src_points)


class FeatureExtractor:
    def __init__(self, detector: str = "orb", n_features: int = 3000, grid: tuple[int, int] = (6, 4), clahe: bool = True):
        self.kind = detector
        self.n_features = n_features
        self.grid = grid
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) if clahe else None
        # Over detect, then bucket down to n_features.
        if detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=n_features * 3, scaleFactor=1.2, nlevels=8, fastThreshold=7)
            self.norm = cv2.NORM_HAMMING
        elif detector == "sift":
            self.detector = cv2.SIFT_create(nfeatures=n_features * 2, contrastThreshold=0.02)
            self.norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown detector {detector!r}; use orb or sift")

    def __call__(self, image: np.ndarray) -> Features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        h, w = gray.shape
        keypoints = self.detector.detect(gray, None)
        keypoints = bucket_keypoints(keypoints, w, h, self.grid, self.n_features)
        keypoints, descriptors = self.detector.compute(gray, keypoints)
        points = np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)
        return Features(points=points, descriptors=descriptors, size=(w, h))


def bucket_keypoints(keypoints, width: int, height: int, grid: tuple[int, int], limit: int) -> list:
    """Keep the strongest keypoints in each grid cell, up to ``limit`` in total."""
    if not keypoints:
        return []
    cols, rows = grid
    per_cell = max(1, limit // (cols * rows))
    cells: dict[tuple[int, int], list] = {}
    for kp in keypoints:
        cx = min(cols - 1, int(kp.pt[0] * cols / width))
        cy = min(rows - 1, int(kp.pt[1] * rows / height))
        cells.setdefault((cx, cy), []).append(kp)
    kept = []
    for bucket in cells.values():
        bucket.sort(key=lambda kp: kp.response, reverse=True)
        kept.extend(bucket[:per_cell])
    return kept


class Matcher:
    def __init__(self, norm: int, ratio: float = 0.8):
        self.ratio = ratio
        self.matcher = cv2.BFMatcher(norm, crossCheck=False)

    def _ratio_matches(self, a: np.ndarray, b: np.ndarray) -> dict[int, int]:
        pairs = self.matcher.knnMatch(a, b, k=2)
        good = {}
        for pair in pairs:
            if (len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance) or len(pair) == 1:
                good[pair[0].queryIdx] = pair[0].trainIdx
        return good

    def __call__(self, src: Features, dst: Features) -> np.ndarray:
        """Symmetric ratio test matches as an (M, 2) array of index pairs."""
        if src.descriptors is None or dst.descriptors is None or len(src) < 2 or len(dst) < 2:
            return np.empty((0, 2), np.int32)
        forward = self._ratio_matches(src.descriptors, dst.descriptors)
        backward = self._ratio_matches(dst.descriptors, src.descriptors)
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
        return np.array(pairs, np.int32).reshape(-1, 2)


def estimate_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 5000,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC + LM refined similarity. Returns (3x3 matrix or None, inlier mask)."""
    if len(src) < 3:
        return None, np.zeros(len(src), bool)
    affine, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=max_iters, confidence=confidence, refineIters=20
    )
    if affine is None:
        return None, np.zeros(len(src), bool)
    return np.vstack([affine, [0, 0, 1]]), mask.ravel().astype(bool)


def similarity_scale(matrix: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))


def align(
    src: Features,
    dst: Features,
    matcher: Matcher,
    threshold: float = 3.0,
    min_inliers: int = 25,
    min_inlier_ratio: float = 0.2,
    max_scale_change: float = 0.3,
) -> tuple[Alignment | None, str]:
    """Align two feature sets. Returns (alignment, "") or (None, reason)."""
    pairs = matcher(src, dst)
    if len(pairs) < min_inliers:
        return None, f"{len(pairs)} symmetric matches"
    src_pts = src.points[pairs[:, 0]]
    dst_pts = dst.points[pairs[:, 1]]
    matrix, inliers = estimate_similarity(src_pts, dst_pts, threshold)
    if matrix is None:
        return None, "RANSAC failed"
    count = int(inliers.sum())
    if count < min_inliers or count < min_inlier_ratio * len(pairs):
        return None, f"{count}/{len(pairs)} inliers"
    scale = similarity_scale(matrix)
    if abs(scale - 1) > max_scale_change:
        return None, f"implausible scale {scale:.2f}"
    return Alignment(matrix, src_pts[inliers], dst_pts[inliers], len(pairs)), ""
