"""Geometric shape classification for single color blobs.

This is the "what is it" step that follows "where is it" in a SUAS style
object detection pipeline: a red blob at (x, y) becomes a red octagon.

Hand tuned vertex counting breaks down on small, pixelated blobs, so shapes
are classified by template matching in a normalized space instead:

1. The blob is translated so its centroid is at the origin and scaled so its
   area equals a fixed reference area. This removes position and size.
2. Every canonical shape (circle, octagon, star, ...) is normalized the same
   way and rasterized at a sweep of rotations within its symmetry period.
   Rectangles are generated with the blob's own aspect ratio, since real
   rectangles come in many proportions.
3. The shape whose best rotation has the highest intersection over union
   (IoU) with the blob wins; the IoU doubles as a confidence score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

CANVAS = 128
REFERENCE_RADIUS = 44.0
REFERENCE_AREA = math.pi * REFERENCE_RADIUS**2
MIN_CONFIDENCE = 0.80


def _regular(sides: int, phase: float = 0.0) -> np.ndarray:
    angles = phase + np.arange(sides) * 2 * math.pi / sides
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _star(points: int = 5, inner: float = 0.45) -> np.ndarray:
    angles = np.arange(points * 2) * math.pi / points
    radii = np.where(np.arange(points * 2) % 2 == 0, 1.0, inner)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def _cross(arm: float = 0.36) -> np.ndarray:
    a = arm
    return np.array([[-a, -1], [a, -1], [a, -a], [1, -a], [1, a], [a, a], [a, 1], [-a, 1], [-a, a], [-1, a], [-1, -a], [-a, -a]], float)


def _semicircle() -> np.ndarray:
    angles = np.linspace(0, math.pi, 64)
    return np.stack([np.cos(angles), -np.sin(angles)], axis=1)


def _quarter_circle() -> np.ndarray:
    angles = np.linspace(0, math.pi / 2, 48)
    return np.vstack([[0.0, 0.0], np.stack([np.cos(angles), np.sin(angles)], axis=1)])


def _trapezoid() -> np.ndarray:
    return np.array([[-0.5, -0.6], [0.5, -0.6], [1.0, 0.6], [-1.0, 0.6]])


def _rectangle(aspect: float) -> np.ndarray:
    return np.array([[-aspect, -1], [aspect, -1], [aspect, 1], [-aspect, 1]], float)


# name -> (unit polygon, rotational symmetry period in radians)
TEMPLATES: dict[str, tuple[np.ndarray, float]] = {
    "circle": (_regular(96), 2 * math.pi / 96),
    "semicircle": (_semicircle(), 2 * math.pi),
    "quarter circle": (_quarter_circle(), 2 * math.pi),
    "triangle": (_regular(3), 2 * math.pi / 3),
    "square": (_regular(4), 2 * math.pi / 4),
    "trapezoid": (_trapezoid(), 2 * math.pi),
    "pentagon": (_regular(5), 2 * math.pi / 5),
    "hexagon": (_regular(6), 2 * math.pi / 6),
    "heptagon": (_regular(7), 2 * math.pi / 7),
    "octagon": (_regular(8), 2 * math.pi / 8),
    "star": (_star(), 2 * math.pi / 5),
    "cross": (_cross(), 2 * math.pi / 4),
}
SHAPE_NAMES = tuple(TEMPLATES) + ("rectangle",)


@dataclass
class ShapeMatch:
    shape: str
    confidence: float
    """IoU between the blob and the best fitting template, 0..1."""
    orientation_deg: float
    runner_up: str
    runner_up_confidence: float


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes such as the letter printed on a target."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _normalize_polygon(points: np.ndarray) -> np.ndarray:
    """Center a polygon on its centroid and scale it to the reference area."""
    moments = cv2.moments(points.astype(np.float32))
    area = abs(moments["m00"])
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return (points - [cx, cy]) * math.sqrt(REFERENCE_AREA / area)


def _rasterize(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rotated = points @ np.array([[c, s], [-s, c]]) + CANVAS / 2
    canvas = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.fillPoly(canvas, [np.round(rotated * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return canvas


@cache
def _template_bank(name: str, steps: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Rasterized, normalized templates over one symmetry period."""
    points, period = TEMPLATES[name]
    normalized = _normalize_polygon(points)
    angles = np.arange(steps) * period / steps
    return np.stack([_rasterize(normalized, a) for a in angles]).reshape(steps, -1), angles


def normalize_blob(mask: np.ndarray) -> np.ndarray | None:
    """Warp a binary blob into the canonical canvas (centroid centered, reference area)."""
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] < 1:
        return None
    area = moments["m00"]
    cx, cy = moments["m10"] / area, moments["m01"] / area
    scale = math.sqrt(REFERENCE_AREA / area)
    matrix = np.array([[scale, 0, CANVAS / 2 - scale * cx], [0, scale, CANVAS / 2 - scale * cy]])
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    warped = cv2.warpAffine(mask.astype(np.float32), matrix, (CANVAS, CANVAS), flags=interpolation)
    return (warped > 127).astype(np.uint8)


def _best_iou(blob: np.ndarray, bank: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    flat = blob.reshape(-1).astype(np.int32)
    intersection = bank @ flat
    union = bank.sum(axis=1) + flat.sum() - intersection
    ious = intersection / np.maximum(union, 1)
    best = int(np.argmax(ious))
    return float(ious[best]), float(angles[best])


def classify(mask: np.ndarray, min_area: int = 150) -> ShapeMatch | None:
    """Classify a single blob given as a 0/255 or 0/1 mask. Holes are ignored."""
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [contour], -1, 255, thickness=cv2.FILLED)
    blob = normalize_blob(solid)
    if blob is None:
        return None

    scores: list[tuple[float, str, float]] = []
    for name in TEMPLATES:
        bank, angles = _template_bank(name)
        iou, angle = _best_iou(blob, bank, angles)
        scores.append((iou, name, angle))

    # Rectangles: build the template from the blob's own aspect ratio.
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = min(rw, rh) / max(rw, rh, 1e-6)
    if aspect < 0.85:
        points = _normalize_polygon(_rectangle(aspect))
        angles = np.arange(90) * math.pi / 90
        bank = np.stack([_rasterize(points, a) for a in angles]).reshape(len(angles), -1)
        iou, angle = _best_iou(blob, bank, angles)
        scores.append((iou, "rectangle", angle))

    scores.sort(reverse=True)
    best_iou, best_name, best_angle = scores[0]
    runner_iou, runner_name, _ = scores[1]
    if best_iou < MIN_CONFIDENCE:
        best_name = "irregular"
    return ShapeMatch(best_name, round(best_iou, 4), round(math.degrees(best_angle), 1), runner_name, round(runner_iou, 4))
