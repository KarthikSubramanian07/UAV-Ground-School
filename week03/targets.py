"""Bonus: the ``shapes.png`` image from the week 3 folder, read as SUAS style targets.

Painted shapes on a textured gray ground. Some are faint (a cream cursor, a
dusty purple dot pattern), so pixels are segmented by their Delta E from the
local background (:func:`week03.dots.contrast_map`) rather than by saturation.
Each region is then classified with the week 2 template matcher, extended
with three generic families it lacks:

* ellipses at the blob's own aspect ratio (as week 2 does for rectangles);
* stars with 4 to 8 points, which covers asterisks;
* outlines: a region with a large hole is reported as the outline of the
  shape it encloses.

Small round blobs of one color close together are reported as a group (the
triangle of ten dots).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

from week02 import shapes as suas

from . import dots


@dataclass
class Target:
    shape: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    color: str
    area: int
    outline: bool = False
    members: int = 1  # more than one for a group of dots
    lab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    circularity: float = 0.0

    def summary(self) -> dict:
        return {
            "shape": self.shape,
            "confidence": round(self.confidence, 3),
            "bbox": [int(v) for v in self.bbox],
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "color": self.color,
            "area": self.area,
            "outline": self.outline,
            "members": self.members,
        }


def _star(points: int, inner: float) -> np.ndarray:
    angles = np.arange(points * 2) * math.pi / points
    radii = np.where(np.arange(points * 2) % 2 == 0, 1.0, inner)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


@cache
def _star_bank(points: int, inner: float) -> tuple[np.ndarray, np.ndarray]:
    poly = suas._normalize_polygon(_star(points, inner))
    period = 2 * math.pi / points
    angles = np.linspace(0, period, 24, endpoint=False)
    bank = np.stack([suas._rasterize(poly, a) for a in angles]).reshape(len(angles), -1)
    return bank, angles


def classify_region(mask: np.ndarray) -> tuple[str, float]:
    """Best shape name and IoU for one filled region."""
    base = suas.classify(mask, min_area=50)
    best = (base.shape, base.confidence) if base else ("irregular", 0.0)
    blob = suas.normalize_blob(mask)
    if blob is None:
        return best
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    candidates: list[tuple[float, str]] = [(best[1], best[0])]
    # Ellipse with the region's own aspect ratio.
    if len(contour) >= 5:
        (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
        aspect = min(d1, d2) / max(d1, d2, 1e-6)
        if aspect < 0.9:
            t = np.linspace(0, 2 * math.pi, 96, endpoint=False)
            poly = suas._normalize_polygon(np.stack([np.cos(t), aspect * np.sin(t)], axis=1))
            angles = np.arange(90) * math.pi / 90
            bank = np.stack([suas._rasterize(poly, a) for a in angles]).reshape(len(angles), -1)
            iou, _ = suas._best_iou(blob, bank, angles)
            candidates.append((iou, "ellipse"))
    # Stars and asterisks.
    for points in range(4, 9):
        for inner in (0.25, 0.35, 0.45):
            bank, angles = _star_bank(points, inner)
            iou, _ = suas._best_iou(blob, bank, angles)
            candidates.append((iou, f"{points} point star" if inner >= 0.4 else f"{points} arm asterisk"))
    iou, name = max(candidates)
    if iou < suas.MIN_CONFIDENCE:
        return "irregular", iou
    return name, iou


def find_targets(image: np.ndarray, min_contrast: float = 16.0, min_area: int = 300, group: bool = True) -> list[Target]:
    """Segment by Delta E from the local background, classify each region, and (with ``group``) merge dot clusters."""
    lab = dots.lab_image(cv2.medianBlur(image, 5))
    delta = dots.contrast_map(lab, max_radius=0.15 * max(image.shape[:2]), smooth=2.0)
    mask = (delta >= min_contrast).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    raw: list[Target] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region = (labels == i).astype(np.uint8)
        filled = np.zeros_like(region)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(filled, contours, -1, 1, -1)
        outline = bool(filled.sum() - region.sum() > 0.25 * filled.sum())
        shape, confidence = classify_region(filled * 255)
        median = np.median(lab[region > 0], axis=0)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * cv2.contourArea(contour) / max(perimeter**2, 1e-6)
        x, y, w, h = (int(v) for v in stats[i, :4])
        raw.append(
            Target(
                shape,
                float(confidence),
                (x, y, w, h),
                (float(centroids[i][0]), float(centroids[i][1])),
                dots.color_name(median),
                area,
                outline,
                1,
                tuple(float(v) for v in median),
                float(circularity),
            )
        )
    return _group_dots(raw) if group else raw


def is_dot(t: Target) -> bool:
    """A small, round, filled region: a dot candidate."""
    return t.circularity >= 0.75 and t.area < 3000 and not t.outline


def _group_dots(targets: list[Target]) -> list[Target]:
    """Merge small round regions of similar color (Delta E < 20) that sit close together into one group."""
    small = [t for t in targets if is_dot(t)]
    others = [t for t in targets if t not in small]
    groups: list[list[Target]] = []
    for t in small:
        size = math.sqrt(t.area)

        def near(g: list[Target], t: Target = t, size: float = size) -> bool:
            similar = np.linalg.norm(np.array(g[0].lab) - np.array(t.lab)) < 20
            return similar and min(math.hypot(t.center[0] - o.center[0], t.center[1] - o.center[1]) for o in g) < 2.5 * size

        home = next((g for g in groups if near(g)), None)
        if home is None:
            groups.append([t])
        else:
            home.append(t)
    for g in groups:
        if len(g) < 3:
            others.extend(g)
            continue
        x0 = min(t.bbox[0] for t in g)
        y0 = min(t.bbox[1] for t in g)
        x1 = max(t.bbox[0] + t.bbox[2] for t in g)
        y1 = max(t.bbox[1] + t.bbox[3] for t in g)
        mean_lab = np.mean([t.lab for t in g], axis=0)
        others.append(
            Target(
                f"group of {len(g)} dots",
                float(np.mean([t.circularity for t in g])),
                (x0, y0, x1 - x0, y1 - y0),
                (float(np.mean([t.center[0] for t in g])), float(np.mean([t.center[1] for t in g]))),
                dots.color_name(mean_lab),
                int(sum(t.area for t in g)),
                members=len(g),
                lab=tuple(float(v) for v in mean_lab),
            )
        )
    others.sort(key=lambda t: (t.center[1] // 200, t.center[0]))
    return others


def annotate(image: np.ndarray, targets: list[Target]) -> np.ndarray:
    """Boxes and labels; each label goes above, below or inside its box, wherever it overlaps nothing yet."""
    out = image.copy()
    scale = max(image.shape[:2]) / 1280
    font = 0.55 * scale
    pad = round(5 * scale)
    placed: list[tuple[int, int, int, int]] = []

    def free(r: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = r
        inside = x0 >= 0 and y0 >= 0 and x1 <= image.shape[1] and y1 <= image.shape[0]
        return inside and all(x1 <= a0 or a1 <= x0 or y1 <= b0 or b1 <= y0 for a0, b0, a1, b1 in placed)

    for t in targets:
        x, y, w, h = t.bbox
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), (255, 255, 255), max(2, round(3 * scale)), cv2.LINE_AA)
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), (30, 30, 30), 1, cv2.LINE_AA)
    for t in targets:
        x, y, w, h = t.bbox
        label = f"{t.color} {t.shape}{' outline' if t.outline else ''}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font, 1)
        lw, lh = tw + 2 * pad, th + 2 * pad
        options = [
            (x - pad, y - pad - lh),  # above
            (x - pad, y + h + pad),  # below
            (x + pad, y + pad),  # inside, top left
            (x + w + pad - lw, y + h + pad),  # below, right aligned
        ]
        lx, ly = next(((ox, oy) for ox, oy in options if free((ox, oy, ox + lw, oy + lh))), options[0])
        placed.append((lx, ly, lx + lw, ly + lh))
        cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), -1)
        cv2.putText(out, label, (lx + pad, ly + lh - pad), cv2.FONT_HERSHEY_DUPLEX, font, (30, 30, 30), 1, cv2.LINE_AA)
    return out
