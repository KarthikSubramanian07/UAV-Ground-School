"""Challenge 2: detect and classify cones, cubes and rings with classical vision.

No training data, so the pipeline has to *understand* the scene:

1. **Foreground.** Game pieces are saturated plastic; floor tiles are gray and
   cardboard is dull brown. Seeds are pixels with high saturation and
   brightness; each color then grows by hysteresis into weaker saturation of
   the same hue (the shaded side of a cube) as long as it stays connected.
2. **Colors.** Foreground hues are clustered (1D k-means on the hue circle),
   so pieces of different colors that touch still separate. Nothing assumes
   which color a class has.
3. **Instances.** Each color's mask is cleaned (a wide closing bridges
   printed logos), then touching pieces of the same color (the two stacked
   cubes) are cut between the notches where they meet (convexity defects),
   when the distance transform shows more than one thick center.
4. **Shape.** Every instance gets color free descriptors: how much of it is
   a hole, solidity (area over convex hull), how well it fills its minimum
   area rectangle, and how close its convex hull is to a triangle.
5. **Classify.** Geometric facts, not colors: a ring has a central hole, a
   cube is a convex polyhedron (convex silhouette), a cone is concave where
   the body meets the base and its hull is nearly a triangle.
6. **Contours.** Optionally, GrabCut refines each instance boundary from its
   mask, which helps where shadows desaturate the edge.

``python -m week03 objects docs/week03/photos/objects.jpg`` prints the detections and
writes an annotated image; ``--truth`` scores masks and boxes against the SAM
annotations.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .evaluate import ANNOTATIONS, box_iou, mask_iou

CLASSES = ("cone", "cube", "ring")
CLASS_BGR = {"cone": (0, 200, 255), "cube": (230, 90, 120), "ring": (60, 60, 240), "unknown": (180, 180, 180)}


@dataclass
class Features:
    area: float
    hole_ratio: float  # largest hole area / filled area
    hole_centered: float  # distance of the hole center from the blob center, / equivalent radius
    rect_fill: float  # area / minimum area rectangle
    triangle_fill: float  # area / minimum enclosing triangle
    triangularity: float  # convex hull area / minimum enclosing triangle (1 for a triangle, 0.5 to 0.7 for boxes)
    solidity: float  # area / convex hull area
    elongation: float  # minor / major side of the minimum area rectangle

    def as_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


@dataclass
class Piece:
    label: str
    confidence: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    center: tuple[float, float]
    hue: float
    features: Features
    scores: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": [int(v) for v in self.bbox],
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "area": int(self.features.area),
            "hue": round(self.hue, 1),
            "features": self.features.as_dict(),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


@dataclass
class ObjectSettings:
    min_saturation: int = 110  # seeds: clearly colored plastic
    grow_saturation: int = 70  # hysteresis: shaded sides of a piece, if connected to a seed of the same hue
    hue_tolerance: float = 8.0  # OpenCV hue units (half degrees)
    min_value: int = 120
    min_area_fraction: float = 0.004  # of the image
    hue_clusters: int | None = None  # chosen automatically when None
    grabcut: bool = False
    split_touching: bool = True


def foreground_mask(image: np.ndarray, s: ObjectSettings) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] >= s.min_saturation) & (hsv[..., 2] >= s.min_value)).astype(np.uint8) * 255
    return mask


def hue_clusters(hues: np.ndarray, k: int | None = None, max_k: int = 6, merge_degrees: float = 20.0) -> np.ndarray:
    """Cluster centers on the hue circle (OpenCV hue, 0..180)."""
    angles = hues.astype(np.float64) * 2 * math.pi / 180
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    if len(points) > 20000:
        points = points[np.random.default_rng(0).choice(len(points), 20000, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    cv2.setRNGSeed(0)
    best = None
    for n in [k] if k else range(1, max_k + 1):
        if len(points) < n:
            break
        compactness, _, centers = cv2.kmeans(points, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        # Elbow: stop adding clusters once the spread barely improves.
        if best is None or compactness < 0.6 * best[0]:
            best = (compactness, centers)
        else:
            break
    centers = best[1]
    degrees = (np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360) / 2  # back to 0..180
    merged: list[float] = []
    for h in sorted(degrees):
        if all(min(abs(h - m), 180 - abs(h - m)) * 2 > merge_degrees for m in merged):
            merged.append(float(h))
    return np.array(merged)


def _hue_distance(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a.astype(np.float64) - b)
    return np.minimum(d, 180 - d)


def _distance_peaks(filled: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Well separated maxima of the distance transform: roughly one per convex piece."""
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return [], dist
    window = max(3, int(0.9 * peak) | 1)
    local_max = (dist >= cv2.dilate(dist, np.ones((window, window), np.uint8))) & (dist >= 0.55 * peak)
    _, _, _, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8), 8)
    seeds = [tuple(int(v) for v in np.round(c)) for c in centroids[1:]]
    merged: list[tuple[int, int]] = []
    for sx, sy in sorted(seeds, key=lambda c: -dist[c[1], c[0]]):
        if all(math.hypot(sx - mx, sy - my) > 1.2 * peak for mx, my in merged):
            merged.append((sx, sy))
    return merged, dist


def _side(p: np.ndarray, q: np.ndarray, point: tuple[int, int]) -> float:
    return float((q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0]))


def _defect_cut(filled: np.ndarray, peaks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    """The shortest segment between two convexity defects that separates two peaks, inside the mask."""
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, np.sort(hull, axis=0))
    except cv2.error:
        return None
    if defects is None:
        return None
    scale = math.sqrt(float(filled.sum()))
    points = [(contour[f][0].astype(np.float64), d / 256.0) for _, _, f, d in defects.reshape(-1, 4) if d / 256.0 >= 0.04 * scale]
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p, dp = points[i]
            q, dq = points[j]
            length = float(np.linalg.norm(q - p))
            if length < 1:
                continue
            if _side(p, q, peaks[0]) * _side(p, q, peaks[1]) >= 0:
                continue  # both peaks on the same side
            steps = np.linspace(0, 1, max(2, int(length)))
            line = p[None, :] + steps[:, None] * (q - p)[None, :]
            inside = filled[
                np.clip(line[:, 1].astype(int), 0, filled.shape[0] - 1), np.clip(line[:, 0].astype(int), 0, filled.shape[1] - 1)
            ].mean()
            if inside < 0.85:
                continue
            score = length / (dp + dq)
            if best is None or score < best[0]:
                best = (score, p, q)
    return None if best is None else (best[1], best[2])


def _split_touching(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Split touching pieces of one color.

    Each piece here is convex or nearly so, so two touching pieces form a
    blob with two thick centers (distance transform peaks) and a notch on
    each side where they meet (convexity defects). The blob is cut along the
    shortest defect to defect segment that separates the peaks. A single cone
    is concave too, but has one thick center, so it is left alone.
    """
    count, labels = cv2.connectedComponents(mask, 8)
    out: list[np.ndarray] = []
    queue = [(labels == i).astype(np.uint8) for i in range(1, count)]
    while queue:
        part = queue.pop()
        if part.sum() < min_area:
            continue
        filled = fill_holes(part)
        peaks, _ = _distance_peaks(filled)
        cut = _defect_cut(filled, peaks[:2]) if len(peaks) >= 2 else None
        if cut is None:
            out.append(part)
            continue
        p, q = cut
        severed = filled.copy()
        cv2.line(severed, tuple(int(v) for v in np.round(p)), tuple(int(v) for v in np.round(q)), 0, 3)
        n, pieces_labels = cv2.connectedComponents(severed, 8)
        pieces = [(pieces_labels == j).astype(np.uint8) for j in range(1, n)]
        pieces = [pc for pc in pieces if pc.sum() >= min_area]
        if len(pieces) < 2:
            out.append(part)
            continue
        # Give the cut line back to the nearest piece, then restore the part's own holes.
        owner = np.zeros(mask.shape, np.int32)
        for j, pc in enumerate(pieces, start=1):
            owner[pc > 0] = j
        _, nearest = cv2.distanceTransformWithLabels((owner == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        label_of_pixel = np.zeros(nearest.max() + 1, np.int32)
        label_of_pixel[nearest[owner > 0]] = owner[owner > 0]
        full = label_of_pixel[nearest]
        for j in range(1, len(pieces) + 1):
            queue.append(((full == j) & (part > 0)).astype(np.uint8))
        if len(out) + len(queue) > 32:
            break
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


def describe(mask: np.ndarray) -> tuple[Features, np.ndarray]:
    """Shape descriptors of one instance mask, and its outer contour."""
    filled = fill_holes(mask)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(filled.sum())
    holes = cv2.subtract(filled, (mask > 0).astype(np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(holes, 8)
    hole_ratio, hole_centered = 0.0, 1.0
    m = cv2.moments(filled, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    radius = math.sqrt(area / math.pi)
    if n > 1:
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hole_ratio = float(stats[j, cv2.CC_STAT_AREA]) / area
        hole_centered = math.hypot(centroids[j][0] - cx, centroids[j][1] - cy) / radius
    rect = cv2.minAreaRect(contour)
    rect_area = max(rect[1][0] * rect[1][1], 1e-6)
    tri_area, _ = cv2.minEnclosingTriangle(contour.astype(np.float32))
    hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1e-6)
    side = sorted(rect[1])
    # Fill ratios compare polygon areas with polygon areas (all measured through pixel centers).
    polygon_area = float(cv2.contourArea(contour))
    return (
        Features(
            area=area,
            hole_ratio=hole_ratio,
            hole_centered=hole_centered,
            rect_fill=polygon_area / rect_area,
            triangle_fill=polygon_area / max(tri_area, 1e-6),
            triangularity=hull_area / max(tri_area, 1e-6),
            solidity=polygon_area / hull_area,
            elongation=side[0] / max(side[1], 1e-6),
        ),
        contour,
    )


def classify(f: Features) -> tuple[str, float, dict[str, float]]:
    """Scores in [0, 1] for each class from shape alone; the best one wins.

    The rules are geometric facts about the pieces, not colors:

    * ring: a large hole near the middle.
    * cube: a convex polyhedron, so its silhouette is convex (solidity near
      1) and a hexagon or square that fills most of its minimum rectangle.
    * cone: concave where the body meets the square base flange, and its
      convex hull is close to a triangle, standing or lying down.
    """

    def ramp(x: float, lo: float, hi: float) -> float:
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    no_hole = 1.0 - ramp(f.hole_ratio, 0.03, 0.12)
    convex = ramp(f.solidity, 0.90, 0.94)
    ring = ramp(f.hole_ratio, 0.05, 0.15) * (1.0 - ramp(f.hole_centered, 0.3, 0.6))
    cube = no_hole * convex * ramp(f.rect_fill, 0.68, 0.74) * (1.0 - ramp(f.triangularity, 0.72, 0.80))
    cone = no_hole * ramp(f.solidity, 0.72, 0.80) * max(1.0 - convex, ramp(f.triangularity, 0.66, 0.72))
    scores = {"cone": cone, "cube": cube, "ring": ring}
    label = max(scores, key=scores.get)
    if scores[label] < 0.3:
        return "unknown", scores[label], scores
    return label, scores[label], scores


def _grabcut(image: np.ndarray, mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Refine an instance mask: certain inside (eroded), certain outside (far away), probable in between."""
    x, y, w, h = cv2.boundingRect(mask)
    pad = max(12, int(0.15 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    crop = image[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    k = max(3, int(0.03 * max(w, h)) | 1)
    sure_fg = cv2.erode(m, np.ones((k, k), np.uint8))
    maybe = cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return mask
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    refined = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # Keep the component that overlaps the original mask most.
    n, labels = cv2.connectedComponents(refined, 8)
    if n > 2:
        best = max(range(1, n), key=lambda j: int(((labels == j) & (m > 0)).sum()))
        refined = (labels == best).astype(np.uint8)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = refined
    return out


@dataclass
class ObjectReport:
    pieces: list[Piece]
    seconds: float
    hues: list[float]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.pieces:
            counts[p.label] = counts.get(p.label, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "counts": self.counts(),
            "seconds": round(self.seconds, 3),
            "hue_clusters": [round(h, 1) for h in self.hues],
            "pieces": [p.summary() for p in self.pieces],
        }


def detect_objects(image: np.ndarray, settings: ObjectSettings | None = None) -> ObjectReport:
    s = settings or ObjectSettings()
    start = time.perf_counter()
    h, w = image.shape[:2]
    min_area = int(s.min_area_fraction * h * w)
    fg = foreground_mask(image, s)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hues = hsv[..., 0][fg > 0]
    centers = hue_clusters(hues, s.hue_clusters) if len(hues) else np.zeros(0)
    pieces: list[Piece] = []
    if len(centers):
        distance = np.stack([_hue_distance(hsv[..., 0], c) for c in centers])
        nearest = np.argmin(distance, axis=0)
        blurred = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
        weak = (blurred[..., 1] >= s.grow_saturation) & (blurred[..., 2] >= s.min_value)
        for i, center in enumerate(centers):
            seeds = (fg > 0) & (nearest == i)
            grow = (weak & (nearest == i) & (distance[i] <= s.hue_tolerance)) | seeds
            n_grow, grow_labels = cv2.connectedComponents(grow.astype(np.uint8), 8)
            seeded = np.unique(grow_labels[seeds])
            layer = np.isin(grow_labels, seeded[seeded > 0]).astype(np.uint8)
            # A wide closing bridges printed logos and glare that notch the outline.
            close = max(7, int(0.02 * math.hypot(h, w)) | 1)
            layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
            layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            parts = _split_touching(layer, min_area) if s.split_touching else [p for p in [layer] if p.sum() >= min_area]
            for part in parts:
                # Close small holes (printed logos, glare) but keep a ring's real hole.
                part = _close_small_holes(part, 0.02)
                if s.grabcut:
                    part = _grabcut(image, part)
                    part = _close_small_holes(part, 0.02)
                if part.sum() < min_area:
                    continue
                features, contour = describe(part)
                label, confidence, scores = classify(features)
                x, y, bw, bh = cv2.boundingRect(part)
                m = cv2.moments(part, binaryImage=True)
                pieces.append(
                    Piece(
                        label,
                        confidence,
                        part,
                        (x, y, bw, bh),
                        contour,
                        (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                        float(center),
                        features,
                        scores,
                    )
                )
    pieces.sort(key=lambda p: (p.label, p.center[1]))
    return ObjectReport(pieces, time.perf_counter() - start, [float(c) for c in centers])


def _close_small_holes(mask: np.ndarray, fraction: float) -> np.ndarray:
    """Fill holes smaller than ``fraction`` of the filled area."""
    filled = fill_holes(mask)
    holes = cv2.subtract(filled, (mask > 0).astype(np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = (mask > 0).astype(np.uint8)
    limit = fraction * filled.sum()
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < limit:
            out[labels == j] = 1
    return out


def annotate(image: np.ndarray, pieces: list[Piece]) -> np.ndarray:
    """Tinted masks, outlines, boxes and labels, sized relative to a 1280 px wide frame."""
    scale = max(image.shape[:2]) / 1280
    thick = max(1, round(2 * scale))
    out = image.copy()
    overlay = image.copy()
    for p in pieces:
        overlay[p.mask > 0] = CLASS_BGR.get(p.label, CLASS_BGR["unknown"])
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for p in pieces:
        color = CLASS_BGR.get(p.label, CLASS_BGR["unknown"])
        contours, _ = cv2.findContours(p.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), thick * 2, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, color, thick, cv2.LINE_AA)
        x, y, w, h = p.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(1, thick // 2), cv2.LINE_AA)
        text = f"{p.label} {p.confidence:.2f}"
        font = 0.6 * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font, 1)
        pad = round(4 * scale)
        ty = y - 2 * pad if y - th - 3 * pad > 0 else y + h + th + 2 * pad
        cv2.rectangle(out, (x, ty - th - pad), (x + tw + 2 * pad, ty + pad), color, -1)
        cv2.putText(out, text, (x + pad, ty), cv2.FONT_HERSHEY_DUPLEX, font, (20, 20, 20), max(1, round(scale)), cv2.LINE_AA)
    return out


# ------------------------------------------------------------- evaluation ----


@dataclass
class TruthObject:
    label: str
    name: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


def load_objects(path: str | Path | None = None) -> list[TruthObject]:
    data = json.loads(Path(path or ANNOTATIONS / "objects.json").read_text())
    w, h = data["size"]
    out = []
    for o in data["objects"]:
        mask = np.zeros((h, w), np.uint8)
        for poly in o["polygons"]:
            cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)
        for poly in o["holes"]:
            pts = np.array(poly, np.int32)
            cv2.fillPoly(mask, [pts], 0)
            cv2.polylines(mask, [pts], True, 1, 1)  # the hole outline belongs to the object
        out.append(TruthObject(o["label"], o["name"], mask, tuple(o["bbox"])))
    return out


def score_objects(pieces: list[Piece], truth: list[TruthObject]) -> dict:
    """Match detections to truth by mask IoU (greedy, IoU >= 0.5)."""
    pairs = sorted(((mask_iou(p.mask, t.mask), i, j) for i, p in enumerate(pieces) for j, t in enumerate(truth)), reverse=True)
    used_p, used_t = set(), set()
    matches = []
    for iou, i, j in pairs:
        if iou < 0.5 or i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matches.append(
            {
                "truth": truth[j].name,
                "label": pieces[i].label,
                "correct": pieces[i].label == truth[j].label,
                "mask_iou": round(iou, 4),
                "box_iou": round(box_iou(pieces[i].bbox, truth[j].bbox), 4),
            }
        )
    correct = sum(m["correct"] for m in matches)
    return {
        "truth_objects": len(truth),
        "detections": len(pieces),
        "matched": len(matches),
        "correct_class": correct,
        "false_positives": len(pieces) - len(matches),
        "missed": len(truth) - len(matches),
        "mean_mask_iou": round(float(np.mean([m["mask_iou"] for m in matches])), 4) if matches else 0.0,
        "mean_box_iou": round(float(np.mean([m["box_iou"] for m in matches])), 4) if matches else 0.0,
        "matches": matches,
    }
