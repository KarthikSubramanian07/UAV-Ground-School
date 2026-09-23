"""Scoring detections against ground truth.

Dots
----
A detection matches a truth dot when its center is within half the truth
radius and its radius is within a factor of 1.5 either way; matching is
greedy by distance, one to one. Truth dots marked ``difficult`` (mostly out
of frame, say) and detections inside ``ignore`` polygons (defocused
background) are neither rewarded nor penalized, the PASCAL VOC convention.

Objects
-------
Instance masks and boxes are matched by intersection over union.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .blobs import Blob

ANNOTATIONS = Path(__file__).parent / "annotations"


@dataclass
class TruthSet:
    image: str
    size: tuple[int, int]
    dots: list[Blob]
    difficult: list[bool]
    ignore: list[np.ndarray] = field(default_factory=list)
    notes: str = ""

    @property
    def required(self) -> int:
        return sum(not d for d in self.difficult)

    def ignored(self, x: float, y: float) -> bool:
        return any(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for poly in self.ignore)


def load_truth(path: str | Path) -> TruthSet:
    data = json.loads(Path(path).read_text())
    dots = [Blob(d["x"], d["y"], d["r"], color=d.get("color")) for d in data["dots"]]
    difficult = [bool(d.get("difficult", False)) for d in data["dots"]]
    ignore = [np.array(poly, np.float32) for poly in data.get("ignore", [])]
    return TruthSet(data["image"], tuple(data["size"]), dots, difficult, ignore, data.get("notes", ""))


def truth_for(image_name: str) -> TruthSet:
    return load_truth(ANNOTATIONS / f"{Path(image_name).stem}.json")


@dataclass
class DotScore:
    tp: int
    fp: int
    fn: int
    center_error: float  # median, pixels
    radius_error: float  # median relative error
    matches: list[tuple[int, int]] = field(default_factory=list)  # (detection, truth)
    false_positives: list[int] = field(default_factory=list)
    misses: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def summary(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "center_error_px": None if math.isnan(self.center_error) else round(self.center_error, 3),
            "radius_error": None if math.isnan(self.radius_error) else round(self.radius_error, 4),
        }


def match_dots(
    detections: list[Blob],
    truth: list[Blob],
    difficult: list[bool] | None = None,
    ignored: list[bool] | None = None,
    center_tolerance: float = 0.5,
    radius_ratio: float = 1.5,
) -> DotScore:
    """One to one greedy matching of detections to truth dots."""
    difficult = difficult or [False] * len(truth)
    ignored = ignored or [False] * len(detections)
    pairs: list[tuple[float, int, int]] = []
    for i, d in enumerate(detections):
        for j, t in enumerate(truth):
            dist = math.hypot(d.x - t.x, d.y - t.y)
            if dist > center_tolerance * t.radius:
                continue
            ratio = d.radius / max(t.radius, 1e-9)
            if ratio > radius_ratio or ratio < 1 / radius_ratio:
                continue
            pairs.append((dist / t.radius, i, j))
    pairs.sort()
    used_d: set[int] = set()
    used_t: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        matches.append((i, j))
    tp_pairs = [(i, j) for i, j in matches if not difficult[j]]
    false_positives = [i for i in range(len(detections)) if i not in used_d and not ignored[i]]
    misses = [j for j in range(len(truth)) if j not in used_t and not difficult[j]]
    center = [math.hypot(detections[i].x - truth[j].x, detections[i].y - truth[j].y) for i, j in tp_pairs]
    radius = [abs(detections[i].radius - truth[j].radius) / truth[j].radius for i, j in tp_pairs]
    return DotScore(
        tp=len(tp_pairs),
        fp=len(false_positives),
        fn=len(misses),
        center_error=float(np.median(center)) if center else float("nan"),
        radius_error=float(np.median(radius)) if radius else float("nan"),
        matches=matches,
        false_positives=false_positives,
        misses=misses,
    )


def score_against(detections: list[Blob], truth: TruthSet) -> DotScore:
    ignored = [truth.ignored(d.x, d.y) for d in detections]
    return match_dots(detections, truth.dots, truth.difficult, ignored)


# ---------------------------------------------------------------- objects ----


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
