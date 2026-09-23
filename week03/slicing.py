"""Sliced inference (the SAHI idea) for small objects in large frames.

A detector trained at 640 px sees a 1920x1080 frame shrunk three times, which
turns a 12 px ball into 4 px. Running it on overlapping tiles at their native
resolution, plus once on the whole frame for the large objects, then merging
the results recovers the small objects.

Boxes touching an inner tile border are dropped first (the neighboring tile
sees that object whole), then the rest are merged with class aware non
maximum suppression. Overlap can be measured as IoU (the default) or as
intersection over the smaller box (IoS, as in SAHI), which also merges half
boxes into whole ones but suppresses players standing in front of each other.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .detmetrics import Detections
from .football import tile_grid

Predictor = Callable[[np.ndarray], Detections]


def overlap_matrix(a: np.ndarray, b: np.ndarray, metric: str = "ios") -> np.ndarray:
    """Pairwise overlap of x1 y1 x2 y2 boxes, as IoU or intersection over the smaller area."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if metric == "iou":
        return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)
    if metric == "ios":
        return inter / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-12)
    raise ValueError("metric must be iou or ios")


def nms(det: Detections, threshold: float = 0.5, metric: str = "iou") -> Detections:
    """Class aware greedy non maximum suppression."""
    if not len(det):
        return det
    keep: list[int] = []
    for c in np.unique(det.classes):
        idx = np.nonzero(det.classes == c)[0]
        idx = idx[np.argsort(-det.scores[idx], kind="stable")]
        overlaps = overlap_matrix(det.boxes[idx], det.boxes[idx], metric)
        suppressed = np.zeros(len(idx), bool)
        for i in range(len(idx)):
            if suppressed[i]:
                continue
            keep.append(int(idx[i]))
            suppressed |= overlaps[i] > threshold
    keep_arr = np.array(sorted(keep), int)
    return Detections(det.boxes[keep_arr], det.scores[keep_arr], det.classes[keep_arr])


def sliced_predict(
    predict: Predictor,
    image: np.ndarray,
    tile: tuple[int, int] = (960, 540),
    overlap: float = 0.2,
    full_frame: bool = True,
    merge_threshold: float = 0.5,
    metric: str = "iou",
    edge_margin: int = 2,
) -> Detections:
    """Run ``predict`` on overlapping tiles (and the full frame), then merge.

    Boxes that touch an inner tile border are dropped when a neighboring tile
    covers that region, because they are usually truncated objects the
    neighbor sees whole.
    """
    height, width = image.shape[:2]
    parts: list[Detections] = []
    for x0, y0 in tile_grid(width, height, tile, overlap):
        crop = image[y0 : y0 + tile[1], x0 : x0 + tile[0]]
        det = predict(crop)
        if not len(det):
            continue
        boxes = det.boxes + np.array([x0, y0, x0, y0], np.float64)
        th, tw = crop.shape[:2]
        # Inner borders only: the image border is a real border.
        cut = np.zeros(len(boxes), bool)
        if x0 > 0:
            cut |= det.boxes[:, 0] <= edge_margin
        if y0 > 0:
            cut |= det.boxes[:, 1] <= edge_margin
        if x0 + tw < width:
            cut |= det.boxes[:, 2] >= tw - edge_margin
        if y0 + th < height:
            cut |= det.boxes[:, 3] >= th - edge_margin
        parts.append(Detections(boxes[~cut], det.scores[~cut], det.classes[~cut]))
    if full_frame:
        parts.append(predict(image))
    if not parts:
        return Detections.empty()
    merged = Detections(
        np.concatenate([p.boxes for p in parts]),
        np.concatenate([p.scores for p in parts]),
        np.concatenate([p.classes for p in parts]).astype(int),
    )
    return nms(merged, merge_threshold, metric)
