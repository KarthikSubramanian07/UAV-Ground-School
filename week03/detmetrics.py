"""COCO style detection metrics: AP at IoU 0.50 and averaged over 0.50:0.95.

Ultralytics reports these during validation, but only for its own full frame
inference. Sliced inference (see :mod:`week03.slicing`) produces detections
outside that loop, so every model here is scored by this one implementation
for a like for like comparison. It follows COCO: predictions sorted by score,
greedy one to one matching per IoU threshold, and precision interpolated at
101 recall points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class Detections:
    """Boxes for one image: (n, 4) x1 y1 x2 y2 in pixels, scores and integer classes."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    @staticmethod
    def empty() -> Detections:
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))

    def __len__(self) -> int:
        return len(self.boxes)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) x1 y1 x2 y2 boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101 point interpolated AP from a precision/recall curve (sorted by score).

    At each recall level r in 0, 0.01, ..., 1 the interpolated precision is the
    best precision at any recall >= r, or 0 if r is never reached. A class
    with no true positives therefore scores exactly 0.
    """
    if not len(recall):
        return 0.0
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    reached = idx < len(recall)
    values = np.where(reached, envelope[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(np.mean(values))


@dataclass
class ClassAP:
    name: str
    instances: int
    ap50: float
    ap: float  # 0.50:0.95
    precision: float  # at the best F1 confidence, IoU 0.5
    recall: float


@dataclass
class DetectionReport:
    classes: list[ClassAP] = field(default_factory=list)

    @property
    def map50(self) -> float:
        valid = [c for c in self.classes if c.instances]
        return float(np.mean([c.ap50 for c in valid])) if valid else 0.0

    @property
    def map(self) -> float:
        valid = [c for c in self.classes if c.instances]
        return float(np.mean([c.ap for c in valid])) if valid else 0.0

    def summary(self) -> dict:
        return {
            "mAP50": round(self.map50, 4),
            "mAP50-95": round(self.map, 4),
            "classes": {
                c.name: {
                    "instances": c.instances,
                    "AP50": round(c.ap50, 4),
                    "AP50-95": round(c.ap, 4),
                    "P": round(c.precision, 4),
                    "R": round(c.recall, 4),
                }
                for c in self.classes
            },
        }


def evaluate(predictions: list[Detections], truths: list[Detections], names: tuple[str, ...]) -> DetectionReport:
    """Score predictions against truth, image by image (same order in both lists)."""
    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must describe the same images")
    report = DetectionReport()
    for c, name in enumerate(names):
        scores: list[np.ndarray] = []
        hits: list[np.ndarray] = []  # (n_pred, n_thresholds) true positive flags
        instances = 0
        for pred, truth in zip(predictions, truths):
            p_mask = pred.classes == c
            t_mask = truth.classes == c
            pb, ps = pred.boxes[p_mask], pred.scores[p_mask]
            tb = truth.boxes[t_mask]
            instances += len(tb)
            if not len(pb):
                continue
            order = np.argsort(-ps, kind="stable")
            pb, ps = pb[order], ps[order]
            iou = box_iou_matrix(pb, tb)
            tp = np.zeros((len(pb), len(IOU_THRESHOLDS)), bool)
            for k, threshold in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(len(tb), bool)
                for i in range(len(pb)):
                    if not len(tb):
                        break
                    candidates = np.where(~taken & (iou[i] >= threshold), iou[i], -1.0)
                    j = int(np.argmax(candidates))
                    if candidates[j] >= threshold:
                        taken[j] = True
                        tp[i, k] = True
            scores.append(ps)
            hits.append(tp)
        if not scores or instances == 0:
            report.classes.append(ClassAP(name, instances, 0.0, 0.0, 0.0, 0.0))
            continue
        all_scores = np.concatenate(scores)
        all_hits = np.concatenate(hits)
        order = np.argsort(-all_scores, kind="stable")
        all_hits = all_hits[order]
        cum_tp = np.cumsum(all_hits, axis=0)
        cum_fp = np.cumsum(~all_hits, axis=0)
        recall = cum_tp / instances
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        aps = [average_precision(recall[:, k], precision[:, k]) for k in range(len(IOU_THRESHOLDS))]
        f1 = 2 * precision[:, 0] * recall[:, 0] / np.maximum(precision[:, 0] + recall[:, 0], 1e-12)
        best = int(np.argmax(f1))
        report.classes.append(ClassAP(name, instances, aps[0], float(np.mean(aps)), float(precision[best, 0]), float(recall[best, 0])))
    return report
