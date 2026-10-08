"""Challenge 1 beyond the polka dots: LoG and contours on the real objects and the distorted shapes.

The extra challenge reads "use other methods such as LoG or contours, and
test them out on the real objects and the distorted shapes images". The dot
pipeline was built for flat colored disks; this module runs the same kinds of
detector on the other two photos from the week's Drive folder and scores
them against ground truth, to show what transfers and what does not.

``objects.jpg``
    Detectors that know nothing about cones, cubes or rings, scored like any
    object detector: a detection's box must reach IoU 0.5 with a SAM 2.1 truth
    box (one to one, greedy). Scale space blobs (LoG, DoG, DoH at object
    sizes), the textbook SimpleBlobDetector on grayscale, contours of an Otsu
    threshold on saturation, and for reference the purpose built
    :mod:`week03.objects`.

``shapes.png``
    The ten dots of the dot group (``annotations/shapes.json``) under the
    image's heavy color noise. Every dot method raw and after non-local means
    denoising, and the contour regions of :mod:`week03.targets`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import blobs, dots, evaluate, objects, targets

DENOISE = 10.0  # non-local means strength for shapes.png
SHAPE_DOTS = {"min_radius": 8.0, "max_radius": 40.0}  # the dots are 15 px; set from a look at the image
OBJECT_RADII = (20.0, 160.0)  # the pieces are 70 to 300 px across


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x, y, w, h
    score: float  # detector strength, used to rank
    circle: tuple[float, float, float] | None = None  # x, y, r for blob detectors
    contour: np.ndarray | None = None


# ---------------------------------------------------------------- objects ----


def _circles(found: list[blobs.Blob]) -> list[Detection]:
    return [Detection((b.x - b.radius, b.y - b.radius, 2 * b.radius, 2 * b.radius), float(b.response), (b.x, b.y, b.radius)) for b in found]


def scale_space_objects(image: np.ndarray, kind: str, min_contrast: float = 20.0) -> list[Detection]:
    """LoG, DoG or DoH peaks at object scales, overlapping peaks pruned, strongest first."""
    lab = dots.lab_image(image)
    found = blobs.detect_scale_space(
        lab, kind, min_radius=OBJECT_RADII[0], max_radius=OBJECT_RADII[1], threshold=blobs.DISK_PEAK * min_contrast * 0.5
    )
    return _circles(blobs.prune_overlaps(sorted(found, key=lambda b: -b.response), 0.3))


def simple_blob_objects(image: np.ndarray) -> list[Detection]:
    """The textbook one liner: SimpleBlobDetector on grayscale, with only an area filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = blobs.BlobFilter(min_area=800, min_circularity=None, min_convexity=None, min_inertia=None, blob_color=None)
    return _circles(sorted(blobs.detect_simple(gray, f), key=lambda b: -b.radius))


def contour_objects(image: np.ndarray, min_area_fraction: float = 0.004) -> list[Detection]:
    """Otsu threshold on saturation (colored plastic against a dull floor), then external contours by area."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = min_area_fraction * image.shape[0] * image.shape[1]
    out = [
        Detection(tuple(float(v) for v in cv2.boundingRect(c)), float(cv2.contourArea(c)), contour=c)
        for c in contours
        if cv2.contourArea(c) >= min_area
    ]
    return sorted(out, key=lambda d: -d.score)


def purpose_built_objects(image: np.ndarray) -> list[Detection]:
    report = objects.detect_objects(image)
    out = []
    for p in report.pieces:
        contours, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out.append(Detection(tuple(float(v) for v in p.bbox), float(p.confidence), contour=max(contours, key=cv2.contourArea)))
    return out


OBJECT_METHODS = {
    "log": ("Laplacian of Gaussian", lambda im: scale_space_objects(im, "log")),
    "dog": ("Difference of Gaussians", lambda im: scale_space_objects(im, "dog")),
    "doh": ("Determinant of Hessian", lambda im: scale_space_objects(im, "doh")),
    "gray": ("SimpleBlobDetector, grayscale", simple_blob_objects),
    "contours": ("Contours of Otsu on saturation", contour_objects),
    "objects.py": ("Purpose built (objects.py)", purpose_built_objects),
}


def match_boxes(found: list[Detection], truth: list[objects.TruthObject], min_iou: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy one to one matching by box IoU: (detection, truth, IoU)."""
    pairs = sorted(((evaluate.box_iou(d.box, t.bbox), i, j) for i, d in enumerate(found) for j, t in enumerate(truth)), reverse=True)
    used_d: set[int] = set()
    used_t: set[int] = set()
    out = []
    for iou, i, j in pairs:
        if iou < min_iou or i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        out.append((i, j, iou))
    return out


def score_objects(found: list[Detection], truth: list[objects.TruthObject]) -> dict:
    matches = match_boxes(found, truth)
    top = match_boxes(found[: len(truth)], truth)  # the strongest as many detections as there are objects
    hit = {j for _, j, _ in matches}
    return {
        "detections": len(found),
        "matched": len(matches),
        "false_positives": len(found) - len(matches),
        "missed": [truth[j].name for j in range(len(truth)) if j not in hit],
        "mean_box_iou": round(float(np.mean([m[2] for m in matches])), 3) if matches else 0.0,
        "strongest_matched": len(top),
    }


def objects_study(image: np.ndarray, truth: list[objects.TruthObject]) -> tuple[dict, dict[str, list[Detection]]]:
    results, found = {}, {}
    for key, (label, fn) in OBJECT_METHODS.items():
        found[key] = fn(image)
        results[key] = {"label": label, **score_objects(found[key], truth)}
    return results, found


# ----------------------------------------------------------------- shapes ----


def contour_dots(image: np.ndarray) -> list[blobs.Blob]:
    """Round regions from targets.py's Delta E segmentation, as dots of the same area."""
    return [
        blobs.Blob(t.center[0], t.center[1], float(np.sqrt(t.area / np.pi)))
        for t in targets.find_targets(image, group=False)
        if targets.is_dot(t)
    ]


def shapes_study(image: np.ndarray, truth: evaluate.TruthSet) -> tuple[dict, dict[str, list[blobs.Blob]]]:
    rows: dict[str, dict] = {}
    found: dict[str, list[blobs.Blob]] = {}

    def record(key: str, label: str, detected: list[blobs.Blob]) -> None:
        score = evaluate.score_against(detected, truth)
        rows[key] = {"label": label, **score.summary()}
        found[key] = detected

    for method in dots.METHODS:
        record(method, method, dots.find_dots(image, method=method, **SHAPE_DOTS).dots)
        record(f"{method}+nlm", f"{method}, denoised", dots.find_dots(image, method=method, denoise=DENOISE, **SHAPE_DOTS).dots)
    relaxed = dots.find_dots(image, method="log", denoise=DENOISE, min_roundness=0.90, **SHAPE_DOTS).dots
    record("log+nlm+0.90", "log, denoised, roundness 0.90", relaxed)
    record("contours", "contours (targets.py regions)", contour_dots(image))
    return rows, found


# ----------------------------------------------------------------- figure ----

GREEN, RED, WHITE = (90, 200, 90), (70, 70, 235), (245, 245, 245)


def _draw_objects(image: np.ndarray, found: list[Detection], truth: list[objects.TruthObject]) -> np.ndarray:
    out = image.copy()
    good = {i for i, _, _ in match_boxes(found, truth)}
    for i, d in reversed(list(enumerate(found))):
        color = GREEN if i in good else RED
        width = 4 if i in good else 2
        if d.contour is not None:
            cv2.drawContours(out, [d.contour], -1, color, width, cv2.LINE_AA)
        elif d.circle is not None:
            x, y, r = d.circle
            cv2.circle(out, (round(x), round(y)), round(r), color, width, cv2.LINE_AA)
    return out


def _draw_dots(image: np.ndarray, found: list[blobs.Blob], truth: evaluate.TruthSet) -> np.ndarray:
    out = image.copy()
    score = evaluate.score_against(found, truth)
    good = {i for i, _ in score.matches}
    for i, b in enumerate(found):
        cv2.circle(out, (round(b.x), round(b.y)), max(round(b.radius), 2), GREEN if i in good else RED, 3, cv2.LINE_AA)
    for j in score.misses:
        t = truth.dots[j]
        cv2.circle(out, (round(t.x), round(t.y)), round(t.radius) + 4, WHITE, 1, cv2.LINE_AA)
    return out


def figure(objects_image, object_found, object_rows, truth_objects, shapes_image, shape_found, shape_rows, truth_dots) -> np.ndarray:
    from .docs import _grid

    tiles = []
    for key in ("log", "contours", "objects.py"):
        r = object_rows[key]
        tiles.append(
            (
                _draw_objects(objects_image, object_found[key], truth_objects),
                f"{r['label']}: {r['matched']}/6 found, {r['false_positives']} false",
            )
        )
    crop = (slice(90, 360), slice(100, 380))  # the dot group and the arrow
    for key in ("log", "log+nlm", "contours"):
        r = shape_rows[key]
        view = _draw_dots(shapes_image, shape_found[key], truth_dots)[crop]
        tiles.append((view, f"shapes, {r['label']}: {r['tp']}/10 dots, {r['fp']} false"))
    return _grid(tiles, 3, 2400, 0.62)


def run(out: str | Path, photos: str | Path | None = None) -> dict:
    """Write ``beyond.json`` and ``beyond.jpg`` to ``out``."""
    from .docs import PHOTOS

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    photos = Path(photos) if photos else out / "photos"
    objects_image = cv2.imread(str(photos / "objects.jpg"))
    shapes_image = cv2.imread(str(photos / "shapes.png"))
    truth_objects = objects.load_objects()
    truth_dots = evaluate.truth_for("shapes.png")
    object_rows, object_found = objects_study(objects_image, truth_objects)
    shape_rows, shape_found = shapes_study(shapes_image, truth_dots)
    noise = {
        name: [round(v, 2) for v in dots.estimate_noise(cv2.imread(str(photos / name)))] for name in (*PHOTOS, "objects.jpg", "shapes.png")
    }
    result = {"noise_lab": noise, "denoise_strength": DENOISE, "shape_dot_radii": SHAPE_DOTS, "objects": object_rows, "shapes": shape_rows}
    (out / "beyond.json").write_text(json.dumps(result, indent=1) + "\n")
    sheet = figure(objects_image, object_found, object_rows, truth_objects, shapes_image, shape_found, shape_rows, truth_dots)
    cv2.imwrite(str(out / "beyond.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return result
