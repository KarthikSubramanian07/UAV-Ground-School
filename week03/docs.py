"""Regenerate every figure and data file in ``docs/week03`` from the code.

``python -m week03 docs docs/week03`` runs the detectors on the ground school
photos in ``docs/week03/photos`` and writes:

* ``dots/<photo>.json``: every method's dots and scores, which drive the
  detector explorer on the showcase site;
* ``dots/<photo>_<method>.jpg``, ``dots/<photo>_compare.jpg`` and ``hero.jpg``;
* ``scale_space.jpg``: the LoG response of one photo at four scales;
* ``synthetic.jpg``: one scene per synthetic preset;
* ``objects.jpg``, ``objects.json``, ``objects_robustness.jpg``;
* ``targets.jpg`` and ``targets.json``;
* ``dots_benchmark.md`` and ``dots_benchmark.json`` (unless ``--skip-benchmark``).

The YOLO figures need trained weights and are made by ``yolo figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import benchmark, blobs, dots, evaluate, objects, synth, targets

PHOTOS = ("polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg")


def _label(image: np.ndarray, text: str, scale: float = 1.0) -> np.ndarray:
    """A dark caption bar above an image, sized for the image's final width."""
    height = int(46 * scale)
    bar = np.full((height, image.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, text, (int(16 * scale), int(31 * scale)), cv2.FONT_HERSHEY_DUPLEX, 0.72 * scale, (238, 236, 232), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox an image into a fixed frame."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    frame = np.full((height, width, 3), 22, np.uint8)
    y0, x0 = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    frame[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return frame


def _grid(tiles: list[tuple[np.ndarray, str]], columns: int, width: int, aspect: float, gap: int = 10) -> np.ndarray:
    """Captioned tiles of equal size in a grid, ``aspect`` = tile height / width."""
    tile_w = (width - gap * (columns - 1)) // columns
    tile_h = round(tile_w * aspect)
    cells = [_label(_fit(image, tile_w, tile_h), text) for image, text in tiles]
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], 22))
    rows = []
    for i in range(0, len(cells), columns):
        row = []
        for cell in cells[i : i + columns]:
            row += [cell, np.full((cell.shape[0], gap, 3), 22, np.uint8)]
        rows += [np.hstack(row[:-1]), np.full((gap, width - (width - gap * (columns - 1)) % columns, 3), 22, np.uint8)]
    return np.vstack(rows[:-1])


def dot_figures(photos: Path, out: Path) -> dict:
    from .cli import comparison_sheet

    (out / "dots").mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in PHOTOS:
        image = cv2.imread(str(photos / name))
        truth = evaluate.truth_for(name)
        stem = Path(name).stem
        record = {"image": name, "width": image.shape[1], "height": image.shape[0], "required": truth.required, "methods": {}}
        found = {}
        for method in dots.METHODS:
            report = dots.find_dots(image, method=method)
            score = evaluate.score_against(report.dots, truth)
            matched = {i for i, _ in score.matches}
            false = set(score.false_positives)
            record["methods"][method] = {
                "seconds": round(report.seconds, 3),
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
                "tp": score.tp,
                "fp": score.fp,
                "fn": score.fn,
                "dots": [
                    {
                        "x": round(d.x, 2),
                        "y": round(d.y, 2),
                        "r": round(d.radius, 2),
                        "color": d.color,
                        "rgb": list(reversed(d.bgr)) if d.bgr else None,
                        "contrast": round(d.contrast or 0, 1),
                        "status": "tp" if i in matched else "fp" if i in false else "ignored",
                    }
                    for i, d in enumerate(report.dots)
                ],
                "missed": [
                    {"x": round(truth.dots[j].x, 2), "y": round(truth.dots[j].y, 2), "r": round(truth.dots[j].radius, 2)}
                    for j in score.misses
                ],
            }
            found[method] = report.dots
            if method in ("log", "gray", "simple"):
                cv2.imwrite(str(out / "dots" / f"{stem}_{method}.jpg"), dots.annotate(image, report.dots), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  {name:18s} {method:9s} F1 {score.f1:.3f}  ({len(report.dots)} dots, {report.seconds:.1f} s)", flush=True)
        (out / "dots" / f"{stem}.json").write_text(json.dumps(record, separators=(",", ":")))
        cv2.imwrite(str(out / "dots" / f"{stem}_compare.jpg"), comparison_sheet(image, found), [cv2.IMWRITE_JPEG_QUALITY, 88])
        summary[name] = {m: record["methods"][m]["f1"] for m in dots.METHODS}
    hero_figure(out)
    return summary


def hero_figure(out: Path, height: int = 900, gap: int = 12) -> None:
    """The three photos with their LoG detections, side by side at one height."""
    parts = []
    for name in PHOTOS:
        tile = cv2.imread(str(out / "dots" / f"{Path(name).stem}_log.jpg"))
        parts.append(cv2.resize(tile, (round(tile.shape[1] * height / tile.shape[0]), height), interpolation=cv2.INTER_AREA))
        parts.append(np.full((height, gap, 3), 22, np.uint8))
    cv2.imwrite(str(out / "hero.jpg"), np.hstack(parts[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def scale_space_figure(photos: Path, out: Path) -> None:
    """LoG response magnitude of photo 1 at four scales, each normalized to the same color range."""
    image = cv2.imread(str(photos / "polka_dots_1.png"))
    lab = dots.lab_image(image)
    radii = (6.0, 12.0, 24.0, 48.0)
    sigmas = np.array([r / blobs.SQRT2 for r in radii])
    stack, _ = blobs.scale_space(lab, sigmas, "log")
    top = float(np.percentile(stack, 99.7))
    tiles = [(image, "input")]
    for radius, layer in zip(radii, stack):
        heat = cv2.applyColorMap(np.clip(layer / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        tiles.append((heat, f"sigma {radius / blobs.SQRT2:.1f}: radius {radius:.0f} px"))
    cv2.imwrite(str(out / "scale_space.jpg"), _grid(tiles, 5, 2400, 1.0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def synthetic_figure(out: Path) -> None:
    tiles = []
    for preset in synth.PRESETS:
        scene = synth.polka_scene(synth.SceneConfig.preset(preset, seed=1))
        report = dots.find_dots(scene.image, method="log")
        tiles.append((dots.annotate(scene.image, report.dots), f"{preset}: {len(scene.dots)} dots, {len(report.dots)} found"))
    cv2.imwrite(str(out / "synthetic.jpg"), _grid(tiles, 4, 2400, 0.75), [cv2.IMWRITE_JPEG_QUALITY, 88])


def object_figures(photos: Path, out: Path) -> dict:
    image = cv2.imread(str(photos / "objects.jpg"))
    report = objects.detect_objects(image)
    score = objects.score_objects(report.pieces, objects.load_objects())
    cv2.imwrite(str(out / "objects.jpg"), objects.annotate(image, report.pieces), [cv2.IMWRITE_JPEG_QUALITY, 90])
    gc = objects.detect_objects(image, objects.ObjectSettings(grabcut=True))
    gc_score = objects.score_objects(gc.pieces, objects.load_objects())

    truth = objects.load_objects()
    overlay = image.copy()
    for t in truth:
        overlay[t.mask > 0] = objects.CLASS_BGR[t.label]
    truth_view = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    for t in truth:
        contours, _ = cv2.findContours(t.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(truth_view, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "objects_truth.jpg"), truth_view, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def hue_shift(im: np.ndarray, d: int) -> np.ndarray:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + d) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def rotate(im: np.ndarray, angle: float) -> np.ndarray:
        h, w = im.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * s + w * c), int(h * c + w * s)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(im, m, (nw, nh), borderValue=(200, 200, 200))

    variants = {
        "rotated 35 degrees": rotate(image, 35),
        "rotated 90 degrees": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        "mirrored": cv2.flip(image, 1),
        "half size": cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        "hue shifted 90 degrees": hue_shift(image, 45),
        "hue shifted 180 degrees": hue_shift(image, 90),
    }
    tiles = []
    robustness = {}
    for name, variant in variants.items():
        r = objects.detect_objects(variant)
        counts = r.counts()
        robustness[name] = counts
        text = ", ".join(f"{counts.get(c, 0)} {c}s" for c in objects.CLASSES)
        tiles.append((objects.annotate(variant, r.pieces), f"{name}: {text}"))
    cv2.imwrite(str(out / "objects_robustness.jpg"), _grid(tiles, 3, 2400, 0.75), [cv2.IMWRITE_JPEG_QUALITY, 86])
    result = {
        **report.summary(),
        "score": score,
        "grabcut_score": {k: v for k, v in gc_score.items() if k != "matches"},
        "robustness": robustness,
    }
    (out / "objects.json").write_text(json.dumps(result, indent=1))
    return result


def target_figures(photos: Path, out: Path) -> list[dict]:
    image = cv2.imread(str(photos / "shapes.png"))
    found = targets.find_targets(image)
    cv2.imwrite(str(out / "targets.jpg"), targets.annotate(image, found), [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary = [t.summary() for t in found]
    (out / "targets.json").write_text(json.dumps(summary, indent=1))
    return summary


def run(out: str | Path, photos: str | Path | None = None, run_benchmark: bool = True, seeds: int = 3) -> None:
    out = Path(out)
    photos = Path(photos) if photos else out / "photos"
    missing = [n for n in (*PHOTOS, "objects.jpg", "shapes.png") if not (photos / n).is_file()]
    if missing:
        raise FileNotFoundError(f"missing ground school photos in {photos}: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    print("dots:", flush=True)
    dot_figures(photos, out)
    scale_space_figure(photos, out)
    synthetic_figure(out)
    result = object_figures(photos, out)
    print(f"objects: {result['counts']}, mean mask IoU {result['score']['mean_mask_iou']:.3f}", flush=True)
    found = target_figures(photos, out)
    print(f"targets: {', '.join(t['shape'] for t in found)}", flush=True)
    if run_benchmark:
        report = benchmark.run(out, photos, seeds=seeds)
        overall = ", ".join(f"{m} {v['f1']:.3f}" for m, v in report["synthetic_overall"].items())
        print(f"benchmark: synthetic F1 {overall}")
