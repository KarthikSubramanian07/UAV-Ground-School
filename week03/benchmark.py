"""Head to head blob detector benchmark.

Two parts:

* **Synthetic scenes** (:mod:`week03.synth`) where the truth is exact, so
  center and radius errors are meaningful. Seven presets, several seeds each.
* **The three real photos** from the ground school, scored against the hand
  checked annotations in ``week03/annotations``. The annotation geometry comes
  from the same edge fit the pipeline uses, so only precision and recall are
  reported there.

``python -m week03 benchmark docs/week03`` writes ``dots_benchmark.json`` and
``dots_benchmark.md``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import dots, evaluate, synth
from .blobs import Blob

REAL_IMAGES = ("polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg")

METHOD_LABELS = {
    "gray": "SimpleBlobDetector on grayscale",
    "simple": "SimpleBlobDetector per palette color",
    "contrast": "SimpleBlobDetector on background contrast",
    "contour": "Contour filtering on color edges",
    "log": "Laplacian of Gaussian (CIELAB)",
    "dog": "Difference of Gaussians (CIELAB)",
    "doh": "Determinant of Hessian (CIELAB)",
}


def _truth_blobs(scene: synth.Scene) -> tuple[list[Blob], list[bool]]:
    truth = [Blob(d.x, d.y, d.radius) for d in scene.dots]
    # Dots cut by the frame are optional.
    h, w = scene.image.shape[:2]
    difficult = [d.x < 0 or d.y < 0 or d.x > w - 1 or d.y > h - 1 for d in scene.dots]
    return truth, difficult


def _totals(scores: list[evaluate.DotScore], seconds: list[float], centers: list[float], radii: list[float]) -> dict:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "seconds": round(float(np.mean(seconds)), 4),
    }
    if centers:
        out["center_error_px"] = round(float(np.median(centers)), 3)
        out["radius_error"] = round(float(np.median(radii)), 4)
    return out


def synthetic_benchmark(methods=dots.METHODS, presets=None, seeds: int = 3, refine: bool = True) -> dict:
    results: dict = {}
    presets = presets or synth.PRESETS
    for preset in presets:
        scenes = [synth.polka_scene(synth.SceneConfig.preset(preset, seed=s)) for s in range(seeds)]
        results[preset] = {}
        for method in methods:
            scores, seconds, centers, radii = [], [], [], []
            for scene in scenes:
                report = dots.find_dots(scene.image, method=method, refine=refine)
                truth, difficult = _truth_blobs(scene)
                score = evaluate.match_dots(report.dots, truth, difficult)
                scores.append(score)
                seconds.append(report.seconds)
                for i, j in score.matches:
                    if difficult[j]:
                        continue
                    centers.append(math.hypot(report.dots[i].x - truth[j].x, report.dots[i].y - truth[j].y))
                    radii.append(abs(report.dots[i].radius - truth[j].radius) / truth[j].radius)
            results[preset][method] = _totals(scores, seconds, centers, radii)
    return results


def real_benchmark(data_dir: str | Path, methods=dots.METHODS) -> dict:
    results: dict = {}
    data_dir = Path(data_dir)
    for name in REAL_IMAGES:
        path = data_dir / name
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        truth = evaluate.truth_for(name)
        results[name] = {"required_dots": truth.required}
        for method in methods:
            report = dots.find_dots(image, method=method)
            score = evaluate.score_against(report.dots, truth)
            results[name][method] = {**score.summary(), "seconds": round(report.seconds, 4)}
            del results[name][method]["center_error_px"]
            del results[name][method]["radius_error"]
    return results


def _overall(section: dict, methods) -> dict:
    overall = {}
    for method in methods:
        tp = sum(v[method]["tp"] for v in section.values() if method in v)
        fp = sum(v[method]["fp"] for v in section.values() if method in v)
        fn = sum(v[method]["fn"] for v in section.values() if method in v)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        overall[method] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        }
    return overall


def to_markdown(report: dict) -> str:
    methods = report["methods"]
    lines = ["# Week 3 blob detector benchmark", "", f"Generated by `python -m week03 benchmark` in {report['seconds']:.0f} s.", ""]
    if report.get("real"):
        lines += [
            "## The three ground school photos",
            "",
            "Precision / recall / F1 against hand checked annotations. Dots cut by the frame are optional; detections on the defocused cards of photo 3 are not scored.",
            "",
        ]
        names = list(report["real"])
        header = "| Method | " + " | ".join(f"{n} ({report['real'][n]['required_dots']} dots)" for n in names) + " | All photos F1 |"
        lines += [header, "| --- | " + " | ".join("---:" for _ in names) + " | ---: |"]
        overall = report["real_overall"]
        for m in methods:
            cells = []
            for n in names:
                r = report["real"][n][m]
                cells.append(f"{r['precision']:.2f} / {r['recall']:.2f} / {r['f1']:.3f}")
            lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(cells) + f" | **{overall[m]['f1']:.3f}** |")
        lines.append("")
    syn = report["synthetic"]
    lines += [
        "## Synthetic scenes with exact truth",
        "",
        f"F1 per preset ({report['seeds']} seeds each), then the median center error and relative radius error over all matched dots.",
        "",
    ]
    presets = list(syn)
    lines += [
        "| Method | " + " | ".join(presets) + " | Overall F1 | Center error | Radius error | Time |",
        "| --- | " + " | ".join("---:" for _ in presets) + " | ---: | ---: | ---: | ---: |",
    ]
    for m in methods:
        cells = [f"{syn[p][m]['f1']:.3f}" for p in presets]
        centers = [syn[p][m].get("center_error_px") for p in presets if syn[p][m].get("center_error_px") is not None]
        radii = [syn[p][m].get("radius_error") for p in presets if syn[p][m].get("radius_error") is not None]
        seconds = float(np.mean([syn[p][m]["seconds"] for p in presets]))
        lines.append(
            f"| {METHOD_LABELS[m]} | "
            + " | ".join(cells)
            + f" | **{report['synthetic_overall'][m]['f1']:.3f}** | {np.median(centers) if centers else float('nan'):.2f} px | {100 * np.median(radii) if radii else float('nan'):.1f}% | {seconds * 1000:.0f} ms |"
        )
    lines += [
        "",
        "Presets: `flat` clean print; `fabric` woven cloth with folds and JPEG; `pastel` light dots on white cards;",
        "`distractors` squares, triangles, stars, crosses and thin ellipses in the same colors; `tilted` perspective,",
        "defocus and uneven light; `tiny` radius 2.5 to 7 px; `crowded` 90 dots on fabric.",
        "",
    ]
    if "ablation" in report:
        lines += [
            "## Ablation: the edge fit",
            "",
            "Overall synthetic F1 and median center error with and without the circle fit to color edges.",
            "",
        ]
        lines += [
            "| Method | F1 without fit | F1 with fit | Center error without | Center error with |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for m in methods:
            a = report["ablation"][m]
            lines.append(
                f"| {METHOD_LABELS[m]} | {a['f1_raw']:.3f} | {a['f1_refined']:.3f} | {a['center_raw']:.2f} px | {a['center_refined']:.2f} px |"
            )
        lines.append("")
    return "\n".join(lines)


def run(
    out_dir: str | Path, data_dir: str | Path | None = None, seeds: int = 3, methods=dots.METHODS, ablation: bool = True, presets=None
) -> dict:
    start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    syn = synthetic_benchmark(methods, presets, seeds=seeds)
    report: dict = {"methods": list(methods), "seeds": seeds, "synthetic": syn, "synthetic_overall": _overall(syn, methods)}
    if ablation:
        raw = synthetic_benchmark(methods, presets, seeds=seeds, refine=False)
        raw_overall = _overall(raw, methods)
        report["ablation"] = {}
        for m in methods:

            def median_center(section: dict, method: str = m) -> float:
                values = [
                    section[p][method].get("center_error_px") for p in section if section[p][method].get("center_error_px") is not None
                ]
                return float(np.median(values)) if values else float("nan")

            report["ablation"][m] = {
                "f1_raw": raw_overall[m]["f1"],
                "f1_refined": report["synthetic_overall"][m]["f1"],
                "center_raw": median_center(raw),
                "center_refined": median_center(syn),
            }
    if data_dir is not None:
        real = real_benchmark(data_dir, methods)
        if real:
            report["real"] = real
            report["real_overall"] = _overall({k: {m: v[m] for m in methods} for k, v in real.items()}, methods)
    report["seconds"] = time.perf_counter() - start
    (out_dir / "dots_benchmark.json").write_text(json.dumps(report, indent=2))
    (out_dir / "dots_benchmark.md").write_text(to_markdown(report))
    return report
