"""Command line interface: ``python -m week03 <command>``.

Option 1, OpenCV blob detection:
  dots       Find polka dots with SimpleBlobDetector, LoG, DoG, DoH or contours
  compare    Run every detector on one image and draw them side by side
  objects    Challenge 2: detect and classify cones, cubes and rings
  targets    Bonus: read shapes.png as SUAS style targets
  synth      Render synthetic polka dot scenes with exact ground truth
  benchmark  Score every detector on synthetic scenes and the real photos
  docs       Regenerate every figure and data file in docs/week03

Option 2, YOLOv8:
  yolo fetch     Download the football dataset (no Roboflow key needed)
  yolo stats     Frames, boxes and clip leakage per split
  yolo augment   Write a Roboflow style augmented copy of a dataset
  yolo train     Train and score named experiments
  yolo report    Turn results.json into a markdown table
  yolo figures   Write docs/week03/yolo (tables, notes, whole versus sliced example)
  yolo predict   Run a trained model on an image, optionally sliced
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from week02.colors import load_image


def _has_display() -> bool:
    import os

    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _show(title: str, image: np.ndarray) -> None:
    limit = 1400
    if max(image.shape[:2]) > limit:
        f = limit / max(image.shape[:2])
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _write(path: str | None, image: np.ndarray) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        print(f"wrote {path}")


# ------------------------------------------------------------------- dots ----


def _settings(args: argparse.Namespace):
    from . import dots

    colors = tuple(c.strip() for c in args.colors.split(",")) if args.colors else None
    return dots.DotSettings(
        method=args.method,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_contrast=args.min_contrast,
        colors=colors,
        refine=not args.no_refine,
        verify=not args.no_verify,
    )


def cmd_dots(args: argparse.Namespace) -> int:
    from . import dots, evaluate

    image = load_image(args.image)
    report = dots.find_dots(image, _settings(args))
    print(f"{args.image}: {len(report.dots)} dots with {args.method} in {report.seconds:.2f} s ({report.candidates} candidates)")
    for color, count in report.counts().items():
        print(f"  {color:12s} {count}")
    if args.verbose:
        for i, d in enumerate(report.dots):
            print(
                f"  {i:3d} ({d.x:7.1f}, {d.y:7.1f})  r={d.radius:6.1f}  {d.color}  contrast {d.contrast:5.1f}  roundness {d.roundness or 0:.3f}"
            )
    truth_path = evaluate.ANNOTATIONS / f"{Path(args.image).stem}.json"
    if args.truth or (args.truth is None and truth_path.is_file()):
        truth = evaluate.load_truth(args.truth or truth_path)
        score = evaluate.score_against(report.dots, truth)
        print(
            f"against {Path(args.truth or truth_path).name}: precision {score.precision:.3f}, recall {score.recall:.3f}, F1 {score.f1:.3f} ({score.tp} found, {score.fp} false, {score.fn} missed)"
        )
    annotated = dots.annotate(image, report.dots, label=args.verbose)
    _write(args.out, annotated)
    if args.json:
        Path(args.json).write_text(json.dumps(report.summary(), indent=2))
        print(f"wrote {args.json}")
    if args.show and _has_display():
        _show(f"{args.method}: {len(report.dots)} dots", annotated)
    return 0


def comparison_sheet(image: np.ndarray, method_dots: dict[str, list], width: int = 1600) -> np.ndarray:
    """A grid of annotated copies, one per method, with a caption bar each."""
    from . import dots

    tiles = []
    columns = 4 if len(method_dots) > 4 else len(method_dots)
    tile_w = width // columns
    for name, found in method_dots.items():
        tile = dots.annotate(image, found)
        scale = tile_w / tile.shape[1]
        tile = cv2.resize(tile, (tile_w, int(tile.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        bar = np.full((34, tile_w, 3), 24, np.uint8)
        cv2.putText(bar, f"{name}: {len(found)}", (10, 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((height - t.shape[0], tile_w, 3), 24, np.uint8)]) for t in tiles]
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 24))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def cmd_compare(args: argparse.Namespace) -> int:
    from . import dots, evaluate

    image = load_image(args.image)
    found = {}
    truth_path = evaluate.ANNOTATIONS / f"{Path(args.image).stem}.json"
    truth = evaluate.load_truth(truth_path) if truth_path.is_file() else None
    for method in dots.METHODS:
        report = dots.find_dots(image, method=method)
        found[method] = report.dots
        line = f"{method:9s} {len(report.dots):4d} dots  {report.seconds:6.2f} s"
        if truth:
            score = evaluate.score_against(report.dots, truth)
            line += f"  P {score.precision:.3f}  R {score.recall:.3f}  F1 {score.f1:.3f}"
        print(line)
    sheet = comparison_sheet(image, found)
    _write(args.out, sheet)
    if args.show and _has_display():
        _show("detectors", sheet)
    return 0


# ---------------------------------------------------------------- objects ----


def cmd_objects(args: argparse.Namespace) -> int:
    from . import objects

    image = load_image(args.image)
    report = objects.detect_objects(image, objects.ObjectSettings(grabcut=args.grabcut))
    print(f"{args.image}: {report.counts()} in {report.seconds:.2f} s")
    for p in report.pieces:
        x, y, w, h = p.bbox
        f = p.features
        print(
            f"  {p.label:7s} {p.confidence:.2f}  box ({x}, {y}, {w}, {h})  solidity {f.solidity:.2f}  hole {f.hole_ratio:.2f}  triangularity {f.triangularity:.2f}"
        )
    # The SAM annotation is used automatically for the ground school's objects.jpg.
    truth_path = Path(args.truth) if args.truth else objects.ANNOTATIONS / "objects.json"
    if args.truth or Path(args.image).stem == "objects":
        score = objects.score_objects(report.pieces, objects.load_objects(truth_path))
        print(
            f"against {truth_path.name}: {score['matched']}/{score['truth_objects']} found, {score['correct_class']} classified correctly, "
            f"{score['false_positives']} false positives, mean mask IoU {score['mean_mask_iou']:.3f}, mean box IoU {score['mean_box_iou']:.3f}"
        )
    annotated = objects.annotate(image, report.pieces)
    _write(args.out, annotated)
    if args.json:
        Path(args.json).write_text(json.dumps(report.summary(), indent=2))
        print(f"wrote {args.json}")
    if args.show and _has_display():
        _show("objects", annotated)
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    from . import targets

    image = load_image(args.image)
    found = targets.find_targets(image)
    for t in found:
        print(
            f"  {t.color} {t.shape}{' outline' if t.outline else ''}  ({t.confidence:.2f})  center ({t.center[0]:.0f}, {t.center[1]:.0f})"
        )
    annotated = targets.annotate(image, found)
    _write(args.out, annotated)
    if args.json:
        Path(args.json).write_text(json.dumps([t.summary() for t in found], indent=2))
        print(f"wrote {args.json}")
    if args.show and _has_display():
        _show("targets", annotated)
    return 0


# ------------------------------------------------------ synth, benchmark ----


def cmd_synth(args: argparse.Namespace) -> int:
    from . import synth

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    presets = synth.PRESETS if args.preset == "all" else (args.preset,)
    for preset in presets:
        scene = synth.polka_scene(synth.SceneConfig.preset(preset, seed=args.seed))
        cv2.imwrite(str(out / f"{preset}.png"), scene.image)
        truth = {
            "preset": preset,
            "config": asdict(scene.config),
            "dots": [asdict(d) for d in scene.dots],
            "distractors": scene.distractors,
        }
        (out / f"{preset}.json").write_text(json.dumps(truth, indent=1))
        print(f"{preset:12s} {len(scene.dots)} dots, {len(scene.distractors)} distractors")
    print(f"wrote {len(presets)} scenes to {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from . import benchmark

    data = Path(args.data) if args.data else None
    if data is not None and not data.is_dir():
        print(f"note: {data} not found, scoring synthetic scenes only")
        data = None
    presets = tuple(args.presets.split(",")) if args.presets else None
    report = benchmark.run(args.out, data, seeds=args.seeds, ablation=not args.no_ablation, presets=presets)
    print(benchmark.to_markdown(report))
    print(f"wrote {args.out}/dots_benchmark.md and dots_benchmark.json")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    from . import docs

    docs.run(args.out, args.photos, run_benchmark=not args.skip_benchmark, seeds=args.seeds)
    return 0


# ------------------------------------------------------------------- yolo ----


def cmd_yolo_fetch(args: argparse.Namespace) -> int:
    from . import football

    data = football.fetch(args.root)
    print(f"dataset in {data}")
    print(football.summary_json(data))
    return 0


def cmd_yolo_stats(args: argparse.Namespace) -> int:
    from . import football

    print(football.summary_json(args.data))
    return 0


def cmd_yolo_augment(args: argparse.Namespace) -> int:
    from . import augment

    yaml = augment.augment_dataset(args.split_dir, args.out, augment.Recipe.preset(args.recipe), copies=args.copies, seed=args.seed)
    print(f"wrote {yaml}")
    return 0


def cmd_yolo_train(args: argparse.Namespace) -> int:
    from . import detector

    names = list(detector.EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    unknown = [n for n in names if n not in detector.EXPERIMENTS]
    if unknown:
        print(f"unknown experiments: {', '.join(unknown)}; choose from {', '.join(detector.EXPERIMENTS)}", file=sys.stderr)
        return 2
    detector.main_run(names, args.data, args.root)
    return 0


def cmd_yolo_report(args: argparse.Namespace) -> int:
    from . import detector

    results = json.loads(Path(args.results).read_text())
    text = detector.to_markdown(results)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_yolo_figures(args: argparse.Namespace) -> int:
    from . import detector

    dataset = detector.figures(args.root, args.out, args.data)
    print(f"wrote {args.out}: {dataset.get('headline', '')}")
    return 0


def cmd_yolo_predict(args: argparse.Namespace) -> int:
    from . import detector, football, slicing

    image = load_image(args.image)
    predict = detector.ultralytics_predictor(Path(args.weights), args.imgsz, conf=args.conf)
    det = slicing.sliced_predict(predict, image) if args.sliced else predict(image)
    keep = det.scores >= args.conf
    out = image.copy()
    colors = [(0, 215, 255), (255, 120, 0), (80, 220, 80), (60, 60, 240)]
    counts: dict[str, int] = {}
    for box, score, c in zip(det.boxes[keep], det.scores[keep], det.classes[keep]):
        name = football.NAMES[int(c)]
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[int(c) % 4], 2, cv2.LINE_AA)
        cv2.putText(out, f"{name} {score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[int(c) % 4], 1, cv2.LINE_AA)
    print(f"{args.image}: {counts}")
    _write(args.out, out)
    return 0


# ------------------------------------------------------------------- main ----


def build_parser() -> argparse.ArgumentParser:
    from . import dots, synth

    parser = argparse.ArgumentParser(prog="week03", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dots", help="find polka dots")
    p.add_argument("image")
    p.add_argument("--method", choices=dots.METHODS, default="log", help="detector (default: log)")
    p.add_argument("--min-radius", type=float, default=3.0, help="smallest dot radius in pixels")
    p.add_argument("--max-radius", type=float, default=None, help="largest dot radius (default: a sixth of the short side)")
    p.add_argument("--min-contrast", type=float, default=10.0, help="minimum Delta E between a dot and its surroundings")
    p.add_argument("--colors", help="keep only these colors, comma separated (red,green,light blue,...)")
    p.add_argument("--no-refine", action="store_true", help="skip the edge fit")
    p.add_argument("--no-verify", action="store_true", help="keep every candidate")
    p.add_argument("--truth", default=None, help="annotation JSON to score against (found automatically for the ground school photos)")
    p.add_argument("--out", help="write the annotated image")
    p.add_argument("--json", help="write every dot as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="list every dot and number them in the image")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.set_defaults(func=cmd_dots)

    p = sub.add_parser("compare", help="every detector on one image, side by side")
    p.add_argument("image")
    p.add_argument("--out", help="write the comparison sheet")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("objects", help="detect and classify cones, cubes and rings")
    p.add_argument("image")
    p.add_argument("--grabcut", action="store_true", help="refine contours with GrabCut")
    p.add_argument("--truth", help="annotation JSON (default: the SAM annotation for objects.jpg)")
    p.add_argument("--out")
    p.add_argument("--json")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.set_defaults(func=cmd_objects)

    p = sub.add_parser("targets", help="read shapes.png as SUAS style targets")
    p.add_argument("image")
    p.add_argument("--out")
    p.add_argument("--json")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("synth", help="render synthetic polka dot scenes with ground truth")
    p.add_argument("out")
    p.add_argument("--preset", choices=(*synth.PRESETS, "all"), default="all")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("benchmark", help="score every detector against ground truth")
    p.add_argument("out")
    p.add_argument("--data", default="docs/week03/photos", help="folder with the ground school photos (skipped if missing)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--presets", help="comma separated synthetic presets (default: all)")
    p.add_argument("--no-ablation", action="store_true", help="skip the with and without edge fit comparison")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("docs", help="regenerate the figures and data in docs/week03")
    p.add_argument("out", nargs="?", default="docs/week03")
    p.add_argument("--photos", default=None, help="folder with the ground school photos (default: OUT/photos)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--skip-benchmark", action="store_true")
    p.set_defaults(func=cmd_docs)

    yolo = sub.add_parser("yolo", help="Option 2: YOLOv8 on the football dataset").add_subparsers(dest="yolo_command", required=True)

    p = yolo.add_parser("fetch", help="download the dataset from Hugging Face")
    p.add_argument("root", nargs="?", default="data/football")
    p.set_defaults(func=cmd_yolo_fetch)

    p = yolo.add_parser("stats", help="describe a YOLO dataset directory")
    p.add_argument("data", nargs="?", default="data/football/data")
    p.set_defaults(func=cmd_yolo_stats)

    p = yolo.add_parser("augment", help="Roboflow style offline augmentation of a split directory")
    p.add_argument("split_dir")
    p.add_argument("out")
    p.add_argument("--recipe", default="broadcast", choices=["none", "broadcast", "heavy"])
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_yolo_augment)

    p = yolo.add_parser("train", help="train and score experiments (see week03/detector.py)")
    p.add_argument("experiments", nargs="+", help="experiment names, or all")
    p.add_argument("--data", default="data/football/data")
    p.add_argument("--root", default="out/yolo")
    p.set_defaults(func=cmd_yolo_train)

    p = yolo.add_parser("report", help="markdown table from results.json")
    p.add_argument("results", nargs="?", default="out/yolo/results.json")
    p.add_argument("--out")
    p.set_defaults(func=cmd_yolo_report)

    p = yolo.add_parser("figures", help="write docs/week03/yolo from trained experiments")
    p.add_argument("--root", default="out/yolo")
    p.add_argument("--out", default="docs/week03/yolo")
    p.add_argument("--data", default="data/football/data")
    p.set_defaults(func=cmd_yolo_figures)

    p = yolo.add_parser("predict", help="run a trained model on one image")
    p.add_argument("weights")
    p.add_argument("image")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--sliced", action="store_true", help="tile the image (SAHI style) and merge")
    p.add_argument("--out", default="prediction.jpg")
    p.set_defaults(func=cmd_yolo_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
