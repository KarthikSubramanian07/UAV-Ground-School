"""Command line interface: ``python -m week02 <command>``.

Commands:
  colors     Color Me Impressed: split an image by color, find object centers
  frames     Save every Nth frame of a video as PNG files
  stitch     I'll be Needin' Stitches: stitch a drone video into one image
  simulate   Generate test images and a survey flight video with ground truth
  benchmark  Compare stitching strategies against simulator ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


def _has_display() -> bool:
    import os

    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ----------------------------------------------------------------- colors ----


def cmd_colors(args: argparse.Namespace) -> int:
    from . import colors

    image = colors.load_image(args.image)
    layers = colors.split_colors(image, min_coverage=args.min_coverage, mode=args.mode, k=args.k, classify_shapes=not args.no_shapes)
    print(f"{args.image}: {image.shape[1]}x{image.shape[0]} px, {len(layers)} colors ({args.mode} mode)\n")
    print(colors.format_report(layers))

    if args.out:
        written = colors.save_layers(image, layers, args.out)
        print(f"\nwrote {len(written)} files to {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps([layer.summary() for layer in layers], indent=2))
        print(f"wrote {args.json}")
    if args.show and _has_display():
        colors.show_layers(image, layers)
    return 0


# ----------------------------------------------------------------- frames ----


def cmd_frames(args: argparse.Namespace) -> int:
    from .stitching import extract_frames

    written = extract_frames(args.video, args.out, every=args.every, scale=args.scale)
    print(f"wrote {len(written)} frames to {args.out}")
    return 0


# ----------------------------------------------------------------- stitch ----


def _live_preview():
    """Callback that shows the mosaic growing as keyframes arrive."""
    from . import mosaic

    def show(tracker) -> None:
        keyframes = tracker.keyframes
        if len(keyframes) % 3:
            return
        small = [cv2.resize(k.frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for k in keyframes]
        shrink, grow = np.diag([0.25, 0.25, 1.0]), np.diag([4.0, 4.0, 1.0])
        transforms = [shrink @ k.transform @ grow for k in keyframes]
        preview, _, _ = mosaic.render(small, transforms, "overwrite")
        cv2.imshow("live mosaic (press q to hide)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    return show


def cmd_stitch(args: argparse.Namespace) -> int:
    from . import stitching

    config = stitching.StitchConfig(
        every=args.every,
        scale=args.scale,
        k1=args.k1,
        detector=args.detector,
        n_features=args.features,
        loop_closure=not args.no_loops,
        bundle_adjust=not args.no_adjust,
        gain_compensation=not args.no_gains,
        blend=args.blend,
        max_keyframes=args.max_keyframes,
    )
    log = print if args.verbose else None
    on_keyframe = _live_preview() if args.live and _has_display() else None
    try:
        result = stitching.stitch_video(args.video, config, method=args.method, log=log, on_keyframe=on_keyframe)
    except stitching.StitchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result.panorama)
    print(result.summary())
    print(f"wrote {out}")
    if args.path_overlay and result.transforms:
        cv2.imwrite(args.path_overlay, stitching.draw_flight_path(result))
        print(f"wrote {args.path_overlay}")
    if args.transforms and result.transforms:
        result.save_transforms(args.transforms)
        print(f"wrote {args.transforms}")
    if args.truth and result.transforms:
        from . import evaluate, synth

        truth = synth.load_truth(args.truth)
        world = cv2.imread(args.world) if args.world else None
        evaluation = evaluate.evaluate(result.transforms, result.frame_indices, truth, result.panorama, world, frame_scale=args.scale)
        print(f"accuracy        pose RMSE {evaluation.rmse_px:.2f} px (max {evaluation.max_px:.2f} px)" + (f", ZNCC {evaluation.zncc:.3f}" if evaluation.zncc is not None else ""))
    if args.show and _has_display():
        preview = result.panorama
        limit = 1600
        if max(preview.shape[:2]) > limit:
            factor = limit / max(preview.shape[:2])
            preview = cv2.resize(preview, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cv2.imshow("stitched", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------- simulate ----


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import synth

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "stop_sign.jpg"), synth.stop_sign())
    cv2.imwrite(str(out / "apple.jpg"), synth.apple())
    targets, truths = synth.suas_targets()
    cv2.imwrite(str(out / "targets.jpg"), targets)
    (out / "targets.json").write_text(json.dumps([asdict(t) for t in truths], indent=2))

    if args.quick:
        plan = synth.FlightPlan(frame_width=320, frame_height=180, passes=2, speed=5.0, seed=args.seed)
        world = synth.voxel_world(width_blocks=200, height_blocks=110, seed=args.seed)
    elif args.hard:
        plan = synth.FlightPlan.hard(seed=args.seed)
        world = synth.voxel_world(height_blocks=300, seed=args.seed)
    else:
        plan = synth.FlightPlan(seed=args.seed)
        world = synth.voxel_world(height_blocks=220, seed=args.seed)
    truth = synth.simulate_flight(world, plan)
    cv2.imwrite(str(out / "world.png"), world)
    synth.write_flight_video(out / "flight.mp4", world, truth)
    synth.save_truth(out / "flight_truth.csv", truth)
    seconds = len(truth.poses) / plan.fps
    print(f"wrote test images, world.png and a {seconds:.0f}s flight ({len(truth.poses)} frames, {plan.passes} passes) to {out}")
    if plan.distortion:
        print(f"the camera has k1={plan.distortion}; pass --k1 {plan.distortion} to stitch to undistort")
    return 0


# -------------------------------------------------------------- benchmark ----


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    run_benchmark(Path(args.out), seed=args.seed, include_stitcher=not args.skip_stitcher)
    return 0


# ------------------------------------------------------------------- main ----


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="week02", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("colors", help="split an image into one image per color and locate objects")
    p.add_argument("image", help="path to an image file")
    p.add_argument("--mode", choices=["named", "auto"], default="named", help="named HSV bands or k-means palette discovery")
    p.add_argument("--k", type=int, default=None, help="number of colors in auto mode (default: silhouette score picks)")
    p.add_argument("--min-coverage", type=float, default=0.01, help="ignore colors below this fraction unless they form objects")
    p.add_argument("--out", help="directory for transparent PNG layers and a contact sheet")
    p.add_argument("--json", help="write the report as JSON")
    p.add_argument("--no-shapes", action="store_true", help="skip shape classification")
    p.add_argument("--no-show", dest="show", action="store_false", help="do not open windows")
    p.set_defaults(func=cmd_colors)

    p = sub.add_parser("frames", help="save every Nth frame of a video")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("stitch", help="stitch a nadir drone video into one image")
    p.add_argument("video")
    p.add_argument("--out", default="stitched.jpg")
    p.add_argument("--method", choices=["features", "stitcher"], default="features")
    p.add_argument("--every", type=int, default=5, help="sample one frame out of every N")
    p.add_argument("--scale", type=float, default=1.0, help="resize frames first (0.5 halves them)")
    p.add_argument("--k1", type=float, default=0.0, help="radial lens distortion to remove")
    p.add_argument("--detector", choices=["orb", "sift"], default="orb")
    p.add_argument("--features", type=int, default=2500)
    p.add_argument("--blend", choices=["multiband", "feather", "overwrite"], default="multiband")
    p.add_argument("--no-loops", action="store_true", help="disable loop closure detection")
    p.add_argument("--no-adjust", action="store_true", help="disable bundle adjustment")
    p.add_argument("--no-gains", action="store_true", help="disable exposure gain compensation")
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--path-overlay", help="also write the panorama with keyframe footprints and flight path")
    p.add_argument("--transforms", help="write keyframe transforms as CSV")
    p.add_argument("--truth", help="ground truth CSV from `simulate`, to report accuracy")
    p.add_argument("--world", help="world.png from `simulate`, for photometric accuracy")
    p.add_argument("--live", action="store_true", help="show the mosaic growing while tracking")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_stitch)

    p = sub.add_parser("simulate", help="generate test images and a drone flight with ground truth")
    p.add_argument("out")
    p.add_argument("--hard", action="store_true", help="5 passes with lens distortion, blur, heavy noise")
    p.add_argument("--quick", action="store_true", help="a small, fast 2 pass flight at 320x180")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("benchmark", help="compare stitching strategies against ground truth")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-stitcher", action="store_true", help="skip the slow cv2.Stitcher baseline")
    p.set_defaults(func=cmd_benchmark)
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
