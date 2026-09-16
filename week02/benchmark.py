"""Reproducible stitching benchmark against simulator ground truth.

``python -m week02 benchmark docs/week02`` renders two survey flights, stitches
them with progressively more of the pipeline enabled, scores every run with
``evaluate.py`` and writes ``benchmark.md``, ``benchmark.json`` and preview
images.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import evaluate, stitching, synth


@dataclass
class Scenario:
    name: str
    description: str
    plan: synth.FlightPlan
    world_height_blocks: int


def scenarios(seed: int) -> list[Scenario]:
    return [
        Scenario("calm", "3 passes, light noise, no lens distortion", synth.FlightPlan(seed=seed), 220),
        Scenario(
            "rough",
            "5 passes, barrel distortion k1=-0.06, motion blur, heavy noise, +/-18% exposure, +/-5% altitude",
            synth.FlightPlan.hard(seed=seed),
            300,
        ),
    ]


def variants(plan: synth.FlightPlan) -> list[tuple[str, stitching.StitchConfig]]:
    runs = [
        ("sequential chaining", stitching.StitchConfig(loop_closure=False, bundle_adjust=False, gain_compensation=False, blend="feather")),
        ("+ loop closure and bundle adjustment", stitching.StitchConfig(gain_compensation=False, blend="feather")),
        ("+ gain compensation and multi band blending", stitching.StitchConfig()),
    ]
    if plan.distortion:
        runs.append(("+ lens undistortion (full pipeline)", stitching.StitchConfig(k1=plan.distortion)))
        runs.append(("full pipeline with SIFT", stitching.StitchConfig(k1=plan.distortion, detector="sift")))
    return runs


def _thumbnail(image: np.ndarray, width: int = 1800) -> np.ndarray:
    if image.shape[1] <= width:
        return image
    factor = width / image.shape[1]
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def _save(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), _thumbnail(image), [cv2.IMWRITE_JPEG_QUALITY, 86])


def run_benchmark(out: Path, seed: int = 42, include_stitcher: bool = True, log=print) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in scenarios(seed):
            world = synth.voxel_world(height_blocks=scenario.world_height_blocks, seed=seed)
            truth = synth.simulate_flight(world, scenario.plan)
            video = Path(tmp) / f"{scenario.name}.mp4"
            synth.write_flight_video(video, world, truth)
            _save(out / f"{scenario.name}_world.jpg", world)
            rng = np.random.default_rng(0)
            middle = len(truth.poses) // 3
            cv2.imwrite(str(out / f"{scenario.name}_frame.jpg"), synth.render_frame(world, truth.poses[middle], scenario.plan, middle, rng))
            log(f"\n{scenario.name}: {len(truth.poses)} frames, {scenario.description}")

            for label, config in variants(scenario.plan):
                result = stitching.stitch_video(video, config)
                score = evaluate.evaluate(result.transforms, result.frame_indices, truth, result.panorama, world)
                row = {
                    "scenario": scenario.name,
                    "variant": label,
                    "keyframes": result.keyframes,
                    "loop_closures": result.loop_closures,
                    "rmse_px": round(score.rmse_px, 2),
                    "max_px": round(score.max_px, 2),
                    "zncc": round(score.zncc, 3) if score.zncc is not None else None,
                    "seconds": round(sum(result.timings.values()), 1),
                }
                rows.append(row)
                log(f"  {label:<48} RMSE {row['rmse_px']:>6.2f} px  max {row['max_px']:>6.2f} px  ZNCC {row['zncc']}  {row['seconds']}s")
                if label.startswith("sequential"):
                    _save(out / f"{scenario.name}_sequential.jpg", result.panorama)
                if label == variants(scenario.plan)[-1][0] or (not scenario.plan.distortion and label.startswith("+ gain")):
                    _save(out / f"{scenario.name}_panorama.jpg", result.panorama)
                    _save(out / f"{scenario.name}_path.jpg", stitching.draw_flight_path(result))

            if include_stitcher and not scenario.plan.distortion:
                frames = list(stitching.iter_frames(video, every=15))
                start = time.perf_counter()
                try:
                    baseline = stitching.stitch_with_opencv(frames)
                    status = "finished"
                    _save(out / f"{scenario.name}_cv2_stitcher.jpg", baseline.panorama)
                except stitching.StitchError as error:
                    status = str(error)
                seconds = round(time.perf_counter() - start, 1)
                rows.append({"scenario": scenario.name, "variant": f"cv2.Stitcher SCANS ({len(frames)} frames)", "status": status, "seconds": seconds})
                log(f"  cv2.Stitcher: {status} in {seconds}s")

    (out / "benchmark.json").write_text(json.dumps(rows, indent=2))
    (out / "benchmark.md").write_text(_markdown(rows, seed))
    return rows


def _markdown(rows: list[dict], seed: int) -> str:
    lines = [
        "# Stitching benchmark",
        "",
        f"Generated by `python -m week02 benchmark docs/week02` (seed {seed}). Pose error is measured on a 5x5 grid of points",
        "in every keyframe after a least squares similarity alignment to the true poses; ZNCC compares the mosaic",
        "with the true world (1.0 is a perfect match).",
        "",
        "| Scenario | Variant | Keyframes | Loop closures | Pose RMSE (px) | Max error (px) | ZNCC | Time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "rmse_px" in row:
            lines.append(
                f"| {row['scenario']} | {row['variant']} | {row['keyframes']} | {row['loop_closures']} | {row['rmse_px']:.2f} | {row['max_px']:.2f} | {row['zncc']:.3f} | {row['seconds']} |"
            )
        else:
            lines.append(f"| {row['scenario']} | {row['variant']} | | | {row['status']} (no poses) | | | {row['seconds']} |")
    return "\n".join(lines) + "\n"
