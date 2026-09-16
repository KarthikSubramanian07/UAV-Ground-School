"""Parity checks between the Python pipeline and the C++17 port in week02/cpp.

Skipped unless WEEK02_CPP_BUILD points at a build directory that contains the
``color_me_impressed`` and ``needin_stitches`` executables, for example::

    cmake -S week02/cpp -B build/cpp && cmake --build build/cpp -j
    WEEK02_CPP_BUILD=build/cpp python -m pytest tests/test_cpp_parity.py -v
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import pytest

from week02 import colors, evaluate, synth

BUILD = os.environ.get("WEEK02_CPP_BUILD")


def _binary(name: str) -> Path | None:
    if not BUILD:
        return None
    for candidate in (Path(BUILD) / name, Path(BUILD) / "Release" / name, Path(BUILD) / f"{name}.exe"):
        if candidate.is_file():
            return candidate.resolve()
    return None


COLOR_BIN = _binary("color_me_impressed")
STITCH_BIN = _binary("needin_stitches")

pytestmark = pytest.mark.skipif(
    COLOR_BIN is None or STITCH_BIN is None,
    reason="set WEEK02_CPP_BUILD to a build directory containing color_me_impressed and needin_stitches",
)


def _run(args: list, timeout: float) -> subprocess.CompletedProcess:
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"{args[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    return result


@pytest.mark.parametrize("name", ["suas_targets", "stop_sign"])
def test_colors_match_python(tmp_path: Path, name: str) -> None:
    image = synth.suas_targets()[0] if name == "suas_targets" else synth.stop_sign()
    # PNG is lossless, so both implementations see exactly the same pixels.
    path = tmp_path / f"{name}.png"
    assert cv2.imwrite(str(path), image)
    image = cv2.imread(str(path))

    python_layers = colors.split_colors(image, mode="named")
    report = tmp_path / f"{name}.json"
    _run([COLOR_BIN, path, "--json", report, "--no-show"], timeout=120)
    cpp_layers = {layer["name"]: layer for layer in json.loads(report.read_text())}

    assert {layer.name for layer in python_layers} == set(cpp_layers)
    shaped = 0
    for layer in python_layers:
        for obj in layer.objects:
            if obj.shape is None:
                continue
            shaped += 1
            matches = [
                other
                for other in cpp_layers[layer.name]["objects"]
                if math.dist(other["center"], obj.center) <= 1.5 and other["shape"] == obj.shape
            ]
            assert matches, f"no C++ {layer.name} {obj.shape} near {obj.center}"
    assert shaped > 0


def test_stitching_accuracy(tmp_path: Path) -> None:
    world = synth.voxel_world(width_blocks=260, height_blocks=150, seed=1)
    plan = synth.FlightPlan(frame_width=320, frame_height=180, passes=2, speed=5.0, seed=1)
    truth = synth.simulate_flight(world, plan)
    video = synth.write_flight_video(tmp_path / "flight.mp4", world, truth)

    transforms_csv = tmp_path / "transforms.csv"
    _run(
        [STITCH_BIN, video, "--every", "4", "--out", tmp_path / "panorama.jpg", "--transforms", transforms_csv, "--no-show"],
        timeout=600,
    )
    transforms, indices = evaluate.load_transforms(transforms_csv)
    assert len(transforms) >= 5
    result = evaluate.evaluate(transforms, indices, truth)
    assert result.rmse_px < 2.0, result
