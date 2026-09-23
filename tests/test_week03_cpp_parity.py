"""Parity checks between the week 03 Python pipeline and the C++17 port in week03/cpp.

Skipped unless WEEK03_CPP_BUILD points at a build directory that contains the
``polka_dots`` and ``game_pieces`` executables, for example::

    cmake -S week03/cpp -B build/cpp3 && cmake --build build/cpp3 -j
    WEEK03_CPP_BUILD=build/cpp3 python -m pytest tests/test_week03_cpp_parity.py -v

Every input is written as PNG first, so both implementations see exactly the
same pixels (JPEG decoders can differ between OpenCV builds).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from week03 import dots, evaluate, objects, synth
from week03.blobs import Blob

BUILD = os.environ.get("WEEK03_CPP_BUILD")
DATA = Path(__file__).resolve().parents[1] / "docs" / "week03" / "photos"


def _binary(name: str) -> Path | None:
    if not BUILD:
        return None
    for candidate in (Path(BUILD) / name, Path(BUILD) / "Release" / name, Path(BUILD) / f"{name}.exe"):
        if candidate.is_file():
            return candidate.resolve()
    return None


DOTS_BIN = _binary("polka_dots")
PIECES_BIN = _binary("game_pieces")

pytestmark = pytest.mark.skipif(
    DOTS_BIN is None or PIECES_BIN is None,
    reason="set WEEK03_CPP_BUILD to a build directory containing polka_dots and game_pieces",
)

METHODS = ["log", "doh", "contrast", "gray"]
SCENES = [("flat", 0), ("fabric", 1), ("pastel", 2), ("distractors", 3), ("tilted", 4), ("crowded", 5)]
PHOTOS = ["polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg"]


def _run(args: list, timeout: float) -> subprocess.CompletedProcess:
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"{args[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    return result


def _as_png(image: np.ndarray, path: Path) -> np.ndarray:
    """Write losslessly and read back, so Python and C++ decode identical pixels."""
    assert cv2.imwrite(str(path), image)
    return cv2.imread(str(path))


def _cpp_dots(path: Path, method: str, tmp_path: Path) -> list[Blob]:
    out = tmp_path / f"{path.stem}_{method}.json"
    _run([DOTS_BIN, path, "--method", method, "--json", out, "--no-show"], timeout=600)
    data = json.loads(out.read_text())
    assert data["method"] == method and data["count"] == len(data["dots"])
    return [Blob(d["x"], d["y"], d["radius"], color=d["color"]) for d in data["dots"]]


def _assert_dot_parity(python: list[Blob], cpp: list[Blob]) -> None:
    """C++ dots scored against Python's as truth: F1, center agreement and color names."""
    if not python and not cpp:
        return
    score = evaluate.match_dots(cpp, python)
    assert score.f1 >= 0.97, f"{score.summary()} ({len(python)} Python dots, {len(cpp)} C++ dots)"
    moved = [math.hypot(cpp[i].x - python[j].x, cpp[i].y - python[j].y) for i, j in score.matches]
    assert statistics.median(moved) < 0.1, f"median center difference {statistics.median(moved):.4f} px"
    same = sum(cpp[i].color == python[j].color for i, j in score.matches)
    assert same >= 0.95 * len(score.matches), f"{same} of {len(score.matches)} color names agree"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("preset", "seed"), SCENES)
def test_dots_match_python_on_synthetic_scenes(tmp_path: Path, preset: str, seed: int, method: str) -> None:
    scene = synth.polka_scene(synth.SceneConfig.preset(preset, seed=seed))
    path = tmp_path / f"{preset}.png"
    image = _as_png(scene.image, path)
    python = dots.find_dots(image, method=method).dots
    if method in ("log", "doh"):
        assert python, "the scene should contain dots"
    _assert_dot_parity(python, _cpp_dots(path, method, tmp_path))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", PHOTOS)
def test_dots_match_python_on_photos(tmp_path: Path, name: str, method: str) -> None:
    if not (DATA / name).is_file():
        pytest.skip(f"{DATA / name} not present")
    path = tmp_path / f"{Path(name).stem}.png"
    image = _as_png(cv2.imread(str(DATA / name)), path)
    python = dots.find_dots(image, method=method).dots
    _assert_dot_parity(python, _cpp_dots(path, method, tmp_path))


# ------------------------------------------------------------ game pieces ----
# Scene helpers copied from tests/test_week03_objects.py.


def cone(image, x, y, s, color, angle=0.0):
    """A traffic cone: a tapered body on a square flange, optionally rotated about (x, y)."""
    body = np.array([[-0.18, -1.0], [0.18, -1.0], [0.42, 0.55], [-0.42, 0.55]])
    base = np.array([[-0.62, 0.5], [0.62, 0.5], [0.62, 0.75], [-0.62, 0.75]])
    c, si = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -si], [si, c]])
    for poly in (body, base):
        pts = (poly @ rot.T) * s + [x, y]
        cv2.fillPoly(image, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)


def cube(image, x, y, s, color):
    """A cube seen from above a corner: a hexagon."""
    a = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    pts = np.stack([x + s * np.cos(a), y + s * np.sin(a)], axis=1)
    cv2.fillPoly(image, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)


def ring(image, x, y, r, color, thickness):
    cv2.ellipse(image, (x, y), (r, int(r * 0.8)), 10, 0, 360, color, thickness, cv2.LINE_AA)


def pieces_scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = (np.full((600, 900, 3), 185, np.float32) + rng.normal(0, 6, (600, 900, 1))).clip(0, 255).astype(np.uint8)
    ring(image, 140, 140, 90, (40, 60, 235), 30)
    cube(image, 420, 130, 80, (190, 60, 110))
    cube(image, 440, 265, 80, (190, 60, 110))  # touching the first cube
    cone(image, 700, 170, 120, (20, 200, 245))
    cone(image, 200, 430, 120, (20, 200, 245), angle=1.3)  # lying on its side
    ring(image, 650, 460, 100, (40, 60, 235), 34)
    return cv2.GaussianBlur(image, (3, 3), 0)


def _assert_piece_parity(image: np.ndarray, path: Path, tmp_path: Path) -> None:
    python = objects.detect_objects(image).pieces
    out = tmp_path / f"{path.stem}_pieces.json"
    _run([PIECES_BIN, path, "--json", out, "--no-show"], timeout=300)
    cpp = json.loads(out.read_text())
    assert sorted(p.label for p in python) == sorted(p["label"] for p in cpp)
    unused = list(range(len(cpp)))
    for piece in python:
        ious = [(evaluate.box_iou(piece.bbox, cpp[k]["bbox"]), k) for k in unused if cpp[k]["label"] == piece.label]
        assert ious, f"no C++ {piece.label}"
        iou, k = max(ious)
        assert iou >= 0.9, f"{piece.label} at {piece.bbox}: best C++ box {cpp[k]['bbox']} has IoU {iou:.3f}"
        unused.remove(k)


@pytest.mark.parametrize("variant", ["upright", "rotated", "hue_shifted"])
def test_pieces_match_python_on_synthetic_scene(tmp_path: Path, variant: str) -> None:
    image = pieces_scene()
    if variant == "rotated":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif variant == "hue_shifted":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    path = tmp_path / f"pieces_{variant}.png"
    image = _as_png(image, path)
    _assert_piece_parity(image, path, tmp_path)


def test_pieces_match_python_on_photo(tmp_path: Path) -> None:
    if not (DATA / "objects.jpg").is_file():
        pytest.skip(f"{DATA / 'objects.jpg'} not present")
    path = tmp_path / "objects.png"
    image = _as_png(cv2.imread(str(DATA / "objects.jpg")), path)
    _assert_piece_parity(image, path, tmp_path)
