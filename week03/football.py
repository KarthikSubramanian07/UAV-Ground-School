"""The Option 2 dataset: football players, goalkeepers, referees and the ball.

The Roboflow notebook downloads ``football-players-detection`` with an API
key. The same Roboflow Universe dataset (CC BY 4.0, 372 frames at 1920x1080
from Bundesliga broadcast clips) is mirrored on Hugging Face, so
``fetch`` needs no account.

Two things about this dataset shape every experiment here:

* **The split leaks.** Frames are named ``<clip>_<n>_<m>``, and the random
  train / valid / test split puts frames of the same clip, often a second
  apart, on both sides. A model can score well by remembering the clip.
  ``clip_split`` holds out whole clips instead.
* **The ball is tiny.** It is about 11 px across at 1920x1080 and under 4 px after
  resizing to 640, which is why ``tile_dataset`` exists.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HF_REPO = "martinjolif/football-player-detection"
NAMES = ("ball", "goalkeeper", "player", "referee")
SPLITS = ("train", "valid", "test")

# Held out clips for the leakage free split (about 20% test, 10% validation by frames).
TEST_CLIPS = ("4b770a", "798b45")
VALID_CLIPS = ("573e61", "54745b")


@dataclass
class Sample:
    image: Path
    label: Path
    split: str

    @property
    def clip(self) -> str:
        return clip_id(self.image.name)


def clip_id(name: str) -> str:
    """``08fd33_3_6_png.rf.<hash>.jpg`` -> ``08fd33``."""
    match = re.match(r"([0-9a-f]{6})_", name)
    return match.group(1) if match else name.split("_")[0]


def fetch(root: str | Path) -> Path:
    """Download the dataset from Hugging Face into ``root``; returns the YOLO data directory."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(root))
    data = root / "data"
    for cache in data.glob("*/labels.cache"):
        cache.unlink()
    return data


def samples(data_dir: str | Path) -> list[Sample]:
    data_dir = Path(data_dir)
    out = []
    for split in SPLITS:
        for image in sorted((data_dir / split / "images").glob("*.jpg")):
            out.append(Sample(image, data_dir / split / "labels" / f"{image.stem}.txt", split))
    return out


def read_labels(path: str | Path) -> np.ndarray:
    """YOLO labels as an (n, 5) array of class, cx, cy, w, h (normalized)."""
    path = Path(path)
    if not path.is_file() or not path.read_text().strip():
        return np.zeros((0, 5), np.float64)
    rows = [list(map(float, line.split()[:5])) for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, np.float64).reshape(-1, 5)


def write_labels(path: str | Path, labels: np.ndarray) -> None:
    lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized cx, cy, w, h -> pixel x1, y1, x2, y2."""
    if not len(labels):
        return np.zeros((0, 4))
    cx, cy, w, h = labels[:, 1] * width, labels[:, 2] * height, labels[:, 3] * width, labels[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def from_xyxy(classes: np.ndarray, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if not len(boxes):
        return np.zeros((0, 5))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    w = (boxes[:, 2] - boxes[:, 0]) / width
    h = (boxes[:, 3] - boxes[:, 1]) / height
    return np.column_stack([classes, cx, cy, w, h])


def stats(data_dir: str | Path) -> dict:
    """Frames and boxes per split, class, and clip, plus the clip overlap between splits."""
    per_split: dict[str, Counter] = defaultdict(Counter)
    clips: dict[str, set[str]] = defaultdict(set)
    sizes: dict[str, list[float]] = defaultdict(list)
    frames = Counter()
    for s in samples(data_dir):
        frames[s.split] += 1
        clips[s.split].add(s.clip)
        labels = read_labels(s.label)
        for c, _, _, w, h in labels:
            per_split[s.split][NAMES[int(c)]] += 1
            sizes[NAMES[int(c)]].append(float(np.sqrt(w * 1920 * h * 1080)))
    return {
        "frames": dict(frames),
        "boxes": {k: dict(v) for k, v in per_split.items()},
        "clips": {k: sorted(v) for k, v in clips.items()},
        "test_clips_seen_in_train": sorted(clips["test"] & clips["train"]),
        "median_box_size_px": {k: round(float(np.median(v)), 1) for k, v in sizes.items()},
    }


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(out_dir: Path, names=NAMES) -> Path:
    text = f"path: {out_dir.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: {len(names)}\nnames: {list(names)}\n"
    path = out_dir / "data.yaml"
    path.write_text(text)
    return path


def original_split(data_dir: str | Path, out_dir: str | Path) -> Path:
    """The Roboflow split as is (symlinked), with a data.yaml that uses absolute paths."""
    out_dir = Path(out_dir)
    for s in samples(data_dir):
        _link(s.image, out_dir / s.split / "images" / s.image.name)
        _link(s.label, out_dir / s.split / "labels" / s.label.name)
    return write_yaml(out_dir)


def clip_split(data_dir: str | Path, out_dir: str | Path, test_clips=TEST_CLIPS, valid_clips=VALID_CLIPS) -> Path:
    """Re-split by clip, so no clip appears in more than one split."""
    out_dir = Path(out_dir)
    for s in samples(data_dir):
        split = "test" if s.clip in test_clips else "valid" if s.clip in valid_clips else "train"
        _link(s.image, out_dir / split / "images" / s.image.name)
        _link(s.label, out_dir / split / "labels" / s.label.name)
    return write_yaml(out_dir)


def tile_boxes(labels: np.ndarray, width: int, height: int, x0: int, y0: int, tw: int, th: int, min_visible: float = 0.4) -> np.ndarray:
    """Labels for one tile: boxes clipped to the tile, dropped if less than ``min_visible`` of their area remains."""
    boxes = to_xyxy(labels, width, height)
    if not len(boxes):
        return np.zeros((0, 5))
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x0, x0 + tw)
    clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y0, y0 + th)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept_area = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(clipped[:, 3] - clipped[:, 1], 0)
    keep = kept_area >= min_visible * np.maximum(area, 1e-9)
    clipped = clipped[keep] - np.array([x0, y0, x0, y0])
    return from_xyxy(labels[keep, 0], clipped, tw, th)


def tile_grid(width: int, height: int, tile: tuple[int, int], overlap: float = 0.2) -> list[tuple[int, int]]:
    """Top left corners of overlapping tiles that cover the whole image."""
    tw, th = tile
    corners = []

    def starts(size: int, t: int) -> list[int]:
        if t >= size:
            return [0]
        step = max(1, int(t * (1 - overlap)))
        out = list(range(0, size - t + 1, step))
        if out[-1] != size - t:
            out.append(size - t)
        return out

    for y in starts(height, th):
        for x in starts(width, tw):
            corners.append((x, y))
    return corners


def tile_dataset(
    split_dir: str | Path, out_dir: str | Path, tile: tuple[int, int] = (960, 540), overlap: float = 0.2, splits=("train",)
) -> Path:
    """Cut every image of the given splits into overlapping tiles (Roboflow's "Tile" preprocessing).

    Splits not listed are linked unchanged, so validation still runs on
    whole frames.
    """
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    for split in SPLITS:
        images = sorted((split_dir / split / "images").glob("*.jpg"))
        for image_path in images:
            label_path = split_dir / split / "labels" / f"{image_path.stem}.txt"
            if split not in splits:
                _link(image_path, out_dir / split / "images" / image_path.name)
                _link(label_path, out_dir / split / "labels" / label_path.name)
                continue
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            labels = read_labels(label_path)
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            for x0, y0 in tile_grid(w, h, tile, overlap):
                name = f"{image_path.stem}_t{x0}_{y0}"
                crop = image[y0 : y0 + tile[1], x0 : x0 + tile[0]]
                cv2.imwrite(str(out_dir / split / "images" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_labels(out_dir / split / "labels" / f"{name}.txt", tile_boxes(labels, w, h, x0, y0, crop.shape[1], crop.shape[0]))
    return write_yaml(out_dir)


def summary_json(data_dir: str | Path) -> str:
    return json.dumps(stats(data_dir), indent=2)
