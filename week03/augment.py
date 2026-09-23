"""Roboflow style offline augmentation, with boxes that follow the pixels.

Roboflow's "Generate version" step applies preprocessing once, then writes
N augmented copies of every training image. Its catalogue is reproduced here
so it can be tried without an account: flips, 90 degree rotations, crop,
rotation, shear, grayscale, hue, saturation, brightness, exposure, blur,
noise and cutout. Geometric augmentations warp the four corners of every box
and take their bounding rectangle, then clip to the image; boxes that lose
more than ``min_visible`` of their area are dropped, as Roboflow does.

Ultralytics already augments online (mosaic, HSV jitter, flips, scale and
translation), so offline augmentation is an addition, not a replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import football


@dataclass(frozen=True)
class Recipe:
    """Probability or strength of each augmentation (0 disables it)."""

    flip_horizontal: float = 0.5
    flip_vertical: float = 0.0
    rotate90: float = 0.0
    crop: float = 0.0  # maximum zoom in, as a fraction of the side removed (Roboflow "Crop: 0% to N%")
    rotation: float = 0.0  # degrees, uniform in [-rotation, rotation]
    shear: float = 0.0  # degrees, horizontal and vertical
    grayscale: float = 0.0  # probability
    hue: float = 0.0  # degrees on a 360 wheel
    saturation: float = 0.0  # fraction, +-
    brightness: float = 0.0  # fraction, +-
    exposure: float = 0.0  # fraction, +- (a gamma change)
    blur: float = 0.0  # maximum Gaussian sigma in pixels
    noise: float = 0.0  # fraction of pixels replaced by salt and pepper noise
    cutout: int = 0  # number of 10% boxes blacked out
    min_visible: float = 0.4

    @staticmethod
    def preset(name: str) -> Recipe:
        presets = {
            "none": Recipe(flip_horizontal=0.0),
            # What the Roboflow football tutorial versions typically use, tuned for broadcast footage:
            # no vertical flips or 90 degree turns (the pitch has a sky side), mild photometric changes.
            "broadcast": Recipe(
                flip_horizontal=0.5, crop=0.2, rotation=5, hue=10, saturation=0.25, brightness=0.2, exposure=0.15, blur=1.0, noise=0.005
            ),
            "heavy": Recipe(
                flip_horizontal=0.5,
                flip_vertical=0.2,
                crop=0.3,
                rotation=12,
                shear=8,
                grayscale=0.1,
                hue=20,
                saturation=0.4,
                brightness=0.3,
                exposure=0.25,
                blur=2.0,
                noise=0.02,
                cutout=3,
            ),
        }
        if name not in presets:
            raise ValueError(f"unknown recipe {name!r}; choose from {', '.join(presets)}")
        return presets[name]


def warp_boxes(boxes: np.ndarray, matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform x1 y1 x2 y2 boxes by a 2x3 or 3x3 matrix; returns clipped boxes and their visible fraction."""
    if not len(boxes):
        return boxes.reshape(0, 4), np.zeros(0)
    m = np.vstack([matrix, [0, 0, 1]]) if matrix.shape == (2, 3) else matrix
    corners = np.stack(
        [
            np.stack([boxes[:, 0], boxes[:, 1]], 1),
            np.stack([boxes[:, 2], boxes[:, 1]], 1),
            np.stack([boxes[:, 2], boxes[:, 3]], 1),
            np.stack([boxes[:, 0], boxes[:, 3]], 1),
        ],
        axis=1,
    )  # (n, 4, 2)
    homogeneous = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], axis=2) @ m.T
    points = homogeneous[..., :2] / homogeneous[..., 2:3]
    warped = np.concatenate([points.min(axis=1), points.max(axis=1)], axis=1)
    clipped = warped.copy()
    clipped[:, [0, 2]] = np.clip(warped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(warped[:, [1, 3]], 0, height)
    full = np.maximum((warped[:, 2] - warped[:, 0]) * (warped[:, 3] - warped[:, 1]), 1e-9)
    kept = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(clipped[:, 3] - clipped[:, 1], 0)
    return clipped, kept / full


def _geometric(image: np.ndarray, rng: np.random.Generator, r: Recipe) -> np.ndarray:
    """A single 3x3 matrix combining every geometric augmentation."""
    h, w = image.shape[:2]
    m = np.eye(3)

    def then(t: np.ndarray) -> None:
        nonlocal m
        m = t @ m

    center = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)
    back = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], float)
    if rng.random() < r.flip_horizontal:
        then(np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], float))
    if rng.random() < r.flip_vertical:
        then(np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], float))
    if r.rotation or r.shear or r.crop:
        a = math.radians(rng.uniform(-r.rotation, r.rotation)) if r.rotation else 0.0
        sx = math.tan(math.radians(rng.uniform(-r.shear, r.shear))) if r.shear else 0.0
        sy = math.tan(math.radians(rng.uniform(-r.shear, r.shear))) if r.shear else 0.0
        zoom = 1.0 / (1.0 - rng.uniform(0, r.crop)) if r.crop else 1.0
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        shear = np.array([[1, sx, 0], [sy, 1, 0], [0, 0, 1]])
        scale = np.diag([zoom, zoom, 1.0])
        shift = np.eye(3)
        if zoom > 1:
            # Crop somewhere in the frame, not always the middle.
            free_x, free_y = (zoom - 1) * w / 2, (zoom - 1) * h / 2
            shift = np.array([[1, 0, rng.uniform(-free_x, free_x)], [0, 1, rng.uniform(-free_y, free_y)], [0, 0, 1]])
        then(back @ shift @ scale @ shear @ rot @ center)
    return m


def apply(image: np.ndarray, labels: np.ndarray, recipe: Recipe, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Augment one image and its YOLO labels."""
    h, w = image.shape[:2]
    out = image
    boxes = football.to_xyxy(labels, w, h)
    classes = labels[:, 0] if len(labels) else np.zeros(0)

    if recipe.rotate90 and rng.random() < recipe.rotate90:
        k = int(rng.integers(1, 4))
        out = np.ascontiguousarray(np.rot90(out, k))
        for _ in range(k):  # counterclockwise quarter turns: (x, y) -> (y, W - x)
            ch, cw = h, w
            boxes = np.stack([boxes[:, 1], cw - boxes[:, 2], boxes[:, 3], cw - boxes[:, 0]], axis=1) if len(boxes) else boxes
            h, w = cw, ch

    m = _geometric(out, rng, recipe)
    if not np.allclose(m, np.eye(3)):
        out = cv2.warpPerspective(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        boxes, visible = warp_boxes(boxes, m, w, h)
        keep = visible >= recipe.min_visible
        boxes, classes = boxes[keep], classes[keep]

    out = out.astype(np.float32)
    if recipe.hue or recipe.saturation:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if recipe.hue:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-recipe.hue, recipe.hue) / 2) % 180
        if recipe.saturation:
            hsv[..., 1] *= 1 + rng.uniform(-recipe.saturation, recipe.saturation)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if recipe.brightness:
        out *= 1 + rng.uniform(-recipe.brightness, recipe.brightness)
    if recipe.exposure:
        gamma = 1 + rng.uniform(-recipe.exposure, recipe.exposure)
        out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** gamma
    if recipe.grayscale and rng.random() < recipe.grayscale:
        gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if recipe.blur:
        sigma = rng.uniform(0, recipe.blur)
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
    if recipe.noise:
        mask = rng.random((h, w)) < recipe.noise
        out[mask] = rng.choice([0.0, 255.0], size=(int(mask.sum()), 1))
    for _ in range(recipe.cutout):
        cw, ch = int(0.1 * w), int(0.1 * h)
        x0, y0 = int(rng.integers(0, w - cw)), int(rng.integers(0, h - ch))
        out[y0 : y0 + ch, x0 : x0 + cw] = 0
    image_out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return image_out, football.from_xyxy(classes, boxes, w, h)


def augment_dataset(
    split_dir: str | Path, out_dir: str | Path, recipe: Recipe, copies: int = 2, seed: int = 0, keep_original: bool = True
) -> Path:
    """Write ``copies`` augmented versions of every training image (plus the original)."""
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    rng = np.random.default_rng(seed)
    for split in football.SPLITS:
        images = sorted((split_dir / split / "images").glob("*.jpg"))
        for image_path in images:
            label_path = split_dir / split / "labels" / f"{image_path.stem}.txt"
            if split != "train":
                football._link(image_path, out_dir / split / "images" / image_path.name)
                football._link(label_path, out_dir / split / "labels" / label_path.name)
                continue
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            if keep_original:
                football._link(image_path, out_dir / split / "images" / image_path.name)
                football._link(label_path, out_dir / split / "labels" / label_path.name)
            image = cv2.imread(str(image_path))
            labels = football.read_labels(label_path)
            for k in range(copies):
                aug, aug_labels = apply(image, labels, recipe, rng)
                name = f"{image_path.stem}_aug{k}"
                cv2.imwrite(str(out_dir / split / "images" / f"{name}.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
                football.write_labels(out_dir / split / "labels" / f"{name}.txt", aug_labels)
    return football.write_yaml(out_dir)
