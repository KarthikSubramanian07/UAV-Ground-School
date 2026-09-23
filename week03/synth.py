"""Synthetic polka dot scenes with exact ground truth.

Three hand annotated photos are enough to eyeball a detector, not to compare
six of them fairly. ``polka_scene`` renders unlimited test images where the
center and radius of every dot are known exactly, with the nuisances real
photos have:

* backgrounds: flat, woven fabric with folds, a lit wall, or white cards;
* dots from a random palette, anti aliased, from tiny to large;
* distractors in the same colors that are *not* dots (squares, triangles,
  stars, crosses, thin ellipses), so precision means something;
* uneven lighting, perspective tilt, defocus blur, sensor noise and JPEG.

Truth under perspective: a tilted circle becomes an ellipse, so each dot's
truth radius is ``r * sqrt(|det J|)`` where ``J`` is the Jacobian of the
homography at the dot center (the radius of the circle with the same area).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

SUPERSAMPLE = 4

PALETTES = {
    "bright": [(40, 40, 220), (60, 190, 60), (220, 170, 20), (200, 60, 150), (30, 200, 240), (20, 130, 250), (140, 50, 110), (30, 30, 30)],
    "pastel": [(235, 210, 170), (200, 180, 245), (190, 235, 200), (170, 215, 250), (225, 190, 225), (150, 160, 170)],
}


@dataclass(frozen=True)
class SceneConfig:
    width: int = 640
    height: int = 480
    dots: int = 40
    radius: tuple[float, float] = (5.0, 45.0)
    background: str = "flat"  # flat, fabric, wall, cards
    palette: str = "bright"
    distractors: int = 0
    lighting: float = 0.0  # strength of a smooth multiplicative light field, 0..0.6
    tilt: float = 0.0  # perspective strength, 0..0.35
    blur: float = 0.0  # Gaussian defocus sigma in pixels
    noise: float = 0.0  # sensor noise sigma in 8-bit levels
    jpeg: int | None = None  # JPEG quality, None for lossless
    seed: int = 0

    @staticmethod
    def preset(name: str, seed: int = 0) -> SceneConfig:
        presets = {
            "flat": SceneConfig(seed=seed),
            "fabric": SceneConfig(background="fabric", lighting=0.35, noise=4, jpeg=85, seed=seed),
            "pastel": SceneConfig(background="cards", palette="pastel", radius=(6, 40), noise=2, seed=seed),
            "distractors": SceneConfig(distractors=25, dots=30, noise=3, seed=seed),
            "tilted": SceneConfig(background="wall", tilt=0.3, blur=1.2, lighting=0.3, noise=3, jpeg=80, seed=seed),
            "tiny": SceneConfig(dots=120, radius=(2.5, 7.0), noise=3, seed=seed),
            "crowded": SceneConfig(dots=90, radius=(8, 30), background="fabric", noise=3, seed=seed),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; choose from {', '.join(presets)}")
        return presets[name]


PRESETS = ("flat", "fabric", "pastel", "distractors", "tilted", "tiny", "crowded")


@dataclass
class TruthDot:
    x: float
    y: float
    radius: float
    bgr: tuple[int, int, int]
    truncated: bool = False  # crosses the image border


@dataclass
class Scene:
    image: np.ndarray
    dots: list[TruthDot]
    distractors: list[tuple[float, float, float, str]] = field(default_factory=list)
    config: SceneConfig = field(default_factory=SceneConfig)


def _fixed(v: float | np.ndarray) -> np.ndarray:
    """Output pixel coordinate to a 1/16 pixel fixed point supersampled coordinate.

    Output pixel ``i`` covers supersampled pixels ``[s i, s i + s)`` so its
    center ``i`` sits at ``s (i + 0.5) - 0.5`` in the supersampled grid.
    """
    return np.round(((np.asarray(v, np.float64) + 0.5) * SUPERSAMPLE - 0.5) * 16).astype(np.int64)


def _background(cfg: SceneConfig, rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """Background image (float BGR 0..255) and the card color if cards are used."""
    if cfg.background == "flat":
        base = np.array(rng.choice([(245, 245, 245), (235, 240, 250), (60, 110, 200)]), np.float32)
        return np.ones((h, w, 3), np.float32) * base, None
    if cfg.background == "fabric":
        base = np.array(rng.choice([(40, 110, 210), (90, 60, 40), (200, 200, 205)]), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        period = 3.0 * SUPERSAMPLE
        weave = 0.06 * np.sin(2 * math.pi * xx / period) * np.sin(2 * math.pi * yy / period)
        grain = cv2.GaussianBlur(rng.normal(0, 0.05, (h, w)).astype(np.float32), (0, 0), SUPERSAMPLE * 0.7)
        return (np.ones((h, w, 3), np.float32) * base) * (1 + weave + grain)[:, :, None], None
    if cfg.background == "wall":
        base = np.array((150, 155, 160), np.float32)
        noise = cv2.GaussianBlur(rng.normal(0, 8, (h, w)).astype(np.float32), (0, 0), SUPERSAMPLE * 2)
        return np.ones((h, w, 3), np.float32) * base + noise[:, :, None], None
    if cfg.background == "cards":
        wall = np.ones((h, w, 3), np.float32) * np.array((150, 158, 165), np.float32)
        card = (245, 245, 242)
        cv2.rectangle(wall, (int(0.05 * w), int(0.05 * h)), (int(0.95 * w), int(0.95 * h)), card, -1)
        return wall, card
    raise ValueError("background must be flat, fabric, wall or cards")


def _star_points(cx: float, cy: float, r: float, angle: float, points: int = 5) -> np.ndarray:
    a = angle + np.arange(points * 2) * math.pi / points
    radii = np.where(np.arange(points * 2) % 2 == 0, r, 0.45 * r)
    return np.stack([cx + radii * np.cos(a), cy + radii * np.sin(a)], axis=1)


def _polygon(kind: str, cx: float, cy: float, r: float, angle: float) -> np.ndarray:
    if kind == "square":
        a = angle + np.arange(4) * math.pi / 2 + math.pi / 4
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    if kind == "triangle":
        a = angle + np.arange(3) * 2 * math.pi / 3
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    if kind == "star":
        return _star_points(cx, cy, r, angle)
    if kind == "cross":
        t = 0.33
        base = np.array([[-t, -1], [t, -1], [t, -t], [1, -t], [1, t], [t, t], [t, 1], [-t, 1], [-t, t], [-1, t], [-1, -t], [-t, -t]])
        c, s = math.cos(angle), math.sin(angle)
        rot = base @ np.array([[c, s], [-s, c]])
        return np.stack([cx + r * rot[:, 0], cy + r * rot[:, 1]], axis=1)
    raise ValueError(kind)


def polka_scene(cfg: SceneConfig | None = None) -> Scene:
    """Render one scene and its ground truth."""
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(cfg.seed)
    s = SUPERSAMPLE
    h, w = cfg.height * s, cfg.width * s
    canvas, card = _background(cfg, rng, h, w)
    palette = PALETTES[cfg.palette]
    margin = 0.08 if cfg.background == "cards" else 0.0

    placed: list[tuple[float, float, float]] = []

    def free(x: float, y: float, r: float, gap: float = 3.0) -> bool:
        return all(math.hypot(x - px, y - py) > r + pr + gap for px, py, pr in placed)

    dots: list[TruthDot] = []
    lo, hi = cfg.radius
    for _ in range(cfg.dots * 40):
        if len(dots) >= cfg.dots:
            break
        r = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        if card is not None:
            x = rng.uniform(margin * cfg.width + r + 2, (1 - margin) * cfg.width - r - 2)
            y = rng.uniform(margin * cfg.height + r + 2, (1 - margin) * cfg.height - r - 2)
        else:
            x = rng.uniform(-0.3 * r, cfg.width + 0.3 * r)
            y = rng.uniform(-0.3 * r, cfg.height + 0.3 * r)
        if not free(x, y, r):
            continue
        color = palette[int(rng.integers(len(palette)))]
        placed.append((x, y, r))
        truncated = bool(x - r < 0 or y - r < 0 or x + r > cfg.width - 1 or y + r > cfg.height - 1)
        dots.append(TruthDot(x, y, r, color, truncated))
        cv2.circle(canvas, (int(_fixed(x)), int(_fixed(y))), int(round(r * s * 16)), color, -1, cv2.LINE_AA, shift=4)

    distractors: list[tuple[float, float, float, str]] = []
    kinds = ["square", "triangle", "star", "cross", "ellipse"]
    for _ in range(cfg.distractors * 40):
        if len(distractors) >= cfg.distractors:
            break
        r = rng.uniform(max(lo, 8.0), hi)
        x = rng.uniform(r, cfg.width - r)
        y = rng.uniform(r, cfg.height - r)
        if not free(x, y, r, gap=4.0):
            continue
        kind = kinds[len(distractors) % len(kinds)]
        color = palette[int(rng.integers(len(palette)))]
        angle = rng.uniform(0, 2 * math.pi)
        placed.append((x, y, r))
        distractors.append((x, y, r, kind))
        if kind == "ellipse":
            axes = (int(round(r * s * 16)), int(round(r * s * 16 * rng.uniform(0.3, 0.55))))
            cv2.ellipse(canvas, (int(_fixed(x)), int(_fixed(y))), axes, math.degrees(angle), 0, 360, color, -1, cv2.LINE_AA, shift=4)
        else:
            pts = _fixed(_polygon(kind, x, y, r, angle)).astype(np.int32)
            cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA, shift=4)

    if cfg.background == "fabric":
        # Dots are printed on the fabric, so they carry its weave too.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        period = 3.0 * s
        weave = 0.05 * np.sin(2 * math.pi * xx / period) * np.sin(2 * math.pi * yy / period)
        canvas *= (1 + weave)[:, :, None]

    image = cv2.resize(canvas, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

    if cfg.lighting > 0:
        yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float32)
        field_ = np.zeros_like(xx)
        for _ in range(3):
            cx, cy = rng.uniform(0, cfg.width), rng.uniform(0, cfg.height)
            sigma = rng.uniform(0.3, 0.8) * max(cfg.width, cfg.height)
            field_ += rng.uniform(-1, 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        field_ = field_ / max(np.abs(field_).max(), 1e-6)
        image = image * (1 + cfg.lighting * field_)[:, :, None]

    if cfg.tilt > 0:
        # A quadrilateral inside the rendered image is stretched to fill the
        # frame, so every output pixel comes from real content (no borders to invent).
        t = cfg.tilt
        inset = rng.uniform(0, t, (4, 2)) * np.float32([cfg.width, cfg.height]) * 0.5
        src = np.float32(
            [
                [inset[0, 0], inset[0, 1]],
                [cfg.width - inset[1, 0], inset[1, 1] * 0.3],
                [cfg.width - inset[2, 0] * 0.3, cfg.height],
                [inset[3, 0] * 0.3, cfg.height - inset[3, 1] * 0.3],
            ]
        )
        dst = np.float32([[0, 0], [cfg.width, 0], [cfg.width, cfg.height], [0, cfg.height]])
        hmat = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, hmat, (cfg.width, cfg.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped: list[TruthDot] = []
        for d in dots:
            p = hmat @ np.array([d.x, d.y, 1.0])
            x, y = p[0] / p[2], p[1] / p[2]
            # Jacobian of the homography at the dot center.
            den = p[2]
            j = np.array(
                [
                    [hmat[0, 0] / den - p[0] * hmat[2, 0] / den**2, hmat[0, 1] / den - p[0] * hmat[2, 1] / den**2],
                    [hmat[1, 0] / den - p[1] * hmat[2, 0] / den**2, hmat[1, 1] / den - p[1] * hmat[2, 1] / den**2],
                ]
            )
            r = d.radius * math.sqrt(abs(np.linalg.det(j)))
            truncated = bool(x - r < 0 or y - r < 0 or x + r > cfg.width - 1 or y + r > cfg.height - 1)
            warped.append(TruthDot(float(x), float(y), float(r), d.bgr, truncated))
        dots = warped

    if cfg.blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), cfg.blur)
    if cfg.noise > 0:
        image = image + rng.normal(0, cfg.noise, image.shape).astype(np.float32)
    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    if cfg.jpeg:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg)])
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Dots that are mostly outside the frame are not visible enough to count.
    visible = [
        d
        for d in dots
        if -0.5 * d.radius <= d.x <= cfg.width - 1 + 0.5 * d.radius and -0.5 * d.radius <= d.y <= cfg.height - 1 + 0.5 * d.radius
    ]
    return Scene(image, visible, distractors, cfg)
