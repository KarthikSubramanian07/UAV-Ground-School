"""Procedural test data with exact ground truth.

The club's flight video is not public, so this module builds everything the
assignments need from scratch and deterministically (seeded):

* photographic style test images for Color Me Impressed (stop sign, apple,
  SUAS style ground targets with known centers and shapes);
* a Minecraft style voxel world seen from above;
* a lawnmower survey flight over that world rendered as a real video, with
  exposure drift, vignetting, altitude wobble and sensor noise, together with
  the true camera pose of every frame.

Because the true poses are known, stitching quality can be measured in
pixels instead of eyeballed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------ utilities ----


def fractal_noise(height: int, width: int, rng: np.random.Generator, octaves: int = 5, base: int = 4) -> np.ndarray:
    """Smooth value noise in [0, 1] made of upsampled random grids."""
    total = np.zeros((height, width), np.float32)
    amplitude, norm = 1.0, 0.0
    for octave in range(octaves):
        cells = base * 2**octave
        grid = rng.random((max(2, height * cells // max(height, width) + 2), cells + 2)).astype(np.float32)
        layer = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        total += amplitude * layer
        norm += amplitude
        amplitude *= 0.5
    total /= norm
    total -= total.min()
    return total / max(float(total.max()), 1e-6)


def photo_finish(image: np.ndarray, rng: np.random.Generator, noise: float = 4.0, jpeg_quality: int = 88) -> np.ndarray:
    """Make a clean render look like a camera photo: blur, noise, JPEG artifacts."""
    out = cv2.GaussianBlur(image, (3, 3), 0.8).astype(np.float32)
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def regular_polygon(center: tuple[float, float], radius: float, sides: int, rotation: float = 0.0) -> np.ndarray:
    angles = rotation + np.arange(sides) * 2 * math.pi / sides
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)


def star_polygon(center: tuple[float, float], radius: float, points: int = 5, rotation: float = 0.0, inner: float = 0.45) -> np.ndarray:
    angles = rotation + np.arange(points * 2) * math.pi / points
    radii = np.where(np.arange(points * 2) % 2 == 0, radius, radius * inner)
    return np.stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)], axis=1)


def cross_polygon(center: tuple[float, float], radius: float, rotation: float = 0.0, arm: float = 0.36) -> np.ndarray:
    a, r = radius * arm, radius
    pts = np.array([[-a, -r], [a, -r], [a, -a], [r, -a], [r, a], [a, a], [a, r], [-a, r], [-a, a], [-r, a], [-r, -a], [-a, -a]])
    return _rotate(pts, rotation) + np.array(center)


def semicircle_polygon(center: tuple[float, float], radius: float, rotation: float = 0.0) -> np.ndarray:
    angles = np.linspace(0, math.pi, 64)
    pts = np.stack([radius * np.cos(angles), -radius * np.sin(angles) + radius * 0.42], axis=1)
    return _rotate(pts, rotation) + np.array(center)


def quarter_circle_polygon(center: tuple[float, float], radius: float, rotation: float = 0.0) -> np.ndarray:
    angles = np.linspace(0, math.pi / 2, 48)
    arc = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    pts = np.vstack([[0.0, 0.0], arc]) - radius * 0.42
    return _rotate(pts, rotation) + np.array(center)


def rectangle_polygon(center: tuple[float, float], radius: float, rotation: float = 0.0, aspect: float = 0.55) -> np.ndarray:
    pts = np.array([[-radius, -radius * aspect], [radius, -radius * aspect], [radius, radius * aspect], [-radius, radius * aspect]])
    return _rotate(pts, rotation) + np.array(center)


def trapezoid_polygon(center: tuple[float, float], radius: float, rotation: float = 0.0) -> np.ndarray:
    pts = np.array([[-radius * 0.5, -radius * 0.6], [radius * 0.5, -radius * 0.6], [radius, radius * 0.6], [-radius, radius * 0.6]])
    return _rotate(pts, rotation) + np.array(center)


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


SHAPE_BUILDERS = {
    "circle": lambda c, r, rot: regular_polygon(c, r, 96, rot),
    "semicircle": semicircle_polygon,
    "quarter circle": quarter_circle_polygon,
    "triangle": lambda c, r, rot: regular_polygon(c, r, 3, rot),
    "square": lambda c, r, rot: regular_polygon(c, r, 4, rot),
    "rectangle": rectangle_polygon,
    "trapezoid": trapezoid_polygon,
    "pentagon": lambda c, r, rot: regular_polygon(c, r, 5, rot),
    "hexagon": lambda c, r, rot: regular_polygon(c, r, 6, rot),
    "heptagon": lambda c, r, rot: regular_polygon(c, r, 7, rot),
    "octagon": lambda c, r, rot: regular_polygon(c, r, 8, rot),
    "star": lambda c, r, rot: star_polygon(c, r, 5, rot),
    "cross": cross_polygon,
}


def fill_polygon(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    """Anti-aliased filled polygon with 1/16 pixel precision."""
    shift = 4
    pts = np.round(points * (1 << shift)).astype(np.int32)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA, shift=shift)


def draw_centered_text(image: np.ndarray, text: str, center: tuple[float, float], height: float, color, thickness_ratio: float = 0.16) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = max(1, int(round(height * thickness_ratio)))
    (w, h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = height / h
    thickness = max(1, int(round(height * thickness_ratio)))
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(round(center[0] - w / 2)), int(round(center[1] + h / 2)))
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------- color test images ----


@dataclass
class TargetTruth:
    shape: str
    color: str
    letter: str
    letter_color: str
    center: tuple[float, float]
    radius: float
    rotation: float


def stop_sign(width: int = 900, height: int = 675, seed: int = 7) -> np.ndarray:
    """A stop sign against sky and grass."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    sky_top, sky_bottom = np.array([205, 140, 60], np.float32), np.array([240, 200, 140], np.float32)
    image[:] = np.clip(sky_top * (1 - t[..., None]) + sky_bottom * t[..., None], 0, 255).astype(np.uint8)[:, :1, :]
    horizon = int(height * 0.74)
    grass_noise = fractal_noise(height - horizon, width, rng, octaves=6, base=8)
    grass = np.stack([30 + 25 * grass_noise, 130 + 50 * grass_noise, 45 + 25 * grass_noise], axis=2)
    image[horizon:] = grass.astype(np.uint8)

    cx, cy, r = width * 0.5, height * 0.40, height * 0.28
    cv2.rectangle(image, (int(cx - r * 0.07), int(cy)), (int(cx + r * 0.07), height), (125, 125, 128), -1)
    rotation = math.pi / 8
    fill_polygon(image, regular_polygon((cx, cy), r, 8, rotation), (245, 245, 245))
    fill_polygon(image, regular_polygon((cx, cy), r * 0.92, 8, rotation), (38, 30, 205))
    draw_centered_text(image, "STOP", (cx, cy), r * 0.42, (250, 250, 250), 0.13)
    return photo_finish(image, rng)


def apple(width: int = 800, height: int = 800, seed: int = 11) -> np.ndarray:
    """A shaded red apple with a stem and a leaf on a pale backdrop."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float32)
    backdrop = 238 - 18 * (yy / height)
    image = np.repeat(backdrop[..., None], 3, axis=2).astype(np.uint8)

    cx, cy, r = width * 0.5, height * 0.56, min(width, height) * 0.30
    # Two overlapping lobes give the apple its shape.
    body = np.zeros((height, width), np.uint8)
    cv2.ellipse(body, (int(cx - r * 0.33), int(cy)), (int(r * 0.78), int(r * 0.92)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(body, (int(cx + r * 0.33), int(cy)), (int(r * 0.78), int(r * 0.92)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    # Soft contact shadow.
    shadow = np.zeros((height, width), np.float32)
    cv2.ellipse(shadow, (int(cx), int(cy + r * 0.95)), (int(r * 1.0), int(r * 0.16)), 0, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), r * 0.08)
    image = (image.astype(np.float32) * (1 - 0.18 * shadow[..., None])).astype(np.uint8)

    light = np.clip(1.0 - np.hypot(xx - (cx - r * 0.35), yy - (cy - r * 0.4)) / (r * 2.6), 0.55, 1.0)
    red = np.stack([30 * light, 25 * light, 215 * light], axis=2)
    alpha = (body.astype(np.float32) / 255)[..., None]
    image = (image * (1 - alpha) + red * alpha).astype(np.uint8)
    cv2.ellipse(image, (int(cx - r * 0.35), int(cy - r * 0.42)), (int(r * 0.16), int(r * 0.09)), -30, 0, 360, (150, 150, 250), -1, cv2.LINE_AA)

    cv2.line(image, (int(cx), int(cy - r * 0.78)), (int(cx + r * 0.08), int(cy - r * 1.18)), (25, 60, 105), max(3, int(r * 0.07)), cv2.LINE_AA)
    leaf = np.array([[cx + r * 0.1, cy - r * 1.02], [cx + r * 0.45, cy - r * 1.28], [cx + r * 0.78, cy - r * 1.12], [cx + r * 0.45, cy - r * 0.94]])
    fill_polygon(image, leaf, (40, 165, 55))
    return photo_finish(image, rng, noise=3.0)


DEFAULT_TARGETS = (
    ("octagon", (38, 30, 205), "red", "R", (250, 250, 250), "white"),
    ("triangle", (200, 90, 20), "blue", "A", (30, 220, 240), "yellow"),
    ("star", (30, 215, 235), "yellow", "U", (190, 50, 130), "purple"),
    ("circle", (170, 40, 150), "purple", "V", (250, 250, 250), "white"),
    ("cross", (30, 140, 250), "orange", "S", (200, 90, 20), "blue"),
)


def suas_targets(width: int = 1280, height: int = 800, seed: int = 3) -> tuple[np.ndarray, list[TargetTruth]]:
    """SUAS style ground targets (colored shape with a letter) on a grass field."""
    rng = np.random.default_rng(seed)
    n1 = fractal_noise(height, width, rng, octaves=7, base=6)
    n2 = fractal_noise(height, width, rng, octaves=3, base=3)
    image = np.stack([40 + 40 * n1 + 10 * n2, 110 + 60 * n1 + 20 * n2, 55 + 30 * n1], axis=2).astype(np.uint8)
    # A dirt path through the field for extra realism.
    path = np.zeros((height, width), np.float32)
    xs = np.arange(width)
    ys = (height * (0.62 + 0.12 * np.sin(xs / width * 5.0))).astype(np.int32)
    cv2.polylines(path, [np.stack([xs, ys], axis=1).astype(np.int32)], False, 1.0, int(height * 0.06))
    path = cv2.GaussianBlur(path, (0, 0), 6)[..., None]
    dirt = np.array([70, 100, 125], np.float32) + 25 * n1[..., None]
    image = (image * (1 - path) + dirt * path).astype(np.uint8)

    slots = [(0.16, 0.28), (0.5, 0.25), (0.84, 0.30), (0.30, 0.78), (0.72, 0.80)]
    truths = []
    for (shape, color, color_name, letter, letter_bgr, letter_name), (fx, fy) in zip(DEFAULT_TARGETS, slots):
        center = (width * fx + rng.uniform(-20, 20), height * fy + rng.uniform(-15, 15))
        radius = height * rng.uniform(0.11, 0.13)
        rotation = rng.uniform(0, 2 * math.pi) if shape not in ("circle",) else 0.0
        fill_polygon(image, SHAPE_BUILDERS[shape](center, radius, rotation), color)
        draw_centered_text(image, letter, center, radius * 0.62, letter_bgr, 0.18)
        truths.append(TargetTruth(shape, color_name, letter, letter_name, center, radius, rotation))
    return photo_finish(image, rng, noise=3.5), truths


# ------------------------------------------------------- voxel world -------

WATER_DEEP = (150, 70, 35)
WATER = (205, 110, 55)
SAND = (150, 205, 220)
GRASS = (60, 150, 85)
FOREST_FLOOR = (45, 115, 60)
LEAVES = (40, 95, 35)
STONE = (125, 125, 125)
SNOW = (245, 245, 240)


def voxel_world(width_blocks: int = 520, height_blocks: int = 220, block: int = 6, seed: int = 42) -> np.ndarray:
    """A top down, Minecraft style world: water, beaches, grass, forest, stone, snow."""
    rng = np.random.default_rng(seed)
    hb, wb = height_blocks, width_blocks
    elevation = fractal_noise(hb, wb, rng, octaves=5, base=5)
    moisture = fractal_noise(hb, wb, rng, octaves=4, base=4)
    detail = rng.random((hb, wb)).astype(np.float32)

    colors = np.zeros((hb, wb, 3), np.float32)
    material = np.zeros((hb, wb), np.uint8)  # used to pick texture strength

    def paint(where, color, mat):
        colors[where] = color
        material[where] = mat

    paint(elevation >= 0, GRASS, 1)
    paint(elevation < 0.30, WATER_DEEP, 0)
    paint((elevation >= 0.30) & (elevation < 0.36), WATER, 0)
    paint((elevation >= 0.36) & (elevation < 0.40), SAND, 2)
    forest = (elevation >= 0.42) & (elevation < 0.70) & (moisture > 0.5)
    paint(forest, FOREST_FLOOR, 1)
    paint((elevation >= 0.70) & (elevation < 0.80), STONE, 3)
    paint(elevation >= 0.80, SNOW, 4)

    # Tree canopies: blobs of leaf blocks with a lighter top.
    tree_seeds = forest & (detail < 0.16)
    canopy = cv2.dilate(tree_seeds.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    canopy &= elevation >= 0.40
    paint(canopy, LEAVES, 5)
    colors[tree_seeds] = np.array(LEAVES, np.float32) * 1.35
    # Flowers and tall grass on open meadows.
    meadow = (material == 1) & ~forest
    flowers = meadow & (detail > 0.985)
    colors[flowers] = (40, 40, 210)
    colors[meadow & (detail > 0.975) & (detail <= 0.985)] = (40, 215, 235)
    # Stone outcrops inside snow and gravel in beaches.
    colors[(material == 4) & (detail < 0.08)] = STONE
    colors[(material == 2) & (detail < 0.05)] = (150, 150, 150)

    # Hillshade from the elevation gradient.
    gy, gx = np.gradient(cv2.GaussianBlur(elevation, (0, 0), 1.2))
    shade = np.clip(1.0 + 9.0 * (-gx - gy), 0.75, 1.2)
    shade[material == 0] = 1.0 + 0.25 * (elevation[material == 0] - 0.3)
    colors *= shade[..., None]
    colors *= (0.92 + 0.16 * rng.random((hb, wb, 1))).astype(np.float32)

    # Per block 16 bit style textures.
    tiles = rng.normal(0, 1, (24, block, block)).astype(np.float32)
    tile_ids = rng.integers(0, len(tiles), (hb, wb))
    strength = np.array([4, 14, 10, 16, 6, 18], np.float32)[material]
    texture = tiles[tile_ids] * strength[..., None, None]
    texture = texture.transpose(0, 2, 1, 3).reshape(hb * block, wb * block)

    pixels = np.repeat(np.repeat(colors, block, axis=0), block, axis=1)
    pixels += texture[..., None]
    return np.clip(pixels, 0, 255).astype(np.uint8)


# ------------------------------------------------------- survey flight -----


@dataclass
class FlightPlan:
    frame_width: int = 640
    frame_height: int = 360
    passes: int = 3
    sidelap: float = 0.40
    speed: float = 7.0
    """Ground speed in world pixels per video frame."""
    margin: float = 30.0
    fps: float = 30.0
    altitude_wobble: float = 0.03
    exposure_wobble: float = 0.10
    vignette: float = 0.25
    noise: float = 2.5
    distortion: float = 0.0
    """Radial lens distortion k1 (negative is barrel). Breaks the similarity model slightly."""
    motion_blur: int = 0
    """Length in pixels of a motion blur streak along the direction of travel."""
    seed: int = 5

    @classmethod
    def hard(cls, **overrides) -> FlightPlan:
        """A rougher flight: barrel distortion, blur, heavier noise and wobble."""
        values = dict(passes=5, speed=9.0, altitude_wobble=0.05, exposure_wobble=0.18, noise=6.0, distortion=-0.06, motion_blur=5)
        values.update(overrides)
        return cls(**values)


@dataclass
class FlightTruth:
    poses: np.ndarray
    """(N, 4) rows of ``x, y, heading, scale`` in world pixels and radians."""
    plan: FlightPlan

    def frame_to_world(self, index: int) -> np.ndarray:
        return pose_matrix(self.poses[index], self.plan.frame_width, self.plan.frame_height)


def pose_matrix(pose: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    """3x3 similarity mapping frame pixels to world pixels."""
    x, y, heading, scale = pose
    c, s = math.cos(heading) * scale, math.sin(heading) * scale
    to_center = np.array([[1, 0, -frame_width / 2], [0, 1, -frame_height / 2], [0, 0, 1]])
    rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    to_world = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return to_world @ rotate @ to_center


def lawnmower_path(world_width: int, world_height: int, plan: FlightPlan) -> np.ndarray:
    """Boustrophedon survey path sampled at constant speed. Returns (N, 4) poses."""
    fw, fh = plan.frame_width, plan.frame_height
    spacing = fh * (1 - plan.sidelap)
    turn_radius = spacing / 2
    half_diag = math.hypot(fw, fh) / 2 * (1 + plan.altitude_wobble)
    x0 = plan.margin + half_diag + turn_radius
    x1 = world_width - plan.margin - half_diag - turn_radius
    y0 = plan.margin + half_diag
    needed = y0 + spacing * (plan.passes - 1) + half_diag + plan.margin
    if x1 <= x0 or needed > world_height:
        raise ValueError("World is too small for this flight plan")

    waypoints = []
    for p in range(plan.passes):
        y = y0 + p * spacing
        left_to_right = p % 2 == 0
        xs = np.arange(0, x1 - x0, 1.0)
        line_x = x0 + xs if left_to_right else x1 - xs
        waypoints += [(x, y) for x in line_x]
        if p < plan.passes - 1:
            # Semicircular turn onto the next pass.
            cx = x1 if left_to_right else x0
            cy = y + turn_radius
            angles = np.linspace(-math.pi / 2, math.pi / 2, max(8, int(math.pi * turn_radius)))
            direction = 1 if left_to_right else -1
            waypoints += [(cx + direction * turn_radius * math.cos(a), cy + turn_radius * math.sin(a)) for a in angles[1:-1]]
    points = np.array(waypoints, np.float64)

    segment = np.hypot(*np.diff(points, axis=0).T)
    arclength = np.concatenate([[0], np.cumsum(segment)])
    samples = np.arange(0, arclength[-1], plan.speed)
    xs = np.interp(samples, arclength, points[:, 0])
    ys = np.interp(samples, arclength, points[:, 1])
    heading = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    heading = cv2.GaussianBlur(heading.reshape(-1, 1), (1, 0), 3).ravel() if heading.size > 7 else heading
    t = np.arange(samples.size) / plan.fps
    scale = 1.0 + plan.altitude_wobble * np.sin(2 * math.pi * t / 9.0)
    return np.stack([xs, ys, heading, scale], axis=1)


_DISTORTION_CACHE: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = {}


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics with focal length equal to half the larger frame side."""
    f = max(width, height) / 2
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def _distortion_maps(width: int, height: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """For every distorted output pixel, where it samples the ideal image.

    Uses OpenCV's radial model (``x_d = x_u (1 + k1 r_u^2)``) so that
    ``cv2.initUndistortRectifyMap`` with the same ``k1`` inverts it exactly.
    """
    key = (width, height, k1)
    if key not in _DISTORTION_CACHE:
        K = camera_matrix(width, height)
        yy, xx = np.indices((height, width), dtype=np.float32)
        distorted = np.stack([xx.ravel(), yy.ravel()], axis=1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(distorted, K, np.array([k1, 0, 0, 0], np.float64), P=K).reshape(height, width, 2)
        _DISTORTION_CACHE[key] = (ideal[..., 0].copy(), ideal[..., 1].copy())
    return _DISTORTION_CACHE[key]


def render_frame(world: np.ndarray, pose: np.ndarray, plan: FlightPlan, index: int, rng: np.random.Generator) -> np.ndarray:
    """What the nadir camera sees at ``pose``, including camera imperfections."""
    fw, fh = plan.frame_width, plan.frame_height
    matrix = pose_matrix(pose, fw, fh)
    frame = cv2.warpAffine(world, matrix[:2], (fw, fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    if plan.distortion:
        map_x, map_y = _distortion_maps(fw, fh, plan.distortion)
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if plan.motion_blur > 1:
        kernel = np.zeros((plan.motion_blur, plan.motion_blur), np.float32)
        kernel[plan.motion_blur // 2, :] = 1.0 / plan.motion_blur
        frame = cv2.filter2D(frame, -1, kernel)
    frame = frame.astype(np.float32)

    t = index / plan.fps
    exposure = 1.0 + plan.exposure_wobble * math.sin(2 * math.pi * t / 7.0 + 0.6)
    yy, xx = np.indices((fh, fw), dtype=np.float32)
    r2 = ((xx - fw / 2) / (fw / 2)) ** 2 + ((yy - fh / 2) / (fh / 2)) ** 2
    vignette = 1.0 - plan.vignette * r2 / 2
    frame *= (exposure * vignette)[..., None]
    frame += rng.normal(0, plan.noise, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def simulate_flight(world: np.ndarray, plan: FlightPlan | None = None) -> FlightTruth:
    plan = plan or FlightPlan()
    poses = lawnmower_path(world.shape[1], world.shape[0], plan)
    return FlightTruth(poses=poses, plan=plan)


def write_flight_video(path: str | Path, world: np.ndarray, truth: FlightTruth) -> Path:
    """Render the flight to an MP4 file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = truth.plan
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), plan.fps, (plan.frame_width, plan.frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    rng = np.random.default_rng(plan.seed)
    try:
        for index, pose in enumerate(truth.poses):
            writer.write(render_frame(world, pose, plan, index, rng))
    finally:
        writer.release()
    return path


def save_truth(path: str | Path, truth: FlightTruth) -> Path:
    path = Path(path)
    header = f"frame_width={truth.plan.frame_width} frame_height={truth.plan.frame_height} fps={truth.plan.fps}\nx,y,heading,scale"
    np.savetxt(path, truth.poses, delimiter=",", header=header, fmt="%.6f")
    return path


def load_truth(path: str | Path) -> FlightTruth:
    path = Path(path)
    first = path.read_text().splitlines()[0].lstrip("# ")
    meta = dict(item.split("=") for item in first.split())
    plan = FlightPlan(frame_width=int(meta["frame_width"]), frame_height=int(meta["frame_height"]), fps=float(meta["fps"]))
    return FlightTruth(poses=np.loadtxt(path, delimiter=",", ndmin=2), plan=plan)
