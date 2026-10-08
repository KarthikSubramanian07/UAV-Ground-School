"""Polka dot detection: the Option 1 pipeline.

``find_dots`` runs any of the detectors in :mod:`week03.blobs` on a photo and
turns raw detections into measured dots:

1. **Detect.** ``gray`` is the one line baseline: SimpleBlobDetector on the
   grayscale image. ``simple`` learns the palette with k-means in CIELAB and
   runs SimpleBlobDetector on a similarity map per color. ``contrast`` runs it
   on the Delta E from a median filtered background. ``contour`` keeps round
   regions enclosed by color edges. ``log``, ``dog`` and ``doh`` search scale
   space on the CIELAB image directly.
2. **Colors.** The dot color comes from each candidate's core, the local
   background from a ring just outside.
3. **Edge fit.** Along 48 rays from the center, the edge is where the color
   crosses halfway between dot and background. A robust circle fit re-centers
   the rays; an ellipse fit to the final edge points gives sub pixel centers,
   radii that do not depend on the detector's scale sampling, a roundness
   score and an aspect ratio.
4. **Verify.** Inside one color, a different color just outside, and a
   surround that is itself one color. This rejects squares, stars, fabric
   texture and gaps of background enclosed by dots.
5. **Filter.** By radius, color name, contrast and roundness.

Noisy images (``shapes.png`` carries per pixel color noise five times that of
the polka dot photos, see :func:`estimate_noise`) can be cleaned first with
non-local means (``denoise``), which averages patches that look alike and so
keeps edges sharp where a blur would round them off.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from week02.colors import _silhouette, name_color

from .blobs import DISK_PEAK, Blob, BlobFilter, detect_contours, detect_scale_space, detect_simple, prune_overlaps

METHODS = ("gray", "simple", "contrast", "contour", "log", "dog", "doh")


def estimate_noise(image: np.ndarray) -> tuple[float, float, float]:
    """Standard deviation of pixel noise in each CIELAB channel (Immerkaer, 1996).

    The mask below cancels any image that is locally a plane, so what is left
    is mostly noise; the median absolute value makes edges barely count.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return tuple(float(np.median(np.abs(cv2.filter2D(lab[..., c], -1, kernel))) * 1.4826 / 6.0) for c in range(3))  # type: ignore[return-value]


def denoise(image: np.ndarray, strength: float) -> np.ndarray:
    """Non-local means on the color image; ``strength`` is OpenCV's ``h`` (10 suits ``shapes.png``)."""
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21) if strength > 0 else image


def lab_image(image: np.ndarray) -> np.ndarray:
    """Float CIELAB image in Delta E units (L in 0..100)."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    bgr = cv2.cvtColor(np.asarray(lab, np.float32).reshape(1, 1, 3), cv2.COLOR_Lab2BGR).reshape(3) * 255
    return tuple(int(v) for v in np.clip(np.round(bgr), 0, 255))


def color_name(lab: np.ndarray) -> str:
    """Name a CIELAB color, calling light low chroma colors pastel rather than white."""
    lab = np.asarray(lab, np.float64)
    chroma = math.hypot(lab[1], lab[2])
    name = name_color(lab_to_bgr(lab))
    if name in ("white", "gray") and chroma >= 8:
        hue = math.degrees(math.atan2(lab[2], lab[1])) % 360
        # Pastels: hue angle in CIELAB.
        if 200 <= hue < 290:
            return "light blue"
        if hue >= 290 or hue < 30:
            return "pink"
        if 30 <= hue < 70:
            return "beige"
        if 70 <= hue < 125:
            return "cream"
        return "light green"
    return name


# ----------------------------------------------------------------- palette ----


@dataclass
class Palette:
    centers: np.ndarray  # (k, 3) CIELAB
    names: list[str]

    def __len__(self) -> int:
        return len(self.centers)


def learn_palette(
    lab: np.ndarray, k: int | None = None, k_range: tuple[int, int] = (6, 14), merge_distance: float = 10.0, seed: int = 0
) -> Palette:
    """k-means palette in CIELAB, k by silhouette score when not given."""
    rng = np.random.default_rng(seed)
    pixels = lab.reshape(-1, 3)
    sample = pixels[rng.choice(len(pixels), size=min(20000, len(pixels)), replace=False)].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    cv2.setRNGSeed(seed)

    def run(n: int) -> tuple[np.ndarray, np.ndarray]:
        _, labels, centers = cv2.kmeans(sample, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        return labels.ravel(), centers

    if k is None:
        probe = rng.choice(len(sample), size=min(1500, len(sample)), replace=False)
        best: tuple[float, np.ndarray] | None = None
        for n in range(k_range[0], k_range[1] + 1):
            labels, centers = run(n)
            score = _silhouette(sample[probe].astype(np.float64), labels[probe])
            if best is None or score > best[0]:
                best = (score, centers)
        centers = best[1]
    else:
        centers = run(k)[1]
    merged: list[np.ndarray] = []
    for center in sorted(centers, key=lambda c: c[0]):
        if all(np.linalg.norm(center - other) >= merge_distance for other in merged):
            merged.append(center)
    centers = np.array(merged, np.float32)
    return Palette(centers, [color_name(c) for c in centers])


def similarity_maps(lab: np.ndarray, palette: Palette, smooth: float = 1.5) -> list[np.ndarray]:
    """One 8-bit map per palette color: 255 on the color, fading to 0 halfway to its nearest neighbor color."""
    maps = []
    for i, center in enumerate(palette.centers):
        others = np.delete(palette.centers, i, axis=0)
        reach = 0.5 * float(np.min(np.linalg.norm(others - center, axis=1))) if len(others) else 30.0
        reach = max(reach, 6.0)
        distance = np.sqrt(((lab - center) ** 2).sum(axis=2))
        similarity = np.clip(255.0 * (1.0 - distance / reach), 0, 255).astype(np.float32)
        # Texture (fabric weave, JPEG) punches holes in the map; smooth it before thresholding.
        maps.append(cv2.GaussianBlur(similarity, (0, 0), smooth).astype(np.uint8) if smooth > 0 else similarity.astype(np.uint8))
    return maps


def background_estimate(lab: np.ndarray, max_radius: float) -> np.ndarray:
    """Local background color: a median filter wider than the largest dot.

    Computed on a downsampled image (medians are robust to resampling) so the
    kernel stays small, then upsampled.
    """
    h, w = lab.shape[:2]
    diameter = 2.5 * max_radius
    factor = max(1, math.ceil(diameter / 31))
    small = cv2.resize(lab, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    ksize = int(diameter / factor) | 1
    ksize = max(3, min(ksize, (min(small.shape[:2]) - 1) | 1 if min(small.shape[:2]) > 3 else 3))
    offsets = np.array([0.0, 128.0, 128.0], np.float32)
    scale = np.array([2.55, 1.0, 1.0], np.float32)
    as_u8 = np.clip((small + offsets) * scale, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(as_u8, ksize).astype(np.float32) / scale - offsets
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def contrast_map(lab: np.ndarray, max_radius: float, smooth: float = 1.5) -> np.ndarray:
    """Delta E between every pixel and its local background, as a float image."""
    delta = np.sqrt(((lab - background_estimate(lab, max_radius)) ** 2).sum(axis=2))
    return cv2.GaussianBlur(delta, (0, 0), smooth) if smooth > 0 else delta


def color_edge_regions(lab: np.ndarray, min_contrast: float, smooth: float = 2.5) -> np.ndarray:
    """Regions enclosed by color edges: Canny on CIELAB, then everything that is not edge.

    OpenCV's Canny takes the strongest gradient over the three channels, so a
    hue change with no brightness change still makes an edge. Each dot becomes
    a region bounded by its edge ring, whatever the background looks like.
    """
    blurred = cv2.GaussianBlur(lab, (0, 0), smooth) if smooth > 0 else lab
    as_u8 = np.clip((blurred + np.array([0.0, 128.0, 128.0], np.float32)) * np.array([2.55, 1.0, 1.0], np.float32), 0, 255).astype(np.uint8)
    high = 1.2 * min_contrast * 4.0 * 0.5
    edges = cv2.Canny(as_u8, high / 2, high, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    return cv2.bitwise_not(edges)


# ------------------------------------------------------ verify and refine ----


@dataclass
class DotSettings:
    method: str = "log"
    min_radius: float = 3.0
    max_radius: float | None = None  # default: a sixth of the short image side
    min_contrast: float = 10.0  # Delta E between dot and surroundings
    per_octave: int = 5
    peak_fraction: float = 0.5  # scale space peaks must reach this fraction of an ideal dot's response at min_contrast
    overlap: float = 0.3
    verify: bool = True
    refine: bool = True
    min_fill: float = 0.8  # fraction of the inside that matches the dot color
    max_leak: float = 0.35  # fraction of the outside ring that also matches it
    min_surround: float = 0.6  # fraction of the outside ring that matches the local background
    min_roundness: float = 0.93  # 1 - RMS distance of the edge from the fitted ellipse / radius
    min_aspect: float = 0.6  # minor / major axis; dots seen at an angle are ellipses
    colors: tuple[str, ...] | None = None  # keep only these color names
    blob_filter: BlobFilter = field(
        default_factory=lambda: BlobFilter(blob_color=255, min_threshold=40, max_threshold=250, threshold_step=15)
    )
    palette_k: int | None = None
    denoise: float = 0.0  # non-local means strength applied first; 0 is off


def _sample(lab: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear samples of a float image at arbitrary points, plus an inside-the-image mask."""
    h, w = lab.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    shape = xs.shape
    mx = np.ascontiguousarray(xs.reshape(-1, 1).astype(np.float32))
    my = np.ascontiguousarray(ys.reshape(-1, 1).astype(np.float32))
    # cv2.remap is limited to 32767 columns/rows, so sample in a column.
    values = np.empty((mx.shape[0], 3), np.float32)
    step = 32000
    for start in range(0, mx.shape[0], step):
        values[start : start + step] = cv2.remap(
            lab, mx[start : start + step], my[start : start + step], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        ).reshape(-1, 3)
    return values.reshape(*shape, 3), valid


_ANGLES = np.linspace(0, 2 * math.pi, 24, endpoint=False)
_CORE = np.array([0.0, 0.15, 0.3])
_INNER = np.array([0.0, 0.3, 0.5, 0.7])
_OUTER = np.array([1.3, 1.55])


def _rings(lab: np.ndarray, blobs: list[Blob], fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples on circles at ``fractions`` of each blob's radius: (n, m, 3) values and (n, m) validity."""
    xy = np.array([[b.x, b.y, b.radius] for b in blobs])
    cos, sin = np.cos(_ANGLES), np.sin(_ANGLES)
    r = xy[:, 2, None, None] * fractions[None, :, None]
    xs = xy[:, 0, None, None] + r * cos[None, None, :]
    ys = xy[:, 1, None, None] + r * sin[None, None, :]
    values, valid = _sample(lab, xs, ys)
    return values.reshape(len(blobs), -1, 3), valid.reshape(len(blobs), -1)


def estimate_colors(lab: np.ndarray, blobs: list[Blob], min_contrast: float) -> list[Blob]:
    """Attach the dot color (from its core) and the local background color (from a ring just outside).

    Only the core is trusted at this stage because detectors can overestimate
    the radius, for example on shaded fabric.
    """
    if not blobs:
        return []
    core, core_ok = _rings(lab, blobs, _CORE)
    outer, outer_ok = _rings(lab, blobs, _OUTER)
    kept: list[Blob] = []
    for i, blob in enumerate(blobs):
        if core_ok[i].mean() < 0.4 or outer_ok[i].mean() < 0.25:
            continue
        dot = np.median(core[i][core_ok[i]], axis=0)
        background = np.median(outer[i][outer_ok[i]], axis=0)
        contrast = float(np.linalg.norm(dot - background))
        if contrast < min_contrast:
            continue
        blob.contrast = contrast
        blob._lab = (dot, background)  # type: ignore[attr-defined]
        kept.append(blob)
    return kept


def verify(lab: np.ndarray, blobs: list[Blob], settings: DotSettings) -> list[Blob]:
    """Keep blobs that look like dots at their current center and radius.

    * the inside is one color (``fill``),
    * the ring just outside is a different color (``leak``),
    * that ring is mostly one color (``surround``): a real dot sits on a
      background, while a gap of background enclosed by several dots is
      ringed by many colors.
    """
    if not blobs:
        return []
    inner, inner_ok = _rings(lab, blobs, _INNER)
    outer, outer_ok = _rings(lab, blobs, _OUTER)
    kept: list[Blob] = []
    for i, blob in enumerate(blobs):
        if inner_ok[i].mean() < 0.4 or outer_ok[i].mean() < 0.25:
            continue
        inside = inner[i][inner_ok[i]]
        around = outer[i][outer_ok[i]]
        dot = np.median(inside, axis=0)
        background = np.median(around, axis=0)
        contrast = float(np.linalg.norm(dot - background))
        if contrast < settings.min_contrast:
            continue
        tolerance = max(0.5 * contrast, 4.0)
        fill = float(np.mean(np.linalg.norm(inside - dot, axis=1) < tolerance))
        leak = float(np.mean(np.linalg.norm(around - dot, axis=1) < tolerance))
        surround = float(np.mean(np.linalg.norm(around - background, axis=1) < max(0.5 * contrast, 6.0)))
        if fill < settings.min_fill or leak > settings.max_leak or surround < settings.min_surround:
            continue
        blob.contrast = contrast
        blob.bgr = lab_to_bgr(dot)
        blob.color = color_name(dot)
        blob._lab = (dot, background)  # type: ignore[attr-defined]
        kept.append(blob)
    return kept


_RAYS = np.linspace(0, 2 * math.pi, 48, endpoint=False)


def _batched_circle_fit(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Kasa circle fits for many point sets at once: points (n, m, 2), weights (n, m)."""
    x, y = points[..., 0], points[..., 1]
    a = np.stack([x, y, np.ones_like(x)], axis=-1) * weights[..., None]
    b = (x * x + y * y) * weights
    ata = np.einsum("nmi,nmj->nij", a, a) + np.eye(3) * 1e-9
    atb = np.einsum("nmi,nm->ni", a, b)
    sol = np.linalg.solve(ata, atb[..., None])[..., 0]
    cx, cy = sol[:, 0] / 2, sol[:, 1] / 2
    r = np.sqrt(np.maximum(sol[:, 2] + cx * cx + cy * cy, 0.0))
    return cx, cy, r


def _ellipse_roundness(points: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, float, float, float, float]:
    """Fit an ellipse to edge points; return center, equal area radius, roundness and aspect."""
    residual = np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r
    circle = (cx, cy, r, 1.0 - float(np.sqrt(np.mean(residual**2))) / max(r, 1e-6), 1.0)
    if len(points) < 8:
        return circle
    (ex, ey), (d1, d2), angle = cv2.fitEllipse(points.astype(np.float32))
    a, b = max(d1, d2) / 2, min(d1, d2) / 2
    if not (b > 0 and math.hypot(ex - cx, ey - cy) < 0.5 * r and 0.5 < math.sqrt(a * b) / r < 2.0):
        return circle
    t = math.radians(angle)
    du, dv = points[:, 0] - ex, points[:, 1] - ey
    u = du * math.cos(t) + dv * math.sin(t)
    v = -du * math.sin(t) + dv * math.cos(t)
    radius = math.sqrt(a * b)
    residual = (np.sqrt((u / (d1 / 2)) ** 2 + (v / (d2 / 2)) ** 2) - 1.0) * radius
    return ex, ey, radius, 1.0 - float(np.sqrt(np.mean(residual**2))) / radius, b / a


def refine(
    lab: np.ndarray, blobs: list[Blob], settings: DotSettings, iterations: int = 2, samples: int = 96, chunk: int = 256
) -> list[Blob]:
    """Fit an ellipse to the color edge of each dot (vectorized over dots and rays).

    Along 48 rays, the edge is where the color, projected on the line from
    background color to dot color, crosses halfway. A robust circle fit
    (two rounds of outlier rejection) re-centers the rays; the final edge
    points get an ellipse fit whose RMS residual gives the roundness.
    """
    out: list[Blob] = []
    cos, sin = np.cos(_RAYS), np.sin(_RAYS)
    fractions = np.linspace(0.35, 1.8, samples)
    for start in range(0, len(blobs), chunk):
        group = blobs[start : start + chunk]
        n = len(group)
        dot = np.array([b._lab[0] for b in group], np.float64)  # type: ignore[attr-defined]
        background = np.array([b._lab[1] for b in group], np.float64)  # type: ignore[attr-defined]
        direction = dot - background
        contrast = np.maximum(np.linalg.norm(direction, axis=1), 1e-6)
        direction /= contrast[:, None]
        cx = np.array([b.x for b in group])
        cy = np.array([b.y for b in group])
        r = np.array([b.radius for b in group])
        ok = np.ones(n, bool)
        edges = np.zeros((n, len(_RAYS), 2))
        keep = np.zeros((n, len(_RAYS)), bool)
        has = np.zeros((n, len(_RAYS)), bool)
        for _ in range(iterations):
            dist = r[:, None] * fractions[None, :]  # (n, k)
            xs = cx[:, None, None] + cos[None, :, None] * dist[:, None, :]
            ys = cy[:, None, None] + sin[None, :, None] * dist[:, None, :]
            values, valid = _sample(lab, xs, ys)  # (n, rays, k, 3)
            level = np.einsum("nrkc,nc->nrk", values - background[:, None, None, :], direction) / contrast[:, None, None]
            below = level < 0.5
            j = np.argmax(below, axis=2)  # first sample outside the dot
            has = below.any(axis=2) & (j > 0)
            jj = np.clip(j, 1, samples - 1)
            idx_n, idx_r = np.indices(jj.shape)
            a_val = level[idx_n, idx_r, jj - 1]
            b_val = level[idx_n, idx_r, jj]
            has &= valid[idx_n, idx_r, jj] & valid[idx_n, idx_r, jj - 1]
            t = (a_val - 0.5) / np.maximum(a_val - b_val, 1e-6)
            d_edge = dist[idx_n, jj - 1] + t * (dist[idx_n, jj] - dist[idx_n, jj - 1])
            edges = np.stack([cx[:, None] + cos[None, :] * d_edge, cy[:, None] + sin[None, :] * d_edge], axis=-1)
            ok &= has.sum(axis=1) >= 0.35 * len(_RAYS)
            keep = has.copy()
            for _round in range(2):
                fx, fy, fr = _batched_circle_fit(edges, keep.astype(np.float64))
                residual = np.abs(np.hypot(edges[..., 0] - fx[:, None], edges[..., 1] - fy[:, None]) - fr[:, None])
                masked = np.where(keep, residual, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # rows with no edges are handled by ``ok``
                    mad = np.nanmedian(masked, axis=1) * 1.4826
                limit = np.maximum(1.0, 3.0 * np.nan_to_num(mad, nan=0.0))
                keep = has & (residual <= limit[:, None])
            ok &= keep.sum(axis=1) >= 6
            fx, fy, fr = _batched_circle_fit(edges, keep.astype(np.float64))
            good = ok & np.isfinite(fr) & (fr > 0)
            cx = np.where(good, fx, cx)
            cy = np.where(good, fy, cy)
            r = np.where(good, fr, r)
        h, w = lab.shape[:2]
        for i, blob in enumerate(group):
            inside = -0.3 * r[i] <= cx[i] < w - 1 + 0.3 * r[i] and -0.3 * r[i] <= cy[i] < h - 1 + 0.3 * r[i]
            if not (ok[i] and inside and r[i] > 0):
                if not settings.verify:
                    out.append(blob)
                continue
            ex, ey, er, roundness, aspect = _ellipse_roundness(edges[i][keep[i]], cx[i], cy[i], r[i])
            roundness *= (keep[i].sum() / max(has[i].sum(), 1)) ** 0.5  # penalize shapes where many rays disagree
            if settings.verify and (roundness < settings.min_roundness or aspect < settings.min_aspect):
                continue
            blob.x, blob.y, blob.radius, blob.roundness, blob.aspect = ex, ey, er, roundness, aspect
            out.append(blob)
    return out


def _describe(blobs: list[Blob]) -> list[Blob]:
    for blob in blobs:
        dot, _ = blob._lab  # type: ignore[attr-defined]
        blob.bgr = lab_to_bgr(dot)
        blob.color = color_name(dot)
    return blobs


# --------------------------------------------------------------- pipeline ----


@dataclass
class DotReport:
    dots: list[Blob]
    method: str
    seconds: float
    palette: Palette | None = None
    candidates: int = 0

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for dot in self.dots:
            counts[dot.color or "?"] = counts.get(dot.color or "?", 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        return {
            "method": self.method,
            "count": len(self.dots),
            "counts_by_color": self.counts(),
            "seconds": round(self.seconds, 4),
            "candidates": self.candidates,
            "dots": [d.summary() for d in self.dots],
        }


def _candidates(image: np.ndarray, lab: np.ndarray, s: DotSettings, max_radius: float) -> tuple[list[Blob], Palette | None]:
    min_area = math.pi * s.min_radius**2 * 0.8
    max_area = math.pi * max_radius**2 * 1.2
    if s.method == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = BlobFilter(min_area=min_area, max_area=max_area, blob_color=None)
        return detect_simple(gray, f), None
    shape_filter = {
        "min_area": min_area,
        "max_area": max_area,
        "min_circularity": s.blob_filter.min_circularity,
        "min_convexity": s.blob_filter.min_convexity,
        "min_inertia": s.blob_filter.min_inertia,
    }
    sbd = {
        "min_threshold": s.blob_filter.min_threshold,
        "max_threshold": s.blob_filter.max_threshold,
        "threshold_step": s.blob_filter.threshold_step,
        "min_repeatability": s.blob_filter.min_repeatability,
        "min_dist_between_blobs": s.blob_filter.min_dist_between_blobs,
    }
    if s.method == "simple":
        palette = learn_palette(lab, k=s.palette_k)
        blobs: list[Blob] = []
        for sim in similarity_maps(lab, palette):
            found = detect_simple(sim, BlobFilter(blob_color=255, **shape_filter, **sbd))
            for b in found:
                b.response = b.radius
            blobs.extend(found)
        return blobs, palette
    if s.method == "contrast":
        delta = contrast_map(lab, max_radius)
        # Scale so that 255 is a strong contrast and SimpleBlobDetector's threshold sweep spans useful levels.
        gray = np.clip(delta * (255.0 / (4.0 * s.min_contrast)), 0, 255).astype(np.uint8)
        found = detect_simple(gray, BlobFilter(blob_color=255, **{**sbd, "min_threshold": 32.0, "threshold_step": 16.0}, **shape_filter))
        for b in found:
            b.response = b.radius
        return found, None
    if s.method == "contour":
        found = detect_contours(color_edge_regions(lab, s.min_contrast), BlobFilter(blob_color=None, **shape_filter), nested=True)
        for b in found:
            b.response = b.radius
        return prune_overlaps(found, 0.5), None
    if s.method in ("log", "dog", "doh"):
        threshold = DISK_PEAK * s.min_contrast * s.peak_fraction
        blobs = detect_scale_space(
            lab, s.method, min_radius=s.min_radius, max_radius=max_radius, per_octave=s.per_octave, threshold=threshold, overlap=s.overlap
        )
        return blobs, None
    raise ValueError(f"method must be one of {', '.join(METHODS)}")


def find_dots(image: np.ndarray, settings: DotSettings | None = None, **overrides) -> DotReport:
    """Detect polka dots in a BGR image. Keyword overrides update ``settings``."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3 channel BGR image")
    s = settings or DotSettings()
    if overrides:
        s = DotSettings(**{**s.__dict__, **overrides})
    start = time.perf_counter()
    image = denoise(image, s.denoise)
    lab = lab_image(image)
    max_radius = s.max_radius or min(image.shape[:2]) / 6.0
    candidates, palette = _candidates(image, lab, s, max_radius)
    count = len(candidates)
    dots = estimate_colors(lab, candidates, 0.6 * s.min_contrast)
    if s.refine:
        dots = refine(lab, dots, s)
    dots = verify(lab, dots, s) if s.verify else _describe(dots)
    dots = [d for d in dots if s.min_radius * 0.8 <= d.radius <= max_radius * 1.25]
    dots = prune_overlaps(sorted(dots, key=lambda d: (d.contrast or 0) * (d.roundness or 0.5), reverse=True), s.overlap)
    if s.colors:
        wanted = {c.lower() for c in s.colors}
        dots = [d for d in dots if (d.color or "").lower() in wanted]
    for d in dots:
        if hasattr(d, "_lab"):
            del d._lab
    dots.sort(key=lambda d: (round(d.y / max(d.radius, 1)), d.x))
    return DotReport(dots, s.method, time.perf_counter() - start, palette, count)


def annotate(image: np.ndarray, dots: list[Blob], label: bool = False, thickness: int | None = None) -> np.ndarray:
    """Draw each dot's fitted circle in a contrasting ring, with its center."""
    out = image.copy()
    t = thickness or max(1, round(min(image.shape[:2]) / 400))
    for i, d in enumerate(dots):
        center = (int(round(d.x * 16)), int(round(d.y * 16)))
        radius = int(round(d.radius * 16))
        cv2.circle(out, center, radius, (255, 255, 255), t + 2, cv2.LINE_AA, shift=4)
        cv2.circle(out, center, radius, (20, 20, 20), t, cv2.LINE_AA, shift=4)
        cv2.circle(out, center, 2 * 16, (20, 20, 20), -1, cv2.LINE_AA, shift=4)
        if label:
            cv2.putText(out, str(i), (int(d.x) + 3, int(d.y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, str(i), (int(d.x) + 3, int(d.y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out
