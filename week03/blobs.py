"""Blob detectors: OpenCV's SimpleBlobDetector, scale space LoG, DoG and DoH, and contour filtering.

Every detector returns the same :class:`Blob` records (center, radius,
response), so they can be compared head to head against ground truth.

Scale space detectors
---------------------
A blob of radius ``r`` produces the strongest scale normalized Laplacian of
Gaussian response at ``sigma = r / sqrt(2)``. Searching the (x, y, sigma)
volume for local maxima therefore finds both where a blob is and how big it
is. Three flavours are implemented from scratch with OpenCV filters:

* **LoG**: ``-sigma^2 * Laplacian(G_sigma * I)``, the textbook detector.
* **DoG**: ``(G_sigma * I - G_{k sigma} * I) / (k - 1)``, the cheap
  approximation of LoG used by SIFT.
* **DoH**: ``sigma^4 * det(Hessian(G_sigma * I))``, which is near zero on
  straight edges and so ignores them.

All three accept multi channel images. On a CIELAB image the per channel
responses are combined into one magnitude, which makes the detector color
aware and polarity free: a cyan dot on orange fabric and a pale blue dot on
white card are both blobs, measured in Delta E units. For an ideal disk of
contrast ``c`` all three peak at ``2 c / e`` (about ``0.74 c``), so a single
threshold means the same thing for every method.

Peaks are refined to sub pixel position and fractional scale with a 3D
quadratic fit (as in SIFT), then overlapping detections are pruned greedily.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

SQRT2 = math.sqrt(2.0)
DISK_PEAK = 2.0 / math.e  # normalized LoG response of a unit contrast disk at its matched scale


@dataclass
class Blob:
    """One detected (or annotated) blob."""

    x: float
    y: float
    radius: float
    response: float = 0.0
    color: str | None = None
    bgr: tuple[int, int, int] | None = None
    contrast: float | None = None  # Delta E between the blob and its surroundings
    roundness: float | None = None  # 1 - RMS edge residual / radius, from the edge fit
    aspect: float | None = None  # minor / major axis of the fitted ellipse

    def summary(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v is not None}
        for key in ("x", "y", "radius", "response", "contrast", "roundness", "aspect"):
            if key in out:
                out[key] = round(float(out[key]), 3)
        if "bgr" in out:
            out["bgr"] = [int(c) for c in out["bgr"]]
        return out


def blobs_to_array(blobs: list[Blob]) -> np.ndarray:
    """(n, 3) array of x, y, radius."""
    if not blobs:
        return np.zeros((0, 3), np.float64)
    return np.array([[b.x, b.y, b.radius] for b in blobs], np.float64)


# ------------------------------------------------------ SimpleBlobDetector ----


@dataclass(frozen=True)
class BlobFilter:
    """The knobs of ``cv2.SimpleBlobDetector``, with ``None`` meaning "do not filter".

    SimpleBlobDetector thresholds the image at every level from
    ``min_threshold`` to ``max_threshold``, finds connected components at each
    level, groups components whose centers agree across levels, and keeps
    groups seen at least ``min_repeatability`` times that pass the filters.
    """

    min_area: float | None = 20.0
    max_area: float | None = None
    min_circularity: float | None = 0.75  # 4 pi A / P^2, 1 for a circle, 0.785 for a square
    min_convexity: float | None = 0.9  # A / convex hull area
    min_inertia: float | None = 0.4  # minor / major axis of the second moments, squared
    blob_color: int | None = 0  # 0 dark blobs, 255 bright blobs, None both
    min_threshold: float = 10.0
    max_threshold: float = 250.0
    threshold_step: float = 10.0
    min_repeatability: int = 2
    min_dist_between_blobs: float = 10.0


DEFAULT_FILTER = BlobFilter()
CONTOUR_FILTER = BlobFilter(blob_color=None)


def simple_blob_detector(f: BlobFilter = DEFAULT_FILTER) -> cv2.SimpleBlobDetector:
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = float(f.min_threshold)
    p.maxThreshold = float(f.max_threshold)
    p.thresholdStep = float(f.threshold_step)
    p.minRepeatability = int(f.min_repeatability)
    p.minDistBetweenBlobs = float(f.min_dist_between_blobs)
    p.filterByColor = f.blob_color is not None
    if f.blob_color is not None:
        p.blobColor = int(f.blob_color)
    p.filterByArea = f.min_area is not None or f.max_area is not None
    p.minArea = float(f.min_area if f.min_area is not None else 0.0)
    p.maxArea = float(f.max_area if f.max_area is not None else 1e12)
    p.filterByCircularity = f.min_circularity is not None
    if f.min_circularity is not None:
        p.minCircularity = float(f.min_circularity)
    p.filterByConvexity = f.min_convexity is not None
    if f.min_convexity is not None:
        p.minConvexity = float(f.min_convexity)
    p.filterByInertia = f.min_inertia is not None
    if f.min_inertia is not None:
        p.minInertiaRatio = float(f.min_inertia)
    return cv2.SimpleBlobDetector_create(p)


def detect_simple(gray: np.ndarray, f: BlobFilter = DEFAULT_FILTER) -> list[Blob]:
    """Run SimpleBlobDetector on an 8-bit single channel image."""
    if gray.ndim != 2:
        raise ValueError("SimpleBlobDetector needs a single channel image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    keypoints = simple_blob_detector(f).detect(gray)
    return [Blob(float(k.pt[0]), float(k.pt[1]), float(k.size) / 2.0) for k in keypoints]


# ------------------------------------------------------------ scale space ----


def sigma_ladder(min_sigma: float, max_sigma: float, per_octave: int = 5) -> np.ndarray:
    """Geometrically spaced scales covering [min_sigma, max_sigma]."""
    if min_sigma <= 0 or max_sigma < min_sigma:
        raise ValueError("need 0 < min_sigma <= max_sigma")
    count = max(2, int(math.ceil(per_octave * math.log2(max_sigma / min_sigma))) + 1)
    return np.geomspace(min_sigma, max_sigma, count)


def _channels(image: np.ndarray) -> list[np.ndarray]:
    image = np.asarray(image)
    if image.ndim == 2:
        return [image.astype(np.float32)]
    return [image[:, :, c].astype(np.float32) for c in range(image.shape[2])]


def _blur(channel: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(channel, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


_DXX = np.array([[1.0, -2.0, 1.0]], np.float32)
_DXY = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], np.float32) / 4.0


def _hessian(blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lxx = cv2.filter2D(blurred, cv2.CV_32F, _DXX, borderType=cv2.BORDER_REFLECT)
    lyy = cv2.filter2D(blurred, cv2.CV_32F, _DXX.T, borderType=cv2.BORDER_REFLECT)
    lxy = cv2.filter2D(blurred, cv2.CV_32F, _DXY, borderType=cv2.BORDER_REFLECT)
    return lxx, lyy, lxy


def _combine(per_channel: list[np.ndarray], polarity: str) -> np.ndarray:
    """Signed single channel response or a multi channel magnitude."""
    if len(per_channel) == 1:
        r = per_channel[0]
        if polarity == "bright":
            return r
        if polarity == "dark":
            return -r
        return np.abs(r)
    total = np.zeros_like(per_channel[0])
    for r in per_channel:
        total += r * r
    return np.sqrt(total)


class _Pyramid:
    """Downsampled copies of the channels, so large scales are filtered on small images.

    Blurring with sigma 60 at full resolution costs a 480 tap kernel per
    pixel; the same response computed at 1/8 resolution with sigma 7.5 and
    upsampled is indistinguishable and about 50 times cheaper.
    """

    def __init__(self, channels: list[np.ndarray], base_sigma: float = 4.0, enabled: bool = True):
        self.channels = channels
        self.base_sigma = base_sigma
        self.enabled = enabled
        self.levels: dict[int, list[np.ndarray]] = {1: channels}
        self.height, self.width = channels[0].shape

    def factor(self, sigma: float) -> int:
        if not self.enabled or sigma < 2 * self.base_sigma:
            return 1
        f = 2 ** int(math.floor(math.log2(sigma / self.base_sigma)))
        return int(min(f, max(1, min(self.height, self.width) // 16)))

    def at(self, factor: int) -> list[np.ndarray]:
        if factor not in self.levels:
            size = (max(1, round(self.width / factor)), max(1, round(self.height / factor)))
            self.levels[factor] = [cv2.resize(c, size, interpolation=cv2.INTER_AREA) for c in self.channels]
        return self.levels[factor]

    def upsample(self, response: np.ndarray) -> np.ndarray:
        if response.shape == (self.height, self.width):
            return response
        return cv2.resize(response, (self.width, self.height), interpolation=cv2.INTER_LINEAR)


def _slices(image: np.ndarray, sigmas: np.ndarray, kind: str, polarity: str, pyramid: bool, split: bool):
    """Yield ``(effective_sigma, maps, signed)`` per scale, finest first.

    ``maps[0]`` is the combined response (signed for one channel, a magnitude
    for several). With ``split`` and a multi channel image, ``maps[1:]`` are
    the signed responses of each channel, positive and negative, so a faint
    blob that lives in one channel (pale blue on white is almost pure ``b``)
    is not drowned by a strong neighbor in another channel. ``signed`` holds
    the signed response of every channel, used to tell a blob from the ring
    of opposite sign around a stronger one.
    """
    if polarity not in ("bright", "dark", "both"):
        raise ValueError("polarity must be bright, dark or both")
    if kind not in ("log", "dog", "doh"):
        raise ValueError("kind must be log, dog or doh")
    pyr = _Pyramid(_channels(image), enabled=pyramid)
    sigmas = np.asarray(sigmas, np.float64)
    multi = len(pyr.channels) > 1
    ratio = sigmas[1] / sigmas[0] if len(sigmas) > 1 else 1.6
    for i, sigma in enumerate(sigmas):
        f = pyr.factor(sigma)
        chans = pyr.at(f)
        sl = sigma / f
        if kind == "log":
            per = [-(sl * sl) * cv2.Laplacian(_blur(c, sl), cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT) for c in chans]
            effective = sigma
        elif kind == "dog":
            upper = sigmas[i + 1] if i + 1 < len(sigmas) else sigma * ratio
            k = upper / sigma
            per = [(_blur(c, sl) - _blur(c, upper / f)) / (k - 1.0) for c in chans]
            effective = math.sqrt(sigma * upper)
        else:
            per = []
            for c in chans:
                lxx, lyy, lxy = _hessian(_blur(c, sl))
                det = (sl**4) * (lxx * lyy - lxy * lxy)
                # Signed like the Laplacian, so bright and dark blobs separate per channel.
                per.append(-np.sign(lxx + lyy) * 2.0 * np.sqrt(np.maximum(det, 0.0)))
        if kind == "doh" and multi:
            # 2 sqrt(sum det) equals |LoG| at the center of a round blob.
            total = sum(p * p for p in per)
            combined = np.sqrt(total)
        else:
            combined = _combine(per, polarity)
        maps = [pyr.upsample(combined.astype(np.float32))]
        signed = [pyr.upsample(p.astype(np.float32)) for p in per]
        if split and multi:
            for up in signed:
                maps.append(up)
                maps.append(-up)
        yield (sigma if kind != "dog" else effective), maps, signed


def scale_space(
    image: np.ndarray, sigmas: np.ndarray, kind: str = "log", polarity: str = "both", pyramid: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Stack of scale normalized blob responses, shape (len(scales), H, W).

    Returns the stack and the effective sigma of each slice (for DoG the slice
    between two blur levels sits at their geometric mean). ``polarity`` is
    ``bright``, ``dark`` or ``both`` and only matters for single channel input.
    With ``pyramid`` the large scales are computed on downsampled images.
    Scale normalized responses do not change with resolution, so no
    correction is needed.
    """
    effective, stack = [], []
    for sigma, maps, _ in _slices(image, sigmas, kind, polarity, pyramid, split=False):
        effective.append(sigma)
        stack.append(maps[0])
    return np.stack(stack).astype(np.float32), np.array(effective)


def local_maxima(stack: np.ndarray, threshold: float) -> np.ndarray:
    """(n, 3) integer (scale, y, x) of 3x3x3 local maxima above ``threshold``."""
    kernel = np.ones((3, 3), np.uint8)
    spatial = np.stack([cv2.dilate(layer, kernel, borderType=cv2.BORDER_REPLICATE) for layer in stack])
    neighborhood = spatial.copy()
    neighborhood[1:] = np.maximum(neighborhood[1:], spatial[:-1])
    neighborhood[:-1] = np.maximum(neighborhood[:-1], spatial[1:])
    peaks = (stack >= neighborhood) & (stack > threshold)
    return np.argwhere(peaks)


def refine_peaks(stack: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sub pixel, sub scale refinement of peaks by a 3D quadratic fit.

    Returns (n, 3) float (scale index, y, x) and the interpolated responses.
    Peaks on the border of the volume, or whose fitted offset leaves the
    voxel, keep their integer position.
    """
    peaks = np.asarray(peaks, np.int64)
    out = peaks.astype(np.float64)
    values = stack[peaks[:, 0], peaks[:, 1], peaks[:, 2]].astype(np.float64) if len(peaks) else np.zeros(0)
    if not len(peaks):
        return out, values
    s_count, height, width = stack.shape
    inner = (
        (peaks[:, 0] > 0)
        & (peaks[:, 0] < s_count - 1)
        & (peaks[:, 1] > 0)
        & (peaks[:, 1] < height - 1)
        & (peaks[:, 2] > 0)
        & (peaks[:, 2] < width - 1)
    )
    idx = np.nonzero(inner)[0]
    if len(idx):
        s, y, x = peaks[idx, 0], peaks[idx, 1], peaks[idx, 2]

        def at(ds: int, dy: int, dx: int) -> np.ndarray:
            return stack[s + ds, y + dy, x + dx].astype(np.float64)

        c = at(0, 0, 0)
        g = np.stack([(at(1, 0, 0) - at(-1, 0, 0)) / 2, (at(0, 1, 0) - at(0, -1, 0)) / 2, (at(0, 0, 1) - at(0, 0, -1)) / 2], axis=1)
        hss = at(1, 0, 0) + at(-1, 0, 0) - 2 * c
        hyy = at(0, 1, 0) + at(0, -1, 0) - 2 * c
        hxx = at(0, 0, 1) + at(0, 0, -1) - 2 * c
        hsy = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) / 4
        hsx = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) / 4
        hyx = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) / 4
        hessian = np.stack(
            [np.stack([hss, hsy, hsx], 1), np.stack([hsy, hyy, hyx], 1), np.stack([hsx, hyx, hxx], 1)],
            axis=1,
        )
        det = np.linalg.det(hessian)
        ok = np.abs(det) > 1e-12
        offset = np.zeros_like(g)
        if ok.any():
            offset[ok] = -np.linalg.solve(hessian[ok], g[ok][:, :, None])[:, :, 0]
        ok &= np.all(np.abs(offset) < 1.0, axis=1)
        out[idx[ok]] += offset[ok]
        values[idx[ok]] = c[ok] + 0.5 * np.einsum("ij,ij->i", g[ok], offset[ok])
    return out, values


def _sigma_at(sigmas: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Interpolate sigma at fractional slice indices, geometrically."""
    logs = np.log(sigmas)
    return np.exp(np.interp(index, np.arange(len(sigmas)), logs, left=None, right=None))


def circle_overlap(r1: float | np.ndarray, r2: float | np.ndarray, d: float | np.ndarray) -> np.ndarray:
    """Area of intersection of two circles divided by the area of the smaller one."""
    r1 = np.asarray(r1, np.float64)
    r2 = np.asarray(r2, np.float64)
    d = np.asarray(d, np.float64)
    small = np.minimum(r1, r2)
    big = np.maximum(r1, r2)
    out = np.zeros(np.broadcast(r1, r2, d).shape)
    contained = d <= big - small
    out = np.where(contained, 1.0, out)
    partial = (d < r1 + r2) & ~contained
    if np.any(partial):
        dd = np.maximum(d, 1e-12)
        a1 = np.clip((dd**2 + r1**2 - r2**2) / (2 * dd * r1), -1, 1)
        a2 = np.clip((dd**2 + r2**2 - r1**2) / (2 * dd * r2), -1, 1)
        area = (
            r1**2 * np.arccos(a1)
            + r2**2 * np.arccos(a2)
            - 0.5 * np.sqrt(np.maximum((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2), 0))
        )
        out = np.where(partial, area / (math.pi * small**2), out)
    return out


def prune_overlaps(blobs: list[Blob], overlap: float = 0.5) -> list[Blob]:
    """Greedy non maximum suppression on circles, strongest response first."""
    order = sorted(blobs, key=lambda b: b.response, reverse=True)
    kept: list[Blob] = []
    xs = np.empty(len(order))
    ys = np.empty(len(order))
    rs = np.empty(len(order))
    for blob in order:
        m = len(kept)
        if m:
            d = np.hypot(xs[:m] - blob.x, ys[:m] - blob.y)
            near = d < rs[:m] + blob.radius
            if np.any(near) and np.any(circle_overlap(rs[:m][near], blob.radius, d[near]) > overlap):
                continue
        xs[m], ys[m], rs[m] = blob.x, blob.y, blob.radius
        kept.append(blob)
    return kept


def suppress_sidelobes(blobs: list[Blob], ratio: float = 3.0, reach: float = 1.6) -> list[Blob]:
    """Drop peaks that are the surround of a much stronger blob.

    The Laplacian of a disk is ringed by a weaker response of the opposite
    sign. A signed single channel search never sees it, but a magnitude or a
    per channel search of both signs does, as a necklace of small peaks
    around every strong blob. A peak is dropped when a blob ``ratio`` times
    stronger is within ``reach`` of its radii and, if both carry a per channel
    ``_signature``, the two signatures point in nearly opposite directions
    (cosine below -0.7). A faint real dot next to a strong one keeps its own
    channels: pale pink beside black is mostly ``a``, which a black dot's ring
    cannot produce.
    """
    if len(blobs) < 2:
        return blobs
    order = sorted(blobs, key=lambda b: b.response, reverse=True)
    xs = np.array([b.x for b in order])
    ys = np.array([b.y for b in order])
    rs = np.array([b.radius for b in order])
    vs = np.array([b.response for b in order])
    signatures = [getattr(b, "_signature", None) for b in order]
    have = all(sig is not None for sig in signatures)
    sig = np.array(signatures, np.float64) if have else None
    alive = np.ones(len(order), bool)
    floor = vs.min() * ratio
    for i in range(len(order)):
        if vs[i] < floor:
            break
        if not alive[i]:
            continue
        weaker = slice(i + 1, None)
        hit = (np.hypot(xs[weaker] - xs[i], ys[weaker] - ys[i]) < reach * rs[i] + rs[weaker]) & (vs[weaker] * ratio < vs[i])
        if sig is not None:
            # A ring is the parent's own response with the sign flipped, in the same channels.
            cosine = (sig[weaker] @ sig[i]) / np.maximum(np.linalg.norm(sig[weaker], axis=1) * np.linalg.norm(sig[i]), 1e-12)
            hit &= cosine < -0.7
        alive[weaker] &= ~hit
    return [b for b, keep in zip(order, alive) if keep]


def detect_scale_space(
    image: np.ndarray,
    kind: str = "log",
    min_radius: float = 3.0,
    max_radius: float = 60.0,
    per_octave: int = 5,
    threshold: float = 0.1,
    overlap: float = 0.5,
    polarity: str = "both",
    subpixel: bool = True,
    pyramid: bool = True,
    split_channels: bool = True,
) -> list[Blob]:
    """Find blobs with radius in [min_radius, max_radius] by scale space search.

    ``threshold`` is in the units of the image (for a CIELAB image, Delta E
    times ``DISK_PEAK``). Radii are ``sqrt(2) * sigma``. The scale volume is
    streamed three slices at a time, so memory does not grow with the number
    of scales.
    """
    sigmas = sigma_ladder(min_radius / SQRT2, max_radius / SQRT2, per_octave)
    kernel = np.ones((3, 3), np.uint8)
    window: list[tuple[float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = []
    blobs: list[Blob] = []
    effective: list[float] = []

    def process(prev, cur, nxt, index: int) -> None:
        for m, response in enumerate(cur[1]):
            neighborhood = cur[2][m].copy()
            if prev is not None:
                np.maximum(neighborhood, prev[2][m], out=neighborhood)
            if nxt is not None:
                np.maximum(neighborhood, nxt[2][m], out=neighborhood)
            peaks = np.argwhere((response >= neighborhood) & (response > threshold))
            if not len(peaks):
                continue
            below = prev[1][m] if prev is not None else response
            above = nxt[1][m] if nxt is not None else response
            mini = np.stack([below, response, above])
            local = np.concatenate([np.ones((len(peaks), 1), np.int64), peaks], axis=1)
            if subpixel and prev is not None and nxt is not None:
                position, values = refine_peaks(mini, local)
            else:
                position = local.astype(np.float64)
                values = response[peaks[:, 0], peaks[:, 1]].astype(np.float64)
            signature = np.stack([c[peaks[:, 0], peaks[:, 1]] for c in cur[3]], axis=1)
            for (ds, y, x), v, sig in zip(position, values, signature):
                blob = Blob(float(x), float(y), float(index + ds - 1), float(v))
                blob._signature = sig  # type: ignore[attr-defined]
                blobs.append(blob)

    for sigma, maps, signed in _slices(image, sigmas, kind, polarity, pyramid, split_channels):
        effective.append(sigma)
        dilated = [cv2.dilate(r, kernel, borderType=cv2.BORDER_REPLICATE) for r in maps]
        window.append((sigma, maps, dilated, signed))
        if len(window) == 2:
            process(None, window[0], window[1], 0)
        elif len(window) == 3:
            process(window[0], window[1], window[2], len(effective) - 2)
            window.pop(0)
    if len(window) == 2:
        process(window[0], window[1], None, len(effective) - 1)
    elif len(window) == 1:
        process(None, window[0], None, 0)

    effective_arr = np.array(effective)
    for b in blobs:  # the radius field held the fractional scale index until now
        b.radius = float(SQRT2 * _sigma_at(effective_arr, np.array([b.radius]))[0])
    if polarity == "both" or (split_channels and np.asarray(image).ndim == 3):
        blobs = suppress_sidelobes(blobs)
    for b in blobs:
        b.__dict__.pop("_signature", None)
    return prune_overlaps(blobs, overlap)


def detect_log(image: np.ndarray, **kwargs) -> list[Blob]:
    return detect_scale_space(image, "log", **kwargs)


def detect_dog(image: np.ndarray, **kwargs) -> list[Blob]:
    return detect_scale_space(image, "dog", **kwargs)


def detect_doh(image: np.ndarray, **kwargs) -> list[Blob]:
    return detect_scale_space(image, "doh", **kwargs)


# ------------------------------------------------------ contour filtering ----


@dataclass
class ShapeStats:
    """Geometric descriptors of one contour."""

    area: float
    perimeter: float
    circularity: float  # 4 pi A / P^2
    convexity: float  # A / hull area
    inertia: float  # squared ratio of minor to major axis
    center: tuple[float, float]
    equivalent_radius: float


def shape_stats(contour: np.ndarray) -> ShapeStats | None:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    spread = math.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11**2, 0.0))
    major = (mu20 + mu02 + spread) / 2
    minor = (mu20 + mu02 - spread) / 2
    inertia = minor / major if major > 0 else 0.0
    return ShapeStats(
        area=area,
        perimeter=perimeter,
        circularity=4 * math.pi * area / perimeter**2 if perimeter > 0 else 0.0,
        convexity=area / hull_area if hull_area > 0 else 0.0,
        inertia=inertia,
        center=(cx, cy),
        equivalent_radius=math.sqrt(area / math.pi),
    )


def detect_contours(mask: np.ndarray, f: BlobFilter = CONTOUR_FILTER, nested: bool = False) -> list[Blob]:
    """Blobs from the contours of a binary mask, filtered like SimpleBlobDetector.

    With ``nested`` every contour counts, including regions inside holes of
    other regions (needed when the mask is "everything that is not an edge").
    """
    mode = cv2.RETR_LIST if nested else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), mode, cv2.CHAIN_APPROX_NONE)
    blobs: list[Blob] = []
    for contour in contours:
        stats = shape_stats(contour)
        if stats is None:
            continue
        if f.min_area is not None and stats.area < f.min_area:
            continue
        if f.max_area is not None and stats.area > f.max_area:
            continue
        if f.min_circularity is not None and stats.circularity < f.min_circularity:
            continue
        if f.min_convexity is not None and stats.convexity < f.min_convexity:
            continue
        if f.min_inertia is not None and stats.inertia < f.min_inertia:
            continue
        blobs.append(Blob(stats.center[0], stats.center[1], stats.equivalent_radius, response=stats.circularity))
    return blobs
