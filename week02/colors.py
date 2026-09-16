"""Option 1: Color Me Impressed.

Split an image into one image per color and locate the center of every
individually colored object.

The approach follows the lecture hints:

1. Convert BGR to HSV with ``cv2.cvtColor`` so hue carries the color and
   saturation/value carry "how colorful" and "how bright".
2. Build one binary mask per named color with ``cv2.inRange``.
3. Apply each mask to the original image (masking) to get a single color layer.
4. Find object centers with ``np.where`` and ``np.mean`` on each connected blob.

The named color bands are chosen so that the masks form an exact partition of
the image: every pixel belongs to exactly one color.

Beyond the assignment:

* ``auto`` mode learns the image's own palette with k-means in CIELAB (a
  perceptually uniform color space) and picks the number of colors with the
  silhouette score, so two different reds do not get lumped together.
* Every object is classified into a SUAS target shape (see ``shapes.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import shapes

# OpenCV stores hue as 0..179 (degrees / 2), saturation and value as 0..255.
BLACK_MAX_V = 50
ACHROMATIC_MAX_S = 50
WHITE_MIN_V = 190


@dataclass(frozen=True)
class ColorBand:
    """A named color defined by one or more inclusive HSV ranges."""

    name: str
    ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    swatch_bgr: tuple[int, int, int]


BROWN_MAX_V = 150


def _hue(lo: int, hi: int, v_lo: int = BLACK_MAX_V + 1, v_hi: int = 255) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (lo, ACHROMATIC_MAX_S + 1, v_lo), (hi, 255, v_hi)


COLOR_BANDS: tuple[ColorBand, ...] = (
    # Red wraps around the hue circle, so it needs two ranges.
    ColorBand("red", (_hue(0, 8), _hue(170, 179)), (40, 40, 220)),
    # Brown is dark orange / dark yellow, so it is carved out by brightness.
    ColorBand("brown", (_hue(9, 30, v_hi=BROWN_MAX_V),), (40, 90, 140)),
    ColorBand("orange", (_hue(9, 20, v_lo=BROWN_MAX_V + 1),), (20, 140, 255)),
    ColorBand("yellow", (_hue(21, 30, v_lo=BROWN_MAX_V + 1), _hue(31, 34)), (40, 220, 240)),
    ColorBand("green", (_hue(35, 85),), (60, 190, 60)),
    ColorBand("cyan", (_hue(86, 100),), (220, 210, 40)),
    ColorBand("blue", (_hue(101, 130),), (220, 110, 30)),
    ColorBand("purple", (_hue(131, 150),), (200, 60, 140)),
    ColorBand("pink", (_hue(151, 169),), (180, 110, 250)),
    ColorBand("black", (((0, 0, 0), (179, 255, BLACK_MAX_V)),), (30, 30, 30)),
    ColorBand(
        "gray",
        (((0, 0, BLACK_MAX_V + 1), (179, ACHROMATIC_MAX_S, WHITE_MIN_V - 1)),),
        (140, 140, 140),
    ),
    ColorBand(
        "white",
        (((0, 0, WHITE_MIN_V), (179, ACHROMATIC_MAX_S, 255)),),
        (245, 245, 245),
    ),
)

BANDS_BY_NAME = {band.name: band for band in COLOR_BANDS}


@dataclass
class ColorObject:
    """One connected blob of a single color."""

    center: tuple[float, float]  # (x, y) in pixels
    area: int
    bbox: tuple[int, int, int, int]  # x, y, width, height
    shape: str | None = None
    shape_confidence: float | None = None
    touches_border: bool = False


@dataclass
class ColorLayer:
    """Everything we know about one color in the image."""

    name: str
    mask: np.ndarray = field(repr=False)
    pixels: int
    coverage: float
    center: tuple[float, float] | None
    objects: list[ColorObject]

    def isolate(self, image: np.ndarray) -> np.ndarray:
        """Return the image with every pixel outside this color set to black."""
        return cv2.bitwise_and(image, image, mask=self.mask)

    def to_bgra(self, image: np.ndarray) -> np.ndarray:
        """Return a BGRA image where alpha is the color mask (true cut-out)."""
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = self.mask
        return bgra

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("mask")
        return data


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk, raising a helpful error instead of returning None."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No image found at {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path} as an image")
    return image


def color_mask(hsv: np.ndarray, band: ColorBand) -> np.ndarray:
    """Build a 0/255 mask of all pixels inside any of the band's HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in band.ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    return mask


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    """Center of mass of a mask using np.where + np.mean (the lecture hint)."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def find_objects(
    mask: np.ndarray, min_area: int, open_kernel: int = 3, classify_shapes: bool = True, min_shape_area: int = 150
) -> list[ColorObject]:
    """Split a color mask into connected blobs and locate each blob's center.

    A small morphological opening removes speckle (JPEG noise, anti-aliased
    edges) so that a single stop sign is reported as one object, not fifty.
    """
    cleaned = mask
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    objects: list[ColorObject] = []
    for label in range(1, count):  # label 0 is the background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        center = mask_center((labels == label).astype(np.uint8))
        assert center is not None
        x, y, w, h = (int(v) for v in stats[label, :4])
        touches_border = x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]
        obj = ColorObject(center=center, area=area, bbox=(x, y, w, h), touches_border=touches_border)
        # A region cut off by the frame edge has no meaningful shape.
        if classify_shapes and area >= min_shape_area and not touches_border:
            # Classify inside a padded crop so the search stays cheap.
            pad = 4
            crop = (labels[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] == label).astype(np.uint8) * 255
            match = shapes.classify(crop, min_area=min_shape_area)
            if match is not None and match.shape != "irregular":
                obj.shape, obj.shape_confidence = match.shape, match.confidence
        objects.append(obj)
    objects.sort(key=lambda obj: obj.area, reverse=True)
    return objects


def named_masks(image: np.ndarray, bands: tuple[ColorBand, ...] = COLOR_BANDS) -> list[tuple[str, np.ndarray]]:
    """One mask per named color band. The default bands partition the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return [(band.name, color_mask(hsv, band)) for band in bands]


def name_color(bgr: np.ndarray | tuple[int, int, int]) -> str:
    """Name a single BGR color with the same HSV bands used for masks."""
    pixel = np.uint8([[list(np.clip(np.round(bgr), 0, 255))]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    for band in COLOR_BANDS:
        if color_mask(hsv, band)[0, 0]:
            return band.name
    return "unknown"


def _silhouette(samples: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient, computed exactly (and vectorized) on a sample."""
    distances = np.sqrt(np.maximum(((samples[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2), 0))
    k = int(labels.max()) + 1
    onehot = np.eye(k, dtype=np.float64)[labels]
    counts = onehot.sum(axis=0)
    sums = distances @ onehot  # (n, k): total distance from each sample to each cluster
    own_counts = counts[labels]
    a = sums[np.arange(len(labels)), labels] / np.maximum(own_counts - 1, 1)
    mean_to_cluster = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
    mean_to_cluster[np.arange(len(labels)), labels] = np.inf
    b = mean_to_cluster.min(axis=1)
    scores = np.where(own_counts > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-9), 0.0)
    return float(scores.mean())


def kmeans_masks(
    image: np.ndarray,
    k: int | None = None,
    k_range: tuple[int, int] = (2, 8),
    sample_size: int = 20000,
    seed: int = 0,
    merge_distance: float = 10.0,
) -> list[tuple[str, np.ndarray]]:
    """Discover the image's own palette with k-means in CIELAB space.

    CIELAB is perceptually uniform, so Euclidean distance approximates how
    different two colors look. When ``k`` is None the number of colors is
    chosen by the silhouette score. Clusters closer than ``merge_distance``
    (Delta E) are merged, and every cluster is named by its center color.
    """
    rng = np.random.default_rng(seed)
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    sample = lab[rng.choice(len(lab), size=min(sample_size, len(lab)), replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    cv2.setRNGSeed(seed)

    def run(n: int) -> tuple[np.ndarray, np.ndarray]:
        _, labels, centers = cv2.kmeans(sample, n, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        return labels.ravel(), centers

    if k is None:
        probe = rng.choice(len(sample), size=min(1500, len(sample)), replace=False)
        best = None
        for n in range(k_range[0], k_range[1] + 1):
            labels, centers = run(n)
            score = _silhouette(sample[probe], labels[probe])
            if best is None or score > best[0]:
                best = (score, centers)
        centers = best[1]
    else:
        centers = run(k)[1]

    # Merge perceptually indistinguishable clusters.
    merged: list[np.ndarray] = []
    for center in sorted(centers, key=lambda c: c[0]):
        if all(np.linalg.norm(center - other) >= merge_distance for other in merged):
            merged.append(center)
    centers = np.array(merged, np.float32)

    # Assign every pixel to its nearest center, in chunks to bound memory.
    assignment = np.empty(len(lab), np.int32)
    for start in range(0, len(lab), 262144):
        chunk = lab[start : start + 262144]
        assignment[start : start + 262144] = np.argmin(((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
    assignment = assignment.reshape(image.shape[:2])

    results: list[tuple[str, np.ndarray]] = []
    seen: dict[str, int] = {}
    for index, center in enumerate(centers):
        bgr = cv2.cvtColor(center.reshape(1, 1, 3), cv2.COLOR_Lab2BGR).reshape(3) * 255
        name = name_color(bgr)
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}-{seen[name]}"
        results.append((label, (assignment == index).astype(np.uint8) * 255))
    return results


def split_colors(
    image: np.ndarray,
    min_coverage: float = 0.01,
    min_object_fraction: float = 0.0008,
    mode: str = "named",
    k: int | None = None,
    classify_shapes: bool = True,
) -> list[ColorLayer]:
    """Split a BGR image into color layers, largest first.

    Args:
        image: BGR uint8 image as loaded by OpenCV.
        min_coverage: drop colors covering less than this fraction of the image,
            unless they contain at least one real object (such as a letter).
        min_object_fraction: blobs smaller than this fraction of the image are
            treated as noise when locating objects.
        mode: ``named`` uses fixed HSV bands (red, orange, ...); ``auto`` learns
            the palette with k-means in CIELAB.
        k: number of clusters for ``auto`` mode; chosen automatically if None.
        classify_shapes: also report the geometric shape of each object.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3 channel BGR image")
    if image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit image")
    if mode == "named":
        masks = named_masks(image)
    elif mode == "auto":
        masks = kmeans_masks(image, k=k)
    else:
        raise ValueError("mode must be 'named' or 'auto'")

    total = image.shape[0] * image.shape[1]
    min_area = max(1, int(round(total * min_object_fraction)))
    layers: list[ColorLayer] = []
    for name, mask in masks:
        pixels = int(cv2.countNonZero(mask))
        coverage = pixels / total
        if pixels < min_area:
            continue
        objects = find_objects(mask, min_area, classify_shapes=classify_shapes)
        if coverage < min_coverage and not objects:
            continue
        layers.append(
            ColorLayer(
                name=name,
                mask=mask,
                pixels=pixels,
                coverage=coverage,
                center=mask_center(mask),
                objects=objects,
            )
        )
    layers.sort(key=lambda layer: layer.pixels, reverse=True)
    return layers


def format_report(layers: list[ColorLayer]) -> str:
    """Human readable summary of colors and object centers."""
    lines = []
    for layer in layers:
        header = f"{layer.name:<7} {layer.coverage * 100:5.1f}% of pixels"
        if layer.center is not None:
            header += f", overall center (x={layer.center[0]:.1f}, y={layer.center[1]:.1f})"
        lines.append(header)
        if not layer.objects:
            lines.append("        no objects above the size threshold")
        for i, obj in enumerate(layer.objects, start=1):
            shape = f", {obj.shape} ({obj.shape_confidence:.2f} IoU)" if obj.shape else ""
            lines.append(
                f"        object {i}: center (x={obj.center[0]:.1f}, y={obj.center[1]:.1f}), "
                f"area {obj.area} px{shape}"
            )
    return "\n".join(lines)


def _checkerboard(height: int, width: int, cell: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = ((yy // cell + xx // cell) % 2).astype(np.uint8)
    tile = np.where(board[..., None] == 1, 58, 44).astype(np.uint8)
    return np.repeat(tile, 3, axis=2)


def render_layer(image: np.ndarray, layer: ColorLayer) -> np.ndarray:
    """Color layer on a checkerboard so that even the black layer is visible."""
    out = _checkerboard(*image.shape[:2])
    out[layer.mask > 0] = image[layer.mask > 0]
    return out


def annotate(image: np.ndarray, layers: list[ColorLayer]) -> np.ndarray:
    """Draw a crosshair and label on every detected object center."""
    out = image.copy()
    scale = max(image.shape[:2]) / 800
    thickness = max(1, int(round(2 * scale)))
    for layer in layers:
        base = layer.name.split("-")[0]
        swatch = BANDS_BY_NAME[base].swatch_bgr if base in BANDS_BY_NAME else (0, 255, 0)
        for obj in layer.objects:
            if obj.touches_border:
                continue  # background regions, not objects worth marking
            cx, cy = (int(round(v)) for v in obj.center)
            size = int(14 * scale) + 4
            cv2.drawMarker(out, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, size + 4, thickness + 2)
            cv2.drawMarker(out, (cx, cy), swatch, cv2.MARKER_CROSS, size, thickness)
            label = f"{layer.name} {obj.shape} ({cx},{cy})" if obj.shape else f"{layer.name} ({cx},{cy})"
            org = (cx + size // 2 + 4, cy - size // 2)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def contact_sheet(
    image: np.ndarray,
    layers: list[ColorLayer],
    tile_width: int = 360,
    columns: int = 3,
) -> np.ndarray:
    """One picture with the original, the annotated centers and every color layer."""
    panels = [("original", image), ("object centers", annotate(image, layers))]
    panels += [(f"{layer.name}  {layer.coverage * 100:.1f}%", render_layer(image, layer)) for layer in layers]

    h, w = image.shape[:2]
    tile_height = max(1, int(round(h * tile_width / w)))
    label_height = 30
    rows = (len(panels) + columns - 1) // columns
    sheet = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3), 24, dtype=np.uint8
    )
    for i, (title, panel) in enumerate(panels):
        r, c = divmod(i, columns)
        y0 = r * (tile_height + label_height)
        x0 = c * tile_width
        sheet[y0 + label_height : y0 + label_height + tile_height, x0 : x0 + tile_width] = cv2.resize(
            panel, (tile_width, tile_height), interpolation=cv2.INTER_AREA
        )
        cv2.putText(sheet, title, (x0 + 10, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return sheet


def save_layers(image: np.ndarray, layers: list[ColorLayer], out_dir: str | Path) -> list[Path]:
    """Write every layer as a transparent PNG plus the contact sheet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for layer in layers:
        path = out_dir / f"layer_{layer.name}.png"
        cv2.imwrite(str(path), layer.to_bgra(image))
        written.append(path)
    for name, picture in (("centers.png", annotate(image, layers)), ("contact_sheet.png", contact_sheet(image, layers))):
        path = out_dir / name
        cv2.imwrite(str(path), picture)
        written.append(path)
    return written


def show_layers(image: np.ndarray, layers: list[ColorLayer]) -> None:
    """Display each color in its own window (press any key to close)."""
    cv2.imshow("original", image)
    cv2.imshow("object centers", annotate(image, layers))
    for layer in layers:
        cv2.imshow(f"color: {layer.name}", layer.isolate(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
