"""Global optimization and rendering of an image mosaic.

Three classic building blocks, implemented directly on NumPy and OpenCV:

``bundle_adjust``
    Chaining pairwise transforms accumulates drift: a 0.5 px error per frame
    becomes 50 px after 100 frames. When the flight path revisits an area
    (the next lawnmower pass overlaps the previous one) we get extra "loop
    closure" constraints. Bundle adjustment finds the set of per frame
    similarity transforms that best agrees with every constraint at once.

    A similarity is the complex map ``z -> alpha z + beta``, with
    ``alpha = a + ib`` (rotation and scale) and ``beta = tx + i ty``. For a
    correspondence ``p`` in frame i and ``q`` in frame j, the residual is
    measured in frame j's own pixels, ``(alpha_i p + beta_i - beta_j) / alpha_j - q``,
    plus the mirror residual in frame i. The obvious linear formulation,
    ``G_i(p) - G_j(q)``, measures error in mosaic pixels instead, and its
    cheapest solution is to shrink every frame (scale collapse). The residual
    is holomorphic in the parameters, so its Jacobian comes straight from
    complex derivatives. The problem is solved with Levenberg Marquardt, with
    the first frame fixed to remove the gauge freedom and Huber weights
    (iteratively reweighted least squares) to discount bad correspondences
    that slipped through RANSAC.

``gain_compensation``
    Auto exposure makes the same patch of ground brighter in one frame than
    the next. Following Brown and Lowe (IJCV 2007), one gain per frame is
    solved from the mean intensities of every overlapping pair, with a prior
    that keeps gains near 1.

``multiband_blend``
    Feathering blurs misalignments into ghosts, while a hard seam shows
    exposure steps. Burt and Adelson's multi band blending mixes low
    frequencies over a wide region and high frequencies over a narrow one,
    using Laplacian pyramids of each image and Gaussian pyramids of the seam
    masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ------------------------------------------------------ transformations ----


def to_params(matrix: np.ndarray) -> np.ndarray:
    """3x3 similarity to (a, b, tx, ty), projecting onto the similarity group."""
    a = (matrix[0, 0] + matrix[1, 1]) / 2
    b = (matrix[1, 0] - matrix[0, 1]) / 2
    return np.array([a, b, matrix[0, 2], matrix[1, 2]])


def from_params(params: np.ndarray) -> np.ndarray:
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def frame_corners(width: int, height: int) -> np.ndarray:
    return np.float64([[0, 0], [width, 0], [width, height], [0, height]])


def footprint_overlap(m1: np.ndarray, m2: np.ndarray, width: int, height: int) -> float:
    """Intersection area of two frame footprints divided by one frame's area."""
    c = frame_corners(width, height)
    p1 = apply(m1, c).astype(np.float32)
    p2 = apply(m2, c).astype(np.float32)
    area, _ = cv2.intersectConvexConvex(p1, p2)
    return float(area) / (width * height * abs(np.linalg.det(m1[:2, :2])))


# ---------------------------------------------------- bundle adjustment ----


@dataclass
class Constraint:
    i: int
    j: int
    src: np.ndarray
    """Points in frame i."""
    dst: np.ndarray
    """Matching points in frame j."""
    loop_closure: bool = False


@dataclass
class AdjustmentReport:
    rms_before: float
    rms_after: float
    constraints: int
    loop_closures: int
    correspondences: int


def _complex(params: np.ndarray) -> tuple[complex, complex]:
    a, b, tx, ty = params
    return complex(a, b), complex(tx, ty)


def _frame_residuals(params: np.ndarray, c: Constraint) -> tuple[np.ndarray, np.ndarray]:
    """Residuals of a constraint in frame j pixels and in frame i pixels."""
    alpha_i, beta_i = _complex(params[c.i])
    alpha_j, beta_j = _complex(params[c.j])
    p = c.src[:, 0] + 1j * c.src[:, 1]
    q = c.dst[:, 0] + 1j * c.dst[:, 1]
    forward = (alpha_i * p + beta_i - beta_j) / alpha_j - q
    backward = (alpha_j * q + beta_j - beta_i) / alpha_i - p
    return forward, backward


def constraint_rms(transforms: list[np.ndarray], constraints: list[Constraint]) -> float:
    """RMS residual in frame pixels over all correspondences, both directions."""
    params = np.array([to_params(m) for m in transforms])
    errors = []
    for c in constraints:
        forward, backward = _frame_residuals(params, c)
        errors += [np.abs(forward) ** 2, np.abs(backward) ** 2]
    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def _real_block(derivative: np.ndarray) -> np.ndarray:
    """Complex derivative d r / d x per point to the real (2m x 2) Jacobian columns."""
    u, v = derivative.real, derivative.imag
    return np.vstack([np.stack([u, -v], axis=1), np.stack([v, u], axis=1)])


def _normal_equations(params: np.ndarray, constraints: list[Constraint], robust: list[np.ndarray], n: int):
    H = np.zeros((4 * n, 4 * n))
    g = np.zeros(4 * n)
    cost = 0.0
    for c, w in zip(constraints, robust):
        alpha_i, beta_i = _complex(params[c.i])
        alpha_j, beta_j = _complex(params[c.j])
        p = c.src[:, 0] + 1j * c.src[:, 1]
        q = c.dst[:, 0] + 1j * c.dst[:, 1]
        for (alpha_a, beta_a, x), (alpha_b, beta_b, y), (ia, ib) in (
            ((alpha_i, beta_i, p), (alpha_j, beta_j, q), (c.i, c.j)),
            ((alpha_j, beta_j, q), (alpha_i, beta_i, p), (c.j, c.i)),
        ):
            mapped = alpha_a * x + beta_a - beta_b
            r = mapped / alpha_b - y
            ones = np.ones_like(x)
            # Columns: alpha_a, beta_a, alpha_b, beta_b (each as real and imaginary part).
            J = np.hstack([
                _real_block(x / alpha_b),
                _real_block(ones / alpha_b),
                _real_block(-mapped / alpha_b**2),
                _real_block(-ones / alpha_b),
            ])
            residual = np.concatenate([r.real, r.imag])
            W = np.concatenate([w, w])
            idx = np.r_[4 * ia : 4 * ia + 2, 4 * ia + 2 : 4 * ia + 4, 4 * ib : 4 * ib + 2, 4 * ib + 2 : 4 * ib + 4]
            JW = J * W[:, None]
            H[np.ix_(idx, idx)] += JW.T @ J
            np.add.at(g, idx, JW.T @ residual)
            cost += float(np.sum(W * residual**2))
    return H, g, cost


def bundle_adjust(
    transforms: list[np.ndarray],
    constraints: list[Constraint],
    fixed: int = 0,
    iterations: int = 15,
    huber: float = 2.0,
    max_points: int = 80,
    seed: int = 0,
) -> tuple[list[np.ndarray], AdjustmentReport]:
    """Jointly refine all frame transforms against all pairwise constraints.

    Args:
        transforms: initial 3x3 similarity per frame (frame to mosaic).
        constraints: point correspondences between pairs of frames.
        fixed: index of the frame whose transform is held constant.
        iterations: Levenberg Marquardt iterations, each followed by reweighting.
        huber: Huber loss threshold in pixels.
        max_points: correspondences kept per constraint (spatially random).
    """
    n = len(transforms)
    rng = np.random.default_rng(seed)
    sampled: list[Constraint] = []
    for c in constraints:
        if len(c.src) > max_points:
            keep = rng.choice(len(c.src), max_points, replace=False)
            c = Constraint(c.i, c.j, c.src[keep], c.dst[keep], c.loop_closure)
        sampled.append(c)

    params = np.array([to_params(m) for m in transforms], np.float64)
    report = AdjustmentReport(
        rms_before=constraint_rms(transforms, sampled),
        rms_after=0.0,
        constraints=len(sampled),
        loop_closures=sum(c.loop_closure for c in sampled),
        correspondences=sum(len(c.src) for c in sampled),
    )
    if n < 2 or not sampled:
        report.rms_after = report.rms_before
        return [from_params(p) for p in params], report

    robust = [np.ones(len(c.src)) for c in sampled]
    free = np.ones(4 * n, bool)
    free[4 * fixed : 4 * fixed + 4] = False
    damping = 1e-3
    for _ in range(iterations):
        H, g, cost = _normal_equations(params, sampled, robust, n)
        H_ff, g_f = H[np.ix_(free, free)], g[free]
        for _attempt in range(8):
            step = np.linalg.solve(H_ff + damping * np.diag(np.diag(H_ff) + 1e-9), -g_f)
            candidate = params.reshape(-1).copy()
            candidate[free] += step
            candidate = candidate.reshape(n, 4)
            _, _, new_cost = _normal_equations(candidate, sampled, robust, n)
            if new_cost < cost:
                params = candidate
                damping = max(damping / 10, 1e-7)
                break
            damping *= 10
        # Huber reweighting from residuals in frame pixels.
        for k, c in enumerate(sampled):
            forward, backward = _frame_residuals(params, c)
            error = np.sqrt((np.abs(forward) ** 2 + np.abs(backward) ** 2) / 2)
            robust[k] = np.where(error <= huber, 1.0, huber / np.maximum(error, 1e-12))

    adjusted = [from_params(p) for p in params]
    report.rms_after = constraint_rms(adjusted, sampled)
    return adjusted, report


# ---------------------------------------------------------- canvas math ----


@dataclass
class Canvas:
    width: int
    height: int
    offset: np.ndarray
    """3x3 translation from mosaic coordinates to canvas pixels."""


def plan_canvas(sizes: list[tuple[int, int]], transforms: list[np.ndarray], max_pixels: int, pad_multiple: int = 1) -> Canvas:
    points = np.vstack([apply(m, frame_corners(w, h)) for (w, h), m in zip(sizes, transforms)])
    x0, y0 = np.floor(points.min(axis=0))
    x1, y1 = np.ceil(points.max(axis=0))
    width = int(x1 - x0)
    height = int(y1 - y0)
    width += (-width) % pad_multiple
    height += (-height) % pad_multiple
    if width * height > max_pixels:
        raise MemoryError(
            f"Canvas would be {width}x{height} px; reduce --scale or raise the limit. "
            "A huge canvas usually means alignment drifted."
        )
    offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return Canvas(width, height, offset)


def _roi(matrix: np.ndarray, size: tuple[int, int], canvas: Canvas, margin: int = 0, multiple: int = 1) -> tuple[int, int, int, int]:
    pts = apply(matrix, frame_corners(*size))
    x0 = int(np.floor(pts[:, 0].min())) - margin
    y0 = int(np.floor(pts[:, 1].min())) - margin
    x1 = int(np.ceil(pts[:, 0].max())) + margin
    y1 = int(np.ceil(pts[:, 1].max())) + margin
    x0 -= x0 % multiple
    y0 -= y0 % multiple
    x1 += (-x1) % multiple
    y1 += (-y1) % multiple
    return max(0, x0), max(0, y0), min(canvas.width, x1), min(canvas.height, y1)


def _warp(image: np.ndarray, matrix: np.ndarray, roi: tuple[int, int, int, int], border=cv2.BORDER_CONSTANT) -> np.ndarray:
    x0, y0, x1, y1 = roi
    local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ matrix
    return cv2.warpAffine(image, local[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=border)


def feather_weight(width: int, height: int) -> np.ndarray:
    """1 at the frame center falling to 0 at the border (distance transform)."""
    mask = np.zeros((height, width), np.uint8)
    mask[1:-1, 1:-1] = 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return (dist / max(float(dist.max()), 1.0)).astype(np.float32)


# ----------------------------------------------------- gain compensation ----


def gain_compensation(
    frames: list[np.ndarray],
    transforms: list[np.ndarray],
    sigma_n: float = 10.0,
    sigma_g: float = 0.1,
    work_pixels: int = 1_000_000,
) -> np.ndarray:
    """Per frame exposure gains (Brown and Lowe 2007). Returns an (N,) array."""
    n = len(frames)
    if n < 2:
        return np.ones(n)
    sizes = [(f.shape[1], f.shape[0]) for f in frames]
    full = plan_canvas(sizes, transforms, max_pixels=10**12)
    scale = min(1.0, np.sqrt(work_pixels / (full.width * full.height)))
    shrink = np.diag([scale, scale, 1.0])
    small = [shrink @ full.offset @ m for m in transforms]
    canvas = Canvas(int(np.ceil(full.width * scale)) + 1, int(np.ceil(full.height * scale)) + 1, np.eye(3))

    warped = []
    for frame, matrix, size in zip(frames, small, sizes):
        roi = _roi(matrix, size, canvas)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ones = np.ones(gray.shape, np.float32)
        warped.append((roi, _warp(gray, matrix, roi), _warp(ones, matrix, roi) > 0.99))

    counts = np.zeros((n, n))
    means = np.zeros((n, n))  # means[i, j]: mean of frame i where it overlaps frame j
    for i in range(n):
        (ax0, ay0, ax1, ay1), gi, vi = warped[i]
        for j in range(i + 1, n):
            (bx0, by0, bx1, by1), gj, vj = warped[j]
            x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if x1 <= x0 or y1 <= y0:
                continue
            si = (slice(y0 - ay0, y1 - ay0), slice(x0 - ax0, x1 - ax0))
            sj = (slice(y0 - by0, y1 - by0), slice(x0 - bx0, x1 - bx0))
            both = vi[si] & vj[sj]
            count = int(both.sum())
            if count < 50:
                continue
            counts[i, j] = counts[j, i] = count
            means[i, j] = gi[si][both].mean()
            means[j, i] = gj[sj][both].mean()

    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j or counts[i, j] == 0:
                continue
            A[i, i] += counts[i, j] * (2 * means[i, j] ** 2 / sigma_n**2 + 1 / sigma_g**2)
            A[i, j] -= 2 * counts[i, j] * means[i, j] * means[j, i] / sigma_n**2
            b[i] += counts[i, j] / sigma_g**2
    isolated = A.diagonal() == 0
    A[isolated, isolated] = 1.0
    b[isolated] = 1.0
    return np.linalg.solve(A, b)


# --------------------------------------------------------------- blending ----


def _seam_labels(sizes, transforms, canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
    """Assign each canvas pixel to the frame whose center it is closest to."""
    best = np.zeros((canvas.height, canvas.width), np.float32)
    labels = np.full((canvas.height, canvas.width), -1, np.int32)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for index, (size, matrix) in enumerate(zip(sizes, transforms)):
        if size not in cache:
            cache[size] = feather_weight(*size) + 1e-3
        roi = _roi(matrix, size, canvas)
        x0, y0, x1, y1 = roi
        weight = _warp(cache[size], matrix, roi)
        region = best[y0:y1, x0:x1]
        better = weight > region
        region[better] = weight[better]
        labels[y0:y1, x0:x1][better] = index
    return labels, best > 0


def overwrite_blend(frames, transforms, canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((canvas.height, canvas.width, 3), np.uint8)
    coverage = np.zeros((canvas.height, canvas.width), bool)
    for frame, matrix in zip(frames, transforms):
        size = (frame.shape[1], frame.shape[0])
        roi = _roi(matrix, size, canvas)
        x0, y0, x1, y1 = roi
        warped = _warp(frame, matrix, roi, cv2.BORDER_REPLICATE)
        valid = _warp(np.ones(frame.shape[:2], np.float32), matrix, roi) > 0.5
        out[y0:y1, x0:x1][valid] = warped[valid]
        coverage[y0:y1, x0:x1] |= valid
    return out, coverage


def feather_blend(frames, transforms, canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
    accum = np.zeros((canvas.height, canvas.width, 3), np.float32)
    total = np.zeros((canvas.height, canvas.width), np.float32)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for frame, matrix in zip(frames, transforms):
        size = (frame.shape[1], frame.shape[0])
        if size not in cache:
            cache[size] = feather_weight(*size)
        roi = _roi(matrix, size, canvas)
        x0, y0, x1, y1 = roi
        weight = _warp(cache[size], matrix, roi)
        warped = _warp(frame, matrix, roi, cv2.BORDER_REPLICATE).astype(np.float32)
        accum[y0:y1, x0:x1] += warped * weight[..., None]
        total[y0:y1, x0:x1] += weight
    coverage = total > 1e-4
    out = np.zeros_like(accum)
    out[coverage] = accum[coverage] / total[coverage, None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), coverage


def multiband_blend(frames, transforms, canvas: Canvas, levels: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Burt and Adelson Laplacian pyramid blending over seam masks."""
    multiple = 2**levels
    sizes = [(f.shape[1], f.shape[0]) for f in frames]
    labels, coverage = _seam_labels(sizes, transforms, canvas)

    shapes = [(canvas.height, canvas.width)]
    for _ in range(levels):
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    bands = [np.zeros((h, w, 3), np.float32) for h, w in shapes]
    weights = [np.zeros((h, w), np.float32) for h, w in shapes]

    for index, (frame, matrix, size) in enumerate(zip(frames, transforms, sizes)):
        roi = _roi(matrix, size, canvas, margin=multiple * 2, multiple=multiple)
        x0, y0, x1, y1 = roi
        if (x1 - x0) % multiple or (y1 - y0) % multiple:
            # Clipped by the canvas edge; the canvas itself is padded to a multiple.
            x1 = min(canvas.width, x0 + ((x1 - x0 + multiple - 1) // multiple) * multiple)
            y1 = min(canvas.height, y0 + ((y1 - y0 + multiple - 1) // multiple) * multiple)
        mask = (labels[y0:y1, x0:x1] == index).astype(np.float32)
        if not mask.any():
            continue
        image = _warp(frame, matrix, (x0, y0, x1, y1), cv2.BORDER_REPLICATE).astype(np.float32)

        gaussian = [image]
        mask_pyramid = [mask]
        for _ in range(levels):
            gaussian.append(cv2.pyrDown(gaussian[-1]))
            mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
        for level in range(levels + 1):
            if level < levels:
                h, w = gaussian[level].shape[:2]
                band = gaussian[level] - cv2.pyrUp(gaussian[level + 1], dstsize=(w, h))
            else:
                band = gaussian[level]
            m = mask_pyramid[level]
            lx, ly = x0 >> level, y0 >> level
            h, w = band.shape[:2]
            h = min(h, shapes[level][0] - ly)
            w = min(w, shapes[level][1] - lx)
            bands[level][ly : ly + h, lx : lx + w] += band[:h, :w] * m[:h, :w, None]
            weights[level][ly : ly + h, lx : lx + w] += m[:h, :w]

    result = None
    for level in range(levels, -1, -1):
        normalized = bands[level] / np.maximum(weights[level], 1e-6)[..., None]
        if result is None:
            result = normalized
        else:
            h, w = shapes[level]
            result = cv2.pyrUp(result, dstsize=(w, h)) + normalized
    out = np.clip(result + 0.5, 0, 255).astype(np.uint8)
    out[~coverage] = 0
    return out, coverage


BLENDERS = {"overwrite": overwrite_blend, "feather": feather_blend, "multiband": multiband_blend}


def render(
    frames: list[np.ndarray],
    transforms: list[np.ndarray],
    blend: str = "multiband",
    gains: np.ndarray | None = None,
    max_pixels: int = 150_000_000,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Compose frames into a mosaic.

    Returns (mosaic, transforms into canvas pixels, coverage mask).
    """
    if blend not in BLENDERS:
        raise ValueError(f"blend must be one of {sorted(BLENDERS)}")
    if gains is not None:
        frames = [np.clip(f.astype(np.float32) * g, 0, 255).astype(np.uint8) for f, g in zip(frames, gains)]
    sizes = [(f.shape[1], f.shape[0]) for f in frames]
    multiple = 32 if blend == "multiband" else 1
    canvas = plan_canvas(sizes, transforms, max_pixels, pad_multiple=multiple)
    canvas_transforms = [canvas.offset @ m for m in transforms]
    mosaic, coverage = BLENDERS[blend](frames, canvas_transforms, canvas)
    return mosaic, canvas_transforms, coverage
