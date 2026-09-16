"""Option 2: I'll be Needin' Stitches.

Turn a nadir (straight down) drone video into one orthomosaic style image.

Pipeline (``method="features"``, written from scratch):

1. **Sample** every Nth frame with ``cv2.VideoCapture`` and modular arithmetic.
2. **Track**: detect grid bucketed ORB or SIFT features and align each
   sampled frame to the current keyframe with symmetric ratio test matching
   and a RANSAC similarity. A frame becomes a new keyframe once it has moved
   far enough. If tracking fails, the most recent good frame is promoted to a
   keyframe and, failing that, the frame is relocalized against all earlier
   keyframes.
3. **Close loops**: predict which non consecutive keyframes overlap (the next
   lawnmower pass looking at the same ground), verify each candidate with
   feature matching, and keep the ones consistent with the prediction.
4. **Bundle adjust** all keyframe transforms against every constraint.
5. **Compensate gains** so exposure changes do not show up as tiles.
6. **Blend** with multi band (Laplacian pyramid), feather or overwrite.

``method="stitcher"`` hands the sampled frames to OpenCV's built in
``cv2.Stitcher`` in SCANS mode for comparison.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import mosaic
from .features import Alignment, FeatureExtractor, Features, Matcher, align


@dataclass
class StitchConfig:
    every: int = 5
    """Keep one frame out of every ``every`` frames (modular arithmetic)."""
    scale: float = 1.0
    """Resize frames by this factor before doing anything else."""
    k1: float = 0.0
    """Radial distortion coefficient of the camera; frames are undistorted first
    (focal length assumed to be half the larger frame side, as in the simulator)."""
    detector: str = "orb"
    n_features: int = 2500
    ratio: float = 0.8
    ransac_threshold: float = 3.0
    min_inliers: int = 30
    keyframe_motion: float = 0.18
    """Promote a frame to keyframe after it moved this fraction of the frame diagonal."""
    loop_closure: bool = True
    loop_min_overlap: float = 0.25
    loop_max_per_frame: int = 4
    bundle_adjust: bool = True
    gain_compensation: bool = True
    blend: str = "multiband"
    max_canvas_pixels: int = 150_000_000
    max_keyframes: int | None = None


@dataclass
class StitchResult:
    panorama: np.ndarray
    coverage: np.ndarray
    transforms: list[np.ndarray]
    """3x3 matrices mapping each keyframe into panorama pixels."""
    frame_indices: list[int]
    frame_size: tuple[int, int]
    sampled: int = 0
    rejected: list[tuple[int, str]] = field(default_factory=list)
    relocalized: int = 0
    sequential_constraints: int = 0
    loop_closures: int = 0
    adjustment: mosaic.AdjustmentReport | None = None
    gains: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def keyframes(self) -> int:
        return len(self.frame_indices)

    def summary(self) -> str:
        h, w = self.panorama.shape[:2]
        lines = [
            f"panorama        {w} x {h} px, {self.coverage.mean() * 100:.0f}% covered",
            f"frames          {self.sampled} sampled, {self.keyframes} keyframes, {len(self.rejected)} rejected, {self.relocalized} relocalized",
            f"constraints     {self.sequential_constraints} sequential + {self.loop_closures} loop closures",
        ]
        if self.adjustment:
            a = self.adjustment
            lines.append(f"bundle adjust   residual RMS {a.rms_before:.2f} px -> {a.rms_after:.2f} px over {a.correspondences} correspondences")
        if self.gains is not None and len(self.gains):
            lines.append(f"gains           {self.gains.min():.3f} .. {self.gains.max():.3f}")
        total = sum(self.timings.values())
        lines.append("timing          " + ", ".join(f"{k} {v:.2f}s" for k, v in self.timings.items()) + f" (total {total:.2f}s)")
        return "\n".join(lines)

    def save_transforms(self, path: str | Path) -> Path:
        """CSV of ``frame_index`` and the 2x3 frame to panorama matrix."""
        path = Path(path)
        rows = [[index, *matrix[:2].ravel()] for index, matrix in zip(self.frame_indices, self.transforms)]
        np.savetxt(path, np.array(rows), delimiter=",", header="frame,m00,m01,m02,m10,m11,m12", fmt=["%d"] + ["%.8f"] * 6)
        return path


class StitchError(RuntimeError):
    pass


# ---------------------------------------------------------------- frames ----


class Undistorter:
    """Removes radial lens distortion with a cached remap table."""

    def __init__(self, k1: float):
        self.k1 = k1
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.k1:
            return frame
        h, w = frame.shape[:2]
        if (w, h) not in self._maps:
            f = max(w, h) / 2
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
            dist = np.array([self.k1, 0, 0, 0], np.float64)
            self._maps[(w, h)] = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
        map_x, map_y = self._maps[(w, h)]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def iter_frames(video_path: str | Path, every: int = 1, scale: float = 1.0, k1: float = 0.0) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` for every ``every``-th frame of a video."""
    if every < 1:
        raise ValueError("every must be at least 1")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"No video found at {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise StitchError(f"OpenCV could not open {path} as a video")
    undistort = Undistorter(k1)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every == 0:
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                yield index, undistort(frame)
            index += 1
    finally:
        cap.release()


def extract_frames(video_path: str | Path, out_dir: str | Path, every: int = 5, scale: float = 1.0) -> list[Path]:
    """Save sampled frames as numbered PNG files. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, frame in iter_frames(video_path, every, scale):
        path = out_dir / f"frame_{index:06d}.png"
        cv2.imwrite(str(path), frame)
        written.append(path)
    return written


# -------------------------------------------------------------- tracking ----


@dataclass
class _Keyframe:
    index: int
    frame: np.ndarray
    features: Features
    transform: np.ndarray


class Tracker:
    """Sequential keyframe tracker producing initial poses and constraints."""

    def __init__(self, config: StitchConfig, log: Callable[[str], None]):
        self.config = config
        self.log = log
        self.extract = FeatureExtractor(config.detector, config.n_features)
        self.matcher = Matcher(self.extract.norm, config.ratio)
        self.keyframes: list[_Keyframe] = []
        self.constraints: list[mosaic.Constraint] = []
        self.rejected: list[tuple[int, str]] = []
        self.relocalized = 0
        self._candidate: tuple[int, np.ndarray, Features, Alignment] | None = None

    def _align(self, src: Features, dst: Features) -> tuple[Alignment | None, str]:
        return align(src, dst, self.matcher, self.config.ransac_threshold, self.config.min_inliers)

    def _add_keyframe(self, index: int, frame: np.ndarray, features: Features, alignment: Alignment | None, reference: int) -> None:
        if alignment is None:
            transform = np.eye(3)
        else:
            transform = self.keyframes[reference].transform @ alignment.matrix
            self.constraints.append(mosaic.Constraint(len(self.keyframes), reference, alignment.src_points, alignment.dst_points))
        self.keyframes.append(_Keyframe(index, frame, features, transform))
        self._candidate = None

    def _motion(self, alignment: Alignment, size: tuple[int, int]) -> float:
        w, h = size
        center = np.array([w / 2, h / 2])
        moved = mosaic.apply(alignment.matrix, center[None])[0]
        return float(np.linalg.norm(moved - center) / np.hypot(w, h))

    def _relocalize(self, features: Features) -> tuple[Alignment | None, int]:
        best: tuple[Alignment | None, int] = (None, -1)
        for k in range(len(self.keyframes) - 1, -1, -1):
            alignment, _ = self._align(features, self.keyframes[k].features)
            if alignment is not None and (best[0] is None or alignment.inliers > best[0].inliers):
                best = (alignment, k)
        return best

    def push(self, index: int, frame: np.ndarray) -> None:
        features = self.extract(frame)
        if not self.keyframes:
            self._add_keyframe(index, frame, features, None, 0)
            return

        key = len(self.keyframes) - 1
        alignment, reason = self._align(features, self.keyframes[key].features)
        if alignment is None and self._candidate is not None:
            # Motion outran the keyframe: promote the last good frame and retry.
            c_index, c_frame, c_features, c_alignment = self._candidate
            self._add_keyframe(c_index, c_frame, c_features, c_alignment, key)
            key = len(self.keyframes) - 1
            alignment, reason = self._align(features, self.keyframes[key].features)
        if alignment is None:
            alignment, key = self._relocalize(features)
            if alignment is None:
                self.rejected.append((index, reason))
                self.log(f"frame {index}: rejected ({reason})")
                return
            self.relocalized += 1
            self.log(f"frame {index}: relocalized against keyframe {key}")
            self._add_keyframe(index, frame, features, alignment, key)
            return

        if self._motion(alignment, features.size) >= self.config.keyframe_motion:
            self._add_keyframe(index, frame, features, alignment, key)
            self.log(f"frame {index}: keyframe {len(self.keyframes) - 1} ({alignment.inliers} inliers)")
        else:
            self._candidate = (index, frame, features, alignment)

    def finish(self) -> None:
        """Keep the final frame of the flight so the mosaic reaches the end."""
        if self._candidate is not None:
            c_index, c_frame, c_features, c_alignment = self._candidate
            self._add_keyframe(c_index, c_frame, c_features, c_alignment, len(self.keyframes) - 1)


def find_loop_closures(tracker: Tracker, config: StitchConfig, log: Callable[[str], None]) -> list[mosaic.Constraint]:
    """Verify predicted overlaps between non consecutive keyframes."""
    keyframes = tracker.keyframes
    connected = {(c.i, c.j) for c in tracker.constraints} | {(c.j, c.i) for c in tracker.constraints}
    closures: list[mosaic.Constraint] = []
    for i, ki in enumerate(keyframes):
        w, h = ki.features.size
        diagonal = np.hypot(w, h)
        candidates = []
        for j in range(i - 2, -1, -1):
            if (i, j) in connected:
                continue
            overlap = mosaic.footprint_overlap(ki.transform, keyframes[j].transform, w, h)
            if overlap >= config.loop_min_overlap:
                candidates.append((overlap, j))
        candidates.sort(reverse=True)
        for _overlap, j in candidates[: config.loop_max_per_frame]:
            alignment, _ = tracker._align(ki.features, keyframes[j].features)
            if alignment is None:
                continue
            predicted = np.linalg.inv(keyframes[j].transform) @ ki.transform
            center = np.array([[w / 2, h / 2]])
            disagreement = np.linalg.norm(mosaic.apply(predicted, center) - mosaic.apply(alignment.matrix, center))
            if disagreement > 0.25 * diagonal:
                log(f"loop {i}->{j}: rejected, disagrees with odometry by {disagreement:.0f} px")
                continue
            closures.append(mosaic.Constraint(i, j, alignment.src_points, alignment.dst_points, loop_closure=True))
            connected |= {(i, j), (j, i)}
    return closures


# -------------------------------------------------------------- pipeline ----


def stitch_frames(
    frames: Iterator[tuple[int, np.ndarray]] | list[tuple[int, np.ndarray]],
    config: StitchConfig | None = None,
    log: Callable[[str], None] | None = None,
    on_keyframe: Callable[[Tracker], None] | None = None,
) -> StitchResult:
    """Stitch ``(frame_index, frame)`` pairs with the feature based pipeline."""
    config = config or StitchConfig()
    log = log or (lambda message: None)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    tracker = Tracker(config, log)
    sampled = 0
    for index, frame in frames:
        sampled += 1
        before = len(tracker.keyframes)
        tracker.push(index, frame)
        if on_keyframe and len(tracker.keyframes) != before:
            on_keyframe(tracker)
        if config.max_keyframes and len(tracker.keyframes) >= config.max_keyframes:
            break
    tracker.finish()
    timings["track"] = time.perf_counter() - start
    if not tracker.keyframes:
        raise StitchError("The video produced no frames")

    transforms = [k.transform for k in tracker.keyframes]
    constraints = list(tracker.constraints)
    closures: list[mosaic.Constraint] = []
    if config.loop_closure and len(tracker.keyframes) > 2:
        start = time.perf_counter()
        closures = find_loop_closures(tracker, config, log)
        timings["loops"] = time.perf_counter() - start

    report = None
    if config.bundle_adjust and len(transforms) > 1:
        start = time.perf_counter()
        transforms, report = mosaic.bundle_adjust(transforms, constraints + closures, huber=config.ransac_threshold)
        timings["adjust"] = time.perf_counter() - start

    frames_kept = [k.frame for k in tracker.keyframes]
    gains = None
    if config.gain_compensation and len(frames_kept) > 1:
        start = time.perf_counter()
        gains = mosaic.gain_compensation(frames_kept, transforms)
        timings["gains"] = time.perf_counter() - start

    start = time.perf_counter()
    try:
        panorama, canvas_transforms, coverage = mosaic.render(frames_kept, transforms, config.blend, gains, config.max_canvas_pixels)
    except MemoryError as error:
        raise StitchError(str(error)) from error
    timings["blend"] = time.perf_counter() - start

    first = tracker.keyframes[0].features.size
    return StitchResult(
        panorama=panorama,
        coverage=coverage,
        transforms=canvas_transforms,
        frame_indices=[k.index for k in tracker.keyframes],
        frame_size=first,
        sampled=sampled,
        rejected=tracker.rejected,
        relocalized=tracker.relocalized,
        sequential_constraints=len(constraints),
        loop_closures=len(closures),
        adjustment=report,
        gains=gains,
        timings=timings,
    )


STITCHER_STATUS = {
    0: "ok",
    1: "need more images (not enough overlap or features)",
    2: "homography estimation failed",
    3: "camera parameter adjustment failed",
}


def stitch_with_opencv(frames: list[tuple[int, np.ndarray]], max_frames: int | None = None) -> StitchResult:
    """Stitch with OpenCV's built in ``cv2.Stitcher`` in SCANS mode."""
    frames = list(frames)[:max_frames] if max_frames else list(frames)
    images = [frame for _, frame in frames]
    if len(images) < 2:
        raise StitchError("cv2.Stitcher needs at least two frames")
    start = time.perf_counter()
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise StitchError(f"cv2.Stitcher failed: {STITCHER_STATUS.get(status, f'status {status}')}")
    coverage = panorama.max(axis=2) > 0
    return StitchResult(
        panorama=panorama,
        coverage=coverage,
        transforms=[],
        frame_indices=[index for index, _ in frames],
        frame_size=(images[0].shape[1], images[0].shape[0]),
        sampled=len(frames),
        timings={"stitcher": time.perf_counter() - start},
    )


def stitch_video(
    video_path: str | Path,
    config: StitchConfig | None = None,
    method: str = "features",
    log: Callable[[str], None] | None = None,
    on_keyframe: Callable[[Tracker], None] | None = None,
) -> StitchResult:
    """Read a video, sample frames and stitch them into one image."""
    config = config or StitchConfig()
    frames = iter_frames(video_path, config.every, config.scale, config.k1)
    if method == "features":
        return stitch_frames(frames, config, log=log, on_keyframe=on_keyframe)
    if method == "stitcher":
        return stitch_with_opencv(list(frames), config.max_keyframes)
    raise ValueError("method must be 'features' or 'stitcher'")


def draw_flight_path(result: StitchResult) -> np.ndarray:
    """Overlay keyframe footprints, the camera track and loop closure count."""
    out = result.panorama.copy()
    w, h = result.frame_size
    corners = mosaic.frame_corners(w, h)
    centers = []
    for matrix in result.transforms:
        footprint = mosaic.apply(matrix, corners).astype(np.int32)
        cv2.polylines(out, [footprint], True, (235, 235, 235), 1, cv2.LINE_AA)
        centers.append(mosaic.apply(matrix, np.array([[w / 2, h / 2]]))[0])
    if len(centers) > 1:
        cv2.polylines(out, [np.int32(centers)], False, (20, 20, 20), 6, cv2.LINE_AA)
        cv2.polylines(out, [np.int32(centers)], False, (0, 170, 255), 3, cv2.LINE_AA)
    for point in centers:
        cv2.circle(out, (int(point[0]), int(point[1])), 5, (0, 170, 255), -1, cv2.LINE_AA)
    return out
