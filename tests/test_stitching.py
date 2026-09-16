import numpy as np
import pytest

from week02 import evaluate, stitching


def test_iter_frames_samples_with_modular_arithmetic(small_flight):
    indices = [index for index, _ in stitching.iter_frames(small_flight.video, every=7)]
    assert indices[:3] == [0, 7, 14]
    assert all(index % 7 == 0 for index in indices)
    assert len(indices) == -(-len(small_flight.truth.poses) // 7)


def test_iter_frames_scale_and_errors(small_flight, tmp_path):
    _, frame = next(stitching.iter_frames(small_flight.video, scale=0.5))
    assert frame.shape[:2] == (90, 160)
    with pytest.raises(FileNotFoundError):
        next(stitching.iter_frames(tmp_path / "nope.mp4"))
    with pytest.raises(ValueError):
        next(stitching.iter_frames(small_flight.video, every=0))
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(stitching.StitchError):
        next(stitching.iter_frames(bogus))


def test_extract_frames(small_flight, tmp_path):
    written = stitching.extract_frames(small_flight.video, tmp_path / "frames", every=50)
    assert [p.name for p in written[:2]] == ["frame_000000.png", "frame_000050.png"]


def test_full_pipeline_is_accurate(small_flight):
    result = stitching.stitch_video(small_flight.video, stitching.StitchConfig(every=4))
    score = evaluate.evaluate(result.transforms, result.frame_indices, small_flight.truth, result.panorama, small_flight.world)
    assert score.rmse_px < 1.5, score
    assert score.zncc > 0.93, score
    assert result.loop_closures > 0
    assert result.adjustment.rms_after <= result.adjustment.rms_before
    assert not result.rejected
    assert "loop closures" in result.summary()


def test_bundle_adjustment_beats_sequential_chaining_on_a_rough_flight(rough_flight):
    k1 = rough_flight.truth.plan.distortion
    runs = {}
    for name, config in {
        "sequential": stitching.StitchConfig(every=4, k1=k1, loop_closure=False, bundle_adjust=False, gain_compensation=False, blend="feather"),
        "adjusted": stitching.StitchConfig(every=4, k1=k1),
    }.items():
        result = stitching.stitch_video(rough_flight.video, config)
        runs[name] = evaluate.evaluate(result.transforms, result.frame_indices, rough_flight.truth)
    assert runs["adjusted"].rmse_px < 2.0
    assert runs["adjusted"].rmse_px < 0.7 * runs["sequential"].rmse_px


def test_save_and_load_transforms(small_flight, tmp_path):
    result = stitching.stitch_video(small_flight.video, stitching.StitchConfig(every=8, blend="overwrite", gain_compensation=False))
    path = result.save_transforms(tmp_path / "transforms.csv")
    transforms, indices = evaluate.load_transforms(path)
    assert indices == result.frame_indices
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(transforms, result.transforms))
    assert stitching.draw_flight_path(result).shape == result.panorama.shape


def test_scaled_frames_evaluate_consistently(small_flight):
    result = stitching.stitch_video(small_flight.video, stitching.StitchConfig(every=4, scale=0.75))
    score = evaluate.evaluate(result.transforms, result.frame_indices, small_flight.truth, frame_scale=0.75)
    assert score.rmse_px < 2.5


def test_opencv_stitcher_needs_frames():
    with pytest.raises(stitching.StitchError):
        stitching.stitch_with_opencv([(0, np.zeros((10, 10, 3), np.uint8))])


def test_unknown_method(small_flight):
    with pytest.raises(ValueError):
        stitching.stitch_video(small_flight.video, method="magic")


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (30, 2))
    c, s = np.cos(0.4) * 1.7, np.sin(0.4) * 1.7
    matrix = np.array([[c, -s, 12], [s, c, -4], [0, 0, 1]])
    dst = src @ matrix[:2, :2].T + matrix[:2, 2]
    assert np.allclose(evaluate.umeyama_similarity(src, dst), matrix)
