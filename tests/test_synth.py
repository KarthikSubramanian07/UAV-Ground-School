import cv2
import numpy as np
import pytest

from week02 import stitching, synth


def test_rendered_frame_matches_pose():
    world = synth.voxel_world(width_blocks=120, height_blocks=80, seed=3)
    plan = synth.FlightPlan(frame_width=200, frame_height=120, exposure_wobble=0, vignette=0, noise=0, altitude_wobble=0)
    pose = np.array([400.0, 240.0, 0.7, 1.1])
    frame = synth.render_frame(world, pose, plan, 0, np.random.default_rng(0))
    matrix = synth.pose_matrix(pose, 200, 120)
    for x, y in [(10, 10), (100, 60), (190, 110)]:
        wx, wy = (matrix @ [x, y, 1])[:2]
        expected = cv2.getRectSubPix(world, (1, 1), (float(wx), float(wy)))[0, 0]
        assert np.abs(frame[y, x].astype(int) - expected.astype(int)).max() <= 3


def test_lawnmower_path_is_smooth_and_inside_the_world():
    world = synth.voxel_world(width_blocks=200, height_blocks=110)
    plan = synth.FlightPlan(frame_width=320, frame_height=180, passes=2, speed=5.0)
    poses = synth.simulate_flight(world, plan).poses
    steps = np.hypot(*np.diff(poses[:, :2], axis=0).T)
    assert np.allclose(steps, plan.speed, atol=0.6)
    assert poses[:, 0].min() > 0 and poses[:, 0].max() < world.shape[1]
    assert np.abs(np.diff(poses[:, 2])).max() < 0.2


def test_world_too_small():
    with pytest.raises(ValueError):
        synth.simulate_flight(np.zeros((200, 300, 3), np.uint8), synth.FlightPlan())


def test_truth_round_trip(tmp_path):
    truth = synth.FlightTruth(np.array([[1.0, 2.0, 0.1, 1.0], [3.0, 4.0, 0.2, 0.9]]), synth.FlightPlan(frame_width=64, frame_height=48, fps=24))
    loaded = synth.load_truth(synth.save_truth(tmp_path / "t.csv", truth))
    assert np.allclose(loaded.poses, truth.poses)
    assert (loaded.plan.frame_width, loaded.plan.frame_height, loaded.plan.fps) == (64, 48, 24)


def test_undistorter_inverts_simulated_lens():
    world = synth.voxel_world(width_blocks=100, height_blocks=60, seed=9)
    ideal_plan = synth.FlightPlan(frame_width=320, frame_height=180, exposure_wobble=0, vignette=0, noise=0)
    lens_plan = synth.FlightPlan(frame_width=320, frame_height=180, exposure_wobble=0, vignette=0, noise=0, distortion=-0.08)
    pose = np.array([300.0, 180.0, 0.0, 1.0])
    ideal = synth.render_frame(world, pose, ideal_plan, 0, np.random.default_rng(0))
    distorted = synth.render_frame(world, pose, lens_plan, 0, np.random.default_rng(0))
    restored = stitching.Undistorter(-0.08)(distorted)
    inner = (slice(20, 160), slice(30, 290))
    assert np.abs(distorted.astype(int) - ideal.astype(int))[inner].mean() > 2 * np.abs(restored.astype(int) - ideal.astype(int))[inner].mean()


def test_color_test_images_are_deterministic():
    assert np.array_equal(synth.apple(), synth.apple())
    image, truths = synth.suas_targets()
    assert image.shape == (800, 1280, 3) and len(truths) == 5


def test_fractal_noise_range():
    noise = synth.fractal_noise(40, 90, np.random.default_rng(0))
    assert noise.shape == (40, 90) and noise.min() == 0 and noise.max() == pytest.approx(1)
