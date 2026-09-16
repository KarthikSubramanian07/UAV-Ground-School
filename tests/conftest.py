from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from week02 import synth


@dataclass
class Flight:
    world: np.ndarray
    truth: synth.FlightTruth
    video: Path
    truth_csv: Path
    world_png: Path


def _flight(tmp_path_factory, name: str, plan: synth.FlightPlan, width_blocks: int, height_blocks: int) -> Flight:
    import cv2

    directory = tmp_path_factory.mktemp(name)
    world = synth.voxel_world(width_blocks=width_blocks, height_blocks=height_blocks, seed=plan.seed)
    truth = synth.simulate_flight(world, plan)
    video = synth.write_flight_video(directory / "flight.mp4", world, truth)
    truth_csv = synth.save_truth(directory / "truth.csv", truth)
    world_png = directory / "world.png"
    cv2.imwrite(str(world_png), world)
    return Flight(world, truth, video, truth_csv, world_png)


@pytest.fixture(scope="session")
def small_flight(tmp_path_factory) -> Flight:
    """A 2 pass lawnmower flight at 320x180, about 300 frames."""
    plan = synth.FlightPlan(frame_width=320, frame_height=180, passes=2, speed=5.0, seed=42)
    return _flight(tmp_path_factory, "small", plan, 200, 110)


@pytest.fixture(scope="session")
def rough_flight(tmp_path_factory) -> Flight:
    """3 passes with lens distortion, blur and heavy noise, where drift is real."""
    plan = synth.FlightPlan.hard(frame_width=320, frame_height=180, passes=3, speed=6.0, seed=7)
    return _flight(tmp_path_factory, "rough", plan, 230, 150)


@pytest.fixture(scope="session")
def targets():
    return synth.suas_targets()
