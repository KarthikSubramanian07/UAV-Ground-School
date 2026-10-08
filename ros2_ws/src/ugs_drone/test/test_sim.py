# Copyright 2026 Karthik Subramanian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the simulation core (no ROS needed)."""

import numpy as np
from ugs_drone.sim import detect, pixel_to_ground, render, target_pixel, Vehicle


def test_vehicle_reaches_setpoint_and_stops():
    v = Vehicle(armed=True, mode='GUIDED')
    v.setpoint = np.array([10.0, 5.0, 5.0])
    for _ in range(30 * 50):
        v.step(0.02)
    assert v.distance_to(v.setpoint) < 0.05
    assert np.linalg.norm(v.velocity) < 0.05


def test_vehicle_respects_speed_limit():
    v = Vehicle(armed=True, mode='GUIDED', max_speed=3.0)
    v.setpoint = np.array([100.0, 0.0, 0.0])
    peak = 0.0
    for _ in range(20 * 50):
        v.step(0.02)
        peak = max(peak, float(np.linalg.norm(v.velocity)))
    assert 2.9 < peak <= 3.0 + 1e-9


def test_disarmed_vehicle_does_not_move():
    v = Vehicle()
    v.setpoint = np.array([5.0, 0.0, 0.0])
    v.step(1.0)
    assert np.allclose(v.position, 0.0)


def test_land_mode_descends():
    v = Vehicle(armed=True, mode='GUIDED', position=np.array([0.0, 0.0, 5.0]))
    v.mode = 'LAND'
    for _ in range(10 * 50):
        v.step(0.02)
    assert v.position[2] < 0.05


def test_camera_model_round_trip():
    drone = np.array([3.0, -2.0, 5.0])
    u, v, _ = target_pixel((4.0, -1.5), drone)
    x, y = pixel_to_ground(u, v, drone)
    assert abs(x - 4.0) < 1e-9 and abs(y + 1.5) < 1e-9


def test_detector_finds_rendered_target():
    rng = np.random.default_rng(1)
    drone = np.array([10.0, 6.0, 5.0])
    frame = render((11.0, 7.0), drone, rng)
    found = detect(frame)
    assert found is not None
    x, y = pixel_to_ground(found[0], found[1], drone)
    assert abs(x - 11.0) < 0.1 and abs(y - 7.0) < 0.1
    assert found[4] > 0.8


def test_detector_ignores_grass():
    rng = np.random.default_rng(2)
    frame = render((100.0, 100.0), np.array([0.0, 0.0, 5.0]), rng)
    assert detect(frame) is None
