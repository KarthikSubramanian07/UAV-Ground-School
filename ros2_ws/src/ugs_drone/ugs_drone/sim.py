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

"""
The physics behind the week 5 drone, with no ROS in it so it can be unit tested.

A point mass multicopter in a local east, north, up frame (metres) that flies
toward a target with a speed and acceleration limit, a battery that drains
faster in the air, and a nadir pinhole camera that sees one orange ground
target. The camera driver renders frames with it and the planner uses the same
model, inverted, to turn a detection back into a position on the ground.
"""

from dataclasses import dataclass, field
import math

import numpy as np

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
FOCAL_PX = 200.0  # a 77 degree horizontal field of view
TARGET_RADIUS_M = 0.8
TARGET_RGB = (255, 120, 20)


@dataclass
class Vehicle:
    """A multicopter that tracks a position setpoint, limited in speed and acceleration."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_speed: float = 4.0
    max_accel: float = 3.0
    armed: bool = False
    mode: str = 'STABILIZE'
    battery: float = 1.0  # fraction of charge left
    setpoint: np.ndarray | None = None

    def step(self, dt: float) -> None:
        """Advance the simulation by dt seconds."""
        if not self.armed:
            self.velocity[:] = 0.0
            self.setpoint = None
        elif self.mode == 'LAND':
            self.setpoint = np.array([self.position[0], self.position[1], 0.0])
        target = self.setpoint if self.setpoint is not None else self.position.copy()
        error = target - self.position
        distance = float(np.linalg.norm(error))
        # brake early enough to stop at the target: v = sqrt(2 a d)
        speed = min(self.max_speed, math.sqrt(2.0 * self.max_accel * distance))
        desired = error / distance * speed if distance > 1e-6 else np.zeros(3)
        dv = desired - self.velocity
        limit = self.max_accel * dt
        norm = float(np.linalg.norm(dv))
        if norm > limit:
            dv *= limit / norm
        self.velocity += dv
        self.position += self.velocity * dt
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = max(0.0, self.velocity[2])
        drain = 1.0 / 1200.0 if self.in_air else 1.0 / 20000.0  # 20 minutes of hover
        self.battery = max(0.0, self.battery - drain * dt)

    @property
    def in_air(self) -> bool:
        return self.position[2] > 0.05

    def distance_to(self, point) -> float:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - self.position))


def target_pixel(target_xy, drone_position):
    """Where a ground point appears in the nadir camera, or None if it is out of view."""
    altitude = max(float(drone_position[2]), 0.3)
    dx = float(target_xy[0]) - float(drone_position[0])
    dy = float(target_xy[1]) - float(drone_position[1])
    u = IMAGE_WIDTH / 2.0 + FOCAL_PX * dx / altitude
    v = IMAGE_HEIGHT / 2.0 - FOCAL_PX * dy / altitude  # image rows grow southward
    return u, v, FOCAL_PX * TARGET_RADIUS_M / altitude


def pixel_to_ground(u: float, v: float, drone_position):
    """Invert target_pixel: the ground point under pixel (u, v)."""
    altitude = max(float(drone_position[2]), 0.3)
    x = float(drone_position[0]) + (u - IMAGE_WIDTH / 2.0) * altitude / FOCAL_PX
    y = float(drone_position[1]) - (v - IMAGE_HEIGHT / 2.0) * altitude / FOCAL_PX
    return x, y


def render(target_xy, drone_position, rng: np.random.Generator) -> np.ndarray:
    """Render an RGB frame of grass with the target disc, as the camera driver publishes it."""
    rows, cols = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    # grass: a fixed texture that scrolls with the drone, plus sensor noise
    gx = cols + drone_position[0] * 37.0
    gy = rows - drone_position[1] * 37.0
    texture = 0.5 + 0.25 * np.sin(gx * 0.11) * np.cos(gy * 0.07)
    image = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.float32)
    image[..., 0] = 60 + 30 * texture
    image[..., 1] = 120 + 50 * texture
    image[..., 2] = 50 + 20 * texture
    u, v, radius = target_pixel(target_xy, drone_position)
    inside = (cols - u) ** 2 + (rows - v) ** 2 <= radius ** 2
    image[inside] = TARGET_RGB
    image += rng.normal(0.0, 6.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def detect(image: np.ndarray):
    """
    Find the orange target: (u, v, width, height, score) or None.

    A colour threshold and the centroid of the matching pixels: enough for one
    saturated target on grass (week 3 does this properly). A target touching the
    frame edge is ignored, since only part of it is visible.
    """
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    mask = (r > 190) & (g > 70) & (g < 175) & (b < 90)
    count = int(mask.sum())
    if count < 12:
        return None
    rows, cols = np.nonzero(mask)
    if rows.min() == 0 or cols.min() == 0 or rows.max() == mask.shape[0] - 1 or \
            cols.max() == mask.shape[1] - 1:
        return None  # cut off by the frame edge: its centroid would be biased
    u, v = float(cols.mean()), float(rows.mean())
    width = float(cols.max() - cols.min() + 1)
    height = float(rows.max() - rows.min() + 1)
    fill = count / (math.pi / 4.0 * width * height)  # 1.0 for a perfect disc
    return u, v, width, height, max(0.0, min(1.0, fill))
