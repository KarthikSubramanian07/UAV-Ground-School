# Week 2 notes: computer vision and aerial imagery

Notes from UAVs@Berkeley Software Ground School, 15 September 2026.

## How drones are used for aerial imagery

* **Applications:** search and rescue, agriculture, law enforcement, environmental monitoring, and competitions (SUAS style object detection and mapping).
* **Nadir vs oblique:** a nadir camera points straight down, so the ground is roughly a flat plane viewed head on and neighboring frames differ by a rotation, a translation and a small scale change. Oblique cameras (high or low) look across the terrain, which adds perspective and makes stitching much harder. Everything in `week02/stitching.py` assumes nadir footage.
* **Sensors:** plain RGB, RGB-D (color plus depth) and multi-spectral (extra bands such as near infrared, popular in agriculture).
* **Gimbal cameras** keep the camera level (and nadir) while the airframe pitches and rolls.

## OpenCV

A free, open source computer vision library with Python and C++ APIs: feature detection, 3D reconstruction, motion analysis, optical flow, photo manipulation (deblurring, refining) and video reading and writing.

```bash
python --version
pip --version
pip install opencv-python numpy
python -c "import cv2; print(cv2.__version__)"
```

The parts the club has used so far: feature detection (SIFT, ORB), feature matching and image manipulation.

## Pre made alternative: OpenDroneMap

[OpenDroneMap](https://www.opendronemap.org/) turns geotagged photos into orthophotos, point clouds and 3D models. Workflow: capture photos with GPS tags (important), tune roughly 50 settings, then wait for processing.

| | OpenDroneMap | Custom OpenCV |
| --- | --- | --- |
| Effort | Plug in pictures, it works; has APIs and a web app | Write and tune the pipeline yourself |
| Output | Orthophoto, 3D model, lots of extra data | Exactly what you build (an orthophoto) |
| In flight use | Not designed for it (found out the hard way last summer) | Can be built for real time |
| Speed | Slow unless carefully tuned | As fast as you make it |
| GPS | Needs accurate geotags | Can work from image content alone |

## Where the software team is heading

* A custom stitching and structure from motion pipeline in OpenCV that runs in real time on the drone.
* Splitting OpenDroneMap into small subcomponents so it can also run in real time, automatically.
* Other directions: SLAM, optical flow, structure from motion for terrain maps, and applied drone imagery.

## Skill booster 1

* **Color Me Impressed:** take an image path, split it into one image per color, display them, and (bonus) print the center of each colored object. Solved in [`week02/colors.py`](../../week02/colors.py).
* **I'll be Needin' Stitches:** split a flight video into frames and stitch them into one image spanning the flight path. Solved in [`week02/stitching.py`](../../week02/stitching.py).
* **Extra challenge:** do it with the C++ API. Solved in [`week02/cpp`](../../week02/cpp).
