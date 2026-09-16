# Week 02 in C++17

A C++17 port of the week 02 Python pipeline: the "extra extra challenge". It has two executables:

| binary | ports | what it does |
| --- | --- | --- |
| `color_me_impressed` | `colors.py` (named mode), `shapes.py` | splits an image into HSV color layers, finds object centers, classifies SUAS shapes |
| `needin_stitches` | `stitching.py`, `features.py`, `mosaic.py` | stitches a nadir drone video into a mosaic (tracking, loop closures, bundle adjustment, gains, blending) |

The algorithms, thresholds and output formats match the Python code. The color report is byte for byte identical. Given the same decoded frames, the stitcher produces the same alignments.

## Install OpenCV and CMake

```sh
# macOS (OpenCV 5.x)
brew install opencv cmake

# Ubuntu / Debian (OpenCV 4.x)
sudo apt-get install -y build-essential cmake libopencv-dev
```

The code builds against OpenCV 4.6+ and 5.x. OpenCV 5 moved `moments`, `contourArea`, `estimateAffinePartial2D` and `intersectConvexConvex` into the new `geometry` module. That header is included only when it exists.

## Build

From the repository root:

```sh
cmake -S week02/cpp -B build/cpp
cmake --build build/cpp -j
```

The default is a Release build compiled with `-Wall -Wextra`.

## Run

```sh
# Color Me Impressed
build/cpp/color_me_impressed targets.jpg --out layers/ --json report.json --no-show
#   --min-coverage F   drop colors below this fraction unless they form objects (default 0.01)

# I'll be Needin' Stitches
build/cpp/needin_stitches flight.mp4 --out stitched.jpg --transforms transforms.csv --no-show
#   --every N                           sample one frame out of every N (default 5)
#   --scale F                           resize frames first
#   --k1 F                              undistort radial lens distortion (simulator camera model)
#   --detector orb|sift                 feature detector (default orb)
#   --features N                        keypoints per frame after grid bucketing (default 2500)
#   --blend multiband|feather|overwrite blending (default multiband)
#   --no-loops --no-adjust --no-gains   switch pipeline stages off
#   --method features|stitcher          our pipeline or cv::Stitcher in SCANS mode
#   -v                                  log keyframes, relocalizations and rejected loops
```

`color_me_impressed --out DIR` writes `layer_<color>.png` (BGRA cut outs with the mask as alpha) and `centers.png`. `--json` writes one entry per color with its objects, centers, bounding boxes and shapes.

`needin_stitches --transforms` writes the same CSV as the Python `StitchResult.save_transforms`. `week02.evaluate.load_transforms` reads it.

## Check parity with Python

```sh
WEEK02_CPP_BUILD=build/cpp PYTHONPATH=. python -m pytest tests/test_cpp_parity.py -v
```

The test renders synthetic targets and a short survey flight, runs both implementations, and checks two things:

- the color layers and shaped object centers agree to within 1.5 px;
- the C++ keyframe poses are within 2 px RMSE of the ground truth.

## Notes

- **Bundle adjustment:** the C++ version draws its random 80 point subset per constraint with `std::mt19937_64`, not NumPy's generator. Residuals therefore differ from Python in the last decimals, while accuracy stays the same.
- **Video decoding:** decoded frames can differ by a few gray levels between OpenCV builds, because the pip wheel and system packages bundle different FFmpeg versions. Poses on the same video can differ a little for that reason.
