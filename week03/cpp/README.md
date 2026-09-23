# Week 03 in C++17

A C++17 port of the two classical week 03 pipelines. It has two executables:

| binary | ports | what it does |
| --- | --- | --- |
| `polka_dots` | `dots.py`, the scale space and SimpleBlobDetector parts of `blobs.py`, `name_color` from `week02/colors.py` | finds polka dots, measures each one (sub pixel center, radius, roundness, aspect) and names its color |
| `game_pieces` | `objects.py` (default `ObjectSettings`) | finds cones, cubes and rings and classifies them by shape |

The algorithms, thresholds and defaults match the Python code step for step, including tie breaks, sort orders and NumPy rounding (`round` is half to even, float32 math where NumPy uses float32). Given the same pixels, the dot sets and piece boxes come out the same as Python's, down to float rounding (see [parity](#check-parity-with-python)).

## Install OpenCV and CMake

```sh
# macOS (OpenCV 5.x)
brew install opencv cmake

# Ubuntu / Debian (OpenCV 4.x)
sudo apt-get install -y build-essential cmake libopencv-dev
```

The code builds against OpenCV 4.6+ and 5.x. OpenCV 5 moved `fitEllipse`, `moments`, `contourArea`, `convexHull`, `minAreaRect` and `minEnclosingTriangle` into the new `geometry` module. That header is included only when it exists.

## Build

From the repository root:

```sh
cmake -S week03/cpp -B build/cpp3
cmake --build build/cpp3 -j
```

The default is a Release build compiled with `-Wall -Wextra`. It also passes `-ffp-contract=off`, so the compiler does not fuse `a * b + c` into one instruction: float32 expressions then round exactly like NumPy's.

## Run

```sh
# Polka dots
build/cpp3/polka_dots dots.jpg --method log --out annotated.png --json dots.json --no-show
#   --method gray|contrast|log|dog|doh   candidate detector (default log)
#   --min-radius F                       smallest dot radius in px (default 3)
#   --max-radius F                       largest dot radius (default: a sixth of the short image side)
#   --min-contrast F                     Delta E between dot and background (default 10)

# Game pieces
build/cpp3/game_pieces objects.jpg --out annotated.png --json pieces.json --no-show
```

`polka_dots --json` writes `{"method", "count", "counts_by_color", "seconds", "candidates", "dots": [...]}`, one dot per entry with `x`, `y`, `radius`, `color`, `bgr`, `contrast`, `roundness` and `aspect` (full precision). `--out` draws each fitted circle like `dots.annotate`.

`game_pieces --json` writes a list with one entry per piece: `label`, `confidence`, `bbox` (`[x, y, w, h]`), `center`, `area`, `hue`, and the shape `features` and class `scores` behind the decision (rounded like `Piece.summary`). `--out` draws the masks, outlines and labels like `objects.annotate`.

## What is ported

**`polka_dots`**

- CIELAB conversion identical to Python: the image is divided by 255 as float32, then `COLOR_BGR2Lab`.
- `log`, `dog`, `doh`: the sigma ladder (`np.geomspace`), the pyramid that filters large scales on downsampled channels, per channel responses (LoG, DoG, signed DoH), the combined magnitude plus both signs of every channel, and the streaming three slice peak search with per channel signatures. Peaks get the 3D quadratic sub pixel and sub scale fit, then `suppress_sidelobes` and `prune_overlaps`.
- `contrast`: the median filtered local background, the Delta E map and SimpleBlobDetector on it. `gray`: SimpleBlobDetector on the grayscale image.
- `estimate_colors`, the ray `refine` (48 rays, 96 samples from 0.35 r to 1.8 r, bilinear sampling through `cv::remap` with `INTER_LINEAR` and `BORDER_REPLICATE` plus an inside the image mask, Kasa circle fits with two rounds of MAD outlier rejection, a final `cv::fitEllipse` for roundness and aspect), `verify` (fill, leak, surround), the radius filter, the final overlap pruning and the reading order sort.
- Color naming: the week 02 HSV bands and `name_color`, plus the CIELAB pastel names of `color_name`.

**`game_pieces`**

- Foreground seeds, circular hue k-means with the elbow rule and merging, hysteresis growth from the seeds, the wide elliptical closing, `_split_touching` (distance transform peaks, a cut between convexity defects, cut pixels given back with `distanceTransform` labels), `_close_small_holes`, `describe` and the `classify` rules.

## What is not ported

- **`simple` and `contour` dot methods.** They need the k-means CIELAB palette chosen by silhouette score (`learn_palette`), which draws its sample with NumPy's random generator. The C++ binary rejects them with a clear error.
- **`--colors` filter and `--verbose` labels** of the Python CLI. Filter the JSON instead.
- **GrabCut refinement** in `game_pieces` (`ObjectSettings.grabcut`). It is off by default in Python, so the default outputs are unaffected.
- **Scoring against annotations** (`--truth`). The parity test does the scoring in Python.

## Check parity with Python

```sh
WEEK03_CPP_BUILD=build/cpp3 PYTHONPATH=. python -m pytest tests/test_week03_cpp_parity.py -v
```

The test writes every input as PNG so both sides decode the same pixels, then:

- for six synthetic scenes (`flat`, `fabric`, `pastel`, `distractors`, `tilted`, `crowded`) and the three polka dot photos, and for each of `log`, `doh`, `contrast` and `gray`, it matches the C++ dots one to one against Python's with `evaluate.match_dots`, and requires F1 >= 0.97, a median center difference below 0.1 px and the same color name for at least 95% of the matched dots;
- for the synthetic cones, cubes and rings scene (upright, rotated and hue shifted) and `objects.jpg`, it requires the same labels and a box IoU of at least 0.9 for every piece.

All 40 cases pass. Measured on macOS with OpenCV 5.0 on both sides (the pip wheel for Python, Homebrew for C++):

| method | dots (Python) | dots (C++) | matched | F1 | median center difference | largest center difference | same color name |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `log` | 485 | 485 | 485 | 1.000 | 0 px | 1.2e-4 px | 485 / 485 |
| `doh` | 485 | 485 | 485 | 1.000 | 0 px | 8.8e-3 px | 485 / 485 |
| `contrast` | 462 | 462 | 462 | 1.000 | 0 px | 6.1e-5 px | 462 / 462 |
| `gray` | 351 | 351 | 351 | 1.000 | 0 px | 6.8e-5 px | 351 / 351 |

Totals are over the six synthetic scenes and the three photos (`gray` finds no dots on `polka_dots_2.jpg` on either side). Every image and method has F1 1.000. The small differences come from the last bits of float rounding in the 3D peak fit and the ray refinement. `polka_dots_2.jpg` also produces the same 5286 `log` candidates on both sides. `dog` is not in the test, but a spot check on the `flat`, `fabric` and `crowded` scenes also gave F1 1.000 with centers within 6.1e-5 px.

`game_pieces` gives the same labels, boxes, centers and hue clusters (3.0, 21.1 and 123.0) as Python on `objects.jpg` (2 cones, 2 cubes, 2 rings), and the same labels with box IoU at least 0.9 on the three synthetic scene variants.

## Notes

- **Hue k-means sample.** When there are more than 20000 foreground pixels, Python clusters a random subset drawn with NumPy's generator. The C++ version takes an evenly strided subset instead. k-means centers on the hue circle barely move, so the clusters and pieces stay the same.
- **Speed.** Python spends much of its time in NumPy array passes (the peak comparisons over seven maps per scale, the batched ray arrays). C++ does that work in single loops, so what remains is mostly the OpenCV Gaussian blurs and dilations both sides share. On `polka_dots_2.jpg` (1280 x 1280) with `log`, in-pipeline time over three runs was 6.9, 9.9 and 17.3 s for C++ (Release) against 36.2, 41.2 and 48.3 s for Python `find_dots`. The median ratio is about 4x. On `objects.jpg`, `game_pieces` took 0.68 to 0.88 s against 0.76 to 1.06 s for Python. These runs shared an 8 GB M1 with a training job (load average 25 to 41), so absolute numbers are noisy.
