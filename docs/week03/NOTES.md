# Week 3 notes: object detection

Notes from UAVs@Berkeley Software Ground School, 22 September 2026. Agenda: object detection with OpenCV, neural network basics, convolutional neural networks, making and using models, and the skill booster (blob detection or YOLO).

## 1. Basic object detection with OpenCV

* OpenCV has many filters that isolate one property of an image. **Blurring** and **level curves** (thresholding at a brightness, the "binary image, T=155" of coins on a table) reduce noise before anything else runs.
* `cv2.findContours()` turns a binary image into outlines, and each outline can be filtered on shape measurements.
* `cv2.SimpleBlobDetector` packages that recipe: threshold at many levels, find connected components at each, group components whose centers agree across levels, and filter the groups by **area**, **threshold** (color), **circularity**, **inertia** (elongation) and **convexity**. The slide's example uses it on medical scans (lesions) and on packets on a table.

How each filter works, since the skill booster asks for "size or color":

| Filter | Measure | Circle | Square | Thin ellipse |
| --- | --- | ---: | ---: | ---: |
| Circularity | 4 pi area / perimeter^2 | 1.0 | 0.785 | low |
| Convexity | area / convex hull area | 1.0 | 1.0 | 1.0 |
| Inertia | (minor / major axis)^2 of the second moments | 1.0 | 1.0 | low |
| Area | pixels | any | any | any |
| Color | 0 for dark blobs, 255 for bright | | | |

Two things the slides do not say, found while building `week03/`:

* SimpleBlobDetector only sees brightness. A cyan dot on orange fabric can have almost the same gray value as the cloth; on `polka_dots_2.jpg` the grayscale detector finds none of the 53 dots. Give it a color contrast image instead (Delta E from the local background) and it finds 51.
* "Filter by color" is easiest after detection: measure each dot's color from its inside and keep the colors you want (`python -m week03 dots image.jpg --colors red,green`).

### Beyond SimpleBlobDetector: scale space

The Laplacian of Gaussian (LoG) responds most strongly to a blob whose radius matches the Gaussian: `r = sqrt(2) sigma`. Multiply by `sigma^2` so responses at different scales are comparable, compute the response at a range of sigmas, and local maxima in (x, y, sigma) give each blob's position and size at once.

* **LoG**: `-sigma^2 * Laplacian(G_sigma * I)`.
* **DoG**: `(G_sigma * I - G_{k sigma} * I) / (k - 1)`, an approximation of LoG that costs two blurs (used by SIFT).
* **DoH**: `sigma^4 * det(Hessian)`, near zero on straight edges.

## 2. Neural network basics

* A neural network is a large graph of simple functions. Each node takes a weighted sum of the previous layer's outputs and applies an **activation function** (threshold, sigmoid, ReLU, tanh).
* Networks "learn" by adjusting the weights (backpropagation computes how each weight affects the loss; gradient descent nudges them).
* Demo: [playground.tensorflow.org](https://playground.tensorflow.org).

## 3. Convolutional neural networks

* **Convolution** slides a small kernel over the image to extract features (edges at first, parts of objects deeper in).
* **Pooling** shrinks feature maps by aggregating neighborhoods into one value, which cuts parameters and adds a little translation tolerance.
* Classic example: VGG-16, stacks of 3x3 convolutions and pooling, then fully connected layers.
* YOLO style detectors: a backbone (feature extractor), a neck that mixes features at several resolutions, and heads that predict boxes and classes at each scale in one pass.

## 4. Making and using models

The workflow:

1. **Collect data**: robust, covering as many cases as possible (lighting, angles, distances, backgrounds).
2. **Annotate**: slow by hand; expensive models like **SAM** (Segment Anything) can propose masks from a click or a box. `week03/annotations/objects.json` was made exactly that way: SAM 2.1 prompted with hand drawn boxes, then checked by eye.
3. **Train**: different preprocessing and sampling may be optimal. Roboflow offers preprocessing (auto orient, resize, tile, grayscale, contrast) and augmentation (flip, rotate, crop, shear, hue, saturation, brightness, exposure, blur, noise, cutout).

The club's own example: detecting SUAS targets (a red bullseye, shapes on grass) in aerial frames.

## 5. Skill booster 2

* **Option 1, OpenCV blob detection** ([tutorial](https://opencv.org/blob-detection-using-opencv/)): SimpleBlobDetector on the three polka dot images, filtering by size or color. Challenge 1: other methods (LoG, DoG, DoH, contour filtering), tested on the polka dots and also on the real objects and the distorted shapes images. Challenge 2: detect and classify the cones, cubes and rings in the objects file with accurate contours or bounding boxes.
* **Option 2, YOLO**: follow the Roboflow YOLOv8 notebook, train a model, explore Roboflow's preprocessing and augmentation, and try to improve performance. Deploying locally or on an edge device is not required.

Solutions: [`week03/README.md`](../../week03/README.md).
