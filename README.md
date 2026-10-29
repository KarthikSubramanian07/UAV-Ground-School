# UAV Ground School

**Solutions to the UAVs@Berkeley Software Ground School 2026, written to competition grade: color segmentation, SUAS target shapes and drone video mosaicking (week 2), blob detection, classical game piece detection and YOLOv8 (week 3), a flight controller design verified on real ArduPilot firmware with every protocol on it implemented from scratch (week 4), ROS 2 from the tutorials down to the bytes, with a pure Python DDS participant that talks to real ROS 2 nodes (week 5), and a ROS 2 takeoff action flying real PX4 SITL in Gazebo over uXRCE-DDS, with DDS-XRCE rebuilt in pure Python (week 6), in Python and C++, measured against ground truth.**

[![CI](https://github.com/KarthikSubramanian07/UAV-Ground-School/actions/workflows/ci.yml/badge.svg)](https://github.com/KarthikSubramanian07/UAV-Ground-School/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6%20to%205.0-5c3ee8)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

`computer-vision` `opencv` `uav` `drones` `image-stitching` `orthomosaic` `bundle-adjustment` `suas` `color-segmentation` `blob-detection` `object-detection` `yolov8` `ardupilot` `mavlink` `dronecan` `can-bus` `flight-controller` `ros2` `dds` `rtps` `cdr` `px4` `gazebo` `sitl` `uxrce-dds` `dds-xrce` `ros2-actions` `offboard-control` `cpp`

Live showcase: **[uav-ground-school.pages.dev](https://uav-ground-school.pages.dev/)** (week 2, with an in browser color splitter) **[/week3](https://uav-ground-school.pages.dev/week3)** (with a detector explorer) **[/week4](https://uav-ground-school.pages.dev/week4)** (a clickable wiring diagram, a logic analyser and a PID playground) **[/week5](https://uav-ground-school.pages.dev/week5)** (a replay of the ROS 2 drone mission, a QoS playground and a packet inspector) and **[/week6](https://uav-ground-school.pages.dev/week6)** (the SITL takeoffs replayed against Gazebo's ground truth, a takeoff lab that breaks PX4 on purpose, and an XRCE packet inspector).

Agent entry points: **[/llms.txt](https://uav-ground-school.pages.dev/llms.txt)** (when to use this project and how to fetch it), Markdown via `Accept: text/markdown` on the same page URLs, plus **[/about](https://uav-ground-school.pages.dev/about)**, **[/contact](https://uav-ground-school.pages.dev/contact)**, and **[/privacy](https://uav-ground-school.pages.dev/privacy)**.

<p align="center">
  <img src="docs/week02/rough_path.jpg" alt="A mosaic of a simulated five pass survey flight over a voxel world, with keyframe footprints and the flight path drawn on top" width="100%">
</p>

The ground school hands out a "skill booster" each week: something small that sort of works counts as a win. This repo takes each booster as far as it can reasonably go, then measures the result against ground truth instead of eyeballing it.

## Weeks

| Week | Topic | Challenges | Where |
| --- | --- | --- | --- |
| 2 | Computer vision and aerial imagery | Color Me Impressed, I'll be Needin' Stitches, both again in C++ | [`week02/`](week02/README.md), [lecture notes](docs/week02/NOTES.md) |
| 3 | Object detection | Blob detection (SimpleBlobDetector, LoG, DoG, DoH, contours), cones, cubes and rings, YOLOv8, the classical parts again in C++ | [`week03/`](week03/README.md), [lecture notes](docs/week03/NOTES.md) |
| 4 | Flight controllers | Protocol Pro: a drone designed around the Cube Orange+, wired pin by pin, a design rule checker, the ArduPilot parameters, MAVLink, DroneCAN, CAN, RTCM, CRSF, SBUS and DShot from scratch, ArduCopter 4.7.1 in the loop, the codecs again in C++ | [`week04/`](week04/README.md), [lecture notes](docs/week04/NOTES.md) |
| 5 | ROS 2: nodes and communication | The Jazzy publisher and subscriber and service and client tutorials in Python and C++ with launch tests, the slides' drone as four nodes using topics, services and an action, QoS and callbacks measured, and RTPS, CDR and REP 2011 hashes from scratch, talking to real ROS 2 | [`week05/`](week05/README.md), [`ros2_ws/`](ros2_ws), [lecture notes](docs/week05/NOTES.md), [results](docs/week05/RESULTS.md) |
| 6 | PX4 SITL and uXRCE-DDS (optional) | PX4 v1.17 SITL in Gazebo Harmonic and the Micro XRCE-DDS Agent in Docker on an M1, a `/takeoff` action server and client (offboard and takeoff mode, cancel, `/land`), every failure path tested against a fake PX4, measured against Gazebo's true pose, and DDS-XRCE from scratch flying the same takeoff with no ROS 2 | [`week06/`](week06/README.md), [`ugs_px4`](ros2_ws/src/ugs_px4), [setup notes and problems](docs/week06/NOTES.md), [results](docs/week06/RESULTS.md) |

## Week 2 at a glance

**Color Me Impressed.** Give it an image path; it splits the image into one image per color, shows them, and prints the center of every colored object. Beyond the brief it classifies each object's shape the way a SUAS target detector would (red octagon, blue triangle, yellow star) and can learn an image's palette with k-means in CIELAB.

<p align="center">
  <img src="docs/week02/colors/targets_sheet.jpg" alt="Contact sheet: the original target image, detected object centers, and one panel per color layer" width="85%">
</p>

**I'll be Needin' Stitches.** Give it a nadir drone video; it samples frames, tracks them with ORB or SIFT, closes loops between overlapping lawnmower passes, jointly optimizes every frame pose, fixes exposure differences, and blends a seamless mosaic.

The club's flight video is private, so the repo ships a flight simulator: a seeded Minecraft style world, a survey flight over it rendered to MP4 (lens distortion, motion blur, exposure drift, altitude wobble, sensor noise) and the true camera pose of every frame. On the harder flight:

| Pipeline | Pose RMSE | Max error | ZNCC vs true world |
| --- | ---: | ---: | ---: |
| Sequential chaining (what most solutions do) | 56.72 px | 159.65 px | 0.486 |
| + loop closure and bundle adjustment | 7.21 px | 17.56 px | 0.839 |
| + lens undistortion, gain compensation, multi band blending | **0.37 px** | **0.91 px** | **0.941** |
| OpenCV's built in `cv2.Stitcher` (calm flight, 67 frames) | broken mosaic after 104 s | | |

Full numbers in [`docs/week02/benchmark.md`](docs/week02/benchmark.md), regenerated by `python -m week02 benchmark docs/week02`.

## Week 3 at a glance

<p align="center">
  <img src="docs/week03/hero.jpg" alt="The three polka dot photos with every detected dot outlined" width="100%">
</p>

**Blob detection.** Every dot in the three photos was annotated and checked by hand, so seven detectors are scored instead of eyeballed. SimpleBlobDetector on grayscale finds none of the 53 dots on the orange fabric (they have the cloth's gray value); on a color contrast map it finds 50. LoG, DoG and DoH written from scratch in CIELAB, with an ellipse fit to each dot's color edge, reach 0.991 to 0.998 F1 on the photos, and 0.998 F1 with 0.02 px median center error on 21 synthetic scenes with exact truth.

**Cones, cubes and rings** with no training data: saturation seeds, hysteresis, a convexity defect cut between the touching cubes, and a classifier that uses only geometry (holes, convexity, triangularity). 6 of 6 pieces, mean mask IoU 0.942 against SAM 2 masks, unchanged under rotation, mirroring, scaling and hue shifts.

**YOLOv8** on the notebook's football dataset, without a Roboflow key. Every test clip in Roboflow's split also appears in training, so models are compared on held out clips. Training on tiles and running sliced inference lifts mAP50 from 0.588 to 0.820 and ball AP50 from 0.00 to 0.57; offline augmentation and 1280 px inference did not help. Results: [`docs/week03/yolo/results.md`](docs/week03/yolo/results.md).

## Week 4 at a glance

<p align="center">
  <img src="docs/week04/wiring.svg" alt="The Protocol Pro wiring diagram around a Cube Orange+" width="100%">
</p>

**A drone designed from parts you can buy**, every figure sourced and every unpublished one marked as an estimate: Cube Orange+, Here4 RTK on DroneCAN, Jetson Orin Nano on DDS and MAVLink, SIYI gimbal, ExpressLRS, RFD900x and a TFmini-S on a Holybro X500 V2. The wiring diagram and a pin by pin harness are generated from the design, and a 30 rule checker (logic levels, baud rates, receive DMA, I2C addresses, CAN load, 5 V budgets, bandwidth, firmware features, ESC and connector ratings) reports 0 errors and 17 actionable warnings. Each rule is tested by breaking the design on purpose.

**Every protocol from its specification**: MAVLink from the XML (every `ardupilotmega` message byte identical to pymavlink, signing included), DroneCAN from the DSDL (signatures and transfers identical to pydronecan), CAN bit by bit with stuffing and arbitration, RTCM 3, CRSF, SBUS, DShot with GCR telemetry, UART and I2C. One RTK correction is followed through five of them, from base station to the drone's GPS.

**Checked on the real firmware**: ArduCopter 4.7.1 built from its release tag and driven only by this repo's MAVLink code accepts 72 of the 75 generated parameters (the other three cannot exist in simulation), sends exactly the predicted telemetry, verifies MAVLink signing, and flies this airframe's mass and inertia. The pitch step rises in 0.32 s; the week's own PID simulator with ArduPilot's gains predicts 0.33 s.

## Week 5 at a glance

**The tutorials, done properly**: the Jazzy publisher and subscriber and service and client, in Python and C++, built with colcon in a Docker image of the repo's own (ROS 2 does not run on macOS), every ament linter on, and launch tests that start the nodes and check what they print.

**The slides' drone, for real**: `camera_driver`, `detector`, `planner` and `px4_bridge` talking through topics (best effort odometry, like PX4), services (`/arm`, `/set_mode`) and the slides' `FlyToWaypoint` action with feedback and cancel. The planner finds a target in the camera and lands 0.14 m from it. The lab measures the slides' claims: an incompatible QoS pair receives 0 of 20, a 300 ms callback stalls a 50 ms timer to 318 ms until it gets its own callback group, and `call()` inside a callback deadlocks.

**ROS 2 without ROS 2**: RTPS, CDR and REP 2011 type hashes written from the specifications in pure Python. The participant joins a live ROS 2 graph as a peer: the tutorial's talker, listener, service and both clients all work with it, as does a Cyclone DDS talker, and it recorded the whole drone mission (42 MB of camera frames reassembled from fragments). Against ROS 2 itself: 1,210 of 1,210 type hashes, 1,265,625 of 1,265,625 QoS verdicts, and 300 of 300 CDR messages agree.

## Week 6 at a glance

**The assignment, on real PX4**: [`ugs_px4`](ros2_ws/src/ugs_px4)'s `takeoff_server` takes PX4 v1.17.0 SITL off to 5 m in Gazebo through the Micro XRCE-DDS Agent, and `takeoff_client` asks it to: one command, `python -m week06 fly`, from an image that builds PX4, Gazebo Harmonic, the agent and px4_msgs from source. The x500 held 5.03 m by PX4's estimate (5.06 m by Gazebo's true pose), rising in 4.0 s with 6 cm of overshoot. The first version's reference ran on a fixed clock and left PX4 chasing it at 2.6 m/s while the motors spooled up; a reference that waits for the vehicle brought that to 1.2 m/s and the tracking error from 1.22 to 0.30 m RMS. Both flights are in the results, measured in PX4's simulated time.

**Every way it can go wrong**: a fake PX4 that follows the commander's rules runs under six launch tests: arming refused, offboard refused, a pilot taking over, cancel, a goal out of range. The page's takeoff lab runs a JavaScript port of the same logic, identical to the Python on all nine scenarios.

**PX4 without ROS 2**: DDS-XRCE (the protocol between PX4 and the agent) from the OMG specification and eProsima's binary representations. The dissector rebuilds PX4's whole session from a capture (197 objects, 65 topics, 17,157 submessages, every sample decoded with px4_msgs' field names), and a pure Python client logs in to the agent itself and flies the same takeoff and landing with no ROS 2 installed in the loop.

## Quick start

```bash
git clone https://github.com/KarthikSubramanian07/UAV-Ground-School
cd UAV-Ground-School
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test data: stop sign, apple, SUAS targets, and a survey flight with ground truth
python -m week02 simulate out/sim

# Option 1
python -m week02 colors out/sim/targets.jpg --out out/layers

# Option 2
python -m week02 stitch out/sim/flight.mp4 --out out/mosaic.jpg \
  --path-overlay out/path.jpg --truth out/sim/flight_truth.csv --world out/sim/world.png

# Week 3
python -m week03 compare docs/week03/photos/polka_dots_2.jpg --out out/detectors.jpg
python -m week03 objects docs/week03/photos/objects.jpg --out out/pieces.jpg
pip install -e ".[yolo]" && python -m week03 yolo fetch && python -m week03 yolo train baseline tiles

# Week 4
python -m week04 check
python -m week04 params --out protocol_pro.param
python -m week04 journey

# Week 5 (ROS 2 runs in Docker; week05's own tools need no ROS)
python -m week05 docker build && python -m week05 docker test
python -m week05 docker run ros2 launch ugs_drone drone.launch.py
python -m week05 decode docs/week05/captures/pubsub.pcap

# Week 6 (PX4 SITL and Gazebo run in Docker too)
python -m week06 docker build && python -m week06 fly        # take off to 5 m, hover, land
python -m week06 fly --xrce                                   # the same, from pure Python over XRCE
python -m week06 xrce session docs/week06/flights/offboard_5m.xrce.pcap
```

Using a real flight video works the same way: `python -m week02 stitch your_flight.mp4 --out mosaic.jpg`. Add `--live` to watch the mosaic grow and `--scale 0.5` for large 4K footage.

## Architecture

```mermaid
flowchart LR
    subgraph option1 [Color Me Impressed]
        img[image] --> hsv[BGR to HSV] --> bands[inRange per color band] --> layers[color layers]
        img --> lab[CIELAB k-means] --> layers
        layers --> cc[opening + connected components] --> centers[centers via np.where + np.mean]
        cc --> shapes[IoU template shape classifier]
    end
    subgraph option2 [I'll be Needin' Stitches]
        video[video] --> sample[every Nth frame] --> undistort[undistort] --> feat[CLAHE + grid bucketed ORB/SIFT]
        feat --> track[keyframe tracker + relocalization] --> loops[loop closure verification]
        loops --> ba[Levenberg Marquardt bundle adjustment] --> gains[gain compensation] --> blend[multi band blending] --> mosaic[mosaic]
    end
    sim[flight simulator] -. ground truth .-> eval[pose RMSE + ZNCC]
    mosaic -.-> eval
```

Week 3:

```mermaid
flowchart LR
    subgraph dots [Blob detection]
        photo[photo] --> lab[CIELAB]
        lab --> ss[LoG / DoG / DoH scale space] --> cand[candidates]
        lab --> contrast[background contrast] --> sbd[SimpleBlobDetector] --> cand
        cand --> fit[edge rays + ellipse fit] --> verify[fill, leak, surround] --> filt[radius, color filters]
    end
    subgraph pieces [Cones, cubes, rings]
        img[photo] --> seeds[saturation seeds] --> grow[hue clusters + hysteresis] --> split[convexity defect split] --> geo[holes, solidity, triangularity] --> cls[class]
    end
    subgraph yolo [YOLOv8]
        hf[Hugging Face mirror] --> clip[clip split] --> tiles[tiles + augmentation] --> train[Ultralytics] --> sliced[sliced inference] --> map[COCO mAP]
    end
    truth[hand checked dots, SAM masks, synthetic truth] -. scoring .-> filt
    truth -.-> cls
```

| Module | Responsibility |
| --- | --- |
| [`week02/colors.py`](week02/colors.py) | HSV band partition, k-means palette discovery, object centers, layer rendering |
| [`week02/shapes.py`](week02/shapes.py) | Rotation and scale invariant shape classification by IoU template search |
| [`week02/features.py`](week02/features.py) | Feature extraction, symmetric matching, RANSAC similarity estimation |
| [`week02/stitching.py`](week02/stitching.py) | Frame sampling, tracking, loop closures, the end to end pipeline |
| [`week02/mosaic.py`](week02/mosaic.py) | Bundle adjustment, gain compensation, feather and multi band blending |
| [`week02/synth.py`](week02/synth.py) | Test images, voxel world, survey flight renderer with ground truth |
| [`week02/evaluate.py`](week02/evaluate.py) | Umeyama alignment, pose error, photometric agreement |
| [`week02/cpp/`](week02/cpp) | C++17 ports of both challenges |
| [`week03/blobs.py`](week03/blobs.py) | SimpleBlobDetector wrapper, scale space LoG, DoG and DoH with pyramids, sub pixel peaks and sidelobe suppression, contour filtering |
| [`week03/dots.py`](week03/dots.py) | The polka dot pipeline: candidates, edge fit, verification, color names and filters |
| [`week03/objects.py`](week03/objects.py) | Cones, cubes and rings: seeds, hysteresis, the touching piece split, geometric classification |
| [`week03/synth.py`](week03/synth.py), [`evaluate.py`](week03/evaluate.py), [`benchmark.py`](week03/benchmark.py) | Synthetic scenes with exact truth, scoring rules, the detector benchmark |
| [`week03/football.py`](week03/football.py), [`augment.py`](week03/augment.py), [`slicing.py`](week03/slicing.py), [`detmetrics.py`](week03/detmetrics.py), [`detector.py`](week03/detector.py) | Option 2: dataset and clip split, Roboflow style tiling and augmentation, sliced inference, COCO style mAP, the experiments |
| [`week03/cpp/`](week03/cpp) | C++17 ports of the dot detectors and the game piece classifier |
| [`week04/design.py`](week04/design.py), [`check.py`](week04/check.py), [`params.py`](week04/params.py), [`diagram.py`](week04/diagram.py) | The design as data, the pin by pin harness, the design rule checker, the ArduPilot parameters, the wiring diagram |
| [`week04/mavlink.py`](week04/mavlink.py), [`dronecan.py`](week04/dronecan.py), [`can.py`](week04/can.py), [`rtcm.py`](week04/rtcm.py), [`rc.py`](week04/rc.py), [`dshot.py`](week04/dshot.py), [`uart.py`](week04/uart.py), [`i2c.py`](week04/i2c.py), [`crc.py`](week04/crc.py) | The protocols, from their specifications |
| [`week04/performance.py`](week04/performance.py), [`links.py`](week04/links.py), [`pid.py`](week04/pid.py), [`journey.py`](week04/journey.py), [`sitl.py`](week04/sitl.py) | Flight time, link budgets, the PID simulator, the RTK journey, ArduCopter SITL experiments |
| [`week04/cpp/`](week04/cpp) | C++17 port of the protocol codecs |
| [`week05/rtps.py`](week05/rtps.py), [`participant.py`](week05/participant.py) | RTPS from the specification, and a participant that joins a ROS 2 graph |
| [`week05/cdr.py`](week05/cdr.py), [`idl.py`](week05/idl.py), [`qos.py`](week05/qos.py) | CDR, interface definitions and REP 2011 type hashes, the rmw QoS rules |
| [`week05/experiments.py`](week05/experiments.py), [`docs.py`](week05/docs.py), [`dissect.py`](week05/dissect.py) | The live experiments in Docker, the docs generator, the packet dissector |
| [`week06/flight.py`](week06/flight.py), [`docs.py`](week06/docs.py), [`lab.py`](week06/lab.py), [`truth.py`](week06/truth.py) | Flight measurements, the docs generator and SITL runs, the takeoff lab's reference simulation, Gazebo ground truth |
| [`week06/xrce.py`](week06/xrce.py), [`dissect.py`](week06/dissect.py), [`xrce_client.py`](week06/xrce_client.py), [`xrce_fly.py`](week06/xrce_fly.py) | DDS-XRCE from the specification, the session dissector, a pure Python client of the agent, the takeoff over it |
| [`ros2_ws/src/`](ros2_ws/src) | The tutorial packages, `ugs_interfaces`, `ugs_drone` and week 6's `ugs_px4` (Apache 2.0, like the ROS code they build on) |
| [`ros2_ws/px4/Dockerfile`](ros2_ws/px4/Dockerfile) | PX4 v1.17.0 SITL, Gazebo Harmonic, the Micro XRCE-DDS Agent v2.4.3 and px4_msgs, on the week 5 image |
| [`site/`](site) | The showcase site, built by [`scripts/build_site.py`](scripts/build_site.py) |

## Quality

* `pytest` runs unit tests for every module plus end to end accuracy checks: shapes across 13 classes, 3 sizes and 3 rotations; exact color partitioning of the whole HSV cube; bundle adjustment removing injected drift and shrugging off outliers; a full stitch held to sub pixel pose error against the simulator.
* Week 3 tests cover scale space theory (peak scale and height), polarity, color only blobs, sidelobes, a comparison with scikit-image's `blob_log`, the edge fit on exact synthetic truth, evaluation semantics, object features and invariance, augmentations that move boxes with pixels, tiling, the mAP implementation against Ultralytics, and regression tests on the ground school photos.
* Week 4 tests compare the protocols with pymavlink, pydronecan and pyrtcm, break the design one rule at a time, validate the parameters against ArduCopter 4.7.1's metadata, check the recorded SITL measurements against the predictions, and hold the page's JavaScript PID simulator to bit for bit agreement with Python.
* Week 5 tests hold CDR, type hashes and QoS to fixtures recorded from rclpy, decode the committed captures of the tutorial nodes, and run two participants against each other over real sockets, at 30 percent packet loss too. A CI job in the ROS 2 Jazzy container builds the workspace, runs every ament linter and the launch tests, and checks the participant against the tutorial nodes, Fast DDS and Cyclone DDS live.
* Week 6 tests fly the takeoff logic against a model of PX4's commander (every refusal, offboard loss, takeover, cancel, landing), measure a model flight with exact answers, hold the XRCE codec and dissector to synthetic sessions and the real SITL capture, and hold the page's JavaScript port to identical traces. The ROS 2 job runs `ugs_px4`'s six launch tests against `fake_px4`, and a separate workflow builds PX4 SITL on GitHub's runners and flies three real takeoffs in Gazebo.
* The C++ binaries are checked for parity with Python in CI (week 2: same colors, centers within 1.5 px, same shapes, stitching accuracy; week 3: the same dots with F1 1.000 and centers within 0.01 px, the same game pieces; week 4: 2,908 protocol cases byte for byte).
* GitHub Actions: ruff (lint and formatting enforced everywhere), tests on Python 3.10 and 3.12, the C++ build against Ubuntu's OpenCV 4.6, the ROS 2 workspace in the Jazzy container, and the site build; PX4 SITL flights in their own workflow. Once every check is green on `main`, CI triggers a Cloudflare Pages deploy hook and the site at uav-ground-school.pages.dev rebuilds from that commit.

## License

MIT. See [LICENSE](LICENSE).
