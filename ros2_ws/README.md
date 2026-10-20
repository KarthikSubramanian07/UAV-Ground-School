# ros2_ws: the week 5 and 6 colcon workspace

| Package | What it is |
| --- | --- |
| [`py_pubsub`](src/py_pubsub), [`cpp_pubsub`](src/cpp_pubsub) | The ROS 2 Jazzy publisher and subscriber tutorials (`talker`, `listener`) |
| [`py_srvcli`](src/py_srvcli), [`cpp_srvcli`](src/cpp_srvcli) | The service and client tutorials (`ros2 run py_srvcli client 2 3`) |
| [`tutorial_interfaces`](src/tutorial_interfaces) | The custom msg and srv tutorial: `Num`, `Sphere`, `AddThreeInts` |
| [`ugs_interfaces`](src/ugs_interfaces) | `FlyToWaypoint.action` from the slides, `SetMode`, `GetMission`, `Mission`, `AllTypes` for the CDR tests, and week 6's `Takeoff.action` and `Land.action` |
| [`ugs_drone`](src/ugs_drone) | The slides' drone (`camera_driver`, `detector`, `planner`, `px4_bridge`, `drone.launch.py`), the slides' snippets, and the QoS and callback `lab` |
| [`ugs_px4`](src/ugs_px4) | Week 6: `takeoff_server` and `takeoff_client` for PX4 over uXRCE-DDS, `fake_px4`, `flight_recorder`, `sitl.launch.py` (PX4 SITL, Gazebo, agent) and `takeoff.launch.py`. Needs px4_msgs, so it builds in the week 6 image ([`px4/Dockerfile`](px4/Dockerfile)) |

```bash
docker build -t ugs-ros:jazzy ros2_ws          # or: python -m week05 docker build
docker run --rm -it -v "$PWD":/ugs ugs-ros:jazzy
# inside the container
colcon build && source install/setup.bash
colcon test && colcon test-result --verbose
ros2 launch ugs_drone drone.launch.py
```

Week 6, in the PX4 image (PX4 v1.17.0 SITL, Gazebo Harmonic, the Micro XRCE-DDS Agent and px4_msgs in `/opt/px4_ws`):

```bash
python -m week06 docker build
python -m week06 docker run ros2 launch ugs_px4 sitl.launch.py   # take off to 5 m in Gazebo, hover, land
```

The packages are Apache 2.0, like the ROS 2 examples the tutorial code comes from; the rest of the repository is MIT. See [`week05/README.md`](../week05/README.md) for the full write up.
