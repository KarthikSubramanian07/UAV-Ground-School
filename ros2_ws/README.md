# ros2_ws: the week 5 colcon workspace

| Package | What it is |
| --- | --- |
| [`py_pubsub`](src/py_pubsub), [`cpp_pubsub`](src/cpp_pubsub) | The ROS 2 Jazzy publisher and subscriber tutorials (`talker`, `listener`) |
| [`py_srvcli`](src/py_srvcli), [`cpp_srvcli`](src/cpp_srvcli) | The service and client tutorials (`ros2 run py_srvcli client 2 3`) |
| [`tutorial_interfaces`](src/tutorial_interfaces) | The custom msg and srv tutorial: `Num`, `Sphere`, `AddThreeInts` |
| [`ugs_interfaces`](src/ugs_interfaces) | `FlyToWaypoint.action` from the slides, `SetMode`, `GetMission`, `Mission`, and `AllTypes` for the CDR tests |
| [`ugs_drone`](src/ugs_drone) | The slides' drone (`camera_driver`, `detector`, `planner`, `px4_bridge`, `drone.launch.py`), the slides' snippets, and the QoS and callback `lab` |

```bash
docker build -t ugs-ros:jazzy ros2_ws          # or: python -m week05 docker build
docker run --rm -it -v "$PWD":/ugs ugs-ros:jazzy
# inside the container
colcon build && source install/setup.bash
colcon test && colcon test-result --verbose
ros2 launch ugs_drone drone.launch.py
```

The packages are Apache 2.0, like the ROS 2 examples the tutorial code comes from; the rest of the repository is MIT. See [`week05/README.md`](../week05/README.md) for the full write up.
