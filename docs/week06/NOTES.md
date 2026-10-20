# Week 6 notes: PX4 SITL, Gazebo and uXRCE-DDS

Week 6's assignment is optional and has no lecture slides; these notes are the setup guide the brief points at (the PX4 [development environment](https://docs.px4.io/main/en/dev_setup/dev_env) and [ROS 2 user guide](https://docs.px4.io/main/en/ros2/user_guide)) condensed to what this repo actually ran, followed by every problem it hit on the way.

## The pieces

| Piece | What it is | Version here |
| --- | --- | --- |
| PX4 SITL | The real PX4 firmware compiled for Linux (`px4_sitl_default`), the same flight stack as on a Pixhawk | v1.17.0 |
| Gazebo | The physics and sensor simulator; PX4's `gz_bridge` module drives it in lockstep | Harmonic (gz-sim 8) |
| `gz_x500` | PX4 airframe 4001, a Holybro X500 quadcopter model | from PX4-gazebo-models |
| `uxrce_dds_client` | A PX4 module: speaks DDS-XRCE to an agent, maps uORB topics to and from DDS | in PX4 v1.17.0 |
| Micro XRCE-DDS Agent | Owns the DDS entities the client asks for; to ROS 2 they look like a node's | v2.4.3 (the guide's pick for Jazzy) |
| `px4_msgs` | The uORB messages as ROS 2 `.msg` files; must match the firmware | v1.17.0 |
| ROS 2 | Jazzy, on Ubuntu 24.04 (the guide's recommendation) | Jazzy |

## Setting it up

PX4 recommends Ubuntu 24.04 with ROS 2 Jazzy. On Ubuntu the guides' own steps work:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh          # build tools, Python packages, Gazebo Harmonic
cd PX4-Autopilot && make px4_sitl gz_x500           # builds, starts Gazebo and PX4

git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git   # into a colcon workspace
git clone -b v1.17.0 https://github.com/PX4/px4_msgs.git                   # match the firmware
colcon build && source install/setup.bash
MicroXRCEAgent udp4 -p 8888                          # PX4 SITL connects to UDP 8888 on localhost
ros2 topic list                                      # /fmu/in/... and /fmu/out/...
```

On macOS (this repo's machine is an 8 GB Apple M1) ROS 2 does not run natively, so everything runs in one arm64 Linux container instead: [`ros2_ws/px4/Dockerfile`](../../ros2_ws/px4/Dockerfile) does exactly the steps above, pinned, on top of week 5's ROS 2 image. `python -m week06 docker build` builds it (about an hour the first time, most of it Gazebo's packages and PX4's 1,141 compile steps), and `python -m week06 fly` runs a flight in it. Gazebo runs headless (`HEADLESS=1`): the x500's sensors need no rendering.

## The action

`ros2_ws/src/ugs_px4` holds the assignment:

* `takeoff_server` serves `/takeoff` (`ugs_interfaces/action/Takeoff`: altitude, climb rate, method) and `/land`. It accepts one goal at a time between 1 and 120 m, and reports a phase, the altitude, the vertical speed and the distance left at 10 Hz.
* `takeoff_client` sends a goal (5 m by default), prints the feedback, and optionally cancels it partway or lands afterwards. Its exit code says whether the flight went as asked.
* `sitl.launch.py` starts the agent, PX4 SITL with Gazebo, the server and the client in one command; `takeoff.launch.py` starts only the server and client, for a PX4 started by hand (or the fake one).

Two ways to take off, both implemented, both flown:

1. **Offboard** (the PX4 ROS 2 offboard example's way). Stream `OffboardControlMode` and `TrajectorySetpoint` first, switch to offboard with `VEHICLE_CMD_DO_SET_MODE` (custom main mode 6), arm, then move the setpoint. The server moves it along a trapezoidal velocity profile (1 m/s by default, 1 m/s² at either end) rather than jumping it to 5 m, so PX4's position controller is asked for something it can fly, and the climb rate is the goal's to choose.
2. **Takeoff mode** (PX4's own). Send `VEHICLE_CMD_NAV_TAKEOFF` with the altitude in AMSL (`ref_alt` of the local origin plus the goal), then arm. PX4 climbs at `MPC_TKO_SPEED` and switches itself to hold when it gets there.

The logic is a state machine with no ROS in it ([`takeoff.py`](../../ros2_ws/src/ugs_px4/ugs_px4/takeoff.py)): `WAIT_FOR_PX4`, `PREFLIGHT`, `OFFBOARD` (or `TAKEOFF_MODE`), `ARMING`, `CLIMB`, `SETTLE`. The goal succeeds after the altitude stays within 0.15 m and the vertical speed under 0.15 m/s for a second. It aborts, with the reason, if PX4 refuses a command five times, leaves offboard (a failsafe or a pilot), disarms, goes quiet for a second, or the preflight checks never pass. Canceling holds the altitude reached.

## How it is tested

* 29 unit tests ([`tests/test_week06_takeoff.py`](../../tests/test_week06_takeoff.py)) fly the logic against a model of PX4's commander rules ([`model.py`](../../ros2_ws/src/ugs_px4/ugs_px4/model.py)): the happy paths, every refusal, offboard loss, a pilot takeover, cancel, and landing.
* Six launch tests in `ugs_px4/test` run the real nodes, in ROS 2, against `fake_px4` (that model behind PX4's exact topic names, types and QoS): offboard, takeoff mode, cancel, arming refused, a takeover, and a goal rejected for its altitude. CI runs them in the `ros` job.
* `.github/workflows/px4-sitl.yml` builds the PX4 image and flies three real SITL flights on GitHub's runners: offboard, takeoff mode, and the pure Python XRCE client.
* The flights in [`flights/`](flights) were flown on the M1 and are measured in [`RESULTS.md`](RESULTS.md).

## Problems

Problems 1 to 9 are the ones this repo actually hit, in order. Problems 10 to 13 are PX4 rules the server was built around from the documentation before they could bite; the unit and launch tests break each rule on purpose to check the server's answer.

### 1. The disk fills up

**Symptom:** `no space left on device` from Docker and the shell, partway through the image build.

**Fix:** Budget about 10 GB: the finished image is 8.2 GB, most of it Gazebo Harmonic's packages and PX4's build. The image deletes PX4's object files after linking, and the PX4 clone is shallow with shallow submodules.

### 2. PX4's setup script cannot upgrade numpy inside the ROS image

**Symptom:** `ERROR: Cannot uninstall numpy 1.26.4, RECORD file not found. Hint: The package was installed by debian.`

**Fix:** Run `Tools/setup/ubuntu.sh` with `PIP_IGNORE_INSTALLED=1`, so its Python packages install into /usr/local beside Debian's instead of trying to replace them.

### 3. PX4 will not arm without a ground station

**Symptom:** `Preflight Fail: No connection to the GCS` in the PX4 log, and `VehicleStatus.pre_flight_checks_pass` stays false forever.

**Fix:** The x500 airframe sets `NAV_DLL_ACT 2` (return home if the ground station link drops), which also makes a ground station link an arming requirement. With no QGroundControl running, set `NAV_DLL_ACT 0`. `sitl.launch.py` does it through the running PX4 (`px4-param set NAV_DLL_ACT 0`); pass `gcs:=true` to leave it alone when QGroundControl is connected.

### 4. PX4_PARAM_NAV_DLL_ACT=0 does nothing

**Symptom:** PX4's startup script accepts `PX4_PARAM_<name>` overrides, but with `PX4_PARAM_NAV_DLL_ACT=0` the parameter still reads 2.

**Fix:** The override runs before the airframe file. Setting a parameter to its firmware default (0) leaves it marked as default, so the airframe's later `param set-default NAV_DLL_ACT 2` wins. Set it after startup instead: the launch waits until `px4-param compare NAV_DLL_ACT 2` succeeds, then sets 0.

### 5. The topic you subscribed to is silent

**Symptom:** `ros2 topic echo /fmu/out/vehicle_status` prints nothing, although `ros2 topic list` shows plenty of `/fmu/out` topics.

**Fix:** PX4 1.16 and later suffix versioned messages with their version: it is `/fmu/out/vehicle_status_v1` and `vehicle_local_position_v1` (also `battery_status_v1`, `home_position_v1` and others; `ros2 topic list` tells you). The server subscribes to both names, so it works with older firmware too.

### 6. Subscriptions match nothing at all

**Symptom:** The suffixed topic exists, `ros2 topic info` shows a publisher, and the callback still never runs.

**Fix:** PX4's writers are best effort (the capture in RESULTS.md shows every `/fmu/out` writer's QoS); a default, reliable ROS 2 subscription is incompatible and silently never matches. Subscribe with `qos_profile_sensor_data`, as the PX4 guide says, or `ros2 topic echo --qos-reliability best_effort`. Week 5 measured exactly this: 0 of 20 messages.

### 7. The climb sits still, then races

**Symptom:** The first SITL flight armed, then sat on the ground for 3.2 s while the server's setpoint climbed away to 2.5 m; PX4 then chased it at 2.6 m/s, more than twice the 1 m/s the goal asked for, with 1.2 m RMS tracking error.

**Fix:** After arming PX4 spools the motors up for `COM_SPOOLUP_TIME` (1 s) and ramps its takeoff over `MPC_TKO_RAMP_T` (3 s). A reference on a fixed clock runs ahead of all that. The server's reference now stops whenever it is 0.4 m ahead of the vehicle (`max_lead`). In the final flights: 2.57 m/s and 1.22 m RMS with the reference on a fixed clock (`--max-lead 0` reproduces it), 1.21 m/s and 0.30 m RMS with the governor. Both are in RESULTS.md.

### 8. Gazebo's topics are invisible from a second process

**Symptom:** `gz topic -l` lists nothing, and the ground truth recorder receives no poses, although Gazebo is running.

**Fix:** PX4 starts Gazebo with `GZ_IP=127.0.0.1`, and gz-transport only discovers peers on the same interface. Export `GZ_IP=127.0.0.1` in every process that talks to Gazebo (`week06/truth.py` sets it itself).

### 9. The setpoint stream stalls on a busy executor

**Symptom:** In the launch tests the fake PX4 dropped out of offboard into hold one second after accepting it, with nothing wrong in the logs; the server's arming retries came 1.8 s apart instead of 1 s.

**Fix:** The 20 Hz loop was an rclpy timer on the MultiThreadedExecutor, beside 50 Hz telemetry and two action servers, and went 0.4 to 1.2 s between calls: longer than PX4's 1 s offboard loss failsafe. Giving the timer its own callback group did not help. Three changes, each found by a failure: the loop runs in a thread of its own (rclpy publishers are thread safe) and is never more than 0.3 s late; PX4's telemetry gets its own node on a SingleThreadedExecutor in another thread, after GitHub's runner let it fall 2 s behind; and liveness is judged by the 50 Hz local position (2 s), not `vehicle_status_v1`, which SITL left silent for 2.5 s while Gazebo ran at half speed. All six launch tests now pass with four CPU-burning processes competing on four cores.

### 10. PX4 refuses offboard mode

**Symptom:** `VehicleCommandAck` for `DO_SET_MODE` comes back `TEMPORARILY_REJECTED`, and the vehicle stays in hold.

**Fix:** PX4 only accepts offboard while `OffboardControlMode` is already streaming (at least 2 Hz). The server streams a hold setpoint from the moment PX4 answers, and asks for offboard a second later.

### 11. The drone drops back to hold after the goal succeeds

**Symptom:** The action reports success at 5 m, then PX4 switches to `AUTO_LOITER` a second later.

**Fix:** Losing the offboard stream for `COM_OF_LOSS_T` (1 s) is a failsafe. The server keeps streaming the last setpoint between goals, from its 20 Hz timer, not from the goal's thread. The launch tests and the takeoff lab drop the stream on purpose to check the server reports it.

### 12. Setpoints in the wrong frame

**Symptom:** A setpoint of `z = 5` drives the vehicle into the ground.

**Fix:** PX4 is NED: z points down, so 5 m up is `z = ground_z - 5`, where the ground is wherever z was while landed (0.09 m in SITL, not 0). ROS is ENU; the server converts at the boundary and the action speaks metres up.

### 13. Takeoff mode wants an AMSL altitude

**Symptom:** `NAV_TAKEOFF` with `param7 = 5` climbs to the wrong height or not at all.

**Fix:** param7 is metres above mean sea level. The server sends `ref_alt` (the local origin's AMSL altitude from `VehicleLocalPosition`; 0.24 m in PX4's default Gazebo world) minus the ground's z plus the goal, with NaN latitude, longitude and yaw to take off in place.
