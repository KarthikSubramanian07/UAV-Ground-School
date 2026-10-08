# Week 4 notes: flight controllers

Notes from UAVs@Berkeley Software Ground School, 29 September 2026. Agenda: what a flight controller does, the sensors inside and around it, the protocols that connect them, firmware, and the skill booster ("Protocol Pro").

## 1. Flight controller and onboard computer

The slides compare them to the two halves of a nervous system.

| Flight controller (the Cube) | Onboard computer (a Jetson) |
| --- | --- |
| Autonomic: keeps the drone stable without being asked | Somatic: decides what to do |
| Fuses IMU, GPS, barometer and compass into an attitude and position estimate | Runs computer vision, path planning, SLAM, ROS nodes and mission logic |
| Runs the control loops (PID) at hundreds of Hz | Sends commands to the flight controller and reads its telemetry |
| Not autonomous on its own: follows the RC link, a mission or an onboard computer | Connects over serial, Ethernet or USB |
| Talks MAVLink over UART or CAN | Talks MAVLink or DDS |

The club flies the CubePilot Cube Orange+ (the "orange cube boi"), and this week's design is built around it.

## 2. Sensors

Inside the flight controller:

* **IMU**: accelerometers and gyroscopes. Measures angular rate and linear acceleration; attitude comes from integrating and fusing them. The Cube Orange+ has three, two of them on a heated, vibration isolated board.
* **Magnetometer (compass)**: heading from the Earth's magnetic field. Corrects gyro yaw drift. "Kind of annoying to calibrate": motors and power wires distort it, which is why the best compass is usually the one in the GPS mast.
* **Barometer**: altitude from air pressure, for altitude hold. Sensitive to prop wash and sunlight, hence the foam over it.

Outside:

* **GPS**: position, velocity and time; enables waypoints, return to home and geofences. The club uses **RTK**: a base station on the ground sends corrections that bring accuracy from metres to about a centimetre.
* **Cameras**: can feed both the flight controller (a gimbal takes angle commands) and the onboard computer (video for detection). Sensor size, resolution, field of view, frame rate, dynamic range, interface and digital against optical zoom are the specs that matter.
* **RC transmitter and receiver**: the pilot's sticks, usually on 2.4 GHz, passed to the flight controller as SBUS or CRSF.

## 3. Protocols

| Protocol | Wires | Topology | Used for |
| --- | --- | --- | --- |
| UART (serial) | TX, RX, ground (and optionally RTS and CTS) | exactly two devices | telemetry radios, GPS, companion computers, RC receivers |
| CAN | CAN_H and CAN_L, a differential pair | a bus: many devices, a "group chat" | GPS, smart ESCs, gimbals, power monitors; robust over long cables |
| I2C | SDA and SCL | a bus with one master; each device has an address | barometers, compasses, rangefinders |
| SPI | clock, MOSI, MISO, one chip select per device | one master, many devices | IMUs and other fast sensors inside the flight controller |
| MAVLink | (a message format, not wires) | runs over UART, UDP or CAN | commands, telemetry, parameters, missions |
| DDS | (a message format) | publish and subscribe, used by ROS 2 | real time topics between the flight controller and the onboard computer |

The common hybrid: **MAVLink to the ground station** (over a radio or UDP) and **DDS inside the drone** between the flight controller and the onboard computer.

What the implementations in `week04/` add to the slides:

* **UART has no clock wire.** Both ends must agree on the baud rate, data bits, parity, stop bits and polarity. SBUS is 100000 baud, 8E2 and inverted, which is why it needs an inverter or a flight controller port made for it.
* **CAN arbitration is free.** A dominant 0 beats a recessive 1, every node listens while it talks, and a node that reads a 0 after sending a 1 stops. The lowest identifier wins without a wasted bit. Bit stuffing (an opposite bit after five equal ones) keeps every receiver's clock locked.
* **MAVLink carries its schema in the checksum.** The CRC includes a byte computed from the message's name and field layout (CRC_EXTRA), so two ends that disagree on a message definition reject it instead of misreading it. MAVLink 2 also drops trailing zeros and can sign every packet with SHA-256.
* **DroneCAN (the CAN protocol ArduPilot uses) does the same with a 64 bit signature** of each message definition, seeded into the transfer CRC of multi frame messages.

## 4. Firmware

Firmware is the software on the flight controller that we do not write: **ArduPilot**, **PX4** or **Betaflight**. It is configured through hundreds of parameters, and those parameters decide what the hardware does: which protocol each port speaks, which outputs drive motors and with which ESC protocol, the flight modes, and the control loop gains.

| Firmware | Typical use | Strengths | Limitations |
| --- | --- | --- | --- |
| ArduPilot | autonomous missions, mapping, research | rich features (GPS, geofence, RTL), multirotors, planes, VTOL, rovers, boats | heavier codebase, slower tuning |
| Betaflight | FPV racing and freestyle | very responsive manual control, fast PID loops | little autonomy, no GPS missions |
| PX4 | research, modular development, industry | clean architecture, SITL and HITL, ROS 2 integration | steeper learning curve |

Applied to this week's design, feature by feature with sources ([`docs/week04/firmware.md`](firmware.md)): ArduPilot 4.7.1 supports 14 of its 15 features as designed and the last (DDS) with a custom build; PX4 v1.16.2 would fly it with six changes (no SIYI driver, CRSF and signing not in the default build, the TFmini-S only on UART, the Here4 not named, no terrain following in missions); Betaflight cannot fly it at all, since there is no Cube Orange+ board config and it has no companion computer control, RTK injection or ADS-B.

Two ArduPilot details that cost time this week, both confirmed on the real firmware:

* In 4.7 the per link stream rates are `MAVn_*` (they were `SRn_*`), and `n` counts **MAVLink ports in serial order**, not serial port numbers: USB is MAV1 and TELEM1 is MAV2 on this design.
* Stream rates are read **at boot**. Changing them at runtime does nothing until a reboot (or a `REQUEST_DATA_STREAM` from the ground station).

### PID control

The slides' characters:

* **Preeti (proportional) watches the present**: pushes back in proportion to the error. On her own she overshoots, and with motor lag the oscillation grows.
* **Darren (derivative) watches the future**: damps the motion by reacting to the rate of change. With Preeti, the response settles, and at the right amount it is **critically damped**: as fast as possible without overshooting.
* **Isabelle (integral) watches the past**: accumulates error, so a constant disturbance (a gust, an off centre battery) cannot leave a permanent offset. Too much of her and the response oscillates.

Ideal gains differ from drone to drone because they depend on its inertia and motors. ArduPilot flies a cascade: an angle P loop feeds a rate PID, with input shaping so a step command becomes a smooth one. `week04/pid.py` simulates both.

## 5. Skill booster 3: Protocol Pro

* Design your own drone.
* Pick out a flight controller and some sensors off the internet.
* Draw a model of what the wiring setup looks like to connect everything together.
* Think about communication protocols and ensure compatibility.
* Submit everything to the repo: documentation is key.

Solution: [`week04/README.md`](../../week04/README.md).
