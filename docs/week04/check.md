0 errors, 17 warnings, 61 checks passed.

| Level | Rule | Where | Finding |
| --- | --- | --- | --- |
| warning | logic level | `gimbal` | 3.3 V to 3.3 V (device level is an estimate) |
| warning | connectors | `telemetry` | custom cable: JST-GH-6 to 2x8-0.1in-header-16 |
| warning | connectors | `companion_dds` | custom cable: JST-GH-6 to 40-pin-2.54mm-header (pins 8,10,11,36,6) |
| warning | connectors | `gimbal` | custom cable: JST-GH-8 to unspecified (vendor cable) |
| warning | connectors | `rc` | custom cable: JST-GH-6 to solder-pads |
| warning | connectors | `lidar` | custom cable: JST-GH-4 to GH1.25-4P on sensor; Molex 51021-0400 (1.25 mm) on the 10 cm cable end |
| warning | firmware features | `companion_dds` | dds_xrce needs AP_DDS, which the stable ArduCopter 4.7.1 build for this board leaves out: build custom firmware (custom.ardupilot.org) with it |
| warning | supply voltage | `rc` | rx.UART wants 5 to 5 V; the Cube's 5 V rail spans 4.9 to 5.5 V |
| warning | supply voltage | `lidar` | lidar.MAIN wants 4.9 to 5.1 V; the Cube's 5 V rail spans 4.9 to 5.5 V |
| warning | supply voltage | `pdb` | esc.POWER: 13.2 to 16.8 V against 7.4 to 16.8 V, only 0.00 V margin |
| warning | supply voltage | `fc_power` | fc.POWER1: 5.1 to 5.3 V against 4.1 to 5.7 V, only 0.40 V margin |
| warning | supply voltage | `radio_power` | radio.HEADER: 5.15 to 5.25 V against 5.0 to 5.5 V, only 0.15 V margin |
| warning | ESC current | `esc` | 19.4 A per motor at full throttle on a 20 A ESC |
| warning | power module rating | `pm` | 85 A at full throttle, 22 A at hover; PM02 V3 is 60 A continuous, 100 A burst |
| warning | connector rating | `battery.MAIN` | XT60 with 12 AWG is rated 30 A continuous (Holybro): 22 A at hover, 85 A in full throttle bursts |
| warning | known issues | `companion` | open NVIDIA report: UART1 TX does not drive on JetPack 7 (L4T R39.2); stay on JetPack 6 until fixed |
| warning | known issues | `lidar` | TFmini-S ships in UART mode: send the I2C mode command once before wiring it to I2C |
| ok | physical layer | `telemetry` | MAVLink 2 over uart |
| ok | physical layer | `companion_dds` | DDS XRCE (ROS 2) over uart |
| ok | physical layer | `companion_mavlink` | MAVLink 2 over uart |
| ok | physical layer | `gimbal` | SIYI gimbal serial over uart |
| ok | physical layer | `rc` | CRSF (ExpressLRS) over uart |
| ok | physical layer | `can` | DroneCAN over can |
| ok | physical layer | `lidar` | I2C over i2c |
| ok | physical layer | `video` | Ethernet over ethernet |
| ok | physical layer | `motors` | DShot300 over pwm |
| ok | one device per UART | `all` | 21 ports, none double booked |
| ok | device protocol | `telemetry` | radio.HEADER speaks mavlink2 |
| ok | device protocol | `companion_dds` | companion.HEADER_UART1 speaks dds_xrce |
| ok | device protocol | `companion_mavlink` | companion.USB_A speaks mavlink2 |
| ok | device protocol | `gimbal` | camera.UART speaks siyi |
| ok | device protocol | `rc` | rx.UART speaks crsf |
| ok | device protocol | `can` | gnss.CAN speaks dronecan |
| ok | device protocol | `lidar` | lidar.MAIN speaks i2c |
| ok | device protocol | `video` | companion.ETH speaks ethernet |
| ok | device protocol | `motors` | esc.SIGNAL speaks dshot300 |
| ok | baud rate | `telemetry` | 57600 baud on both ends |
| ok | baud rate | `companion_dds` | 921600 baud on both ends |
| ok | baud rate | `gimbal` | 115200 baud on both ends |
| ok | baud rate | `rc` | 420000 baud on both ends |
| ok | receive DMA | `rc` | fc.GPS2 receives with DMA |
| ok | logic level | `telemetry` | 3.3 V to 3.3 V |
| ok | logic level | `companion_dds` | 3.3 V to 3.3 V |
| ok | logic level | `rc` | 3.3 V to 3.3 V |
| ok | logic level | `can` | CAN is differential; transceivers set the levels |
| ok | logic level | `lidar` | 3.3 V to 3.3 V |
| ok | pins | `telemetry` | 5 wires, TX and RX crossed |
| ok | pins | `companion_dds` | 5 wires, TX and RX crossed |
| ok | pins | `gimbal` | 3 wires, TX and RX crossed |
| ok | pins | `rc` | 4 wires, TX and RX crossed |
| ok | pins | `can` | 4 wires |
| ok | pins | `lidar` | 4 wires |
| ok | I2C address | `fc.I2C2` | 0x10 unique on fc.I2C2 (the internal compass at 0x0C is on a separate internal bus) |
| ok | CAN termination | `fc.CAN1` | terminated at the autopilot and at gnss.CAN |
| ok | CAN node ids | `fc.CAN1` | the autopilot runs DroneCAN dynamic node allocation; no fixed ids to clash |
| ok | CAN bus load | `fc.CAN1` | 3.0 percent at 1000 kbit/s (worst case stuffing) |
| ok | firmware features | `all` | every other protocol is in the stable build |
| ok | MAVLink channels | `fc` | SERIAL0 (USB_GH) is MAV1; SERIAL1 (TELEM1) is MAV2; SERIAL5 (ADSB_INTERNAL) is MAV3 |
| ok | bandwidth | `telemetry` | 2973 B/s of 5760 B/s (52 percent at 57600 8N1) |
| ok | bandwidth | `rc` | CRSF at 150 Hz uses 4500 B/s of 42000 B/s (11 percent) |
| ok | supply voltage | `can` | 4.9 to 5.5 V inside 4.75 to 5.5 V |
| ok | 5 V budget | `fc shared` | 0.41 A typical, 0.55 A peak of 1.0 A (1.5 A peak) |
| ok | radio power | `telemetry` | radio powered by bec5, ground shared |
| ok | supply voltage | `bec12_in` | bec12.VIN: 13.2 to 16.8 V inside 9 to 55 V |
| ok | supply voltage | `bec5_in` | bec5.VIN: 13.2 to 16.8 V inside 9 to 55 V |
| ok | supply voltage | `jetson_power` | companion.DC_IN: 11.09 to 12.12 V inside 9 to 20 V |
| ok | supply voltage | `camera_power` | camera.POWER: 13.2 to 16.8 V inside 11 to 25.2 V |
| ok | regulator load | `bec12.VOUT` | 2.25 A peak of 5 A continuous |
| ok | regulator load | `bec5.VOUT` | 1.00 A peak of 5 A continuous |
| ok | regulator load | `pm.FC_POWER` | 1.10 A for the Cube and its peripherals from the PM02 5.2 V / 3 A max |
| ok | DShot rate | `motors` | DShot300 on AUX1, AUX2, AUX3, AUX4 (FMU timers, no IOMCU firmware needed) |
| ok | thrust to weight | `airframe` | 2.61 at 1.99 kg (2.0 or more leaves control authority in wind) |
| ok | hover throttle | `airframe` | 60 percent throttle to hover |
| ok | battery C rating | `battery` | 85 A at full throttle of 250 A continuous |
| ok | flight time | `battery` | 11.0 min hover to 20 percent charge |
| ok | radio range | `elrs` | 12.3 km free space with a 10 dB fade margin |
| ok | radio range | `sik` | 93.1 km free space with a 10 dB fade margin |
| ok | band overlap | `wifi` | Jetson WiFi on 5 GHz keeps 2.4 GHz clear for ExpressLRS |
