### `telemetry`: fc.TELEM1 to radio.HEADER

JST-GH-6 to 2x8-0.1in-header-16 (custom cable)

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | TX | 7 | RX |  |
| 3 | RX | 9 | TX |  |
| 5 | RTS | 3 | CTS |  |
| 4 | CTS | 11 | RTS |  |
| 6 | GND | 1 | GND |  |

Leave the 5V pins unconnected: radio is powered by bec5, sharing ground only.

### `companion_dds`: fc.TELEM2 to companion.HEADER_UART1

JST-GH-6 to 40-pin-2.54mm-header (pins 8,10,11,36,6) (custom cable)

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | TX | 10 | UART1_RXD (pin 10) |  |
| 3 | RX | 8 | UART1_TXD (pin 8) |  |
| 5 | RTS | 36 | UART1_CTS* (pin 36) |  |
| 4 | CTS | 11 | UART1_RTS* (pin 11) |  |
| 6 | GND | 6 | GND (pin 6) |  |

### `gimbal`: fc.GPS1 to camera.UART

JST-GH-8 to vendor cable (custom cable)

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | TX | 2 | RX |  |
| 3 | RX | 1 | TX |  |
| 8 | GND | 3 | GND |  |

### `rc`: fc.GPS2 to rx.UART

JST-GH-6 to solder-pads (custom cable)

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | TX | 4 | RX |  |
| 3 | RX | 3 | TX |  |
| 6 | GND | 2 | GND |  |
| 1 | 5V | 1 | 5V | device powered by the flight controller |

### `can`: fc.CAN1 to gnss.CAN

JST-GH-4 to JST-GH-4

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | CAN_H | 2 | CAN_H |  |
| 3 | CAN_L | 3 | CAN_L |  |
| 4 | GND | 4 | GND |  |
| 1 | 5V | 1 | 5V | device powered by the flight controller |

### `lidar`: fc.I2C2 to lidar.MAIN

JST-GH-4 to GH1.25-4P on sensor; Molex 51021-0400 (1.25 mm) on the 10 cm cable end (custom cable)

| From pin | Signal | To pin | Signal | Note |
| ---: | --- | ---: | --- | --- |
| 2 | SCL | 4 | TXD/SCL |  |
| 3 | SDA | 3 | RXD/SDA |  |
| 4 | GND | 1 | GND |  |
| 1 | 5V | 2 | 5V | device powered by the flight controller |
