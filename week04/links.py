"""Will it fit? Bandwidth budgets for every serial and radio link in the design.

ArduPilot does not stream MAVLink messages one by one: it groups them into
streams, and the ``SRn_*`` parameters set a rate in Hz per stream for the
port with ``SERIALn``. The table below is ArduCopter 4.7.1's grouping, read
from ``libraries/GCS_MAVLink/GCS_MAVLink_Parameters.cpp`` (messages that only
exist in simulation or need hardware this design lacks are marked).

From stream rates and the message sizes in the dialect this module predicts
the bytes per second on a link; ``week04.sitl`` then measures the same link
on a real ArduCopter build and the two are compared in the docs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mavlink import Dialect, frame_bytes
from .uart import UartConfig

# stream parameter suffix -> MAVLink messages (ArduCopter 4.7.1)
STREAMS: dict[str, list[str]] = {
    "RAW_SENS": ["RAW_IMU", "SCALED_IMU2", "SCALED_IMU3", "SCALED_PRESSURE", "SCALED_PRESSURE2", "SCALED_PRESSURE3"],
    "EXT_STAT": [
        "SYS_STATUS",
        "POWER_STATUS",
        "MCU_STATUS",
        "MEMINFO",
        "MISSION_CURRENT",
        "GPS_RAW_INT",
        "GPS_RTK",
        "GPS2_RAW",
        "GPS2_RTK",
        "NAV_CONTROLLER_OUTPUT",
        "FENCE_STATUS",
        "POSITION_TARGET_GLOBAL_INT",
    ],
    "POSITION": ["GLOBAL_POSITION_INT", "LOCAL_POSITION_NED"],
    "RAW_CTRL": [],
    "RC_CHAN": ["SERVO_OUTPUT_RAW", "RC_CHANNELS"],
    "EXTRA1": ["ATTITUDE", "SIMSTATE", "AHRS2", "RPM", "PID_TUNING", "ESC_TELEMETRY_1_TO_4"],
    "EXTRA2": ["VFR_HUD"],
    "EXTRA3": [
        "AHRS",
        "WIND",
        "DISTANCE_SENSOR",
        "SYSTEM_TIME",
        "TERRAIN_REPORT",
        "TERRAIN_REQUEST",
        "BATTERY_STATUS",
        "GIMBAL_DEVICE_ATTITUDE_STATUS",
        "OPTICAL_FLOW",
        "MAG_CAL_REPORT",
        "MAG_CAL_PROGRESS",
        "EKF_STATUS_REPORT",
        "VIBRATION",
    ],
    "PARAMS": ["PARAM_VALUE", "AVAILABLE_MODES"],
    "ADSB": ["ADSB_VEHICLE"],
}

# Sent only in some situations; excluded from the steady state budget.
CONDITIONAL = {
    "SIMSTATE": "simulation only",
    "GPS_RTK": "only when the GPS reports RTK baseline data",
    "GPS2_RAW": "only with a second GPS",
    "GPS2_RTK": "only with a second GPS",
    "MAG_CAL_REPORT": "only during compass calibration",
    "MAG_CAL_PROGRESS": "only during compass calibration",
    "TERRAIN_REQUEST": "only while terrain tiles are missing",
    "OPTICAL_FLOW": "no optical flow sensor in this design",
    "RPM": "no RPM sensor configured",
    "PID_TUNING": "only when GCS_PID_MASK is set",
    "PARAM_VALUE": "only while a parameter download is in progress",
    "AVAILABLE_MODES": "only on request",
    "ADSB_VEHICLE": "one per aircraft in range",
    "SCALED_IMU3": "only with a third IMU",
    "SCALED_PRESSURE3": "only with a third barometer",
    "ESC_TELEMETRY_1_TO_4": "only with ESC telemetry",
    "MCU_STATUS": "only on boards with MCU monitoring",
    "GIMBAL_DEVICE_ATTITUDE_STATUS": "only with a mount configured",
    "DISTANCE_SENSOR": "only with a rangefinder configured",
    "FENCE_STATUS": "only with the fence enabled",
    "WIND": "Plane only in practice; Copter sends it when wind estimation is on",
    "POSITION_TARGET_GLOBAL_INT": "only while a position target is active (GUIDED or AUTO)",
}

ALWAYS_1HZ = ["HEARTBEAT"]  # sent once a second regardless of stream rates


@dataclass
class LinkBudget:
    name: str
    config: UartConfig
    rows: list[tuple[str, str, float, int]]  # stream, message, Hz, bytes per frame
    extra: list[tuple[str, float]]  # (description, bytes per second)

    @property
    def bytes_per_second(self) -> float:
        return sum(hz * size for _, _, hz, size in self.rows) + sum(b for _, b in self.extra)

    @property
    def utilisation(self) -> float:
        return self.bytes_per_second / self.config.bytes_per_second

    def by_stream(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for stream, _, hz, size in self.rows:
            out[stream] = out.get(stream, 0.0) + hz * size
        return out


def mavlink_budget(
    name: str,
    config: UartConfig,
    rates: dict[str, float],
    dialect: Dialect | None = None,
    include: set[str] | None = None,
    signed: bool = False,
    extra: list[tuple[str, float]] | None = None,
) -> LinkBudget:
    """Predicted MAVLink 2 traffic for SRn_* ``rates`` (keys like "EXTRA1").

    ``include`` lists conditional messages that do apply to this vehicle.
    Sizes are untruncated frames, an upper bound: MAVLink 2 trims trailing zeros.
    """
    dialect = dialect or Dialect.default()
    include = include or set()
    rows = [("HEARTBEAT", "HEARTBEAT", 1.0, frame_bytes(dialect["HEARTBEAT"], signed=signed))]
    for stream, hz in rates.items():
        if hz <= 0:
            continue
        for msg in STREAMS[stream]:
            if msg in CONDITIONAL and msg not in include:
                continue
            if msg == "ESC_TELEMETRY_1_TO_4":
                size = frame_bytes(dialect["ESC_TELEMETRY_1_TO_4"], signed=signed)
            else:
                size = frame_bytes(dialect[msg], signed=signed)
            rows.append((stream, msg, hz, size))
    return LinkBudget(name, config, rows, extra or [])


def rtcm_msm4_bytes(satellites: int, signals_per_satellite: int) -> int:
    """Size of one RTCM 3 MSM4 message (RTCM 10403.3 field widths), transport framing included.

    MSM header 169 bits (message number through signal mask), a cell mask of
    one bit per satellite and signal, 18 bits per satellite (rough range) and
    48 bits per signal (fine pseudorange 15, fine phase range 22, lock time 4,
    half cycle 1, CNR 6). The frame adds 3 bytes of header and a 3 byte CRC-24Q.
    """
    cells = satellites * signals_per_satellite
    bits = 169 + cells + 18 * satellites + 48 * cells
    return 3 + (bits + 7) // 8 + 3


def rtcm_base_bytes_per_second(constellations: dict[str, tuple[int, int]] | None = None) -> float:
    """1 Hz MSM4 for each constellation, plus 1005 (base position, 25 bytes) every second and 1230 (GLONASS biases, 14 bytes).

    Default sky: GPS 10 satellites, GLONASS 7, Galileo 8 and BeiDou 9, two signals each (a dual band base like the ZED-F9P).
    """
    constellations = constellations or {"GPS": (10, 2), "GLONASS": (7, 2), "Galileo": (8, 2), "BeiDou": (9, 2)}
    return sum(rtcm_msm4_bytes(n, s) for n, s in constellations.values()) + 25 + 14


def rtcm_over_mavlink(bytes_per_second: float, dialect: Dialect | None = None) -> float:
    """RTCM reaches the drone inside GPS_RTCM_DATA messages carrying up to 180 bytes each."""
    import math

    dialect = dialect or Dialect.default()
    per_msg = frame_bytes(dialect["GPS_RTCM_DATA"])
    return math.ceil(bytes_per_second / 180.0) * per_msg


# ---------------------------------------------------------- radio range ----


def free_space_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    import math

    return 20 * math.log10(distance_m) + 20 * math.log10(frequency_hz) - 147.55


def max_range_m(
    tx_dbm: float, tx_gain_dbi: float, rx_gain_dbi: float, sensitivity_dbm: float, frequency_hz: float, fade_margin_db: float = 10.0
) -> float:
    """Free space range at which the received power drops to sensitivity plus a fade margin."""
    budget = tx_dbm + tx_gain_dbi + rx_gain_dbi - sensitivity_dbm - fade_margin_db
    import math

    return 10 ** ((budget + 147.55 - 20 * math.log10(frequency_hz)) / 20)


def applicable_messages(design) -> set[str]:
    """Conditional messages this vehicle does send, from what the design contains."""
    roles = {inst.part["role"] for inst in design.parts.values()}
    out = {"WIND", "FENCE_STATUS"}  # Copter estimates wind; the parameter file enables the fence
    if "rangefinder" in roles:
        out.add("DISTANCE_SENSOR")
    if "camera" in roles:
        out.add("GIMBAL_DEVICE_ATTITUDE_STATUS")
    if any(p["id"] == "ADSB_INTERNAL" for p in design.fc.part["ports"]):
        out.add("ADSB_VEHICLE")
    return out


def design_budget(design, link) -> LinkBudget:
    """The planned traffic on one MAVLink link of the design, RTK corrections included on the telemetry radio."""
    config = UartConfig(link.spec["baud"]) if link.spec.get("baud") else UartConfig(12_000_000)
    extra = []
    if link.id == "telemetry":
        extra.append(("RTCM corrections (uplink)", rtcm_over_mavlink(rtcm_base_bytes_per_second())))
    return mavlink_budget(link.id, config, link.spec["streams"], include=applicable_messages(design), extra=extra)
