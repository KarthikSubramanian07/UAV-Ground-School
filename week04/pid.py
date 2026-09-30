"""Pitch axis PID: the Isabelle, Preeti and Darren slides as a simulator you can tune.

The plant is one axis of a quadcopter::

    I * theta'' = tau_motor - c * theta' + tau_disturbance
    tau_motor'  = (clip(u, -tau_max, tau_max) - tau_motor) / motor_lag

``I`` is the pitch moment of inertia, ``motor_lag`` the time a propeller
needs to change speed, and ``tau_disturbance`` a constant torque (an off
centre battery, or wind) that only the integral term can cancel.

Two controllers:

* ``single``: the textbook loop from the lecture, one PID on pitch angle.
  The derivative acts on the measurement (so a step in the target does not
  kick the motors) through a first order low pass filter, and the integrator
  is clamped (ArduPilot calls the limit IMAX).
* ``cascade``: what ArduPilot actually flies. An outer P loop turns angle
  error into a target rate (``ATC_ANG_PIT_P``), and an inner PID on the gyro
  rate (``ATC_RAT_PIT_P``, ``_I``, ``_D``) turns rate error into torque.

The controller runs at 400 Hz (ArduCopter's main loop) and the physics at
four substeps per control step. ``site/js/pid.js`` is a line by line port;
the tests run both and require identical traces.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class Plant:
    inertia: float = 0.021  # kg m^2, pitch axis of the design (see week04.design)
    damping: float = 0.004  # N m s / rad, aerodynamic
    tau_max: float = 1.9  # N m, most differential torque the motors can make
    motor_lag: float = 0.035  # s
    disturbance: float = 0.05  # N m: a 2 kg drone with its battery 2.5 mm off centre


@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    imax: float = 1.0  # N m, integrator clamp
    d_cutoff_hz: float = 20.0
    mode: str = "single"  # or "cascade"
    angle_p: float = 4.5  # cascade only: ATC_ANG_PIT_P
    max_rate: float = math.radians(200.0)  # cascade only


@dataclass(frozen=True)
class Scenario:
    target: float = math.radians(10.0)  # step in target pitch
    duration: float = 3.0
    rate_hz: int = 400
    substeps: int = 4
    disturbance_at: float = 1.5  # s; the disturbance torque switches on here
    initial: float = 0.0
    gyro_noise: float = 0.0  # rad/s, amplitude of the (triangular) gyro noise
    angle_noise: float = 0.0  # rad
    seed: int = 2463534242


def _xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state


class _Noise:
    """Deterministic triangular noise in [-1, 1], identical in the JavaScript port."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def __call__(self) -> float:
        self.state = _xorshift32(self.state)
        a = self.state / 4294967296.0
        self.state = _xorshift32(self.state)
        b = self.state / 4294967296.0
        return a + b - 1.0


DEFAULT_PLANT = Plant()
DEFAULT_SCENARIO = Scenario()


@dataclass
class Trace:
    t: list[float] = field(default_factory=list)
    angle: list[float] = field(default_factory=list)
    rate: list[float] = field(default_factory=list)
    torque: list[float] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    i: list[float] = field(default_factory=list)
    d: list[float] = field(default_factory=list)
    command: list[float] = field(default_factory=list)  # clipped controller output, N m


def simulate(gains: Gains, plant: Plant = DEFAULT_PLANT, scenario: Scenario = DEFAULT_SCENARIO) -> Trace:
    dt = 1.0 / scenario.rate_hz
    h = dt / scenario.substeps
    alpha = dt / (dt + 1.0 / (2.0 * math.pi * gains.d_cutoff_hz))
    theta, omega, tau = scenario.initial, 0.0, 0.0
    integ, d_filt, prev_gyro = 0.0, 0.0, 0.0
    noise = _Noise(scenario.seed)
    trace = Trace()
    steps = round(scenario.duration * scenario.rate_hz)
    for k in range(steps + 1):
        t = k * dt
        gyro = omega + scenario.gyro_noise * noise()
        angle = theta + scenario.angle_noise * noise()
        if gains.mode == "cascade":
            rate_target = max(-gains.max_rate, min(gains.max_rate, gains.angle_p * (scenario.target - angle)))
            err = rate_target - gyro
            raw_d = -(gyro - prev_gyro) / dt if k > 0 else 0.0  # D acts on angular acceleration
        else:
            err = scenario.target - angle
            raw_d = -gyro  # the derivative of the pitch angle is the gyro rate: no need to differentiate
        prev_gyro = gyro
        p_term = gains.kp * err
        integ = max(-gains.imax, min(gains.imax, integ + gains.ki * err * dt))
        d_filt += alpha * (raw_d - d_filt)
        d_term = gains.kd * d_filt
        u = max(-plant.tau_max, min(plant.tau_max, p_term + integ + d_term))

        trace.t.append(t)
        trace.angle.append(theta)
        trace.rate.append(omega)
        trace.torque.append(tau)
        trace.p.append(p_term)
        trace.i.append(integ)
        trace.d.append(d_term)
        trace.command.append(u)

        dist = plant.disturbance if t >= scenario.disturbance_at else 0.0
        for _ in range(scenario.substeps):
            tau += (u - tau) * h / plant.motor_lag
            omega += (tau - plant.damping * omega + dist) / plant.inertia * h
            theta += omega * h
    return trace


# ------------------------------------------------------------ metrics ----


@dataclass
class Metrics:
    rise_time: float | None  # 10 to 90 percent
    overshoot: float  # fraction of the step
    settling_time: float | None  # within 2 percent of the step and stays there
    steady_state_error: float  # rad, mean over the last 0.25 s before the disturbance
    disturbance_error: float  # rad, mean over the last 0.25 s of the run
    peak_torque: float
    itae: float
    chatter: float  # mean step to step change of the torque command, N m

    def as_dict(self) -> dict:
        return asdict(self)


def metrics(trace: Trace, scenario: Scenario = DEFAULT_SCENARIO) -> Metrics:
    step = scenario.target - scenario.initial
    t, y = trace.t, trace.angle
    before = [i for i, ti in enumerate(t) if ti < scenario.disturbance_at]
    rise = None
    t10 = next((t[i] for i in before if (y[i] - scenario.initial) / step >= 0.1), None)
    t90 = next((t[i] for i in before if (y[i] - scenario.initial) / step >= 0.9), None)
    if t10 is not None and t90 is not None:
        rise = t90 - t10
    peak = max(((y[i] - scenario.initial) / step for i in before), default=0.0)
    band = 0.02 * abs(step)
    settle = None
    for i in reversed(before):
        if abs(y[i] - scenario.target) > band:
            settle = t[i + 1] if i + 1 < len(before) else None
            break
    else:
        settle = 0.0
    window = max(1, round(0.25 * scenario.rate_hz))
    sse = sum(scenario.target - y[i] for i in before[-window:]) / window
    dse = sum(scenario.target - v for v in y[-window:]) / window
    itae = sum(ti * abs(scenario.target - yi) for ti, yi in zip(t, y)) / scenario.rate_hz
    u = trace.command
    chatter = sum(abs(a - b) for a, b in zip(u[1:], u[:-1])) / max(1, len(u) - 1)
    return Metrics(rise, max(0.0, peak - 1.0), settle, sse, dse, max(abs(v) for v in trace.torque), itae, chatter)


# ------------------------------------------------------------ analysis ----


def damping_ratio(gains: Gains, plant: Plant = DEFAULT_PLANT) -> float:
    """Damping ratio of the dominant (slowest) oscillatory closed loop mode of the linearised single loop.

    State: angle, rate, motor torque, integral of error, filtered derivative.
    Returns 1.0 or more when no pole pair oscillates (critically or over damped).
    """
    import numpy as np

    wc = 2 * math.pi * gains.d_cutoff_hz
    inertia, c, lag = plant.inertia, plant.damping, plant.motor_lag
    # d_filt' = wc * (-theta' - d_filt) and u = kp (r - theta) + ki z + kd d_filt, with r = 0
    a = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, -c / inertia, 1 / inertia, 0, 0],
            [-gains.kp / lag, 0, -1 / lag, gains.ki / lag, gains.kd / lag],
            [-1, 0, 0, 0, 0],
            [0, -wc, 0, 0, -wc],
        ],
        dtype=float,
    )
    poles = np.linalg.eigvals(a)
    if np.any(poles.real > 1e-9):
        return -1.0  # unstable
    oscillatory = [p for p in poles if abs(p.imag) > 1e-6]
    if not oscillatory:
        return 1.0
    p = min(oscillatory, key=abs)
    return float(-p.real / abs(p))


def ultimate_gain(plant: Plant = DEFAULT_PLANT, d_cutoff_hz: float = 20.0) -> tuple[float, float]:
    """Ziegler Nichols: the P only gain at which the loop oscillates forever, and that period.

    With motor lag the P only loop is unstable for any gain, so this uses a small
    fixed amount of damping (kd = 0.02) the way a pilot would start tuning.
    """
    import numpy as np

    kd = 0.02
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        zeta = damping_ratio(Gains(mid, 0.0, kd, d_cutoff_hz=d_cutoff_hz), plant)
        if zeta < 0:
            hi = mid
        else:
            lo = mid
    ku = lo
    wc = 2 * math.pi * d_cutoff_hz
    inertia, c, lag = plant.inertia, plant.damping, plant.motor_lag
    a = np.array(
        [
            [0, 1, 0, 0],
            [0, -c / inertia, 1 / inertia, 0],
            [-ku / lag, 0, -1 / lag, kd / lag],
            [0, -wc, 0, -wc],
        ]
    )
    poles = np.linalg.eigvals(a)
    w = max(abs(p.imag) for p in poles)
    return ku, 2 * math.pi / w


def ziegler_nichols(plant: Plant = DEFAULT_PLANT) -> Gains:
    ku, tu = ultimate_gain(plant)
    return Gains(kp=0.6 * ku, ki=1.2 * ku / tu, kd=0.075 * ku * tu)


NOISY = Scenario(gyro_noise=0.03, angle_noise=0.002)


def cost(gains: Gains, plant: Plant, scenario: Scenario) -> float:
    """ITAE, plus penalties for overshoot, motor chatter (noise amplified by D) and a poorly damped mode."""
    trace = simulate(gains, plant, scenario)
    if any(math.isnan(v) or abs(v) > 10 for v in trace.angle):
        return 1e9
    if gains.mode == "single" and damping_ratio(gains, plant) < 0:
        return 1e9
    m = metrics(trace, scenario)
    return m.itae + 2.0 * max(0.0, m.overshoot - 0.05) + 5.0 * m.chatter / plant.tau_max


def tune(
    plant: Plant = DEFAULT_PLANT,
    scenario: Scenario | None = None,
    start: Gains | None = None,
    iterations: int = 150,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> Gains:
    """Nelder Mead on log gains. The step is followed by a disturbance, so the integral term earns its keep.

    ``bounds`` clamps each gain (for example to the ranges the firmware accepts).
    """
    scenario = scenario or NOISY
    start = start or Gains(0.8, 0.8, 0.12)
    plant_d = plant
    names = ("kp", "ki", "kd")
    limits = [(math.log(bounds[n][0]), math.log(bounds[n][1])) if bounds and n in bounds else (-20.0, 20.0) for n in names]

    def clamp(x):
        return [min(max(v, lo), hi) for v, (lo, hi) in zip(x, limits)]

    def gains_of(x):
        x = clamp(x)
        return replace(start, kp=math.exp(x[0]), ki=math.exp(x[1]), kd=math.exp(x[2]))

    def f(x):
        return cost(gains_of(x), plant_d, scenario)

    simplex = [[math.log(start.kp), math.log(start.ki), math.log(start.kd)]]
    for i in range(3):
        v = list(simplex[0])
        v[i] += 0.7
        simplex.append(v)
    values = [f(v) for v in simplex]
    for _ in range(iterations):
        order = sorted(range(4), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = [sum(v[j] for v in simplex[:3]) / 3 for j in range(3)]
        worst = simplex[3]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(reflected)
        if fr < values[0]:
            expanded = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(expanded)
            simplex[3], values[3] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[2]:
            simplex[3], values[3] = reflected, fr
        else:
            contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(contracted)
            if fc < values[3]:
                simplex[3], values[3] = contracted, fc
            else:
                best = simplex[0]
                simplex = [best] + [[b + 0.5 * (v - b) for b, v in zip(best, s)] for s in simplex[1:]]
                values = [values[0]] + [f(v) for v in simplex[1:]]
    return gains_of(simplex[min(range(4), key=lambda i: values[i])])


# ArduCopter 4.7.1 parameter ranges for the rate loop (ATC_RAT_PIT_P, _I, _D), in the firmware's units:
# output as a fraction of full differential thrust per rad/s of rate error.
ARDUPILOT_RATE_RANGES = {"kp": (0.01, 0.35), "ki": (0.01, 0.6), "kd": (0.0005, 0.03)}
ARDUPILOT_DEFAULTS = {"kp": 0.135, "ki": 0.135, "kd": 0.0036}


def tune_cascade(plant: Plant = DEFAULT_PLANT, angle_p: float = 4.5) -> Gains:
    """Tune the inner rate loop of ArduPilot's cascade with the outer angle P fixed (ATC_ANG_PIT_P default 4.5).

    Starts from ArduCopter's default rate gains and stays inside the ranges the
    firmware accepts, both converted to torque by multiplying with tau_max.
    """
    t = plant.tau_max
    start = Gains(ARDUPILOT_DEFAULTS["kp"] * t, ARDUPILOT_DEFAULTS["ki"] * t, ARDUPILOT_DEFAULTS["kd"] * t, mode="cascade", angle_p=angle_p)
    bounds = {k: (lo * t, hi * t) for k, (lo, hi) in ARDUPILOT_RATE_RANGES.items()}
    return tune(plant, NOISY, start, bounds=bounds)


def critical_kd(kp: float, plant: Plant = DEFAULT_PLANT, scenario: Scenario | None = None) -> float:
    """Smallest D gain for which a PD loop reaches the step without overshoot: critical damping, found by bisection."""
    scenario = scenario or Scenario(disturbance_at=10.0, duration=2.0)
    plant = replace(plant, disturbance=0.0)
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if metrics(simulate(Gains(kp, 0.0, mid), plant, scenario), scenario).overshoot > 1e-4:
            lo = mid
        else:
            hi = mid
    return hi


# The characters from the slides, on the default plant (kd is critical_kd(1.5), rounded).
PRESETS = {
    "preeti": Gains(kp=1.5),  # P only: every correction overshoots, and motor lag makes it grow
    "preeti_darren": Gains(kp=1.5, kd=0.274),  # PD, critically damped: no overshoot, but the disturbance leaves an offset
    "isabelle_too_much": Gains(kp=1.5, ki=10.0, kd=0.274),  # too much I: winds up and rings
    "all_three": Gains(kp=1.5, ki=2.0, kd=0.274),  # PID: the offset is gone
}
