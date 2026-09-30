"""Mass, thrust, hover power and flight time, from the parts list and first principles.

Propeller (static, from the propeller's thrust and power coefficients)::

    T = CT * rho * n^2 * D^4        Q = CP * rho * n^2 * D^5 / (2 pi)

Motor (the standard DC motor model: back EMF, winding resistance, no load
current)::

    Kt = 60 / (2 pi KV)            I = Q / Kt + I0            V = I R + omega / (2 pi KV / 60)

For a throttle d the ESC applies d * V_battery. Solving the two equations
for the speed where motor torque equals propeller torque gives thrust and
current at every throttle; hover is the throttle where total thrust equals
weight. The propeller's power coefficient and the lumped resistance are not
published anywhere, so :func:`calibrate` fits them to the two numbers
Holybro does publish for the stock X500 V2 kit: about 18 minutes of hover on
a 5000 mAh pack, and about 1 kg of payload at 70 percent throttle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .design import Design

G = 9.80665
RHO = 1.225  # kg/m^3, sea level standard


@dataclass
class Propulsion:
    kv: float
    resistance: float
    no_load_a: float
    diameter_m: float
    ct: float
    cp: float
    esc_efficiency: float = 0.93

    def operating_point(self, volts: float) -> tuple[float, float, float]:
        """(rev/s, thrust N, motor current A) with ``volts`` applied to the motor."""
        kt = 60.0 / (2 * math.pi * self.kv)
        lo, hi = 0.0, volts * self.kv / 60.0  # rev/s; the top is the no load speed
        for _ in range(60):
            n = (lo + hi) / 2
            q = self.cp * RHO * n * n * self.diameter_m**5 / (2 * math.pi)
            current = q / kt + self.no_load_a
            back_emf = n * 60.0 / self.kv
            if current * self.resistance + back_emf > volts:
                hi = n
            else:
                lo = n
        n = lo
        q = self.cp * RHO * n * n * self.diameter_m**5 / (2 * math.pi)
        return n, self.ct * RHO * n * n * self.diameter_m**4, q / kt + self.no_load_a

    def at_throttle(self, throttle: float, battery_v: float) -> tuple[float, float]:
        """(thrust N, battery current A) for one motor."""
        _, thrust, motor_a = self.operating_point(throttle * battery_v)
        return thrust, motor_a * throttle / self.esc_efficiency

    def hover(self, weight_n: float, motors: int, battery_v: float) -> tuple[float, float]:
        """(throttle, battery current for all motors) at hover."""
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.at_throttle(mid, battery_v)[0] * motors < weight_n:
                lo = mid
            else:
                hi = mid
        return hi, motors * self.at_throttle(hi, battery_v)[1]


@dataclass
class Performance:
    mass_items: list[tuple[str, float, bool]]  # name, grams, estimated
    mass_kg: float
    max_thrust_n: float
    thrust_to_weight: float
    hover_throttle: float
    hover_current_a: float
    avionics_w: float
    full_throttle_current_a: float
    motor_full_throttle_a: float
    hover_minutes: float
    reference_minutes: float  # the stock X500 V2 kit through the same model
    inertia: tuple[float, float, float]  # kg m^2 about x, y, z

    def as_dict(self) -> dict:
        return {
            "mass_kg": round(self.mass_kg, 3),
            "max_thrust_n": round(self.max_thrust_n, 1),
            "thrust_to_weight": round(self.thrust_to_weight, 2),
            "hover_throttle": round(self.hover_throttle, 3),
            "hover_current_a": round(self.hover_current_a, 2),
            "avionics_w": round(self.avionics_w, 1),
            "full_throttle_current_a": round(self.full_throttle_current_a, 1),
            "motor_full_throttle_a": round(self.motor_full_throttle_a, 1),
            "hover_minutes": round(self.hover_minutes, 1),
            "reference_minutes": round(self.reference_minutes, 1),
            "inertia_kg_m2": [round(v, 4) for v in self.inertia],
            "mass_items": [{"item": n, "grams": g, "estimated": e} for n, g, e in self.mass_items],
        }


def propulsion(design: Design) -> Propulsion:
    motor = design.parts["motor"].part["specs"]
    prop = design.parts["prop"].part["specs"]
    esc = design.parts["esc"].part["specs"]
    return Propulsion(
        motor["kv"],
        motor["resistance_ohm"],
        motor["no_load_a"],
        prop["diameter_in"] * 0.0254,
        prop["ct"],
        prop["cp"],
        esc.get("efficiency", 0.93),
    )


# What Holybro publishes for the stock kit (docs.holybro.com, PX4 Development Kit X500 v2):
# about 18 minutes of hover with a 5000 mAh pack and no payload, and about 1 kg of payload at 70 percent throttle.
HOLYBRO_HOVER_MINUTES = 18.0
HOLYBRO_PAYLOAD_KG_AT_70 = 1.0
REFERENCE_AVIONICS_G = {"Pixhawk 6C": 60, "M8N GPS": 32, "SiK radio": 30, "PM02": 20}


def reference_mass_kg(design: Design) -> float:
    kit = sum(design.parts[k].part["mass_g"] * design.parts[k].count for k in ("frame", "motor", "esc", "prop", "battery"))
    return (kit + sum(REFERENCE_AVIONICS_G.values())) / 1000.0


def _hover_minutes(prop: Propulsion, mass_kg: float, capacity_ah: float, volts: float, avionics_w: float, motors: int = 4) -> float:
    _, amps = prop.hover(mass_kg * G, motors, volts)
    return 60 * capacity_ah * 0.8 / (amps + avionics_w / volts)


def calibrate(design: Design) -> tuple[Propulsion, dict]:
    """Fit the two least known numbers, propeller power coefficient and lumped resistance, to Holybro's two claims.

    The lumped resistance stands for winding, ESC, wiring and battery internal
    resistance together, which is why it comes out several times the motor's own.
    """
    from dataclasses import replace

    start = propulsion(design)
    battery = design.parts["battery"].part["specs"]
    volts, capacity = battery["nominal_v"], battery["capacity_mah"] / 1000.0
    ref = reference_mass_kg(design)
    target_thrust = (ref + HOLYBRO_PAYLOAD_KG_AT_70) * G / 4

    def error(cp: float, r: float) -> float:
        prop = replace(start, cp=cp, resistance=r)
        thrust = prop.at_throttle(0.7, volts)[0]
        minutes = _hover_minutes(prop, ref, capacity, volts, 3.0)
        return ((thrust - target_thrust) / target_thrust) ** 2 + ((minutes - HOLYBRO_HOVER_MINUTES) / HOLYBRO_HOVER_MINUTES) ** 2

    cp, r, step_cp, step_r = start.cp, start.resistance, 0.01, 0.2
    best = error(cp, r)
    for _ in range(80):
        improved = False
        for dcp, dr in ((step_cp, 0), (-step_cp, 0), (0, step_r), (0, -step_r)):
            c, rr = cp + dcp, r + dr
            if 0.02 <= c <= 0.08 and 0.02 <= rr <= 1.0:
                e = error(c, rr)
                if e < best:
                    cp, r, best, improved = c, rr, e, True
        if not improved:
            step_cp, step_r = step_cp / 2, step_r / 2
    fitted = replace(start, cp=cp, resistance=r)
    report = {
        "cp_estimate": start.cp,
        "cp_fitted": round(cp, 4),
        "resistance_estimate_ohm": start.resistance,
        "resistance_fitted_ohm": round(r, 3),
        "fit_error": best,
        "reference_mass_kg": round(ref, 3),
        "reference_minutes": round(_hover_minutes(fitted, ref, capacity, volts, 3.0), 1),
        "thrust_at_70_n": round(fitted.at_throttle(0.7, volts)[0], 2),
        "target_thrust_at_70_n": round(target_thrust, 2),
    }
    return fitted, report


def mass_budget(design: Design) -> list[tuple[str, float, bool]]:
    items = []
    for inst in design.parts.values():
        grams = inst.part.get("mass_g")
        if grams is None:
            raise ValueError(f"{inst.name} ({inst.part['id']}) has no mass, not even an estimate")
        estimated = "mass_g" in inst.part.get("estimates", {})
        label = f"{inst.count} x {inst.label}" if inst.count > 1 else inst.label
        items.append((label, grams * inst.count, estimated))
    for name, grams in design.spec.get("payload_mass_g", {}).items():
        items.append((name, grams, True))
    return items


def avionics_power(design: Design) -> float:
    """Watts drawn from the battery by everything except the motors, at typical load."""
    watts = 0.0
    for inst in design.parts.values():
        supply = inst.part.get("supply") or {}
        role = inst.part["role"]
        if role in ("motor", "esc", "battery", "regulator", "power_module", "frame", "propeller"):
            continue
        if supply.get("typical_w"):
            watts += supply["typical_w"]
        elif supply.get("typical_a"):
            watts += 5.0 * supply["typical_a"]
        elif role == "flight_controller":
            watts += 5.0 * design.fc.part["specs"].get("fmu_plus_io_power_budget_a", 0.55)
    return watts / 0.9  # regulators


def inertia(design: Design, mass_kg: float) -> tuple[float, float, float]:
    """Motors (with props and ESCs) as point masses on the arms, the rest as a 16 cm by 16 cm by 8 cm box."""
    arm = design.parts["frame"].part["specs"].get("arm_length_m", 0.25)
    per_corner = sum(design.parts[k].part["mass_g"] for k in ("motor", "esc", "prop")) / 1000.0
    rotor_mass = 4 * per_corner
    body = mass_kg - rotor_mass
    a, b, c = 0.16, 0.16, 0.08
    ixx_body = body * (b * b + c * c) / 12
    iyy_body = body * (a * a + c * c) / 12
    izz_body = body * (a * a + b * b) / 12
    # X frame: each motor sits arm/sqrt(2) from both the roll and pitch axes
    d = arm / math.sqrt(2)
    return ixx_body + rotor_mass * d * d, iyy_body + rotor_mass * d * d, izz_body + rotor_mass * arm * arm


def analyse(design: Design) -> Performance:
    items = mass_budget(design)
    mass = sum(g for _, g, _ in items) / 1000.0
    battery = design.parts["battery"].part
    cells = battery["specs"]["cells"]
    nominal = battery["specs"]["nominal_v"]
    capacity_ah = battery["specs"]["capacity_mah"] / 1000.0
    prop, _ = calibrate(design)
    motors = design.parts["motor"].count
    thrust_max, current_max = prop.at_throttle(1.0, cells * 4.2)
    throttle, hover_a = prop.hover(mass * G, motors, nominal)
    avionics = avionics_power(design)
    usable = 0.8  # never fly a LiPo below 20 percent
    total_a = hover_a + avionics / nominal
    minutes = 60 * capacity_ah * usable / total_a

    ref_minutes = _hover_minutes(prop, reference_mass_kg(design), capacity_ah, nominal, 3.0, motors)

    return Performance(
        items,
        mass,
        thrust_max * motors,
        thrust_max * motors / (mass * G),
        throttle,
        hover_a,
        avionics,
        current_max * motors + avionics / nominal,
        current_max * prop.esc_efficiency,
        minutes,
        ref_minutes,
        inertia(design, mass),
    )


def sitl_frame(design: Design, perf: Performance) -> dict:
    """An ArduPilot SITL frame model (``--model quad:frame.json``) for this drone."""
    battery = design.parts["battery"].part["specs"]
    prop_d = design.parts["prop"].part["specs"]["diameter_in"] * 0.0254
    motors = design.parts["motor"].count
    return {
        "mass": round(perf.mass_kg, 3),
        "diagonal_size": 2 * design.parts["frame"].part["specs"].get("arm_length_m", 0.25),
        "refVoltage": battery["nominal_v"],
        "maxVoltage": battery["cells"] * 4.2,
        "battCapacityAh": battery["capacity_mah"] / 1000.0,
        "hoverThrOut": round(perf.hover_throttle, 3),
        "disc_area": round(motors * math.pi * (prop_d / 2) ** 2, 4),
        "moment_inertia": [round(v, 5) for v in perf.inertia],
        "num_motors": motors,
    }
