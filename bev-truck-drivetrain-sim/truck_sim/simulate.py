"""Backward-facing quasi-static simulation (the approach used by VECTO).

Speed and grade are prescribed by the cycle; at each time step the model works
backwards from the force needed at the wheels to the power drawn from the battery.
"""
from dataclasses import dataclass
import numpy as np

from .vehicle import Vehicle
from .components import Battery


@dataclass
class SimResult:
    topology: str
    cycle: str
    distance_km: float
    energy_kwh: dict          # energy breakdown
    soc_end: float
    t: np.ndarray
    p_battery_w: np.ndarray
    op_points: list           # (machine, torque, speed, dt)
    modes: list
    max_shortfall_w: float

    @property
    def kwh_per_km(self):
        return self.energy_kwh["battery_internal"] / self.distance_km


def simulate(topology, cycle, vehicle: Vehicle = None, battery: Battery = None, dt=1.0):
    vehicle = vehicle or Vehicle()
    battery = battery or Battery()
    soc = battery.soc_init
    J = 1.0 / 3.6e6  # W*s -> kWh
    e = {k: 0.0 for k in ["wheel_traction", "wheel_braking", "aero", "rolling", "grade_net",
                          "motor", "inverter", "transmission", "friction_brake",
                          "battery_loss", "aux", "battery_internal"]}
    p_bat, ops, modes, shortfall = [], [], [], 0.0

    for v, a, g in zip(cycle.v, cycle.a, cycle.grade):
        f, parts = vehicle.tractive_force(v, a, g)
        w_wheel = v / vehicle.wheel_radius_m
        t_wheel = f * vehicle.wheel_radius_m
        p_wheel = f * v
        e["aero"] += parts["aero"] * v * dt * J
        e["rolling"] += parts["rolling"] * v * dt * J
        e["grade_net"] += parts["grade"] * v * dt * J
        if p_wheel >= 0:
            e["wheel_traction"] += p_wheel * dt * J
        else:
            e["wheel_braking"] += -p_wheel * dt * J

        if v < 0.05:                       # standstill: drivetrain off
            p_dc, mode = 0.0, "standstill"
        else:
            r = topology.step(t_wheel, w_wheel)
            if r is None:
                raise RuntimeError("No feasible operating point at v=%.1f m/s" % v)
            # respect the battery charge-power limit: excess braking goes to friction brakes
            p_dc = r.p_dc_w
            if p_dc < -battery.p_charge_max_w:
                r.friction_brake_w += -battery.p_charge_max_w - p_dc
                p_dc = -battery.p_charge_max_w
            e["motor"] += r.loss_motor_w * dt * J
            e["inverter"] += r.loss_inverter_w * dt * J
            e["transmission"] += r.loss_transmission_w * dt * J
            e["friction_brake"] += r.friction_brake_w * dt * J
            shortfall = max(shortfall, r.shortfall_w)
            ops += [(m, tq, w, dt) for (m, tq, w) in r.op_points]
            mode = r.mode

        p_term = p_dc + vehicle.aux_power_w
        soc, _, loss, p_int = battery.step(p_term, soc, dt)
        e["battery_loss"] += loss * dt * J
        e["aux"] += vehicle.aux_power_w * dt * J
        e["battery_internal"] += p_int * dt * J
        p_bat.append(p_term)
        modes.append(mode)

    return SimResult(topology.name, cycle.name, cycle.distance_km, e, soc, cycle.t,
                     np.array(p_bat), ops, modes, shortfall)
