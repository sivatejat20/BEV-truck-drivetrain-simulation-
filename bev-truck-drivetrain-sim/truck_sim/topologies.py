"""Drivetrain topologies compared in the study.

Each topology answers one question per time step: given the torque and speed the
wheels need, how much DC power must the battery deliver (or can it absorb), and
where are the losses? Operating strategies pick, at every step, the option with
the lowest electrical power (gear choice or torque split).
"""
from dataclasses import dataclass, field
import numpy as np

from .components import EMachine, Inverter, Transmission


@dataclass
class StepResult:
    p_dc_w: float                 # power at the DC link (+ = drawn from battery)
    loss_motor_w: float = 0.0
    loss_inverter_w: float = 0.0
    loss_transmission_w: float = 0.0
    friction_brake_w: float = 0.0  # mechanical braking power dissipated (>= 0)
    shortfall_w: float = 0.0       # traction power that could not be delivered
    op_points: list = field(default_factory=list)  # (machine name, torque Nm, speed rad/s)
    mode: str = ""


def _drive_path(machine: EMachine, inv: Inverter, trans: Transmission, t_wheel, w_wheel):
    """Evaluate one machine + inverter + gear path for a wheel torque/speed share.
    Regenerative torque beyond the machine limit goes to the friction brake."""
    t_m, w_m = trans.motor_side(t_wheel, w_wheel)
    if w_m > machine.w_max_rads:
        return None
    t_lim = machine.torque_limit(w_m)
    friction = 0.0
    shortfall = 0.0
    if t_m < -t_lim:                       # more braking than the machine can absorb
        friction = (-t_lim - t_m) * w_m / trans.efficiency
        t_m = -t_lim
    elif t_m > t_lim:                      # more traction than available
        shortfall = (t_m - t_lim) * w_m
        t_m = t_lim
    p_mech = t_m * w_m
    loss_m = machine.losses(t_m, w_m)
    p_ac = p_mech + loss_m
    loss_inv = inv.losses(p_ac)
    p_dc = p_ac + loss_inv
    # mesh losses: motoring -> share of motor output; recuperating -> share of wheel input
    if p_mech >= 0:
        loss_tr = p_mech * (1 - trans.efficiency)
    else:
        loss_tr = -p_mech * (1 / trans.efficiency - 1)
    return StepResult(p_dc_w=p_dc, loss_motor_w=loss_m, loss_inverter_w=loss_inv,
                      loss_transmission_w=loss_tr, friction_brake_w=friction,
                      shortfall_w=shortfall, op_points=[(machine.name, t_m, w_m)])


class CentralDrive:
    """One central e-machine + multi-speed gearbox + rear-axle differential.
    The gear with the lowest DC power is chosen at each step (ideal shift strategy)."""

    def __init__(self, machine: EMachine, gear_ratios, final_drive, eta_gear=0.975,
                 eta_final=0.97, inverter=None):
        self.name = "Central drive (1 motor, %d-speed)" % len(gear_ratios)
        self.machine = machine
        self.inverter = inverter or Inverter()
        self.paths = [Transmission(g * final_drive, eta_gear * eta_final) for g in gear_ratios]

    def step(self, t_wheel, w_wheel):
        best = None
        for gear, trans in enumerate(self.paths, start=1):
            res = _drive_path(self.machine, self.inverter, trans, t_wheel, w_wheel)
            if res is None:
                continue
            res.mode = "gear %d" % gear
            # rank: first deliver the demand, then lowest DC power
            key = (round(res.shortfall_w), res.p_dc_w)
            if best is None or key < (round(best.shortfall_w), best.p_dc_w):
                best = res
        return best


class DualMotorEAxle:
    """Electric axle with two e-machines, each on a fixed single-speed reduction.

    Operating modes chosen per step (lowest DC power wins):
      - 'single': one machine delivers all torque, the other is unpowered;
        it still spins with the wheels, so its drag losses load the active path
        (unless `disconnect=True`, which models a decoupling clutch)
      - 'shared': both machines share torque equally
    """

    def __init__(self, machine: EMachine, ratio, eta=0.965, inverter=None, disconnect=False):
        suffix = " + disconnect" if disconnect else ""
        self.name = "E-axle (2 motors, 1-speed%s)" % suffix
        self.m1 = machine
        self.m2 = EMachine(**{**machine.__dict__, "name": machine.name.replace("1", "2")})
        self.inverter = inverter or Inverter()
        self.trans = Transmission(ratio, eta)
        self.disconnect = disconnect

    def step(self, t_wheel, w_wheel):
        options = []
        # --- shared mode
        a = _drive_path(self.m1, self.inverter, self.trans, t_wheel / 2, w_wheel)
        b = _drive_path(self.m2, self.inverter, self.trans, t_wheel / 2, w_wheel)
        if a is not None and b is not None:
            options.append(StepResult(
                p_dc_w=a.p_dc_w + b.p_dc_w,
                loss_motor_w=a.loss_motor_w + b.loss_motor_w,
                loss_inverter_w=a.loss_inverter_w + b.loss_inverter_w,
                loss_transmission_w=a.loss_transmission_w + b.loss_transmission_w,
                friction_brake_w=a.friction_brake_w + b.friction_brake_w,
                shortfall_w=a.shortfall_w + b.shortfall_w,
                op_points=a.op_points + b.op_points, mode="shared"))
        # --- single-motor mode
        w_m = w_wheel * self.trans.ratio
        drag_w = 0.0 if self.disconnect else self.m2.speed_losses(w_m)
        # drag torque of the idle machine must be supplied through the wheels
        t_drag_wheel = drag_w / max(w_wheel, 1e-3) if w_wheel > 0.1 else 0.0
        s = _drive_path(self.m1, self.inverter, self.trans, t_wheel + t_drag_wheel, w_wheel)
        if s is not None:
            s.loss_motor_w += drag_w
            s.mode = "single"
            options.append(s)
        if not options:
            return None
        return min(options, key=lambda r: (round(r.shortfall_w), r.p_dc_w))
