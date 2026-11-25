"""Quasi-static component models: e-machine, inverter, transmission, battery.

All models are deliberately simple and parameterised so they can be replaced
with measured efficiency maps later (see README, "Next steps").
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class EMachine:
    """Permanent-magnet synchronous machine with an analytical loss model.

    P_loss = P_max * (c0 + c_cu * (T/T_max)^2 + c_fe * (w/w_max)^1.5 + c_fw * (w/w_max)^2)

    - c_cu: copper (I^2 R) losses, scale with torque squared
    - c_fe: iron losses, scale roughly with speed^1.5
    - c_fw: friction and windage, scale with speed^2
    - c0:   constant (control electronics, cooling pump share)
    Coefficients give a peak efficiency of about 96-97 %, typical of traction PMSMs.
    """
    name: str
    t_max_nm: float
    p_max_w: float
    w_max_rads: float
    c0: float = 0.004
    c_cu: float = 0.022
    c_fe: float = 0.010
    c_fw: float = 0.006

    def torque_limit(self, w):
        """Maximum torque [Nm] at speed w [rad/s] (constant torque, then constant power)."""
        w = np.maximum(np.abs(w), 1e-6)
        return np.minimum(self.t_max_nm, self.p_max_w / w)

    def feasible(self, t, w):
        return (np.abs(w) <= self.w_max_rads) & (np.abs(t) <= self.torque_limit(w) + 1e-6)

    def speed_losses(self, w):
        """Losses that occur whenever the rotor spins, even at zero torque [W]."""
        wn = np.abs(w) / self.w_max_rads
        return self.p_max_w * (self.c_fe * wn ** 1.5 + self.c_fw * wn ** 2)

    def losses(self, t, w, active=True):
        """Total electrical-to-mechanical losses [W] at torque t [Nm] and speed w [rad/s]."""
        if not active:
            return self.speed_losses(w)
        tn = np.abs(t) / self.t_max_nm
        return self.p_max_w * (self.c0 + self.c_cu * tn ** 2) + self.speed_losses(w)

    def electrical_power(self, t, w, active=True):
        """DC-link-side power [W] of the machine (positive = motoring, negative = generating)."""
        return t * w + self.losses(t, w, active)


@dataclass
class Inverter:
    """Inverter with a proportional + constant loss model."""
    k_prop: float = 0.012    # 1.2 % of transferred power
    p_const_w: float = 150.0  # switching/gate-drive base load per active inverter

    def losses(self, p_ac, active=True):
        return (self.k_prop * np.abs(p_ac) + self.p_const_w) if active else 0.0


@dataclass
class Transmission:
    """Gear stage (gearbox + final drive) with constant mechanical efficiency."""
    ratio: float
    efficiency: float

    def motor_side(self, t_wheel, w_wheel):
        """Map wheel torque/speed to motor torque/speed, including mesh losses."""
        w_m = w_wheel * self.ratio
        if t_wheel >= 0:   # motoring: motor must deliver extra torque for losses
            t_m = t_wheel / (self.ratio * self.efficiency)
        else:              # recuperating: losses reduce the torque reaching the motor
            t_m = t_wheel * self.efficiency / self.ratio
        return t_m, w_m


@dataclass
class Battery:
    """Equivalent-circuit battery (open-circuit voltage + internal resistance)."""
    capacity_kwh: float = 600.0
    r_int_ohm: float = 0.045
    soc_init: float = 0.90
    p_charge_max_w: float = 400_000.0  # max regenerative charging power

    def ocv(self, soc):
        """Pack open-circuit voltage [V] (linearised NMC-like curve)."""
        return 640.0 + 120.0 * soc

    def step(self, p_terminal_w, soc, dt):
        """Apply terminal power for dt seconds. Returns (new_soc, current, loss_w, internal_power_w).
        Positive power = discharge."""
        u0 = self.ocv(soc)
        r = self.r_int_ohm
        # Solve P = (U0 - I*R) * I for I (physical root)
        disc = u0 ** 2 - 4 * r * p_terminal_w
        if disc < 0:
            raise ValueError("Battery power demand exceeds what the pack can deliver")
        i = (u0 - np.sqrt(disc)) / (2 * r)
        loss = i ** 2 * r
        p_internal = u0 * i          # chemical power drawn from the cells
        soc_new = soc - p_internal * dt / (self.capacity_kwh * 3.6e6)
        return soc_new, i, loss, p_internal
