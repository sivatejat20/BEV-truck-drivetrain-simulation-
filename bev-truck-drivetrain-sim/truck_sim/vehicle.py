"""Vehicle parameters and road-load model for a heavy-duty battery-electric truck."""
from dataclasses import dataclass
import numpy as np

G = 9.81      # gravitational acceleration [m/s^2]
RHO_AIR = 1.2  # air density [kg/m^3]


@dataclass
class Vehicle:
    """40 t tractor-semitrailer combination (generic, literature-typical values)."""
    mass_kg: float = 40_000.0        # gross combination mass
    rot_mass_factor: float = 0.03    # equivalent rotating inertia as fraction of mass
    cd: float = 0.50                 # aerodynamic drag coefficient
    frontal_area_m2: float = 10.0    # frontal area
    crr: float = 0.0050              # rolling resistance coefficient
    wheel_radius_m: float = 0.50     # dynamic tyre radius
    aux_power_w: float = 3_000.0     # constant auxiliary load (HVAC, compressors, 24 V)

    def tractive_force(self, v, a, grade):
        """Force required at the wheels [N] for speed v [m/s], acceleration a [m/s^2]
        and road grade [-] (rise/run). Returns the force and its components."""
        theta = np.arctan(grade)
        f_aero = 0.5 * RHO_AIR * self.cd * self.frontal_area_m2 * v ** 2
        f_roll = self.mass_kg * G * self.crr * np.cos(theta) * (v > 0.01)
        f_grade = self.mass_kg * G * np.sin(theta)
        f_inertia = self.mass_kg * (1 + self.rot_mass_factor) * a
        total = f_aero + f_roll + f_grade + f_inertia
        return total, {"aero": f_aero, "rolling": f_roll, "grade": f_grade, "inertia": f_inertia}
