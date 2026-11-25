"""Drive-cycle generation.

The cycles here are synthetic but built in the style of the EU VECTO mission
profiles (target speed and road grade defined over distance, with stops).
Replace them with official VECTO cycles for publishable results: see README.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Segment:
    v_kmh: float        # target cruise speed
    length_m: float     # segment length
    grade_pct: float = 0.0
    stop_s: float = 0.0  # standstill time at the END of the segment (0 = no stop)


@dataclass
class DriveCycle:
    name: str
    t: np.ndarray       # time [s]
    v: np.ndarray       # speed [m/s]
    a: np.ndarray       # acceleration [m/s^2]
    grade: np.ndarray   # road grade [-]
    x: np.ndarray       # distance [m]

    @property
    def distance_km(self):
        return self.x[-1] / 1000.0


def build_cycle(name, segments, dt=1.0, a_acc=0.4, a_dec=0.6, dx=1.0,
                power_limit_w=380e3, mass_kg=40_000.0):
    """Turn a list of segments into a time-based speed trace.

    Method (standard for distance-based mission profiles):
    1. Speed limit over distance from the segment targets; zero at every stop.
    2. Forward pass limits acceleration to a_acc and to what a truck with
       `power_limit_w` at the wheels can achieve on that grade (a loaded truck
       accelerates slowly uphill); backward pass limits braking to a_dec.
    3. Integrate time along distance, insert standstill times, resample to dt.
    """
    seg_ends = np.cumsum([s.length_m for s in segments])
    x = np.arange(0.0, seg_ends[-1] + dx, dx)
    idx = np.minimum(np.searchsorted(seg_ends, x, side="right"), len(segments) - 1)
    v_lim = np.array([segments[i].v_kmh / 3.6 for i in idx])
    grade_x = np.array([segments[i].grade_pct / 100 for i in idx])
    stops = {0: 0.0}
    for i, s in enumerate(segments):
        if s.stop_s > 0 or i == len(segments) - 1:
            k = int(round(seg_ends[i] / dx))
            stops[k] = s.stop_s
    v = v_lim.copy()
    for k in stops:
        v[k] = 0.0
    for i in range(len(v) - 1):                      # forward: acceleration limit
        vi = max(v[i], 1.0)
        f_res = mass_kg * 9.81 * (grade_x[i] + 0.005) + 0.5 * 1.2 * 5.0 * vi ** 2
        a_pow = (power_limit_w / vi - f_res) / (1.03 * mass_kg)
        a_max = max(min(a_acc, a_pow), 0.02)
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_max * dx))
    for i in range(len(v) - 2, -1, -1):              # backward: braking limit
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_dec * dx))

    # time along distance, with standstill inserted at stop points
    t_x = np.zeros_like(x)
    for i in range(1, len(x)):
        t_x[i] = t_x[i - 1] + stops.get(i - 1, 0.0) * (i - 1 > 0) + dx / max(0.5 * (v[i] + v[i - 1]), 1e-3)
    t_end = t_x[-1] + stops.get(len(x) - 1, 0.0)
    t = np.arange(0.0, t_end, dt)
    # standstill phases appear as flat distance; interpolate x(t) then derive v, a
    t_knots, x_knots = [0.0], [0.0]
    for i in range(1, len(x)):
        if stops.get(i - 1, 0.0) > 0 and i - 1 > 0:
            t_knots.append(t_x[i - 1] + stops[i - 1]); x_knots.append(x[i - 1])
        t_knots.append(t_x[i]); x_knots.append(x[i])
    t_knots.append(t_end); x_knots.append(x[-1])
    x_t = np.interp(t, t_knots, x_knots)
    v_t = np.interp(x_t, x, v)
    # at stops x_t is constant -> speed from profile is 0 at the stop point
    v_t[np.r_[False, np.diff(x_t) < 1e-9]] = 0.0
    a_t = np.gradient(v_t, dt)
    g_t = np.interp(x_t, x, grade_x)
    return DriveCycle(name, t, v_t, a_t, g_t, x_t)


def regional_delivery():
    """~60 km regional distribution: town edges, rural roads, short motorway, hills."""
    segs = [
        Segment(50, 1500, 0.0, 30), Segment(50, 2000, 1.0, 20), Segment(70, 4000, 0.0),
        Segment(80, 6000, -1.0), Segment(70, 3000, 2.0, 40), Segment(50, 1500, 0.0, 30),
        Segment(85, 12000, 0.5), Segment(85, 6000, -1.5), Segment(70, 5000, 3.0),
        Segment(70, 4000, -3.0), Segment(50, 2000, 0.0, 60), Segment(80, 8000, 0.0),
        Segment(50, 2500, 1.0, 30), Segment(30, 1500, 0.0, 120),
    ]
    return build_cycle("Regional delivery", segs)


def urban_delivery():
    """~25 km urban delivery with frequent stops, low speeds and short climbs."""
    segs = []
    for k in range(12):
        segs.append(Segment(50 if k % 3 else 30, 1200 + 150 * (k % 4), (2.0 if k % 4 == 1 else -1.5 if k % 4 == 3 else 0.0), 45 if k % 2 else 20))
        segs.append(Segment(30, 600, 0.0, 90 if k % 3 == 0 else 10))
    return build_cycle("Urban delivery", segs)


def long_haul():
    """~120 km motorway long haul with long grades."""
    segs = [Segment(50, 2000, 0.0, 10), Segment(85, 30000, 0.0), Segment(85, 8000, 2.0),
            Segment(85, 8000, -2.0), Segment(85, 40000, 0.3), Segment(80, 10000, 3.0),
            Segment(85, 10000, -2.5), Segment(85, 12000, 0.0), Segment(50, 2000, 0.0, 120)]
    return build_cycle("Long haul", segs)
