"""Sanity checks: energy conservation, plausible consumption, cycle limits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from truck_sim import simulate, urban_delivery, regional_delivery, long_haul
from run_study import build_topologies

CYCLES = [urban_delivery(), regional_delivery(), long_haul()]


@pytest.mark.parametrize("cycle", CYCLES, ids=lambda c: c.name)
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_energy_balance_closes(cycle, idx):
    """Battery energy = net wheel energy + every loss + auxiliaries (within 0.5 %)."""
    r = simulate(build_topologies()[idx], cycle)
    e = r.energy_kwh
    rhs = (e["wheel_traction"] - e["wheel_braking"] + e["motor"] + e["inverter"]
           + e["transmission"] + e["friction_brake"] + e["battery_loss"] + e["aux"])
    assert abs(rhs - e["battery_internal"]) / e["battery_internal"] < 0.005


@pytest.mark.parametrize("cycle", CYCLES, ids=lambda c: c.name)
def test_consumption_is_plausible(cycle):
    """Published values for 40 t BEV trucks are roughly 0.9-1.6 kWh/km."""
    for top in build_topologies():
        assert 0.8 < simulate(top, cycle).kwh_per_km < 1.8


@pytest.mark.parametrize("cycle", CYCLES, ids=lambda c: c.name)
def test_cycle_respects_driver_limits(cycle):
    assert cycle.a.max() <= 0.41 and cycle.a.min() >= -0.61
    assert cycle.v.min() >= 0


@pytest.mark.parametrize("cycle", CYCLES, ids=lambda c: c.name)
def test_disconnect_never_increases_consumption(cycle):
    tops = build_topologies()
    assert simulate(tops[2], cycle).kwh_per_km <= simulate(tops[1], cycle).kwh_per_km + 1e-9
