"""Compare drivetrain topologies of a 40 t battery-electric truck over three drive cycles.

Usage:  python run_study.py            -> writes results/ (CSV, Markdown table, figures)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from truck_sim import (EMachine, CentralDrive, DualMotorEAxle, simulate,
                       urban_delivery, regional_delivery, long_haul)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RPM = 2 * np.pi / 60

# Categorical colours (fixed order) and text inks
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 130,
})


def build_topologies(eta_eaxle=0.965):
    """Both layouts deliver the same peak power (450 kW) and similar wheel torque."""
    central_motor = EMachine("Central motor", t_max_nm=1600, p_max_w=450e3, w_max_rads=7000 * RPM)
    eaxle_motor = EMachine("E-axle motor 1", t_max_nm=800, p_max_w=225e3, w_max_rads=9000 * RPM)
    return [
        CentralDrive(central_motor, gear_ratios=[3.2, 1.0], final_drive=4.2),
        DualMotorEAxle(eaxle_motor, ratio=17.0, eta=eta_eaxle),
        DualMotorEAxle(eaxle_motor, ratio=17.0, eta=eta_eaxle, disconnect=True),
    ]


def main():
    cycles = [urban_delivery(), regional_delivery(), long_haul()]
    topologies = build_topologies()
    results, rows = {}, []
    for cyc in cycles:
        for top in topologies:
            r = simulate(top, cyc)
            assert r.max_shortfall_w == 0, f"{top.name} cannot follow {cyc.name}"
            results[(cyc.name, top.name)] = r
            e = r.energy_kwh
            rows.append({
                "cycle": cyc.name, "topology": top.name, "distance_km": round(r.distance_km, 1),
                "kWh_per_km": round(r.kwh_per_km, 3),
                "motor_loss_kWh": round(e["motor"], 2), "inverter_loss_kWh": round(e["inverter"], 2),
                "transmission_loss_kWh": round(e["transmission"], 2),
                "friction_brake_kWh": round(e["friction_brake"], 2),
                "battery_loss_kWh": round(e["battery_loss"], 2),
                "recuperation_share_pct": round(100 * (1 - e["friction_brake"] / max(e["wheel_braking"], 1e-9)), 1),
            })
    df = pd.DataFrame(rows)

    # Sensitivity: same transmission efficiency for both layouts
    sens = []
    for cyc in cycles:
        base = df[(df.cycle == cyc.name)].set_index("topology")["kWh_per_km"]
        eq = build_topologies(eta_eaxle=0.975 * 0.97)
        r_eq = simulate(eq[1], cyc)
        sens.append({"cycle": cyc.name,
                     "central_kWh_per_km": base[topologies[0].name],
                     "eaxle_kWh_per_km": base[topologies[1].name],
                     "eaxle_equal_gear_eff_kWh_per_km": round(r_eq.kwh_per_km, 3)})
    sens = pd.DataFrame(sens)

    df.to_csv(OUT / "results.csv", index=False)
    sens.to_csv(OUT / "sensitivity_gear_efficiency.csv", index=False)
    with open(OUT / "results.md", "w") as f:
        piv = df.pivot(index="topology", columns="cycle", values="kWh_per_km")[[c.name for c in cycles]]
        f.write("## Energy consumption [kWh/km]\n\n" + piv.to_markdown() + "\n\n")
        ref = piv.loc[topologies[0].name]
        f.write("## Difference to central drive [%]\n\n" + ((piv / ref - 1) * 100).round(1).to_markdown() + "\n\n")
        f.write("## Sensitivity: e-axle with the same gear efficiency as the central drive\n\n"
                + sens.to_markdown(index=False) + "\n")

    plot_cycles(cycles)
    plot_consumption(df, cycles, topologies)
    plot_losses(df, topologies, "Regional delivery")
    plot_operating_points(results, "Regional delivery", topologies)
    print(open(OUT / "results.md").read())


def plot_cycles(cycles):
    """Speed and road grade over distance; grade gets its own small panel (no dual axis)."""
    fig = plt.figure(figsize=(9, 10), layout="constrained")
    for sf, c in zip(fig.subfigures(len(cycles), 1, hspace=0.04), cycles):
        ax_v, ax_g = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        km = c.x / 1000
        ax_v.plot(km, c.v * 3.6, color=C[0], lw=1.5)
        ax_v.set_ylabel("Speed [km/h]"); ax_v.set_ylim(0, 100)
        ax_v.set_title(f"{c.name}: {c.distance_km:.0f} km, {c.t[-1] / 60:.0f} min",
                       loc="left", fontsize=10)
        ax_g.fill_between(km, c.grade * 100, 0, color=INK2, alpha=0.35, lw=0)
        ax_g.set_ylabel("Grade [%]"); ax_g.set_ylim(-4, 4); ax_g.set_yticks([-3, 0, 3])
        ax_g.set_xlabel("Distance [km]")
    fig.savefig(OUT / "drive_cycles.png")
    plt.close(fig)


def plot_consumption(df, cycles, topologies):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    x = np.arange(len(cycles))
    for i, top in enumerate(topologies):
        vals = [df[(df.cycle == c.name) & (df.topology == top.name)].kWh_per_km.iloc[0] for c in cycles]
        bars = ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=C[i], label=top.name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, [c.name for c in cycles])
    ax.set_ylabel("Battery energy [kWh/km]")
    ax.set_ylim(0, 1.6)
    ax.set_title("Energy consumption by drivetrain topology, 40 t BEV truck", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "energy_consumption.png")
    plt.close(fig)


def plot_losses(df, topologies, cycle):
    cols = [("motor_loss_kWh", "E-machine"), ("transmission_loss_kWh", "Gears"),
            ("inverter_loss_kWh", "Inverter"), ("battery_loss_kWh", "Battery"),
            ("friction_brake_kWh", "Friction brake")]
    sub = df[df.cycle == cycle].set_index("topology").loc[[t.name for t in topologies]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(sub))
    y = np.arange(len(sub))
    for i, (col, label) in enumerate(cols):
        ax.barh(y, sub[col], left=left, color=C[i], label=label, height=0.55,
                edgecolor="white", linewidth=1.5)
        left += sub[col].values
    for yi, tot in zip(y, left):
        ax.text(tot + 0.2, yi, f"{tot:.1f} kWh", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, sub.index)
    ax.invert_yaxis()
    ax.set_xlabel("Energy lost on the cycle [kWh]")
    ax.set_xlim(0, left.max() * 1.2)
    ax.set_title(f"Where the losses occur: {cycle} cycle ({sub.distance_km.iloc[0]:.0f} km)", loc="left")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_breakdown.png")
    plt.close(fig)


def plot_operating_points(results, cycle, topologies):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    for ax, top in zip(axes, topologies[:2]):
        m = top.machine if hasattr(top, "machine") else top.m1
        w = np.linspace(1, m.w_max_rads, 200)
        t = np.linspace(-m.t_max_nm, m.t_max_nm, 200)
        W, T = np.meshgrid(w, t)
        P = T * W
        eff = np.where(P >= 0, P / (P + m.losses(T, W)), (-P - m.losses(T, W)) / -P)
        eff = np.where(np.abs(T) <= m.torque_limit(W), eff, np.nan)
        cs = ax.contourf(W / RPM, T, eff * 100, levels=[70, 80, 85, 90, 93, 95, 96, 97],
                         cmap="Blues", extend="both")
        ax.plot(w / RPM, m.torque_limit(w), color=INK2, lw=1)
        ax.plot(w / RPM, -m.torque_limit(w), color=INK2, lw=1)
        ops = [(tq, sp) for (name, tq, sp, dt) in results[(cycle, top.name)].op_points]
        tq, sp = np.array(ops).T
        ax.scatter(sp / RPM, tq, s=6, color=C[1], alpha=0.35, lw=0)
        ax.set_title(top.name, loc="left", fontsize=10)
        ax.set_xlabel("Motor speed [rpm]")
        ax.set_ylabel("Motor torque [Nm]")
        ax.grid(False)
    fig.colorbar(cs, ax=axes, label="E-machine efficiency [%]", shrink=0.9)
    fig.suptitle(f"Motor operating points (orange) over the efficiency map, {cycle} cycle",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "operating_points.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
