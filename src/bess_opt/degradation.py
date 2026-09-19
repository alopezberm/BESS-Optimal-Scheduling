"""Cycle counting and degradation costing.

The question this module answers is narrow and specific: *given one solved schedule, how
much of the asset did it consume, and what is that worth in EUR?* Getting the answer right
turns out to depend almost entirely on being disciplined about one thing - the definition
of the unit cycle.

The anchoring problem
---------------------
Two cycle metrics are in common use and they do not measure the same thing:

* **Equivalent full cycles**, ``throughput / (2 * E_nominal)``. The industry reporting
  convention. Its unit cycle is a *100% depth* cycle.
* **Warranted cycles**, counted against a datasheet figure such as "4,000 cycles at 80%
  DoD". Its unit cycle is a cycle of depth ``reference_dod``.

A naive comparison of a throughput-based cost against a depth-aware one silently compares
these two different units, and the depth-aware figure comes out roughly ``1 / reference_dod``
higher - about +25% at a reference DoD of 0.8 - for reasons that have nothing whatsoever to
do with depth of discharge. That is an artefact, not a result.

This module therefore expresses **both** costs in the same unit, the warranted reference
cycle, and builds both from the same half-cycle decomposition of the state-of-charge
trajectory. The consequence is a clean, testable property:

    at ``dod_stress_exponent = 1`` the two costs are identical, by construction.

Any divergence between them is then genuinely attributable to depth, which is the effect we
actually wanted to isolate. The notebook verifies this identity numerically before using the
depth-aware figure for anything.

The ageing law
--------------
Cycles to failure are modelled with the Wohler-type power law standard in the battery
literature,

    N(DoD) = N_ref * (reference_dod / DoD) ** k

so that a cycle of depth ``DoD`` consumes ``(DoD / reference_dod) ** k`` of a warranted
cycle. The exponent ``k`` carries the entire physical claim: ``k = 1`` means damage is
proportional to energy throughput and depth is irrelevant; ``k > 1`` means deep cycles are
disproportionately damaging, which is what mechanical-stress-driven ageing mechanisms
(particle cracking, SEI fracture) produce in practice. Published values cluster around
1.0-1.3 for LFP and can exceed 2 for high-nickel NMC.

Calibrating ``k`` against a real cell dataset is listed in the project roadmap; until then
it is a parameter to be swept, not a number to be trusted, and the notebook treats it that
way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig

# A solved LP returns values carrying floating-point noise of order 1e-12, and an interior-point
# method can leave a variable at 3e-13 instead of exactly zero. Differenced into a SoC trajectory,
# that noise produces sign flips, and a naive swing-counter reports dozens of "half cycles" of
# essentially zero depth. Both thresholds below exist to reject that numerical artefact, not to
# approximate anything physical: 1e-6 of nominal energy is 4 Wh on a 4 MWh battery.
_IDLE_TOLERANCE = 1e-6  # per-interval SoC change treated as "the battery did not move"
_MIN_SWING_DEPTH = 1e-4  # swings shallower than this are discarded as noise, not counted


# --------------------------------------------------------------------------------------
# Throughput metrics
# --------------------------------------------------------------------------------------
def energy_throughput_mwh(schedule: pd.DataFrame, dt: float) -> float:
    """Total energy moved through the battery, AC side, charging plus discharging.

    PV routed into the battery counts: it is free energy but it is not free *cycling*. A
    model that charges degradation only on grid imports systematically over-cycles hybrid
    plants, because the co-located PV looks costless to the optimiser.
    """
    charge = schedule["P_ch"]
    if "P_pv_to_batt" in schedule:
        charge = charge + schedule["P_pv_to_batt"]
    return float((charge + schedule["P_dis"]).sum() * dt)


def equivalent_full_cycles(schedule: pd.DataFrame, e_nominal_mwh: float, dt: float) -> float:
    """Equivalent full cycles in the industry sense: throughput / (2 * nominal energy).

    The factor two is because one full cycle moves ``E_nominal`` in and ``E_nominal`` out.
    Reported because it is the number every BESS operator quotes and every warranty counts,
    but note its unit cycle is a 100%-depth cycle, which no real asset performs - see the
    module docstring on anchoring.
    """
    return energy_throughput_mwh(schedule, dt) / (2.0 * e_nominal_mwh)


# --------------------------------------------------------------------------------------
# Half-cycle decomposition
# --------------------------------------------------------------------------------------
def dod_half_cycles(schedule: pd.DataFrame, e_nominal_mwh: float) -> pd.DataFrame:
    """Decompose the SoC trajectory into monotonic swings ("half cycles").

    Method: take the state-of-charge series, difference it, discard intervals where the
    battery is idle, and merge consecutive differences of the same sign into one swing. Each
    swing's magnitude is its depth, as a fraction of nominal energy.

    Idle intervals do not break a swing. This matters: a battery that charges, sits still for
    two hours and then keeps charging has performed *one* swing, and a scheme that split it
    into two would report two shallow cycles where there was one deeper one - and, under a
    convex ageing law, would understate the damage.

    **Scope of the method.** This is peak-valley counting. It is exact for the smooth,
    single-pass trajectories a day-ahead schedule produces, but it does not handle nested
    cycles - a small ripple superimposed on a large swing - the way full rainflow counting
    (ASTM E1049) does. If a rolling-horizon or reserve-stacking extension ever produces
    noisy trajectories, this is the first thing that must be upgraded, and the roadmap says so.
    """
    soc = np.concatenate(
        [
            schedule["E"].to_numpy() / e_nominal_mwh,
            schedule["E_end"].to_numpy()[-1:] / e_nominal_mwh,
        ]
    )
    deltas = np.diff(soc)
    moving = np.abs(deltas) > _IDLE_TOLERANCE
    if not moving.any():
        return pd.DataFrame({"depth_of_discharge": [], "direction": []})

    swings = _merge_same_sign_runs(deltas[moving])

    # Discarding a negligible swing re-joins its neighbours, which then have the same sign and
    # are one swing rather than two: a 0.45 charge, a 1e-5 dither and another 0.45 charge is a
    # single 0.90 charge. The merge therefore has to be repeated until nothing more collapses.
    # This is not cosmetic - under a convex ageing law one 0.90 swing and two 0.45 swings differ
    # materially in damage, and the second reading would be an artefact of solver noise.
    while True:
        keep = np.abs(swings) >= _MIN_SWING_DEPTH
        if keep.all():
            break
        swings = _merge_same_sign_runs(swings[keep])

    return pd.DataFrame(
        {
            "depth_of_discharge": np.abs(swings),
            "direction": np.where(swings > 0, "charge", "discharge"),
        }
    )


def _merge_same_sign_runs(values: np.ndarray) -> np.ndarray:
    """Sum consecutive values sharing a sign, collapsing each monotonic run into one swing."""
    if values.size == 0:
        return values
    signs = np.sign(values)
    new_run = np.concatenate([[True], signs[1:] != signs[:-1]])
    return np.bincount(np.cumsum(new_run) - 1, weights=values)


def soc_travel(schedule: pd.DataFrame, e_nominal_mwh: float) -> float:
    """Total distance travelled by the state of charge, in fractions of nominal energy.

    Measured on the storage side of the converter, so it is not equal to AC throughput: the
    two differ by the one-way efficiencies. All depth-based metrics below are built on this
    quantity, which is why they are internally consistent with each other.
    """
    return float(dod_half_cycles(schedule, e_nominal_mwh)["depth_of_discharge"].sum())


# --------------------------------------------------------------------------------------
# Costing
# --------------------------------------------------------------------------------------
def degradation_cost_eur(schedule: pd.DataFrame, cfg: ProjectConfig) -> dict:
    """Cost of the ageing incurred by one solved schedule, on two consistent bases.

    Both figures are expressed in *warranted reference cycles*, so they are directly
    comparable:

    ``reference_cycles_throughput``
        ``sum(depth_i) / (2 * reference_dod)`` - depth-blind. One reference cycle consists of
        two half-cycles of depth ``reference_dod``, hence the denominator.

    ``reference_cycles_dod_aware``
        ``sum( 0.5 * (depth_i / reference_dod) ** k )`` - each swing charged at the damage
        implied by the ageing law.

    At ``k = 1`` the two expressions are algebraically identical, which is the built-in test
    that the comparison is measuring depth rather than a choice of units. Multiplying either
    by ``capex * E_nominal / cycle_life`` converts it into EUR.

    Residual value is ignored, which is conservative: it charges the full installed cost to
    the cycles that consume the asset.
    """
    d = cfg.degradation
    e_nom = cfg.battery.e_nominal_mwh
    dt = cfg.simulation.dt_hours

    cost_per_reference_cycle = d.cost_per_reference_cycle_eur(e_nom)
    half_cycles = dod_half_cycles(schedule, e_nom)

    if half_cycles.empty:
        ref_cycles_throughput = 0.0
        ref_cycles_dod = 0.0
        mean_dod = 0.0
        max_dod = 0.0
    else:
        depths = half_cycles["depth_of_discharge"].to_numpy()
        ref_cycles_throughput = float(depths.sum() / (2.0 * d.reference_dod))
        ref_cycles_dod = float(
            np.sum(0.5 * (depths / d.reference_dod) ** d.dod_stress_exponent)
        )
        mean_dod = float(depths.mean())
        max_dod = float(depths.max())

    throughput_cost = ref_cycles_throughput * cost_per_reference_cycle
    dod_cost = ref_cycles_dod * cost_per_reference_cycle

    return {
        # Reporting metric, industry convention (100%-depth unit cycle).
        "equivalent_full_cycles": equivalent_full_cycles(schedule, e_nom, dt),
        "energy_throughput_mwh": energy_throughput_mwh(schedule, dt),
        # Warranty metric, both bases in the same unit.
        "reference_cycles_throughput": ref_cycles_throughput,
        "reference_cycles_dod_aware": ref_cycles_dod,
        "throughput_only_cost_eur": throughput_cost,
        "dod_aware_cost_eur": dod_cost,
        "dod_cost_ratio": (dod_cost / throughput_cost) if throughput_cost > 0 else float("nan"),
        # Diagnostics on how the energy was cycled.
        "n_half_cycles": int(len(half_cycles)),
        "mean_depth_of_discharge": mean_dod,
        "max_depth_of_discharge": max_dod,
        "cost_per_reference_cycle_eur": cost_per_reference_cycle,
        # What the LP objective actually charged, for reconciliation against the above.
        "lp_charged_cost_eur": cfg.degradation_cost_eur_per_mwh_throughput()
        * energy_throughput_mwh(schedule, dt),
    }


def annual_capacity_fade_frac(
    reference_cycles_per_year: float, cfg: ProjectConfig
) -> dict:
    """Split annual capacity loss into its cycling and calendar components.

    Capacity fade is taken as linear in warranted cycles - the asset loses
    ``1 - end_of_life_capacity_frac`` of its capacity over ``cycle_life_at_ref_dod`` cycles -
    plus a calendar term that accrues with time regardless of use.

    The split matters for the investment case, and for a reason worth stating: only the
    cycling term is a *decision*. Calendar fade happens whether the battery trades or sits
    idle, so it is sunk from the operator's point of view and must not enter the objective,
    or the optimiser would be asked to avoid a cost it cannot avoid. It belongs in the
    economics, where it shortens the asset's life and therefore its revenue stream.
    """
    d = cfg.degradation
    fade_span = 1.0 - d.end_of_life_capacity_frac
    cycling_fade = fade_span * reference_cycles_per_year / d.cycle_life_at_ref_dod
    calendar_fade = d.calendar_fade_frac_per_year
    total = cycling_fade + calendar_fade
    return {
        "cycling_fade_frac_per_year": cycling_fade,
        "calendar_fade_frac_per_year": calendar_fade,
        "total_fade_frac_per_year": total,
        "years_to_end_of_life": fade_span / total if total > 0 else float("inf"),
    }
