"""Battery cycling & degradation analysis.

Two levels of detail, on purpose:

1. A **throughput-based** equivalent-cycle count (``equivalent_full_cycles``) - the
   standard, coarse metric, and the one implicitly assumed by the linear
   ``degradation_eur_per_mwh_throughput`` cost term used inside the LP in ``model.py``.
   It is exact for the *total energy* moved, but treats a full 0-100% cycle and four
   shallow 20-30% swings as equally damaging, which real cells do not.

2. A **depth-of-discharge-aware** half-cycle count (``dod_half_cycles``), using local
   extrema of the state-of-charge trajectory (a simplified rainflow-counting scheme:
   full rainflow counting also handles nested/interrupted cycles, which matters for
   irregular loading but is unlikely to change the picture much for a smooth day-ahead
   SoC trajectory - a good first upgrade, noted in the project roadmap). Combined with a
   documented DoD-stress curve, this gives a *relative* degradation cost that is higher
   for deep cycles than for shallow ones at the same throughput, which is the effect the
   throughput-only metric misses entirely.

``degradation_cost_eur`` turns either metric into a EUR figure given a battery CAPEX and
an assumed cycle life at a reference DoD - see the function docstring for the (explicitly
simplified, meant-to-be-replaced-later) assumptions behind that conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def equivalent_full_cycles(schedule: pd.DataFrame, e_max_mwh: float, dt: float) -> float:
    """Total energy throughput (charge + discharge) divided by 2*E_max.

    2*E_max because one "full cycle" moves E_max MWh in and E_max MWh out.
    """
    throughput_mwh = (schedule["P_ch"] + schedule["P_dis"]).sum() * dt
    return throughput_mwh / (2 * e_max_mwh)


def dod_half_cycles(schedule: pd.DataFrame, e_max_mwh: float) -> pd.DataFrame:
    """Depth-of-discharge of each charge/discharge swing in the SoC trajectory.

    Finds local maxima and minima of the state-of-charge series and reports the depth
    (as a fraction of E_max) of every monotonic swing between them - a "half cycle" in
    rainflow-counting terminology. A flat trajectory (no cycling at all) returns an
    empty frame.
    """
    soc = schedule["E"].to_numpy() / e_max_mwh
    soc = np.concatenate([soc, schedule["E_end"].to_numpy()[-1:] / e_max_mwh])

    maxima = argrelextrema(soc, np.greater_equal, order=1)[0]
    minima = argrelextrema(soc, np.less_equal, order=1)[0]
    turning_points = np.sort(np.unique(np.concatenate([[0], maxima, minima, [len(soc) - 1]])))
    # Drop consecutive duplicates that argrelextrema's >=/<= tolerance can introduce on plateaus.
    turning_points = turning_points[np.concatenate([[True], np.diff(soc[turning_points]) != 0])]

    depths = np.abs(np.diff(soc[turning_points]))
    directions = np.where(np.diff(soc[turning_points]) > 0, "charge", "discharge")

    return pd.DataFrame({"depth_of_discharge": depths, "direction": directions})


def degradation_cost_eur(
    schedule: pd.DataFrame,
    e_max_mwh: float,
    dt: float,
    capex_eur_per_mwh: float,
    cycle_life_at_reference_dod: float = 4000.0,
    reference_dod: float = 0.8,
    dod_stress_exponent: float = 1.0,
) -> dict:
    """Estimate the EUR cost of degradation incurred by one solved schedule.

    Model (deliberately simple - see the project roadmap for the planned, calibrated
    replacement): a cell rated for ``cycle_life_at_reference_dod`` full-equivalent cycles
    at ``reference_dod`` (e.g. 4000 cycles at 80% DoD is a common Li-ion NMC/LFP
    datasheet figure) loses a fraction ``1 / cycle_life`` of its CAPEX per equivalent
    full cycle *at that reference depth*. Cycling more shallowly than the reference is
    assumed less damaging per unit of energy throughput, and more deeply more damaging,
    via a power-law stress factor ``(depth / reference_dod) ** dod_stress_exponent``
    applied to each individual half-cycle before summing - this is the mechanism that
    lets "10 shallow cycles" and "1 deep cycle" of the same total throughput end up with
    different costs, which a flat throughput-based price cannot represent.

    Returns both the throughput-only estimate (what the LP's linear cost term actually
    represents) and the DoD-aware estimate, so the two can be compared directly.
    """
    battery_capex_eur = capex_eur_per_mwh * e_max_mwh
    cost_per_full_cycle_eur = battery_capex_eur / cycle_life_at_reference_dod

    equiv_cycles = equivalent_full_cycles(schedule, e_max_mwh, dt)
    throughput_only_cost_eur = equiv_cycles * cost_per_full_cycle_eur

    half_cycles = dod_half_cycles(schedule, e_max_mwh)
    if half_cycles.empty:
        dod_aware_cost_eur = 0.0
        mean_dod = 0.0
    else:
        stress = (half_cycles["depth_of_discharge"] / reference_dod) ** dod_stress_exponent
        # Two half-cycles make one full cycle, hence the factor 1/2.
        dod_aware_cost_eur = 0.5 * (stress * cost_per_full_cycle_eur).sum()
        mean_dod = half_cycles["depth_of_discharge"].mean()

    return {
        "equivalent_full_cycles": equiv_cycles,
        "n_half_cycles": len(half_cycles),
        "mean_depth_of_discharge": mean_dod,
        "throughput_only_cost_eur": throughput_only_cost_eur,
        "dod_aware_cost_eur": dod_aware_cost_eur,
        "cost_per_full_cycle_eur": cost_per_full_cycle_eur,
    }
