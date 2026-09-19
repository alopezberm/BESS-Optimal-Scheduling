"""Project economics: NPV, IRR, LCOS and a CAPEX break-even sweep.

We deliberately don't know a real battery price, so instead of assuming one, everything
here is parameterized by ``capex_eur_per_mwh`` and swept over a plausible range. The
headline result the notebook builds towards is: *given the arbitrage revenue this BESS
actually earns in the backtest, up to what CAPEX (EUR/MWh) does the project break even?*

All cash-flow assumptions (flat annual revenue, constant OPEX, no augmentation/replacement
when capacity fades) are simplifications, called out where they matter and listed in the
project roadmap as the natural next refinements (e.g. a declining-revenue path as the
battery ages, or a mid-life augmentation once DoD-aware degradation crosses a threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import numpy_financial as npf
except ImportError:  # pragma: no cover - numpy_financial is a required dependency, see requirements.txt
    npf = None


def npv(cash_flows: list[float], discount_rate: float) -> float:
    """Net present value of ``cash_flows`` (index 0 = year 0, e.g. -CAPEX)."""
    years = np.arange(len(cash_flows))
    return float(np.sum(np.asarray(cash_flows) / (1 + discount_rate) ** years))


def irr(cash_flows: list[float]) -> float:
    """Internal rate of return of ``cash_flows`` (index 0 = year 0, e.g. -CAPEX)."""
    if npf is not None:
        return float(npf.irr(cash_flows))
    return _irr_bisection(cash_flows)


def _irr_bisection(cash_flows: list[float], lo: float = -0.99, hi: float = 5.0, tol: float = 1e-6) -> float:
    """Fallback IRR via bisection on NPV(rate) = 0, used only if numpy_financial is absent."""
    f_lo, f_hi = npv(cash_flows, lo), npv(cash_flows, hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(cash_flows, mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def lcos(
    capex_eur_per_mwh: float,
    e_max_mwh: float,
    annual_opex_eur: float,
    annual_discharged_mwh: float,
    discount_rate: float,
    lifetime_years: int,
) -> float:
    """Levelized Cost of Storage [EUR / MWh discharged].

    LCOS = (CAPEX + present value of OPEX) / (present value of energy discharged).
    No replacement/augmentation cost is included (see module docstring).
    """
    capex_total = capex_eur_per_mwh * e_max_mwh
    years = np.arange(1, lifetime_years + 1)
    pv_opex = np.sum(annual_opex_eur / (1 + discount_rate) ** years)
    pv_energy = np.sum(annual_discharged_mwh / (1 + discount_rate) ** years)
    return (capex_total + pv_opex) / pv_energy


def capex_breakeven_sweep(
    annual_arbitrage_revenue_eur: float,
    annual_equivalent_cycles: float,
    e_max_mwh: float,
    discount_rate: float,
    lifetime_years: int,
    capex_range_eur_per_mwh: np.ndarray,
    cycle_life_at_reference_dod: float = 4000.0,
    opex_pct_of_capex: float = 0.02,
) -> pd.DataFrame:
    """NPV, IRR and simple payback for a range of battery CAPEX values.

    For each CAPEX (EUR/MWh) in the sweep:
      - battery cost      = capex * E_max
      - annual OPEX       = opex_pct_of_capex * battery cost
      - annual degradation = annual_equivalent_cycles * battery_cost / cycle_life_at_reference_dod
        (throughput-based; see degradation.py for the DoD-aware refinement)
      - annual net cash flow = arbitrage revenue - OPEX - degradation cost
      - cash flows = [-battery cost, net cash flow, net cash flow, ...] over the project life
    """
    rows = []
    for capex in capex_range_eur_per_mwh:
        battery_cost_eur = capex * e_max_mwh
        opex_annual_eur = opex_pct_of_capex * battery_cost_eur
        degradation_annual_eur = annual_equivalent_cycles * battery_cost_eur / cycle_life_at_reference_dod
        net_annual_cf = annual_arbitrage_revenue_eur - opex_annual_eur - degradation_annual_eur

        cash_flows = [-battery_cost_eur] + [net_annual_cf] * lifetime_years
        project_npv = npv(cash_flows, discount_rate)
        project_irr = irr(cash_flows)
        payback_years = battery_cost_eur / net_annual_cf if net_annual_cf > 0 else np.inf

        rows.append(
            {
                "capex_eur_per_mwh": capex,
                "battery_cost_eur": battery_cost_eur,
                "net_annual_cash_flow_eur": net_annual_cf,
                "npv_eur": project_npv,
                "irr": project_irr,
                "payback_years": payback_years,
            }
        )
    return pd.DataFrame(rows)


def find_breakeven_capex(sweep: pd.DataFrame) -> float:
    """Linearly interpolate the CAPEX (EUR/MWh) where NPV crosses zero in ``sweep``."""
    sweep = sweep.sort_values("capex_eur_per_mwh")
    sign_change = np.where(np.diff(np.sign(sweep["npv_eur"])) < 0)[0]
    if len(sign_change) == 0:
        return float("nan")
    i = sign_change[0]
    x0, x1 = sweep["capex_eur_per_mwh"].iloc[i], sweep["capex_eur_per_mwh"].iloc[i + 1]
    y0, y1 = sweep["npv_eur"].iloc[i], sweep["npv_eur"].iloc[i + 1]
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
