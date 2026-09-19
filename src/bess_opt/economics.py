"""Project economics: discounted cash flow, LCOS, and the CAPEX at which the project breaks even.

Where degradation belongs, and where it does not
------------------------------------------------
This is the one methodological point that has to be right, because getting it wrong changes
the answer by tens of percent and the error is invisible in the output.

The degradation cost derived in ``config.DegradationParams`` plays **two different roles**,
and it must be counted in exactly one of them at a time:

1. **In the operating problem it is an opportunity cost.** Charging the optimiser
   ``c_degr`` EUR per MWh of throughput is what stops it chasing a 5 EUR/MWh spread with an
   asset whose cycles cost 23 EUR/MWh. It changes the *schedule*. This is its role in
   ``model.build_bess_lp_extended``.

2. **In the investment case it is the CAPEX.** A discounted cash flow already accounts for
   the cost of the asset: it appears in full as the year-zero outflow, and the fact that it
   wears out appears as a finite project life. Subtracting an annual "degradation cost" from
   the operating margin *on top of* that year-zero CAPEX charges the same asset twice.

So the cash flows below use the **gross operating margin** - market revenue net of network
tariffs, which is the money that actually reaches the bank account - and never subtract
degradation from it. What degradation does here instead is set the horizon: cycling harder
raises the annual margin and simultaneously shortens the life over which it is earned. That
trade-off is the substance of the investment decision, and expressing it through the lifetime
rather than through a deducted cost is what makes it visible.

The link is ``effective_lifetime_years``: the economic life requested, capped by the life the
cycling pattern actually permits given the warranted cycle count and calendar fade.

Remaining simplifications, stated so they can be challenged
-----------------------------------------------------------
Revenue is extrapolated from a sample of independently-optimised days, so it assumes future
years resemble the price distribution of the sample; capacity fade is applied linearly to
revenue, which is right for an energy-limited asset and approximate otherwise; no mid-life
augmentation is modelled; and no revenue beyond arbitrage is included, which is conservative
for a real BESS, most of which earn a substantial share of their income in reserve markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig

try:
    import numpy_financial as npf
except ImportError:  # pragma: no cover - listed in requirements.txt
    npf = None


# --------------------------------------------------------------------------------------
# Discounting primitives
# --------------------------------------------------------------------------------------
def annual_periods(horizon_years: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a possibly fractional horizon into yearly periods, the last one part-length.

    Returns the end date of each period and its length, both in years. Shared by the cash-flow
    builder and by LCOS so that the two always integrate over exactly the same timeline - a
    mismatch there would make LCOS and NPV describe different projects.
    """
    horizon = max(1e-9, float(horizon_years))
    n_full = int(np.floor(horizon))
    tail = horizon - n_full
    ends = np.arange(1, n_full + 1, dtype=float)
    if tail > 1e-9:
        ends = np.concatenate([ends, [horizon]])
    if ends.size == 0:  # horizon shorter than a single year
        ends = np.array([horizon])
    lengths = np.diff(np.concatenate([[0.0], ends]))
    return ends, lengths


def npv(cash_flows: list[float] | np.ndarray, discount_rate: float, times: np.ndarray | None = None) -> float:
    """Net present value. ``times`` gives each flow's date in years; defaults to 0, 1, 2, ...

    Passing explicit times matters here because a degradation-limited project ends after a
    fractional number of years, and discounting its final stub as if it were a whole period
    would overstate it.
    """
    flows = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(flows), dtype=float) if times is None else np.asarray(times, dtype=float)
    return float(np.sum(flows / (1.0 + discount_rate) ** t))


def irr(cash_flows: list[float] | np.ndarray, times: np.ndarray | None = None) -> float:
    """Internal rate of return: the discount rate at which NPV vanishes.

    Solved by bisection on the time-aware NPV rather than with ``numpy_financial.irr``, which
    assumes equally spaced periods and would therefore misprice the fractional final year.

    Returns NaN when the cash flows never change sign - the honest answer, since a project that
    never turns positive has no internal rate of return, and printing a number there would be
    worse than printing nothing.
    """
    flows = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(flows), dtype=float) if times is None else np.asarray(times, dtype=float)
    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(flows, lo, t), npv(flows, hi, t)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = npv(flows, mid, t)
        if abs(f_mid) < 1e-9 or hi - lo < 1e-12:
            return float(mid)
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def discounted_payback_years(
    cash_flows: np.ndarray, discount_rate: float, times: np.ndarray | None = None
) -> float:
    """Date at which cumulative discounted cash flow first turns positive, interpolated.

    Preferred over simple payback because a project at a 6% discount rate is not indifferent
    between being repaid in year 2 and in year 8, and simple payback says it is.
    """
    flows = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(flows), dtype=float) if times is None else np.asarray(times, dtype=float)
    discounted = flows / (1.0 + discount_rate) ** t
    cumulative = np.cumsum(discounted)
    positive = np.where(cumulative > 0)[0]
    if positive.size == 0:
        return float("inf")
    i = int(positive[0])
    if i == 0:
        return float(t[0])
    # Linear interpolation inside the period in which the sign changes.
    share = -cumulative[i - 1] / discounted[i]
    return float(t[i - 1] + share * (t[i] - t[i - 1]))


# --------------------------------------------------------------------------------------
# Lifetime
# --------------------------------------------------------------------------------------
def effective_lifetime_years(annual_reference_cycles: float, cfg: ProjectConfig) -> dict:
    """Project life, capped by how fast the chosen operating pattern consumes the asset.

    Two clocks run at once. The cycling clock is set by the schedule: at
    ``annual_reference_cycles`` warranted cycles per year, the asset reaches its end-of-life
    capacity after ``cycle_life / annual_reference_cycles`` years. The calendar clock runs
    regardless. End of life arrives when the two together have consumed the available
    capacity fade, and the project life is whichever comes first, that or the economic
    lifetime assumed in the configuration.

    This is the mechanism through which an aggressive trading strategy pays for itself - or
    does not. It is also the reason the notebook reports lifetime alongside margin: a strategy
    that earns 20% more per year while halving the asset's life is not a better strategy.
    """
    d = cfg.degradation
    fade_span = 1.0 - d.end_of_life_capacity_frac
    cycling_fade = fade_span * annual_reference_cycles / d.cycle_life_at_ref_dod
    calendar_fade = d.calendar_fade_frac_per_year
    total_fade = cycling_fade + calendar_fade

    life_from_degradation = fade_span / total_fade if total_fade > 0 else float("inf")
    requested = float(cfg.economics.lifetime_years)
    return {
        "annual_reference_cycles": annual_reference_cycles,
        "cycling_fade_frac_per_year": cycling_fade,
        "calendar_fade_frac_per_year": calendar_fade,
        "life_from_degradation_years": life_from_degradation,
        "requested_lifetime_years": requested,
        "effective_lifetime_years": min(requested, life_from_degradation),
        "binding_constraint": "degradation" if life_from_degradation < requested else "economic lifetime",
    }


# --------------------------------------------------------------------------------------
# Cash flows
# --------------------------------------------------------------------------------------
def project_cashflows(
    annual_gross_margin_eur: float,
    annual_reference_cycles: float,
    cfg: ProjectConfig,
    capex_eur_per_mwh: float | None = None,
    apply_capacity_fade_to_revenue: bool = True,
) -> pd.DataFrame:
    """Year-by-year cash flows for the storage project.

    ``annual_gross_margin_eur``
        Market revenue net of network tariffs, *before* any degradation charge - see the
        module docstring for why degradation is deliberately absent here.
    ``annual_reference_cycles``
        Warranted cycles per year implied by the operating strategy; it sets the life.
    ``apply_capacity_fade_to_revenue``
        When True, revenue in year n is scaled by the remaining capacity. Appropriate for an
        energy-limited asset whose earnings scale with the MWh it can shift; switch it off to
        reproduce the simpler flat-revenue convention and see how much it flatters the result.
    """
    capex_per_mwh = cfg.degradation.capex_eur_per_mwh if capex_eur_per_mwh is None else capex_eur_per_mwh
    capex_total = capex_per_mwh * cfg.battery.e_nominal_mwh
    opex_annual = cfg.economics.opex_pct_of_capex * capex_total

    life = effective_lifetime_years(annual_reference_cycles, cfg)
    horizon = max(1e-6, life["effective_lifetime_years"])
    fade_per_year = life["cycling_fade_frac_per_year"] + life["calendar_fade_frac_per_year"]

    # The asset's life is a continuous quantity - cycling at a given intensity exhausts it after
    # 5.83 years, not 6 - and rounding it to whole years makes NPV jump discontinuously as a
    # parameter is swept. Those jumps are large enough to invert the ranking of two strategies,
    # so the final year is kept fractional: it earns a proportional share of revenue and OPEX and
    # is discounted at its true (fractional) date.
    period_ends, period_lengths = annual_periods(horizon)

    # Mid-period capacity, so a period is charged the average fade it experiences rather than
    # the fade at either endpoint.
    midpoints = period_ends - 0.5 * period_lengths
    capacity = np.clip(1.0 - fade_per_year * midpoints, 0.0, 1.0)
    scale = capacity if apply_capacity_fade_to_revenue else np.ones_like(capacity)

    revenue = annual_gross_margin_eur * cfg.economics.availability * scale * period_lengths
    opex = opex_annual * period_lengths
    net = revenue - opex
    net[-1] += cfg.economics.residual_value_frac_of_capex * capex_total

    return pd.DataFrame(
        {
            "year": np.concatenate([[0.0], period_ends]),
            "period_length_years": np.concatenate([[0.0], period_lengths]),
            "revenue_eur": np.concatenate([[0.0], revenue]),
            "opex_eur": np.concatenate([[0.0], -opex]),
            "capex_eur": np.concatenate([[-capex_total], np.zeros(len(period_ends))]),
            "net_cash_flow_eur": np.concatenate([[-capex_total], net]),
            "remaining_capacity_frac": np.concatenate([[1.0], capacity]),
        }
    )


def evaluate_project(
    annual_gross_margin_eur: float,
    annual_reference_cycles: float,
    annual_discharged_mwh: float,
    cfg: ProjectConfig,
    capex_eur_per_mwh: float | None = None,
    apply_capacity_fade_to_revenue: bool = True,
) -> dict:
    """Headline investment metrics for one strategy at one CAPEX."""
    flows = project_cashflows(
        annual_gross_margin_eur,
        annual_reference_cycles,
        cfg,
        capex_eur_per_mwh=capex_eur_per_mwh,
        apply_capacity_fade_to_revenue=apply_capacity_fade_to_revenue,
    )
    life = effective_lifetime_years(annual_reference_cycles, cfg)
    net = flows["net_cash_flow_eur"].to_numpy()
    times = flows["year"].to_numpy()
    rate = cfg.economics.discount_rate
    capex_per_mwh = cfg.degradation.capex_eur_per_mwh if capex_eur_per_mwh is None else capex_eur_per_mwh

    return {
        "capex_eur_per_mwh": capex_per_mwh,
        "capex_total_eur": capex_per_mwh * cfg.battery.e_nominal_mwh,
        "effective_lifetime_years": life["effective_lifetime_years"],
        "lifetime_binding_constraint": life["binding_constraint"],
        "npv_eur": npv(net, rate, times),
        "irr": irr(net, times),
        "discounted_payback_years": discounted_payback_years(net, rate, times),
        "lcos_eur_per_mwh": lcos(
            annual_discharged_mwh,
            cfg,
            capex_eur_per_mwh=capex_per_mwh,
            lifetime_years=life["effective_lifetime_years"],
        ),
    }


def lcos(
    annual_discharged_mwh: float,
    cfg: ProjectConfig,
    capex_eur_per_mwh: float | None = None,
    lifetime_years: float | None = None,
) -> float:
    """Levelised cost of storage [EUR per MWh discharged].

        LCOS = (CAPEX + PV of OPEX) / (PV of energy discharged)

    Discounting the energy in the denominator is not a typo and not a physical claim about
    electrons: it is what makes LCOS the break-even *price* at which discounted revenue equals
    discounted cost. Undiscounted energy would give a number that cannot be compared against a
    price. Charging cost is excluded, so this is a cost of *storage*, not a levelised cost of
    delivered energy - the two are routinely confused in the literature.
    """
    capex_per_mwh = cfg.degradation.capex_eur_per_mwh if capex_eur_per_mwh is None else capex_eur_per_mwh
    horizon = float(cfg.economics.lifetime_years if lifetime_years is None else lifetime_years)

    capex_total = capex_per_mwh * cfg.battery.e_nominal_mwh
    opex_annual = cfg.economics.opex_pct_of_capex * capex_total
    rate = cfg.economics.discount_rate

    # Same period decomposition as the cash flows, so LCOS and NPV describe one project.
    ends, lengths = annual_periods(horizon)
    discount = (1.0 + rate) ** ends
    pv_opex = float(np.sum(opex_annual * lengths / discount))
    pv_energy = float(np.sum(annual_discharged_mwh * cfg.economics.availability * lengths / discount))
    if pv_energy <= 0:
        return float("nan")
    return (capex_total + pv_opex) / pv_energy


# --------------------------------------------------------------------------------------
# CAPEX sweep
# --------------------------------------------------------------------------------------
def capex_breakeven_sweep(
    annual_gross_margin_eur: float,
    annual_reference_cycles: float,
    annual_discharged_mwh: float,
    cfg: ProjectConfig,
    capex_range_eur_per_mwh: np.ndarray,
    apply_capacity_fade_to_revenue: bool = True,
) -> pd.DataFrame:
    """Investment metrics across a range of installed costs, holding the schedule fixed.

    This answers "given the margin this strategy earns, what may the hardware cost?" and is
    the right question when the operating strategy is taken as given.

    It is however only half the story, and the notebook makes the point explicitly: CAPEX
    feeds back into the *operating* decision through the degradation cost, so a cheaper
    battery is not merely a cheaper battery - it is also one the optimiser is willing to cycle
    harder, which earns more margin. Holding the schedule fixed while sweeping CAPEX therefore
    understates the value of cheap storage. ``experiments.capex_sweep_reoptimised`` re-solves
    the schedule at every CAPEX and quantifies the gap.
    """
    rows = []
    for capex in np.asarray(capex_range_eur_per_mwh, dtype=float):
        row = evaluate_project(
            annual_gross_margin_eur,
            annual_reference_cycles,
            annual_discharged_mwh,
            cfg,
            capex_eur_per_mwh=float(capex),
            apply_capacity_fade_to_revenue=apply_capacity_fade_to_revenue,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def find_breakeven_capex(sweep: pd.DataFrame) -> float:
    """Interpolate the CAPEX at which NPV crosses zero.

    Returns NaN when the sweep does not bracket the crossing, rather than extrapolating past
    the range that was actually evaluated.
    """
    sweep = sweep.sort_values("capex_eur_per_mwh")
    npv_values = sweep["npv_eur"].to_numpy()
    capex_values = sweep["capex_eur_per_mwh"].to_numpy()

    crossings = np.where(np.diff(np.sign(npv_values)) < 0)[0]
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[0])
    x0, x1 = capex_values[i], capex_values[i + 1]
    y0, y1 = npv_values[i], npv_values[i + 1]
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
