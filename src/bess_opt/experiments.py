"""The fine-tuning workbench: run strategies, decompose their P&L, and compare them fairly.

A *strategy*, in this project, is a ``ProjectConfig``. It contains no trading logic of its
own - the operating policy is whatever the optimiser does once the physical envelope,
the cost of cycling and the market frictions are specified. Tuning the strategy therefore
means tuning assumptions, and comparing strategies means solving the same days under
different assumptions and looking at what changed.

Why comparisons need this much care
-----------------------------------
Two schedules are easy to compare badly. Three rules are enforced here:

1. **Same days, same weather.** Every strategy is evaluated on an identical list of dates,
   and PV is seeded from the date rather than from the run, so no strategy gets a luckier
   sample. Without this, differences of a few percent are indistinguishable from noise.

2. **Report the number that reaches the bank account, and the number that drives the
   decision, separately.** The *gross margin* is market revenue net of network tariffs: real
   cash. The *economic margin* additionally deducts the degradation charge, which is an
   opportunity cost rather than a payment. A strategy can raise one while lowering the other,
   and that is precisely the trade-off worth seeing, so both appear side by side.

3. **Quote uncertainty.** Daily arbitrage profit is heavily right-skewed - a handful of
   volatile days carry most of the annual result - so the mean over thirty days carries a
   standard error worth several percent. Any difference between two strategies smaller than
   that is not a finding, and ``compare_strategies`` reports the standard error next to the
   mean so the reader can apply that test themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest, sample_available_days
from .config import ProjectConfig
from .degradation import degradation_cost_eur, energy_throughput_mwh
from .economics import capex_breakeven_sweep, effective_lifetime_years, evaluate_project, find_breakeven_capex

DAYS_PER_YEAR = 365.0


# --------------------------------------------------------------------------------------
# One day
# --------------------------------------------------------------------------------------
def schedule_pnl(schedule: pd.DataFrame, cfg: ProjectConfig) -> dict:
    """Decompose one solved day into its revenue, cost and operating components.

    The decomposition is deliberately redundant: the individual revenue streams are reported
    alongside the totals so that they can be added up and checked against the solver's own
    objective. The notebook performs exactly that reconciliation, because an accounting layer
    that silently disagrees with the model it reports on is worse than no accounting layer.
    """
    b, market = cfg.battery, cfg.market
    dt = cfg.simulation.dt_hours
    price = schedule["price"]

    discharge_revenue = float((price * schedule["P_dis"]).sum() * dt)
    pv_direct_revenue = float((price * schedule.get("P_pv_direct", 0.0)).sum() * dt)
    charge_cost = float((price * schedule["P_ch"]).sum() * dt)
    aux_cost = float(price.sum() * b.aux_load_mw * dt)

    tariff_cost = float(
        (
            market.grid_tariff_charge_eur_per_mwh * schedule["P_ch"]
            + market.grid_tariff_discharge_eur_per_mwh * schedule["P_dis"]
        ).sum()
        * dt
    )

    throughput = energy_throughput_mwh(schedule, dt)
    degradation_charged = cfg.degradation_cost_eur_per_mwh_throughput() * throughput

    gross_margin = discharge_revenue + pv_direct_revenue - charge_cost - aux_cost - tariff_cost
    economic_margin = gross_margin - degradation_charged

    charged_mwh = float(schedule["P_ch"].sum() * dt)
    discharged_mwh = float(schedule["P_dis"].sum() * dt)
    pv_available = float(schedule["P_pv"].sum() * dt)
    pv_curtailed = float(schedule.get("P_pv_curt", pd.Series(0.0, index=schedule.index)).sum() * dt)

    # Volume-weighted prices: what the battery actually bought and sold at, as opposed to the
    # day's average price, which nobody trades.
    buy_price = charge_cost / charged_mwh if charged_mwh > 1e-9 else np.nan
    sell_price = discharge_revenue / discharged_mwh if discharged_mwh > 1e-9 else np.nan

    degradation = degradation_cost_eur(schedule, cfg)

    return {
        # Cash
        "discharge_revenue_eur": discharge_revenue,
        "pv_direct_revenue_eur": pv_direct_revenue,
        "charge_cost_eur": charge_cost,
        "aux_cost_eur": aux_cost,
        "tariff_cost_eur": tariff_cost,
        "gross_margin_eur": gross_margin,
        # Economic signal
        "degradation_charged_eur": degradation_charged,
        "economic_margin_eur": economic_margin,
        # Operation
        "charged_mwh": charged_mwh,
        "discharged_mwh": discharged_mwh,
        "throughput_mwh": throughput,
        "equivalent_full_cycles": degradation["equivalent_full_cycles"],
        "reference_cycles": degradation["reference_cycles_dod_aware"],
        "mean_depth_of_discharge": degradation["mean_depth_of_discharge"],
        "max_depth_of_discharge": degradation["max_depth_of_discharge"],
        "n_half_cycles": degradation["n_half_cycles"],
        # Market behaviour
        "volume_weighted_buy_price": buy_price,
        "volume_weighted_sell_price": sell_price,
        "captured_spread_eur_per_mwh": sell_price - buy_price,
        "mean_price": float(price.mean()),
        "price_std": float(price.std()),
        "price_spread_eur_per_mwh": float(price.max() - price.min()),
        # PV
        "pv_available_mwh": pv_available,
        "pv_curtailed_mwh": pv_curtailed,
        "pv_curtailed_frac": pv_curtailed / pv_available if pv_available > 1e-9 else 0.0,
    }


# --------------------------------------------------------------------------------------
# Many days
# --------------------------------------------------------------------------------------
def daily_metrics(schedules: dict, cfg: ProjectConfig) -> pd.DataFrame:
    """One row of ``schedule_pnl`` per solved day, indexed by date."""
    rows = {day.date(): schedule_pnl(schedule, cfg) for day, schedule in schedules.items()}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "date"
    return frame.sort_index()


def strategy_kpis(daily: pd.DataFrame, cfg: ProjectConfig) -> pd.Series:
    """Aggregate a strategy's daily results into the figures a decision is made on.

    Annual quantities are the daily mean times 365. This extrapolation assumes the sampled
    days are representative of the year, which is exactly why the sample is drawn at random
    across the full price history rather than from one season - and why the standard error
    travels alongside the mean.
    """
    n = len(daily)
    gross = daily["gross_margin_eur"]

    annual_gross_margin = float(gross.mean() * DAYS_PER_YEAR)
    annual_reference_cycles = float(daily["reference_cycles"].mean() * DAYS_PER_YEAR)
    annual_discharged = float(daily["discharged_mwh"].mean() * DAYS_PER_YEAR)

    project = evaluate_project(annual_gross_margin, annual_reference_cycles, annual_discharged, cfg)
    life = effective_lifetime_years(annual_reference_cycles, cfg)

    return pd.Series(
        {
            "days_solved": n,
            # Operating result
            "gross_margin_eur_per_day": float(gross.mean()),
            "gross_margin_stderr_eur_per_day": float(gross.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "gross_margin_median_eur_per_day": float(gross.median()),
            "degradation_cost_eur_per_day": float(daily["degradation_charged_eur"].mean()),
            "economic_margin_eur_per_day": float(daily["economic_margin_eur"].mean()),
            # Utilisation
            "equivalent_full_cycles_per_day": float(daily["equivalent_full_cycles"].mean()),
            "reference_cycles_per_day": float(daily["reference_cycles"].mean()),
            "mean_depth_of_discharge": float(daily["mean_depth_of_discharge"].mean()),
            "discharged_mwh_per_day": float(daily["discharged_mwh"].mean()),
            "eur_per_mwh_discharged": float(gross.sum() / daily["discharged_mwh"].sum())
            if daily["discharged_mwh"].sum() > 1e-9
            else np.nan,
            "captured_spread_eur_per_mwh": float(daily["captured_spread_eur_per_mwh"].mean()),
            "pv_curtailed_frac": float(daily["pv_curtailed_frac"].mean()),
            # Annualised
            "annual_gross_margin_eur": annual_gross_margin,
            "annual_reference_cycles": annual_reference_cycles,
            # Investment case
            "effective_lifetime_years": project["effective_lifetime_years"],
            "lifetime_binding_constraint": life["binding_constraint"],
            "npv_eur": project["npv_eur"],
            "irr": project["irr"],
            "discounted_payback_years": project["discounted_payback_years"],
            "lcos_eur_per_mwh": project["lcos_eur_per_mwh"],
        },
        name=cfg.name,
    )


def run_strategy(cfg: ProjectConfig, prices: pd.Series, days: list[pd.Timestamp] | None = None):
    """Solve every day under one configuration and return ``(daily, kpis, schedules)``."""
    if days is None:
        days = sample_available_days(prices, cfg)
    _, schedules = run_backtest(days, prices, cfg)
    daily = daily_metrics(schedules, cfg)
    return daily, strategy_kpis(daily, cfg), schedules


def compare_strategies(
    configs: list[ProjectConfig],
    prices: pd.Series,
    days: list[pd.Timestamp] | None = None,
    return_daily: bool = False,
):
    """Evaluate several strategies on identical days and tabulate them side by side.

    The day sample is drawn once, from the first configuration, and reused for all of them.
    Drawing it separately per strategy would let a difference in the ``day_sample_seed`` masquerade
    as a difference in strategy - the single most effective way to fool oneself in a study like this.
    """
    if not configs:
        raise ValueError("compare_strategies needs at least one configuration.")
    if days is None:
        days = sample_available_days(prices, configs[0])

    kpis, dailies = [], {}
    for cfg in configs:
        daily, kpi, _ = run_strategy(cfg, prices, days)
        kpis.append(kpi)
        dailies[cfg.name] = daily

    table = pd.DataFrame(kpis)
    return (table, dailies) if return_daily else table


# --------------------------------------------------------------------------------------
# Parameter sweeps
# --------------------------------------------------------------------------------------
def sensitivity(
    cfg: ProjectConfig,
    section: str,
    parameter: str,
    values,
    prices: pd.Series,
    days: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Sweep one parameter and tabulate how every KPI responds.

    The workhorse of the fine-tuning sections: it isolates a single assumption while holding
    the rest of the configuration and the day sample fixed, which is the only way to attribute
    a change in outcome to that assumption. Use it before forming an opinion about a parameter,
    not after.

    Example::

        sensitivity(cfg, "battery", "soc_min_frac", [0.0, 0.05, 0.10, 0.20], prices)
    """
    if days is None:
        days = sample_available_days(prices, cfg)

    variants = [
        cfg.variant(f"{parameter}={value}", **{section: {parameter: value}}) for value in values
    ]
    table = compare_strategies(variants, prices, days)
    table.insert(0, parameter, list(values))
    return table.reset_index(drop=True)


def capex_sweep_reoptimised(
    cfg: ProjectConfig,
    capex_range_eur_per_mwh,
    prices: pd.Series,
    days: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Sweep CAPEX while re-solving the operating problem at every point.

    CAPEX enters the investment case twice, and only one of those is usually modelled. It is
    the year-zero outflow, obviously; but through ``DegradationParams`` it is also the price
    the optimiser puts on a cycle, so a cheaper battery is one the optimiser rationally cycles
    harder, earning more margin than the same schedule would have earned. Holding the schedule
    fixed across the sweep therefore understates the value of cheap storage and overstates the
    break-even cost of expensive storage.

    This function re-solves every day at every CAPEX - a few hundred small LPs, seconds of
    work - so that the feedback is present. The notebook compares it against the fixed-schedule
    sweep in ``economics.capex_breakeven_sweep`` to size the error the shortcut introduces.
    """
    if days is None:
        days = sample_available_days(prices, cfg)

    rows = []
    for capex in np.asarray(capex_range_eur_per_mwh, dtype=float):
        variant = cfg.variant(f"CAPEX={capex:,.0f}", degradation=dict(capex_eur_per_mwh=float(capex)))
        daily, kpi, _ = run_strategy(variant, prices, days)
        rows.append(
            {
                "capex_eur_per_mwh": float(capex),
                "degradation_cost_eur_per_mwh": variant.degradation_cost_eur_per_mwh_throughput(),
                "gross_margin_eur_per_day": kpi["gross_margin_eur_per_day"],
                "equivalent_full_cycles_per_day": kpi["equivalent_full_cycles_per_day"],
                "effective_lifetime_years": kpi["effective_lifetime_years"],
                "npv_eur": kpi["npv_eur"],
                "irr": kpi["irr"],
                "lcos_eur_per_mwh": kpi["lcos_eur_per_mwh"],
            }
        )
    return pd.DataFrame(rows)


def breakeven_capex_both_ways(
    cfg: ProjectConfig,
    capex_range_eur_per_mwh,
    prices: pd.Series,
    days: list[pd.Timestamp] | None = None,
) -> dict:
    """Break-even CAPEX with and without the operating feedback, and the gap between them."""
    reoptimised = capex_sweep_reoptimised(cfg, capex_range_eur_per_mwh, prices, days)
    be_reopt = find_breakeven_capex(reoptimised)

    daily, kpi, _ = run_strategy(cfg, prices, days)
    fixed = capex_breakeven_sweep(
        kpi["annual_gross_margin_eur"],
        kpi["annual_reference_cycles"],
        daily["discharged_mwh"].mean() * DAYS_PER_YEAR,
        cfg,
        capex_range_eur_per_mwh,
    )
    be_fixed = find_breakeven_capex(fixed)

    return {
        "breakeven_capex_fixed_schedule": be_fixed,
        "breakeven_capex_reoptimised": be_reopt,
        "relative_gap": (be_reopt / be_fixed - 1.0) if np.isfinite(be_fixed) and be_fixed else np.nan,
        "sweep_fixed_schedule": fixed,
        "sweep_reoptimised": reoptimised,
    }
