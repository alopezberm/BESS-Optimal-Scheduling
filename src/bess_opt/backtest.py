"""Multi-day backtesting: solve the day-ahead LP independently for many days.

Each day is optimized in isolation (perfect foresight of that day's own prices and PV,
no information carried over except the battery's SoC always restarting from
``battery.initial_energy()``) - i.e. this measures the *arbitrage value under perfect
day-ahead foresight*, repeated across many days, not a realistic rolling deployment.
A forecast-error-aware rolling-horizon backtest is listed in the project roadmap as the
natural next step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import synthetic_pv_profile
from .degradation import equivalent_full_cycles
from .model import BatteryParams, build_bess_lp_extended


def solve_day(
    day: pd.Timestamp,
    prices: pd.Series,
    dt: float,
    battery: BatteryParams,
    pv_capacity_mw: float,
    degradation_eur_per_mwh_throughput: float = 0.0,
    pv_seed_offset: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Build and solve the extended BESS model for a single calendar day."""
    t_steps = round(24 / dt)
    index = pd.date_range(day.normalize(), periods=t_steps, freq=pd.Timedelta(hours=dt))

    prices_day = prices.reindex(index)
    if prices_day.isna().any():
        raise ValueError(f"Missing price data for {day.date()}")

    pv_day = synthetic_pv_profile(index, capacity_mw=pv_capacity_mw, seed=hash((day.date(), pv_seed_offset)) % (2**32))

    model, schedule = build_bess_lp_extended(
        index,
        prices_day.to_numpy(),
        pv_day.to_numpy(),
        dt,
        battery,
        degradation_eur_per_mwh_throughput=degradation_eur_per_mwh_throughput,
    )
    return model.ObjVal, schedule


def run_multi_day_backtest(
    days: list[pd.Timestamp],
    prices: pd.Series,
    dt: float,
    battery: BatteryParams,
    pv_capacity_mw: float,
    degradation_eur_per_mwh_throughput: float = 0.0,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    """Solve one day-ahead LP per day in ``days`` and summarize the results."""
    rows = []
    schedules = {}
    for day in sorted(days):
        try:
            profit, schedule = solve_day(
                day, prices, dt, battery, pv_capacity_mw, degradation_eur_per_mwh_throughput
            )
        except ValueError:
            continue

        rows.append(
            {
                "date": day.date(),
                "profit_eur": profit,
                "mean_price": schedule["price"].mean(),
                "price_std": schedule["price"].std(),
                "pv_energy_mwh": schedule["P_pv"].sum() * dt,
                "charged_mwh": schedule["P_ch"].sum() * dt,
                "discharged_mwh": schedule["P_dis"].sum() * dt,
                "equivalent_full_cycles": equivalent_full_cycles(schedule, battery.e_max_mwh, dt),
            }
        )
        schedules[day] = schedule

    return pd.DataFrame(rows), schedules


def sample_available_days(prices: pd.Series, n_days: int, dt: float, seed: int = 32) -> list[pd.Timestamp]:
    """Pick ``n_days`` random calendar days that have a full day (``round(24/dt)`` records) of price data."""
    expected_intervals = round(24 / dt)
    counts = prices.groupby(prices.index.normalize()).size()
    full_days = counts[counts == expected_intervals].index
    rng = np.random.default_rng(seed)
    chosen = rng.choice(full_days, size=min(n_days, len(full_days)), replace=False)
    return list(pd.to_datetime(chosen))
