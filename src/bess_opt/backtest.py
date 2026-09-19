"""Solving many days: from one schedule to a distribution of outcomes.

What this backtest is
---------------------
Each calendar day is optimised independently, with exact knowledge of that day's own prices
and PV, starting from the configured initial state of charge. It therefore measures the
**arbitrage value available under perfect day-ahead foresight**, sampled across many days.

What it is not
--------------
It is not a simulation of a deployed trading strategy, and the distinction is not pedantic.
A real operator commits a schedule before knowing the prices, using a forecast that is wrong
in ways that systematically hurt: forecast error is most damaging precisely on the volatile
days where most of the value sits. Published comparisons put perfect-foresight arbitrage
value somewhere around 10-30% above what a forecast-driven strategy realises, so every
revenue figure derived from here should be read as an **upper bound**.

Two structural choices follow from that honesty, both visible in the code below:

* The day loop is kept separate from the single-day solve, so a rolling-horizon variant with
  forecast error can reuse everything except the loop - the natural seam for the extension
  listed in the roadmap.
* The state of charge resets to its initial value every morning rather than carrying over.
  Combined with the ``equal_to_initial`` terminal condition this makes the days genuinely
  independent, which is what licenses treating them as a sample and reporting a standard
  error on the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import synthetic_pv_profile
from .model import SolverError, build_bess_lp_extended


def day_pv_seed(day: pd.Timestamp, offset: int = 0) -> int:
    """Deterministic PV seed for a calendar day.

    Note this does *not* use Python's built-in ``hash``. ``hash`` on a date object is derived
    from a bytes hash, which is salted per interpreter process unless ``PYTHONHASHSEED`` is
    fixed - so a seed built that way silently produces different PV series on every run, and
    a "reproducible" backtest quietly is not. The multiplier below is Knuth's, used purely to
    scatter consecutive ordinals into unrelated seeds.
    """
    return (day.toordinal() * 2654435761 + offset) % (2**32)


def build_day_index(day: pd.Timestamp, cfg: ProjectConfig) -> pd.DatetimeIndex:
    """The interval grid for one optimisation horizon starting at midnight local time."""
    return pd.date_range(
        day.normalize(),
        periods=cfg.simulation.n_intervals,
        freq=pd.Timedelta(hours=cfg.simulation.dt_hours),
    )


def solve_day(
    day: pd.Timestamp,
    prices: pd.Series,
    cfg: ProjectConfig,
    pv_seed_offset: int = 0,
):
    """Build and solve the extended model for a single day.

    PV is generated from a seed derived from the date, so the same day always receives the
    same PV series regardless of which strategy is being evaluated. That is what makes a
    strategy comparison a controlled experiment: two configurations differ only in the
    parameters under study, never in the weather they happened to draw.
    """
    index = build_day_index(day, cfg)
    prices_day = prices.reindex(index)
    if prices_day.isna().any():
        raise ValueError(f"Missing price data for {day.date()}")

    pv_day = synthetic_pv_profile(
        index,
        capacity_mw=cfg.plant.pv_capacity_mw,
        seed=day_pv_seed(day, pv_seed_offset),
    )
    return build_bess_lp_extended(index, prices_day.to_numpy(), pv_day.to_numpy(), cfg)


def sample_available_days(prices: pd.Series, cfg: ProjectConfig, n_days: int | None = None) -> list[pd.Timestamp]:
    """Draw calendar days that have a complete price series, without replacement.

    Incomplete days are excluded rather than padded. A day missing its evening peak is not a
    cheap day, it is an unobserved one, and letting it into the sample would bias the mean
    downwards in a way no amount of extra sampling would reveal.
    """
    n_days = cfg.simulation.n_backtest_days if n_days is None else n_days
    expected = cfg.simulation.n_intervals
    counts = prices.groupby(prices.index.normalize()).size()
    full_days = counts[counts == expected].index
    if len(full_days) == 0:
        raise ValueError(
            f"No day in the price series has the expected {expected} intervals. "
            f"Check dt_hours ({cfg.simulation.dt_hours} h) against the data's resolution."
        )
    rng = np.random.default_rng(cfg.simulation.day_sample_seed)
    chosen = rng.choice(full_days, size=min(n_days, len(full_days)), replace=False)
    return sorted(pd.to_datetime(chosen))


def run_backtest(
    days: list[pd.Timestamp],
    prices: pd.Series,
    cfg: ProjectConfig,
    skip_failures: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Solve one day-ahead problem per day and return the per-day schedules.

    ``skip_failures`` controls what happens when a day has no feasible solution, which under
    a tight configuration is informative rather than exceptional: skipping keeps the run going
    and the caller can compare the number of days solved, while ``False`` surfaces the
    infeasibility diagnostics immediately. Either way the count of solved days is returned, so
    a strategy that quietly failed on a third of the sample cannot masquerade as a good one.
    """
    schedules: dict[pd.Timestamp, pd.DataFrame] = {}
    failures: dict[pd.Timestamp, str] = {}

    for day in sorted(days):
        try:
            _, schedule = solve_day(day, prices, cfg)
        except (ValueError, SolverError) as exc:
            if not skip_failures:
                raise
            failures[day] = str(exc).splitlines()[0]
            continue
        schedules[day] = schedule

    if not schedules:
        raise SolverError(
            f"No day could be solved for strategy '{cfg.name}'. "
            f"First failure: {next(iter(failures.values()), 'unknown')}"
        )

    summary = pd.DataFrame(
        {"date": [d.date() for d in schedules], "solved": True}
    )
    summary.attrs["failures"] = failures
    return summary, schedules
