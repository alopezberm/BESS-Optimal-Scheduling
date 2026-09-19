"""Gurobi formulations of the BESS day-ahead scheduling problem.

Three variants, from simplest to richest, so the notebook can show *why* each extension
is added rather than presenting the final model as a given:

- ``build_bess_lp``            baseline: faithful reproduction of the course formulation
                                (all PV is routed through the battery; no direct PV sale).
- ``build_bess_lp_extended``   + direct PV-to-grid sale, PV curtailment, and a linear
                                throughput-based degradation cost in the objective.
- ``build_bess_lp_stochastic`` the ``build_bess_lp`` formulation extended to several
                                price/PV scenarios with a single here-and-now schedule
                                (Pch, Pdis fixed across scenarios) maximizing expected profit.

All variants share the same battery-parameter dataclass and return a tidy per-interval
schedule ``DataFrame`` plus the solved ``gurobipy.Model``, so downstream code (backtest,
degradation, economics) does not need to know which variant produced the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


@dataclass
class BatteryParams:
    """Physical and economic parameters of the BESS."""

    e_max_mwh: float = 4.0
    p_ch_max_mw: float = 2.0
    p_dis_max_mw: float = 2.0
    eta_ch: float = 0.97
    eta_dis: float = 0.97
    e_init_mwh: float | None = None  # defaults to 50% SoC if left as None

    def initial_energy(self) -> float:
        return self.e_max_mwh / 2 if self.e_init_mwh is None else self.e_init_mwh


def _schedule_frame(index, prices, pv, pch, pdis, energy, extra: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": index,
            "price": prices,
            "P_pv": pv,
            "P_ch": pch,
            "P_dis": pdis,
            "E": energy[:-1],
            "E_end": energy[1:],
        }
    )
    if extra:
        for key, values in extra.items():
            frame[key] = values
    return frame


def build_bess_lp(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    dt: float,
    battery: BatteryParams,
    verbose: bool = False,
) -> tuple[gp.Model, pd.DataFrame]:
    """Baseline model: exactly the course formulation, all PV forced through the battery."""
    t_steps = len(prices)
    model = gp.Model("bess_baseline")
    model.Params.OutputFlag = int(verbose)

    p_ch = model.addVars(t_steps, lb=0.0, ub=battery.p_ch_max_mw, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=battery.p_dis_max_mw, name="Pdis")
    energy = model.addVars(t_steps + 1, lb=0.0, ub=battery.e_max_mwh, name="E")

    model.addConstr(energy[0] == battery.initial_energy(), name="E_init")
    model.addConstrs(
        (
            energy[t + 1]
            == energy[t] + battery.eta_ch * (p_ch[t] + pv[t]) * dt - (p_dis[t] / battery.eta_dis) * dt
            for t in range(t_steps)
        ),
        name="energy_balance",
    )

    model.setObjective(
        gp.quicksum(prices[t] * (p_dis[t] - p_ch[t]) * dt for t in range(t_steps)),
        GRB.MAXIMIZE,
    )
    model.optimize()

    schedule = _schedule_frame(
        index,
        prices,
        pv,
        [p_ch[t].X for t in range(t_steps)],
        [p_dis[t].X for t in range(t_steps)],
        [energy[t].X for t in range(t_steps + 1)],
    )
    return model, schedule


def build_bess_lp_extended(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    dt: float,
    battery: BatteryParams,
    degradation_eur_per_mwh_throughput: float = 0.0,
    allow_direct_pv_sale: bool = True,
    verbose: bool = False,
) -> tuple[gp.Model, pd.DataFrame]:
    """Extended model used throughout the rest of the project.

    On top of the baseline it adds:
      1. ``P_pv_direct``: PV can be sold straight to the grid, bypassing the battery
         (and its round-trip losses) instead of being forced through it.
      2. ``P_pv_curt``: PV can be curtailed when neither selling nor storing it is
         worthwhile (e.g. very negative prices and a full battery).
      3. A linear degradation cost, ``degradation_eur_per_mwh_throughput`` EUR per MWh of
         battery throughput (charge + discharge), subtracted from the market profit. Left
         at 0 this collapses back to a pure arbitrage objective; see ``degradation.py`` for
         where this coefficient comes from and its limits (it is throughput-based, not yet
         depth-of-discharge aware - that is a documented future extension).
    """
    t_steps = len(prices)
    model = gp.Model("bess_extended")
    model.Params.OutputFlag = int(verbose)

    p_ch = model.addVars(t_steps, lb=0.0, ub=battery.p_ch_max_mw, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=battery.p_dis_max_mw, name="Pdis")
    energy = model.addVars(t_steps + 1, lb=0.0, ub=battery.e_max_mwh, name="E")

    pv_ub = pv.max() if len(pv) else 0.0
    p_pv_direct = model.addVars(t_steps, lb=0.0, ub=pv_ub, name="Ppv_direct")
    p_pv_curt = model.addVars(t_steps, lb=0.0, ub=pv_ub, name="Ppv_curt")
    p_pv_to_batt = model.addVars(t_steps, lb=0.0, ub=pv_ub, name="Ppv_to_batt")

    model.addConstr(energy[0] == battery.initial_energy(), name="E_init")
    model.addConstrs(
        (
            energy[t + 1]
            == energy[t] + battery.eta_ch * (p_ch[t] + p_pv_to_batt[t]) * dt - (p_dis[t] / battery.eta_dis) * dt
            for t in range(t_steps)
        ),
        name="energy_balance",
    )
    model.addConstrs(
        (p_pv_direct[t] + p_pv_to_batt[t] + p_pv_curt[t] == pv[t] for t in range(t_steps)),
        name="pv_split",
    )
    if not allow_direct_pv_sale:
        model.addConstrs((p_pv_direct[t] == 0 for t in range(t_steps)), name="no_direct_pv")

    market_revenue = gp.quicksum(
        prices[t] * (p_dis[t] + p_pv_direct[t] - p_ch[t]) * dt for t in range(t_steps)
    )
    degradation_cost = degradation_eur_per_mwh_throughput * gp.quicksum(
        (p_ch[t] + p_dis[t]) * dt for t in range(t_steps)
    )
    model.setObjective(market_revenue - degradation_cost, GRB.MAXIMIZE)
    model.optimize()

    schedule = _schedule_frame(
        index,
        prices,
        pv,
        [p_ch[t].X for t in range(t_steps)],
        [p_dis[t].X for t in range(t_steps)],
        [energy[t].X for t in range(t_steps + 1)],
        extra={
            "P_pv_direct": [p_pv_direct[t].X for t in range(t_steps)],
            "P_pv_curt": [p_pv_curt[t].X for t in range(t_steps)],
            "P_pv_to_batt": [p_pv_to_batt[t].X for t in range(t_steps)],
        },
    )
    return model, schedule


def build_bess_lp_stochastic(
    index: pd.DatetimeIndex,
    scenarios: dict[int, dict],
    pv_scenarios: dict[int, np.ndarray],
    dt: float,
    battery: BatteryParams,
    verbose: bool = False,
) -> tuple[gp.Model, dict[int, pd.DataFrame], pd.DataFrame]:
    """Two-stage-flavoured stochastic version of the baseline model.

    The charge/discharge schedule (``Pch``, ``Pdis``) is a single here-and-now decision,
    exactly as a day-ahead bid must be: it cannot depend on which scenario materializes.
    The resulting state of charge, however, is necessarily scenario-indexed because it is
    driven by the scenario's own PV realization; so is PV curtailment (see below). The
    objective maximizes the probability-weighted expected profit across scenarios, i.e. the
    extension the course notebook leaves as an (unsolved) optional exercise.
    """
    t_steps = len(index)
    model = gp.Model("bess_stochastic")
    model.Params.OutputFlag = int(verbose)

    p_ch = model.addVars(t_steps, lb=0.0, ub=battery.p_ch_max_mw, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=battery.p_dis_max_mw, name="Pdis")

    # PV curtailment is scenario-indexed *recourse*: unlike Pch/Pdis (the day-ahead market
    # commitment, necessarily shared across scenarios), how much PV to curtail is a real-time
    # decision that can react to the PV actually realized. Without it, a single shared
    # charge/discharge trajectory can easily be infeasible across scenarios whose PV differs
    # enough (no amount of shared discharging can always make room for every scenario's PV).
    energy = {}
    p_pv_curt = {}
    for s in scenarios:
        pv_ub = float(pv_scenarios[s].max()) if len(pv_scenarios[s]) else 0.0
        p_pv_curt[s] = model.addVars(t_steps, lb=0.0, ub=pv_ub, name=f"Ppv_curt_s{s}")
        energy[s] = model.addVars(t_steps + 1, lb=0.0, ub=battery.e_max_mwh, name=f"E_s{s}")
        model.addConstr(energy[s][0] == battery.initial_energy(), name=f"E_init_s{s}")
        model.addConstrs(
            (
                energy[s][t + 1]
                == energy[s][t]
                + battery.eta_ch * (p_ch[t] + pv_scenarios[s][t] - p_pv_curt[s][t]) * dt
                - (p_dis[t] / battery.eta_dis) * dt
                for t in range(t_steps)
            ),
            name=f"energy_balance_s{s}",
        )

    expected_profit = gp.quicksum(
        scenarios[s]["probability"]
        * gp.quicksum(scenarios[s]["prices"][t] * (p_dis[t] - p_ch[t]) * dt for t in range(t_steps))
        for s in scenarios
    )
    model.setObjective(expected_profit, GRB.MAXIMIZE)
    model.optimize()

    schedules = {}
    for s in scenarios:
        schedules[s] = _schedule_frame(
            index,
            scenarios[s]["prices"],
            pv_scenarios[s],
            [p_ch[t].X for t in range(t_steps)],
            [p_dis[t].X for t in range(t_steps)],
            [energy[s][t].X for t in range(t_steps + 1)],
            extra={"P_pv_curt": [p_pv_curt[s][t].X for t in range(t_steps)]},
        )

    shared = pd.DataFrame(
        {
            "time": index,
            "P_ch": [p_ch[t].X for t in range(t_steps)],
            "P_dis": [p_dis[t].X for t in range(t_steps)],
        }
    )
    return model, schedules, shared
