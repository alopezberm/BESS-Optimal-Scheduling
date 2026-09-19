"""Solver-agnostic replica of ``model.build_bess_lp_extended``, built with PuLP.

Gurobi needs a licence (even the free academic one requires renewal and is tied to a
university network/account), which is a poor fit for a public portfolio repo that other
people may want to actually run. This module reproduces the *exact same* extended LP
using PuLP with the CBC solver it ships with, so anyone can clone the repo and get
Gurobi-identical results with zero licence setup. Used in the notebook purely as a
reproducibility check, not as the primary modelling path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from .model import BatteryParams


def solve_bess_lp_extended_opensource(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    dt: float,
    battery: BatteryParams,
    degradation_eur_per_mwh_throughput: float = 0.0,
    allow_direct_pv_sale: bool = True,
) -> tuple[float, pd.DataFrame]:
    t_steps = len(prices)
    prob = pulp.LpProblem("bess_extended_opensource", pulp.LpMaximize)

    p_ch = pulp.LpVariable.dicts("Pch", range(t_steps), lowBound=0, upBound=battery.p_ch_max_mw)
    p_dis = pulp.LpVariable.dicts("Pdis", range(t_steps), lowBound=0, upBound=battery.p_dis_max_mw)
    energy = pulp.LpVariable.dicts("E", range(t_steps + 1), lowBound=0, upBound=battery.e_max_mwh)

    pv_ub = float(pv.max()) if len(pv) else 0.0
    ub_direct = pv_ub if allow_direct_pv_sale else 0.0
    p_pv_direct = pulp.LpVariable.dicts("Ppv_direct", range(t_steps), lowBound=0, upBound=ub_direct)
    p_pv_curt = pulp.LpVariable.dicts("Ppv_curt", range(t_steps), lowBound=0, upBound=pv_ub)
    p_pv_to_batt = pulp.LpVariable.dicts("Ppv_to_batt", range(t_steps), lowBound=0, upBound=pv_ub)

    prob += energy[0] == battery.initial_energy()
    for t in range(t_steps):
        prob += (
            energy[t + 1]
            == energy[t] + battery.eta_ch * (p_ch[t] + p_pv_to_batt[t]) * dt - (p_dis[t] / battery.eta_dis) * dt
        )
        prob += p_pv_direct[t] + p_pv_to_batt[t] + p_pv_curt[t] == pv[t]

    market_revenue = pulp.lpSum(prices[t] * (p_dis[t] + p_pv_direct[t] - p_ch[t]) * dt for t in range(t_steps))
    degradation_cost = degradation_eur_per_mwh_throughput * pulp.lpSum(
        (p_ch[t] + p_dis[t]) * dt for t in range(t_steps)
    )
    prob += market_revenue - degradation_cost

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Open-source solve did not reach optimality: {pulp.LpStatus[prob.status]}")

    schedule = pd.DataFrame(
        {
            "time": index,
            "price": prices,
            "P_pv": pv,
            "P_ch": [p_ch[t].value() for t in range(t_steps)],
            "P_dis": [p_dis[t].value() for t in range(t_steps)],
            "E": [energy[t].value() for t in range(t_steps)],
            "E_end": [energy[t + 1].value() for t in range(t_steps)],
            "P_pv_direct": [p_pv_direct[t].value() for t in range(t_steps)],
            "P_pv_curt": [p_pv_curt[t].value() for t in range(t_steps)],
            "P_pv_to_batt": [p_pv_to_batt[t].value() for t in range(t_steps)],
        }
    )
    return pulp.value(prob.objective), schedule
