"""The same extended model, written against PuLP and solved with CBC.

Purpose
-------
Gurobi is the right tool for this problem and the one the course uses, but it needs a
licence: the free academic one requires a university account and periodic renewal, and the
licence bundled with the pip package is size-limited. Neither is a reasonable prerequisite
for someone who wants to clone a public repository and check a result.

So the extended formulation is implemented a second time here, independently, against
`PuLP <https://coin-or.github.io/pulp/>`_ and the COIN-OR CBC solver it ships with. Anyone
can run it with nothing but ``pip install pulp``.

Why re-implementing it is worth the duplication
-----------------------------------------------
A second implementation is not only about licences. Two formulations written from the same
mathematics, solved by two unrelated solvers, agreeing to eight significant figures is
evidence that the objective encodes what the write-up claims it encodes. A single
implementation can only ever be self-consistent: it will happily return a confident optimum
for a model that quietly differs from the one described in the report. The notebook treats
the agreement between the two as a test, with an assertion, rather than as a remark.

The mirror has to be exact for that argument to hold, so every term in ``model.py`` appears
here - the SoC window, the terminal condition, the inverter limit on total charging power,
the POI limits, tariffs, self-discharge, station-service load and the degradation charge.
Anything omitted would turn a meaningful cross-check into a coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from .config import ProjectConfig


def solve_bess_lp_extended_opensource(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    cfg: ProjectConfig,
    verbose: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Solve the extended formulation with CBC; returns ``(objective, schedule)``.

    The returned schedule uses the same column names as ``model.build_bess_lp_extended``, so
    downstream analysis code cannot tell the two apart - which is the point.
    """
    b, plant, market = cfg.battery, cfg.plant, cfg.market
    dt = cfg.simulation.dt_hours
    c_degr = cfg.degradation_cost_eur_per_mwh_throughput()
    pv_arr = np.asarray(pv, dtype=float)
    T = len(prices)

    problem = pulp.LpProblem("bess_extended_opensource", pulp.LpMaximize)

    # -- Variables ---------------------------------------------------------------------
    p_ch_max = b.p_ch_max_mw if plant.allow_grid_charging else 0.0
    p_ch = [pulp.LpVariable(f"Pch_{t}", 0, p_ch_max) for t in range(T)]
    p_dis = [pulp.LpVariable(f"Pdis_{t}", 0, b.p_dis_max_mw) for t in range(T)]
    energy = [pulp.LpVariable(f"E_{t}", b.e_min_mwh, b.e_max_mwh) for t in range(T + 1)]

    pv_direct = [
        pulp.LpVariable(f"Ppv_direct_{t}", 0, pv_arr[t] if plant.allow_direct_pv_sale else 0.0)
        for t in range(T)
    ]
    pv_curt = [
        pulp.LpVariable(f"Ppv_curt_{t}", 0, pv_arr[t] if plant.allow_pv_curtailment else 0.0)
        for t in range(T)
    ]
    pv_batt = [pulp.LpVariable(f"Ppv_to_batt_{t}", 0, pv_arr[t]) for t in range(T)]

    # -- Constraints -------------------------------------------------------------------
    retention = 1.0 - b.self_discharge_frac_per_hour() * dt
    problem += energy[0] == b.e_init_mwh, "E_init"
    for t in range(T):
        problem += (
            energy[t + 1]
            == retention * energy[t]
            + b.eta_ch * (p_ch[t] + pv_batt[t]) * dt
            - (p_dis[t] / b.eta_dis) * dt
        ), f"energy_balance_{t}"
        problem += pv_direct[t] + pv_batt[t] + pv_curt[t] == pv_arr[t], f"pv_split_{t}"
        problem += p_ch[t] + pv_batt[t] <= b.p_ch_max_mw, f"charge_power_limit_{t}"

        net_export = p_dis[t] + pv_direct[t] - p_ch[t] - b.aux_load_mw
        if plant.poi_export_limit_mw is not None:
            problem += net_export <= plant.poi_export_limit_mw, f"poi_export_{t}"
        if plant.poi_import_limit_mw is not None:
            problem += net_export >= -plant.poi_import_limit_mw, f"poi_import_{t}"

    terminal_target = b.terminal_energy_mwh()
    if b.soc_terminal_mode == "equal_to_initial":
        problem += energy[T] == terminal_target, "E_terminal"
    elif b.soc_terminal_mode == "at_least":
        problem += energy[T] >= terminal_target, "E_terminal"

    if b.forbid_simultaneous_ch_dis:
        mode = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(T)]
        for t in range(T):
            problem += p_ch[t] + pv_batt[t] <= b.p_ch_max_mw * mode[t], f"excl_charge_{t}"
            problem += p_dis[t] <= b.p_dis_max_mw * (1 - mode[t]), f"excl_discharge_{t}"

    # -- Objective ---------------------------------------------------------------------
    objective = pulp.lpSum(
        (
            prices[t] * (p_dis[t] + pv_direct[t] - p_ch[t] - b.aux_load_mw)
            - market.grid_tariff_charge_eur_per_mwh * p_ch[t]
            - market.grid_tariff_discharge_eur_per_mwh * p_dis[t]
            - c_degr * (p_ch[t] + pv_batt[t] + p_dis[t])
        )
        * dt
        for t in range(T)
    )
    if b.soc_terminal_mode == "valued":
        objective = objective + b.terminal_energy_value_eur_per_mwh * energy[T]
    problem += objective

    status = problem.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"CBC did not reach optimality (status: {pulp.LpStatus[status]}). "
            "The same configuration should be tried with the Gurobi model, whose infeasibility "
            "diagnostics name the conflicting constraints."
        )

    value = lambda var: float(var.value())  # noqa: E731 - local shorthand, used once per column
    schedule = pd.DataFrame(
        {
            "time": index,
            "price": prices,
            "P_pv": pv_arr,
            "P_ch": [value(v) for v in p_ch],
            "P_dis": [value(v) for v in p_dis],
            "E": [value(v) for v in energy[:-1]],
            "E_end": [value(v) for v in energy[1:]],
            "P_pv_direct": [value(v) for v in pv_direct],
            "P_pv_curt": [value(v) for v in pv_curt],
            "P_pv_to_batt": [value(v) for v in pv_batt],
        }
    )
    schedule["net_export"] = (
        schedule["P_dis"] + schedule["P_pv_direct"] - schedule["P_ch"] - b.aux_load_mw
    )
    schedule["soc"] = schedule["E"] / b.e_nominal_mwh
    return float(pulp.value(problem.objective)), schedule
