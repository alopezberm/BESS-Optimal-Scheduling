"""Optimisation models for the BESS day-ahead scheduling problem (Gurobi).

Three formulations, kept separate on purpose so the notebook can show *why* each term is
added rather than presenting a finished model as a given:

``build_bess_lp``
    The course baseline, reproduced faithfully and deliberately left unchanged: pure price
    arbitrage, all PV forced through the battery, state of charge free between 0 and the
    nameplate energy. It exists as the reference every extension is measured against.

``build_bess_lp_extended``
    The formulation used for the rest of the project. It adds, each individually switchable
    from the configuration so its contribution can be isolated: direct PV sale and
    curtailment, a usable SoC window, a terminal-SoC condition, the shared POI connection
    limit, network tariffs, self-discharge and station-service load, a throughput-based
    degradation cost, and an optional non-simultaneity restriction.

``build_bess_lp_stochastic``
    The extended formulation replicated over scenarios, with a single here-and-now market
    schedule shared across them - the structure of an actual day-ahead bid.

A note on what stays linear
---------------------------
Every term above is linear in the decision variables, so the problem remains an LP and is
solved to global optimality in milliseconds. That is not an accident but a modelling
discipline: convexity is what makes the dual variables meaningful (see
``soc_shadow_price`` below), and what lets a 30-day backtest or a scenario sweep run
interactively. The one exception is ``forbid_simultaneous_ch_dis``, which introduces
binaries and turns the problem into a MILP; the notebook uses it to *test* whether the
restriction ever binds rather than as a default.

Dual variables and the trader's reading of the solution
-------------------------------------------------------
The schedule returned by the extended model carries a ``soc_shadow_price`` column: the dual
variable of the energy-balance constraint, in EUR/MWh. It is the marginal value of one extra
MWh stored in the battery at that instant - the storage analogue of a hydro reservoir's water
value. Stationarity of the Lagrangian gives, for an interval where discharging is interior
(strictly between its bounds),

    lambda_t = eta_dis * (price_t - tariff_dis - c_degr)

and, where charging is interior,

    lambda_t = (price_t + tariff_ch + c_degr) / eta_ch

Eliminating lambda between the two recovers exactly the break-even spread implemented in
``ProjectConfig.breakeven_spread_eur_per_mwh``. In other words the LP is not a black box:
it is executing a price rule that can be written in one line, and the notebook verifies the
two numerically agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from .config import BatteryParams, ProjectConfig

# Re-exported for convenience: older code and the notebook import BatteryParams from here.
__all__ = [
    "BatteryParams",
    "build_bess_lp",
    "build_bess_lp_extended",
    "build_bess_lp_stochastic",
    "evaluate_fixed_schedule",
    "SolverError",
]


class SolverError(RuntimeError):
    """Raised when a model does not solve to optimality.

    Infeasibility is a normal event while tuning parameters - demanding the battery return
    to 50% SoC by midnight while forbidding grid charging on an overcast day has no solution -
    so the message names the conflicting constraints rather than reporting a status code.
    """


def _check_status(model: gp.Model, diagnose: bool = True) -> None:
    if model.Status == GRB.OPTIMAL:
        return
    status_name = {
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
        GRB.INF_OR_UNBD: "infeasible or unbounded",
        GRB.TIME_LIMIT: "time limit reached",
    }.get(model.Status, f"status code {model.Status}")

    detail = ""
    if model.Status == GRB.INFEASIBLE and diagnose:
        # The irreducible inconsistent subsystem is the smallest set of constraints that
        # cannot all hold at once: exactly the list of assumptions to revisit.
        try:
            model.computeIIS()
            culprits = sorted({c.ConstrName.split("[")[0] for c in model.getConstrs() if c.IISConstr})
            if culprits:
                detail = (
                    "\nThe conflict involves these constraint families: "
                    + ", ".join(culprits)
                    + ".\nTypical causes: a terminal-SoC target the battery cannot reach with "
                    "grid charging disabled, a POI limit below the PV that must be exported "
                    "while curtailment is forbidden, or an SoC window that excludes the "
                    "initial state."
                )
        except gp.GurobiError:  # pragma: no cover - IIS unavailable under some licences
            pass

    raise SolverError(f"Model '{model.ModelName}' did not solve to optimality ({status_name}).{detail}")


def _schedule_frame(index, prices, pv, pch, pdis, energy, extra: dict | None = None) -> pd.DataFrame:
    """Assemble the tidy per-interval result table shared by every formulation.

    ``E`` is the state of charge at the *start* of each interval and ``E_end`` at its end, so
    a row is self-contained: the energy balance of interval ``t`` can be checked from row ``t``
    alone, which is what makes the schedule auditable by eye.
    """
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


# --------------------------------------------------------------------------------------
# 1. Baseline - the course formulation, preserved
# --------------------------------------------------------------------------------------
def build_bess_lp(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    dt: float,
    battery: BatteryParams,
    verbose: bool = False,
) -> tuple[gp.Model, pd.DataFrame]:
    """Baseline model: the course formulation, reproduced without additions.

    Deliberately kept simple and *not* refactored to use the full configuration object, so
    that it remains a literal transcription of the exercise being extended:

        max  sum_t  price_t * (P_dis_t - P_ch_t) * dt
        s.t. E_{t+1} = E_t + eta_ch * (P_ch_t + P_pv_t) * dt - P_dis_t / eta_dis * dt
             0 <= P_ch_t <= P_ch_max,   0 <= P_dis_t <= P_dis_max
             0 <= E_t <= E_nominal
             E_0 = E_init

    Two properties of this formulation are worth stating explicitly, because the extended
    model exists to address them:

    * All PV is forced through the battery. The plant therefore pays a round-trip loss on
      every photon even when the battery is full and the price is high, and it has no way to
      curtail. It also means PV revenue does not appear in the objective at all - PV enters
      only as free energy injected into storage.
    * The state of charge is free over the entire nameplate range and free at the end of the
      horizon, so the optimiser will empty the battery in the final intervals. Stored energy
      has no value after T, which is true inside the model and false in operation.
    """
    t_steps = len(prices)
    model = gp.Model("bess_baseline")
    model.Params.OutputFlag = int(verbose)

    p_ch = model.addVars(t_steps, lb=0.0, ub=battery.p_ch_max_mw, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=battery.p_dis_max_mw, name="Pdis")
    energy = model.addVars(t_steps + 1, lb=0.0, ub=battery.e_nominal_mwh, name="E")

    model.addConstr(energy[0] == battery.e_init_mwh, name="E_init")
    model.addConstrs(
        (
            energy[t + 1]
            == energy[t]
            + battery.eta_ch * (p_ch[t] + pv[t]) * dt
            - (p_dis[t] / battery.eta_dis) * dt
            for t in range(t_steps)
        ),
        name="energy_balance",
    )

    model.setObjective(
        gp.quicksum(prices[t] * (p_dis[t] - p_ch[t]) * dt for t in range(t_steps)),
        GRB.MAXIMIZE,
    )
    model.optimize()
    _check_status(model)

    schedule = _schedule_frame(
        index,
        prices,
        pv,
        [p_ch[t].X for t in range(t_steps)],
        [p_dis[t].X for t in range(t_steps)],
        [energy[t].X for t in range(t_steps + 1)],
    )
    return model, schedule


# --------------------------------------------------------------------------------------
# 2. Extended - the working model
# --------------------------------------------------------------------------------------
def build_bess_lp_extended(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    cfg: ProjectConfig,
    verbose: bool = False,
) -> tuple[gp.Model, pd.DataFrame]:
    """The formulation used throughout the project, driven entirely by ``cfg``.

    Decision variables per interval t
    ---------------------------------
    ``P_ch``          grid power drawn to charge                          [MW]
    ``P_dis``         battery power injected into the grid                [MW]
    ``P_pv_direct``   PV sold straight to the grid, bypassing storage     [MW]
    ``P_pv_to_batt``  PV routed into the battery                          [MW]
    ``P_pv_curt``     PV spilled                                          [MW]
    ``E``             energy stored at the start of the interval          [MWh]

    Objective
    ---------
        max  sum_t [ price_t * (P_dis + P_pv_direct - P_ch - aux)
                     - tariff_ch * P_ch - tariff_dis * P_dis
                     - c_degr * (P_ch + P_pv_to_batt + P_dis) ] * dt
             ( + terminal_value * E_T , when soc_terminal_mode == 'valued' )

    Three details in that objective are easy to get wrong and are worth naming:

    * **PV charging wears the battery too.** The degradation term counts ``P_pv_to_batt``
      alongside grid charging. Charging from a co-located panel is free in energy terms, not
      in asset terms, and a model that omits this systematically over-cycles hybrid plants.
    * **Tariffs are charged per MWh through the meter**, so they sit outside the price and
      widen the spread required to trade - see ``breakeven_spread_eur_per_mwh``.
    * **Station-service load is a constant.** It shifts the P&L without changing the optimal
      schedule, unless a POI limit binds. It is included so that reported margins are net,
      not because it alters the operating decision.

    Constraints
    -----------
    Energy balance with self-discharge, the PV split, the *inverter* power limits (which
    apply to total charging power, grid plus PV, because both pass through the same power
    conversion system), the POI net-flow limits, the SoC window, and the terminal condition.
    """
    b, plant, market = cfg.battery, cfg.plant, cfg.market
    dt = cfg.simulation.dt_hours
    c_degr = cfg.degradation_cost_eur_per_mwh_throughput()
    t_steps = len(prices)

    model = gp.Model("bess_extended")
    model.Params.OutputFlag = int(verbose)

    # -- Variables ---------------------------------------------------------------------
    p_ch_ub = b.p_ch_max_mw if plant.allow_grid_charging else 0.0
    p_ch = model.addVars(t_steps, lb=0.0, ub=p_ch_ub, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=b.p_dis_max_mw, name="Pdis")
    energy = model.addVars(t_steps + 1, lb=b.e_min_mwh, ub=b.e_max_mwh, name="E")

    # PV routing. Each branch is bounded by the PV actually available in that interval, which
    # keeps the LP tight (a tighter relaxation solves faster and reads more clearly) and makes
    # a disabled branch literally a zero variable rather than a constraint bolted on afterwards.
    pv_arr = np.asarray(pv, dtype=float)
    direct_ub = pv_arr if plant.allow_direct_pv_sale else np.zeros_like(pv_arr)
    curt_ub = pv_arr if plant.allow_pv_curtailment else np.zeros_like(pv_arr)
    p_pv_direct = model.addVars(t_steps, lb=0.0, ub=list(direct_ub), name="Ppv_direct")
    p_pv_curt = model.addVars(t_steps, lb=0.0, ub=list(curt_ub), name="Ppv_curt")
    p_pv_batt = model.addVars(t_steps, lb=0.0, ub=list(pv_arr), name="Ppv_to_batt")

    # -- Energy balance ----------------------------------------------------------------
    retention = 1.0 - b.self_discharge_frac_per_hour() * dt
    model.addConstr(energy[0] == b.e_init_mwh, name="E_init")
    balance = model.addConstrs(
        (
            energy[t + 1]
            == retention * energy[t]
            + b.eta_ch * (p_ch[t] + p_pv_batt[t]) * dt
            - (p_dis[t] / b.eta_dis) * dt
            for t in range(t_steps)
        ),
        name="energy_balance",
    )

    # -- PV must go somewhere ----------------------------------------------------------
    model.addConstrs(
        (p_pv_direct[t] + p_pv_batt[t] + p_pv_curt[t] == pv_arr[t] for t in range(t_steps)),
        name="pv_split",
    )

    # -- Inverter limit applies to total charging power, grid plus PV ------------------
    model.addConstrs(
        (p_ch[t] + p_pv_batt[t] <= b.p_ch_max_mw for t in range(t_steps)),
        name="charge_power_limit",
    )

    # -- Shared point of interconnection -----------------------------------------------
    # Net flow, positive when exporting. PV routed into the battery never crosses the meter,
    # which is precisely the congestion relief a co-located battery provides.
    net_export = {
        t: p_dis[t] + p_pv_direct[t] - p_ch[t] - b.aux_load_mw for t in range(t_steps)
    }
    if plant.poi_export_limit_mw is not None:
        model.addConstrs(
            (net_export[t] <= plant.poi_export_limit_mw for t in range(t_steps)),
            name="poi_export",
        )
    if plant.poi_import_limit_mw is not None:
        model.addConstrs(
            (net_export[t] >= -plant.poi_import_limit_mw for t in range(t_steps)),
            name="poi_import",
        )

    # -- Terminal condition ------------------------------------------------------------
    # Without one of these the optimiser empties the battery before T, because stored energy
    # is worthless at the end of a finite horizon. That is an artefact of the horizon, not a
    # property of the asset, and it inflates single-day profit.
    terminal_target = b.terminal_energy_mwh()
    if b.soc_terminal_mode == "equal_to_initial":
        model.addConstr(energy[t_steps] == terminal_target, name="E_terminal")
    elif b.soc_terminal_mode == "at_least":
        model.addConstr(energy[t_steps] >= terminal_target, name="E_terminal")

    # -- Optional non-simultaneity (LP becomes MILP) -----------------------------------
    if b.forbid_simultaneous_ch_dis:
        mode = model.addVars(t_steps, vtype=GRB.BINARY, name="is_charging")
        model.addConstrs(
            (p_ch[t] + p_pv_batt[t] <= b.p_ch_max_mw * mode[t] for t in range(t_steps)),
            name="excl_charge",
        )
        model.addConstrs(
            (p_dis[t] <= b.p_dis_max_mw * (1 - mode[t]) for t in range(t_steps)),
            name="excl_discharge",
        )

    # -- Objective ---------------------------------------------------------------------
    market_revenue = gp.quicksum(prices[t] * net_export[t] * dt for t in range(t_steps))
    tariff_cost = gp.quicksum(
        (
            market.grid_tariff_charge_eur_per_mwh * p_ch[t]
            + market.grid_tariff_discharge_eur_per_mwh * p_dis[t]
        )
        * dt
        for t in range(t_steps)
    )
    degradation_cost = c_degr * gp.quicksum(
        (p_ch[t] + p_pv_batt[t] + p_dis[t]) * dt for t in range(t_steps)
    )
    objective = market_revenue - tariff_cost - degradation_cost
    if b.soc_terminal_mode == "valued":
        objective = objective + b.terminal_energy_value_eur_per_mwh * energy[t_steps]

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    _check_status(model)

    # -- Extract ------------------------------------------------------------------------
    extra = {
        "P_pv_direct": [p_pv_direct[t].X for t in range(t_steps)],
        "P_pv_curt": [p_pv_curt[t].X for t in range(t_steps)],
        "P_pv_to_batt": [p_pv_batt[t].X for t in range(t_steps)],
        "net_export": [
            p_dis[t].X + p_pv_direct[t].X - p_ch[t].X - b.aux_load_mw for t in range(t_steps)
        ],
    }
    # Dual variables exist only for a continuous problem; a MILP has no meaningful shadow price.
    if not b.forbid_simultaneous_ch_dis:
        extra["soc_shadow_price"] = [balance[t].Pi for t in range(t_steps)]

    schedule = _schedule_frame(
        index,
        prices,
        pv_arr,
        [p_ch[t].X for t in range(t_steps)],
        [p_dis[t].X for t in range(t_steps)],
        [energy[t].X for t in range(t_steps + 1)],
        extra=extra,
    )
    schedule["soc"] = schedule["E"] / b.e_nominal_mwh
    return model, schedule


def evaluate_fixed_schedule(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    pv: np.ndarray,
    cfg: ProjectConfig,
    p_ch_fixed: np.ndarray,
    p_dis_fixed: np.ndarray,
    verbose: bool = False,
) -> tuple[float, pd.DataFrame] | tuple[None, None]:
    """Profit of an already-committed market schedule when a different day actually happens.

    The market commitment is frozen; only the genuinely real-time decisions - how PV is routed
    and, through it, the state of charge - are re-optimised. This is what settlement of a
    day-ahead bid looks like when the forecast was wrong.

    Returns ``(None, None)`` when the committed schedule cannot be honoured under the realised
    conditions. That outcome is information, not a failure: it says the commitment was
    infeasible in that state of the world, which is exactly the risk a here-and-now decision
    carries and the reason the stochastic formulation exists.
    """
    b, plant, market = cfg.battery, cfg.plant, cfg.market
    dt = cfg.simulation.dt_hours
    c_degr = cfg.degradation_cost_eur_per_mwh_throughput()
    pv_arr = np.asarray(pv, dtype=float)
    t_steps = len(prices)

    model = gp.Model("bess_fixed_schedule")
    model.Params.OutputFlag = int(verbose)

    p_ch = [float(v) for v in p_ch_fixed]
    p_dis = [float(v) for v in p_dis_fixed]
    energy = model.addVars(t_steps + 1, lb=b.e_min_mwh, ub=b.e_max_mwh, name="E")
    direct_ub = pv_arr if plant.allow_direct_pv_sale else np.zeros_like(pv_arr)
    curt_ub = pv_arr if plant.allow_pv_curtailment else np.zeros_like(pv_arr)
    p_pv_direct = model.addVars(t_steps, lb=0.0, ub=list(direct_ub), name="Ppv_direct")
    p_pv_curt = model.addVars(t_steps, lb=0.0, ub=list(curt_ub), name="Ppv_curt")
    p_pv_batt = model.addVars(t_steps, lb=0.0, ub=list(pv_arr), name="Ppv_to_batt")

    retention = 1.0 - b.self_discharge_frac_per_hour() * dt
    model.addConstr(energy[0] == b.e_init_mwh, name="E_init")
    model.addConstrs(
        (
            energy[t + 1]
            == retention * energy[t]
            + b.eta_ch * (p_ch[t] + p_pv_batt[t]) * dt
            - (p_dis[t] / b.eta_dis) * dt
            for t in range(t_steps)
        ),
        name="energy_balance",
    )
    model.addConstrs(
        (p_pv_direct[t] + p_pv_batt[t] + p_pv_curt[t] == pv_arr[t] for t in range(t_steps)),
        name="pv_split",
    )
    model.addConstrs(
        (p_ch[t] + p_pv_batt[t] <= b.p_ch_max_mw for t in range(t_steps)), name="charge_power_limit"
    )
    if plant.poi_export_limit_mw is not None:
        model.addConstrs(
            (
                p_dis[t] + p_pv_direct[t] - p_ch[t] - b.aux_load_mw <= plant.poi_export_limit_mw
                for t in range(t_steps)
            ),
            name="poi_export",
        )
    terminal_target = b.terminal_energy_mwh()
    if b.soc_terminal_mode == "equal_to_initial":
        model.addConstr(energy[t_steps] == terminal_target, name="E_terminal")
    elif b.soc_terminal_mode == "at_least":
        model.addConstr(energy[t_steps] >= terminal_target, name="E_terminal")

    model.setObjective(
        gp.quicksum(
            (
                prices[t] * (p_dis[t] + p_pv_direct[t] - p_ch[t] - b.aux_load_mw)
                - market.grid_tariff_charge_eur_per_mwh * p_ch[t]
                - market.grid_tariff_discharge_eur_per_mwh * p_dis[t]
                - c_degr * (p_ch[t] + p_pv_batt[t] + p_dis[t])
            )
            * dt
            for t in range(t_steps)
        ),
        GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None, None

    schedule = _schedule_frame(
        index,
        prices,
        pv_arr,
        p_ch,
        p_dis,
        [energy[t].X for t in range(t_steps + 1)],
        extra={
            "P_pv_direct": [p_pv_direct[t].X for t in range(t_steps)],
            "P_pv_curt": [p_pv_curt[t].X for t in range(t_steps)],
            "P_pv_to_batt": [p_pv_batt[t].X for t in range(t_steps)],
        },
    )
    schedule["soc"] = schedule["E"] / b.e_nominal_mwh
    return float(model.ObjVal), schedule


# --------------------------------------------------------------------------------------
# 3. Stochastic - one commitment, many futures
# --------------------------------------------------------------------------------------
def build_bess_lp_stochastic(
    index: pd.DatetimeIndex,
    price_scenarios: dict[int, dict],
    pv_scenarios: dict[int, np.ndarray],
    cfg: ProjectConfig,
    verbose: bool = False,
) -> tuple[gp.Model, dict[int, pd.DataFrame], pd.DataFrame]:
    """Two-stage stochastic version: a single here-and-now schedule across all scenarios.

    Stage structure, which is the whole point of the formulation:

    * **First stage (here-and-now).** ``P_ch`` and ``P_dis`` are shared by every scenario.
      This is what a day-ahead bid physically is: one quantity schedule committed before the
      uncertainty resolves, which cannot be conditioned on which future materialises.
    * **Second stage (recourse).** The state of charge and PV curtailment are scenario-indexed.
      The SoC necessarily is, since it is driven by that scenario's own PV realisation.
      Curtailment is a real-time action taken after the PV is observed, so modelling it as
      recourse is not a convenience but the correct representation - and without it a single
      shared charge/discharge trajectory is easily infeasible across scenarios whose PV
      differs enough, because no fixed discharge profile can make room for every outcome.

    The objective maximises probability-weighted expected profit, sum_s prob_s * profit_s.
    """
    b, plant, market = cfg.battery, cfg.plant, cfg.market
    dt = cfg.simulation.dt_hours
    c_degr = cfg.degradation_cost_eur_per_mwh_throughput()
    t_steps = len(index)

    model = gp.Model("bess_stochastic")
    model.Params.OutputFlag = int(verbose)

    p_ch_ub = b.p_ch_max_mw if plant.allow_grid_charging else 0.0
    p_ch = model.addVars(t_steps, lb=0.0, ub=p_ch_ub, name="Pch")
    p_dis = model.addVars(t_steps, lb=0.0, ub=b.p_dis_max_mw, name="Pdis")

    retention = 1.0 - b.self_discharge_frac_per_hour() * dt
    terminal_target = b.terminal_energy_mwh()

    energy: dict[int, gp.tupledict] = {}
    p_pv_curt: dict[int, gp.tupledict] = {}
    p_pv_batt: dict[int, gp.tupledict] = {}
    p_pv_direct: dict[int, gp.tupledict] = {}

    for s in price_scenarios:
        pv_s = np.asarray(pv_scenarios[s], dtype=float)
        direct_ub = pv_s if plant.allow_direct_pv_sale else np.zeros_like(pv_s)
        curt_ub = pv_s if plant.allow_pv_curtailment else np.zeros_like(pv_s)

        p_pv_direct[s] = model.addVars(t_steps, lb=0.0, ub=list(direct_ub), name=f"Ppv_direct_s{s}")
        p_pv_curt[s] = model.addVars(t_steps, lb=0.0, ub=list(curt_ub), name=f"Ppv_curt_s{s}")
        p_pv_batt[s] = model.addVars(t_steps, lb=0.0, ub=list(pv_s), name=f"Ppv_to_batt_s{s}")
        energy[s] = model.addVars(t_steps + 1, lb=b.e_min_mwh, ub=b.e_max_mwh, name=f"E_s{s}")

        model.addConstr(energy[s][0] == b.e_init_mwh, name=f"E_init_s{s}")
        model.addConstrs(
            (
                energy[s][t + 1]
                == retention * energy[s][t]
                + b.eta_ch * (p_ch[t] + p_pv_batt[s][t]) * dt
                - (p_dis[t] / b.eta_dis) * dt
                for t in range(t_steps)
            ),
            name=f"energy_balance_s{s}",
        )
        model.addConstrs(
            (
                p_pv_direct[s][t] + p_pv_batt[s][t] + p_pv_curt[s][t] == pv_s[t]
                for t in range(t_steps)
            ),
            name=f"pv_split_s{s}",
        )
        model.addConstrs(
            (p_ch[t] + p_pv_batt[s][t] <= b.p_ch_max_mw for t in range(t_steps)),
            name=f"charge_power_limit_s{s}",
        )
        if plant.poi_export_limit_mw is not None:
            model.addConstrs(
                (
                    p_dis[t] + p_pv_direct[s][t] - p_ch[t] - b.aux_load_mw
                    <= plant.poi_export_limit_mw
                    for t in range(t_steps)
                ),
                name=f"poi_export_s{s}",
            )
        if plant.poi_import_limit_mw is not None:
            model.addConstrs(
                (
                    p_dis[t] + p_pv_direct[s][t] - p_ch[t] - b.aux_load_mw
                    >= -plant.poi_import_limit_mw
                    for t in range(t_steps)
                ),
                name=f"poi_import_s{s}",
            )
        if b.soc_terminal_mode == "equal_to_initial":
            model.addConstr(energy[s][t_steps] == terminal_target, name=f"E_terminal_s{s}")
        elif b.soc_terminal_mode == "at_least":
            model.addConstr(energy[s][t_steps] >= terminal_target, name=f"E_terminal_s{s}")

    def _scenario_profit(s: int) -> gp.LinExpr:
        prices_s = price_scenarios[s]["prices"]
        revenue = gp.quicksum(
            prices_s[t]
            * (p_dis[t] + p_pv_direct[s][t] - p_ch[t] - b.aux_load_mw)
            * dt
            for t in range(t_steps)
        )
        tariffs = gp.quicksum(
            (
                market.grid_tariff_charge_eur_per_mwh * p_ch[t]
                + market.grid_tariff_discharge_eur_per_mwh * p_dis[t]
            )
            * dt
            for t in range(t_steps)
        )
        degradation = c_degr * gp.quicksum(
            (p_ch[t] + p_pv_batt[s][t] + p_dis[t]) * dt for t in range(t_steps)
        )
        profit = revenue - tariffs - degradation
        if b.soc_terminal_mode == "valued":
            profit = profit + b.terminal_energy_value_eur_per_mwh * energy[s][t_steps]
        return profit

    model.setObjective(
        gp.quicksum(price_scenarios[s]["probability"] * _scenario_profit(s) for s in price_scenarios),
        GRB.MAXIMIZE,
    )
    model.optimize()
    _check_status(model)

    schedules = {}
    for s in price_scenarios:
        schedules[s] = _schedule_frame(
            index,
            price_scenarios[s]["prices"],
            np.asarray(pv_scenarios[s], dtype=float),
            [p_ch[t].X for t in range(t_steps)],
            [p_dis[t].X for t in range(t_steps)],
            [energy[s][t].X for t in range(t_steps + 1)],
            extra={
                "P_pv_direct": [p_pv_direct[s][t].X for t in range(t_steps)],
                "P_pv_curt": [p_pv_curt[s][t].X for t in range(t_steps)],
                "P_pv_to_batt": [p_pv_batt[s][t].X for t in range(t_steps)],
                "net_export": [
                    p_dis[t].X + p_pv_direct[s][t].X - p_ch[t].X - b.aux_load_mw
                    for t in range(t_steps)
                ],
            },
        )
        schedules[s]["soc"] = schedules[s]["E"] / b.e_nominal_mwh

    shared = pd.DataFrame(
        {
            "time": index,
            "P_ch": [p_ch[t].X for t in range(t_steps)],
            "P_dis": [p_dis[t].X for t in range(t_steps)],
        }
    )
    return model, schedules, shared
