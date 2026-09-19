"""Single source of truth for every tunable parameter in this project.

Why a configuration module instead of loose variables in the notebook?
----------------------------------------------------------------------
Three reasons, all of which matter for an academic exercise that is meant to be
*re-run under different assumptions*, not just read once:

1. **Every assumption is stated, with units.** A number like ``0.97`` is meaningless on
   its own. Each field below carries its unit, a typical utility-scale range, and a short
   note on *what changes in the result when it moves*. ``ProjectConfig.summary()`` renders
   all of that as a table, so a reader can audit the entire set of assumptions behind any
   figure in the notebook without reading solver code.

2. **Experiments become reproducible objects.** A "trading strategy" in this project is
   nothing more than one ``ProjectConfig``. Comparing strategies (``experiments.py``) is
   therefore comparing configurations on identical price and PV data, which is the only
   way a comparison isolates the effect of the parameter under study.

3. **Model code stops carrying defaults.** Solver functions take a config and honour it.
   There is exactly one place in the repository where a default can hide.

Conventions used throughout
---------------------------
* Power in MW, energy in MWh, prices and costs in EUR/MWh, time in hours.
* Sign convention at the point of interconnection (POI): **positive = export to grid**.
* A *state of charge* (SoC) is always a fraction of nominal energy capacity, in [0, 1];
  an *energy* is always MWh. The two are never mixed in one field name.
* Efficiencies are one-way (charge, discharge) and AC-side, i.e. measured at the POI, so
  round-trip efficiency is their product.

Editing parameters
------------------
Defaults here describe a generic 2 h / 4 MWh utility-scale Li-ion system. Override them
from the notebook's control-panel cell, either by constructing the dataclasses explicitly
or by deriving a named variant::

    aggressive = base.variant("Deep cycling",
                              battery=dict(soc_min_frac=0.0, soc_max_frac=1.0))

``variant`` copies the configuration, applies the overrides and re-runs validation, so an
inconsistent set of parameters fails immediately and loudly rather than silently producing
an optimistic schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import pandas as pd


# --------------------------------------------------------------------------------------
# Field metadata helper
# --------------------------------------------------------------------------------------
def tunable(default: Any, unit: str, typical: str, why: str) -> Any:
    """Declare a parameter together with the documentation a reader needs to challenge it.

    ``unit``     physical unit, so no number in this module is ambiguous.
    ``typical``  the range seen in real utility-scale projects; a value outside it is not
                 forbidden, but the notebook should then say why it was chosen.
    ``why``      what the parameter controls and, crucially, in which direction the result
                 moves when it is increased.
    """
    return field(default=default, metadata={"unit": unit, "typical": typical, "why": why})


# --------------------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------------------
@dataclass
class BatteryParams:
    """Physical envelope of the storage asset.

    The distinction between *nominal* and *usable* energy is the one most often glossed
    over in textbook formulations. A block rated 4 MWh is not operated between 0% and 100%
    SoC: warranties are written around a restricted window, and the energy actually
    available for trading is ``(soc_max - soc_min) * e_nominal_mwh``. Optimising over the
    full 0-100% range overstates achievable revenue and understates degradation, so the
    window is an explicit parameter here rather than an implicit assumption.
    """

    # -- Energy and power rating -------------------------------------------------------
    e_nominal_mwh: float = tunable(
        4.0, "MWh", "0.5-4 h of duration, i.e. 0.5-4 MWh per MW of inverter",
        "Nameplate energy. Sets how much volume a single price spread can capture; raising it "
        "increases revenue sub-linearly, because the extra capacity chases progressively "
        "narrower spreads.",
    )
    p_ch_max_mw: float = tunable(
        2.0, "MW", "0.25-1.0 C, i.e. e_nominal/4 to e_nominal per hour",
        "Maximum charging power at the POI. Binding mainly on days with short, deep price "
        "troughs; a higher C-rate buys the ability to fill the battery inside a narrow window.",
    )
    p_dis_max_mw: float = tunable(
        2.0, "MW", "usually equal to p_ch_max_mw (shared power conversion system)",
        "Maximum discharging power at the POI. Normally symmetric with charging, because both "
        "flow through the same inverter.",
    )

    # -- Usable state-of-charge window -------------------------------------------------
    soc_min_frac: float = tunable(
        0.05, "fraction of e_nominal", "0.02-0.10 for LFP, 0.10-0.20 for NMC",
        "Lower operating limit. Raising it protects the cells (and the warranty) at the cost of "
        "usable energy, so revenue falls roughly in proportion to the window lost.",
    )
    soc_max_frac: float = tunable(
        0.95, "fraction of e_nominal", "0.90-1.00; calendar ageing accelerates near 100%",
        "Upper operating limit. Holding cells at high SoC accelerates calendar fade, which this "
        "LP does not price; keeping the ceiling below 1.0 is the cheap proxy for that.",
    )
    soc_init_frac: float = tunable(
        0.50, "fraction of e_nominal", "0.30-0.60 at gate closure",
        "SoC at the start of the optimisation horizon. On a single day it fixes the starting "
        "inventory; in a rolling deployment it is the state handed over by the previous day.",
    )

    # -- Terminal condition ------------------------------------------------------------
    soc_terminal_mode: Literal["free", "equal_to_initial", "at_least", "valued"] = tunable(
        "equal_to_initial", "-", "'equal_to_initial' for repeated daily operation",
        "How the end of the horizon is closed. 'free' lets the optimiser sell the battery empty "
        "on the last interval, because stored energy has no value after T - a classic "
        "end-of-horizon artefact that inflates single-day profit. 'equal_to_initial' enforces "
        "day-to-day repeatability; 'at_least' sets a floor; 'valued' instead prices leftover "
        "energy at terminal_energy_value_eur_per_mwh, the closest analogue to a water value.",
    )
    soc_terminal_frac: float | None = tunable(
        None, "fraction of e_nominal", "equal to soc_init_frac, or 0.2-0.5 as a floor",
        "Target/floor SoC at the end of the horizon. None means 'reuse soc_init_frac'. Ignored "
        "when soc_terminal_mode is 'free' or 'valued'.",
    )
    terminal_energy_value_eur_per_mwh: float = tunable(
        0.0, "EUR/MWh", "next day's expected peak price, discounted",
        "Opportunity value assigned to energy still stored at time T, used only when "
        "soc_terminal_mode='valued'. This is the single-horizon stand-in for the value function a "
        "multi-day dynamic program would compute endogenously.",
    )

    # -- Conversion efficiency ---------------------------------------------------------
    eta_ch: float = tunable(
        0.97, "-", "0.95-0.98 one-way, AC-side",
        "Charging efficiency measured at the POI (transformer + inverter + cell). Enters the "
        "arbitrage trigger condition: the round-trip loss is the toll paid on every cycle.",
    )
    eta_dis: float = tunable(
        0.97, "-", "0.95-0.98 one-way, AC-side",
        "Discharging efficiency at the POI. eta_ch * eta_dis is the round-trip efficiency; at "
        "0.97/0.97 that is 94.1%, so a spread below ~6% of the purchase price never pays.",
    )

    # -- Parasitic losses --------------------------------------------------------------
    self_discharge_frac_per_day: float = tunable(
        0.002, "fraction of stored energy per day", "0.1-0.3%/day at container level",
        "Standing loss on stored energy, applied per interval. Negligible over 24 h; it matters "
        "once the horizon lengthens or a strategy parks energy for days.",
    )
    aux_load_mw: float = tunable(
        0.0, "MW", "0.5-2% of rated power (HVAC, BMS, controls)",
        "Constant station-service load drawn from the grid. Note it does not change the optimal "
        "*schedule* - a constant load is a constant in the objective - unless a POI limit binds. "
        "It does change the P&L, which is why it belongs in the economics rather than nowhere.",
    )

    # -- Operational restrictions ------------------------------------------------------
    forbid_simultaneous_ch_dis: bool = tunable(
        False, "-", "False unless negative prices are present",
        "Adds one binary per interval preventing charge and discharge at the same time, turning "
        "the LP into a MILP. With positive prices and eta < 1 the LP never does this on its own; "
        "it can become attractive at negative prices, where burning energy in the round-trip loss "
        "is itself profitable. The notebook tests whether the restriction is ever binding.",
    )

    # -- Derived quantities ------------------------------------------------------------
    @property
    def e_usable_mwh(self) -> float:
        """Energy actually available for trading, i.e. the width of the SoC window."""
        return (self.soc_max_frac - self.soc_min_frac) * self.e_nominal_mwh

    @property
    def e_min_mwh(self) -> float:
        return self.soc_min_frac * self.e_nominal_mwh

    @property
    def e_max_mwh(self) -> float:
        return self.soc_max_frac * self.e_nominal_mwh

    @property
    def e_init_mwh(self) -> float:
        return self.soc_init_frac * self.e_nominal_mwh

    @property
    def eta_roundtrip(self) -> float:
        return self.eta_ch * self.eta_dis

    @property
    def duration_h(self) -> float:
        """Nominal discharge duration - how the market labels a BESS (a '2-hour battery')."""
        return self.e_nominal_mwh / self.p_dis_max_mw

    @property
    def c_rate_ch(self) -> float:
        return self.p_ch_max_mw / self.e_nominal_mwh

    @property
    def c_rate_dis(self) -> float:
        return self.p_dis_max_mw / self.e_nominal_mwh

    def terminal_energy_mwh(self) -> float | None:
        """Target/floor energy at time T, or None when the terminal SoC is left free."""
        if self.soc_terminal_mode in ("free", "valued"):
            return None
        frac = self.soc_init_frac if self.soc_terminal_frac is None else self.soc_terminal_frac
        return frac * self.e_nominal_mwh

    def self_discharge_frac_per_hour(self) -> float:
        return self.self_discharge_frac_per_day / 24.0

    def __post_init__(self) -> None:
        if self.e_nominal_mwh <= 0 or self.p_ch_max_mw <= 0 or self.p_dis_max_mw <= 0:
            raise ValueError("Energy and power ratings must be strictly positive.")
        if not 0.0 <= self.soc_min_frac < self.soc_max_frac <= 1.0:
            raise ValueError(
                f"Invalid SoC window: need 0 <= soc_min_frac ({self.soc_min_frac}) "
                f"< soc_max_frac ({self.soc_max_frac}) <= 1."
            )
        if not self.soc_min_frac <= self.soc_init_frac <= self.soc_max_frac:
            raise ValueError(
                f"soc_init_frac ({self.soc_init_frac}) lies outside the operating window "
                f"[{self.soc_min_frac}, {self.soc_max_frac}]."
            )
        target = self.terminal_energy_mwh()
        if target is not None and not (self.e_min_mwh - 1e-9 <= target <= self.e_max_mwh + 1e-9):
            raise ValueError("The terminal SoC target lies outside the operating window.")
        for name in ("eta_ch", "eta_dis"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")


# --------------------------------------------------------------------------------------
# Plant and grid connection
# --------------------------------------------------------------------------------------
@dataclass
class PlantParams:
    """The hybrid plant around the battery: co-located PV and the shared grid connection.

    A co-located PV + BESS plant shares one connection agreement. That shared export limit
    is a large part of why hybrid plants are built at all - the battery absorbs PV that
    would otherwise be clipped - and omitting it makes the two assets look independent when
    they are not.
    """

    pv_capacity_mw: float = tunable(
        1.0, "MW", "0.5-2 MW of PV per MW of BESS in a hybrid plant",
        "Installed PV capacity feeding the same connection point. Raising it increases energy "
        "sold but also the hours in which the POI limit binds and PV must be stored or spilled.",
    )
    poi_export_limit_mw: float | None = tunable(
        2.0, "MW", "usually sized on the BESS rating, below PV + BESS combined",
        "Maximum net injection into the grid; None disables the constraint. When it is smaller "
        "than pv_capacity + p_dis_max the plant is deliberately over-built, and the battery earns "
        "part of its value simply by shifting energy that could not otherwise be exported.",
    )
    poi_import_limit_mw: float | None = tunable(
        2.0, "MW", "usually equal to the export limit",
        "Maximum net withdrawal from the grid, which caps grid charging; None disables it.",
    )
    allow_grid_charging: bool = tunable(
        True, "-", "False where a subsidy or tax regime forbids it",
        "Whether the battery may charge from the grid at all. Setting it False turns the asset "
        "into a pure PV-shifting device, the relevant case under several renewable-support schemes.",
    )
    allow_direct_pv_sale: bool = tunable(
        True, "-", "True; False only to reproduce the course baseline",
        "Whether PV may bypass the battery and go straight to the grid. False reproduces the "
        "course formulation, which forces every watt of PV through the battery and therefore pays "
        "a round-trip loss it did not need to pay.",
    )
    allow_pv_curtailment: bool = tunable(
        True, "-", "True - without it the model can be infeasible",
        "Whether PV may be spilled. Needed whenever the POI limit binds with a full battery, and "
        "the only available response to deeply negative prices.",
    )

    def __post_init__(self) -> None:
        if self.pv_capacity_mw < 0:
            raise ValueError("pv_capacity_mw must be non-negative.")
        for name in ("poi_export_limit_mw", "poi_import_limit_mw"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, or None for unconstrained.")


# --------------------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------------------
@dataclass
class MarketParams:
    """Where the energy is sold, and what it costs to move it across the meter.

    Grid tariffs are the parameter most often missing from academic arbitrage models and
    among the most decisive in a real project: they are charged per MWh *through the meter*,
    so they widen the spread the battery needs before a cycle is worth doing, independently
    of the spread itself.
    """

    price_area: str = tunable(
        "DK1", "-", "DK1 / DK2 for Denmark",
        "Bidding zone whose day-ahead price series drives the optimisation.",
    )
    grid_tariff_charge_eur_per_mwh: float = tunable(
        0.0, "EUR/MWh withdrawn", "2-15 EUR/MWh depending on zone and tariff class",
        "Network charge paid on every MWh imported. Acts exactly like an increase in the purchase "
        "price, so it raises the spread required to trade.",
    )
    grid_tariff_discharge_eur_per_mwh: float = tunable(
        0.0, "EUR/MWh injected", "0-5 EUR/MWh; often zero for storage",
        "Network charge paid on every MWh exported from the battery; reduces the effective sale price.",
    )

    def __post_init__(self) -> None:
        if self.grid_tariff_charge_eur_per_mwh < 0 or self.grid_tariff_discharge_eur_per_mwh < 0:
            raise ValueError("Grid tariffs must be non-negative.")


# --------------------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------------------
@dataclass
class DegradationParams:
    """How cycling consumes the asset, and what that consumption is worth in EUR.

    The central quantity is the *marginal cost of throughput*: EUR charged per MWh moved
    through the battery, which is what the LP objective can actually see. Deriving it from
    CAPEX and warranted cycle life, rather than picking a round number, is what links the
    operating problem to the investment case: the same EUR/MWh that disciplines the schedule
    is the EUR/MWh that must be recovered from the spread.
    """

    capex_eur_per_mwh: float = tunable(
        150_000.0, "EUR/MWh of nominal energy", "100,000-250,000 EUR/MWh installed",
        "Installed cost of the storage block. Drives both the investment case and, through the "
        "derivation below, how expensive a cycle looks to the optimiser.",
    )
    cycle_life_at_ref_dod: float = tunable(
        4_000.0, "equivalent full cycles", "3,000-6,000 (NMC) / 6,000-10,000 (LFP) to 80% SoH",
        "Warranted cycles at reference_dod before end-of-life capacity is reached. Doubling it "
        "halves the marginal cost of a cycle and makes the optimiser markedly more willing to trade.",
    )
    reference_dod: float = tunable(
        0.80, "fraction of e_nominal", "0.80 or 0.90 in most datasheets",
        "Depth of discharge at which cycle_life_at_ref_dod is warranted. It anchors the unit cycle: "
        "every cycle-counting metric in degradation.py is expressed relative to it, so that "
        "throughput-based and depth-aware costs are directly comparable.",
    )
    dod_stress_exponent: float = tunable(
        1.0, "-", "1.0-2.0; near 1.0 for LFP, higher for NMC",
        "Exponent k in the Wohler-type law N(DoD) = N_ref * (reference_dod / DoD)**k. At k = 1 "
        "damage is exactly proportional to energy throughput and the depth-aware metric collapses "
        "onto the throughput one; only k > 1 makes deep cycles genuinely more damaging per MWh.",
    )
    end_of_life_capacity_frac: float = tunable(
        0.80, "fraction of initial capacity", "0.70-0.80",
        "State of health defining end of life. Used in the residual-value and augmentation "
        "discussion; it does not enter the LP.",
    )
    calendar_fade_frac_per_year: float = tunable(
        0.015, "fraction of capacity per year", "1-2.5%/yr at moderate SoC and temperature",
        "Ageing that happens regardless of cycling. From the optimiser's viewpoint it is sunk - no "
        "operating decision avoids it - which is exactly why it belongs in the economics and not "
        "in the objective.",
    )
    marginal_cost_mode: Literal["derived", "manual", "off"] = tunable(
        "derived", "-", "'derived' - it is the defensible choice",
        "'derived' computes the throughput cost from CAPEX and cycle life (see "
        "marginal_throughput_cost_eur_per_mwh); 'manual' uses the value below, for sensitivity "
        "sweeps; 'off' sets it to zero and recovers a pure arbitrage objective.",
    )
    manual_eur_per_mwh_throughput: float = tunable(
        0.0, "EUR/MWh throughput", "0-40 EUR/MWh spans the plausible range",
        "Throughput cost used when marginal_cost_mode='manual'. Sweeping it shows how the schedule "
        "backs off as cycling is priced.",
    )

    def marginal_throughput_cost_eur_per_mwh(self) -> float:
        """EUR charged per MWh of battery throughput (charge + discharge), as seen by the LP.

        Derivation. Over its warranted life the asset delivers

            E_life = cycle_life_at_ref_dod * reference_dod * e_nominal   [MWh discharged]

        and, counting both directions, moves ``2 * E_life`` MWh of throughput. Consuming the
        whole asset costs ``capex_eur_per_mwh * e_nominal`` (less any residual value, which we
        conservatively ignore), so the cost attributable to one MWh of throughput is

            c_degr = capex_eur_per_mwh / (2 * cycle_life_at_ref_dod * reference_dod)

        Note ``e_nominal`` cancels: the marginal cost of cycling is a property of the chemistry
        and its price, not of how large the installation is. At 150,000 EUR/MWh and 4,000 cycles
        at 80% DoD this returns ~23.4 EUR/MWh - which, through the arbitrage trigger condition in
        ``ProjectConfig.breakeven_spread_eur_per_mwh``, means a spread of roughly 48 EUR/MWh is
        needed before a cycle pays for the capacity it consumes.
        """
        if self.marginal_cost_mode == "off":
            return 0.0
        if self.marginal_cost_mode == "manual":
            return self.manual_eur_per_mwh_throughput
        return self.capex_eur_per_mwh / (2.0 * self.cycle_life_at_ref_dod * self.reference_dod)

    def cost_per_reference_cycle_eur(self, e_nominal_mwh: float) -> float:
        """EUR consumed by one full cycle *at the reference depth* - the unit all metrics share."""
        return self.capex_eur_per_mwh * e_nominal_mwh / self.cycle_life_at_ref_dod

    def __post_init__(self) -> None:
        if self.capex_eur_per_mwh < 0:
            raise ValueError("capex_eur_per_mwh must be non-negative.")
        if self.cycle_life_at_ref_dod <= 0:
            raise ValueError("cycle_life_at_ref_dod must be strictly positive.")
        if not 0.0 < self.reference_dod <= 1.0:
            raise ValueError("reference_dod must lie in (0, 1].")
        if self.dod_stress_exponent < 1.0:
            raise ValueError(
                "dod_stress_exponent < 1 would make deep cycles *less* damaging per MWh than "
                "shallow ones, contradicting the published ageing laws this model is based on."
            )


# --------------------------------------------------------------------------------------
# Project economics
# --------------------------------------------------------------------------------------
@dataclass
class EconomicParams:
    """Assumptions turning an operating margin into an investment decision."""

    discount_rate: float = tunable(
        0.06, "-", "5-9% real WACC for a merchant storage project",
        "Weighted average cost of capital. Merchant revenue is volatile, so the rate sits above "
        "that of a contracted renewable asset; raising it lowers break-even CAPEX steeply.",
    )
    lifetime_years: int = tunable(
        12, "years", "10-20 years, often set by the warranty",
        "Economic life of the project. Must stay consistent with cycle life: cycling 365 times a "
        "year for 12 years is 4,380 cycles, which already exceeds a 4,000-cycle warranty.",
    )
    opex_pct_of_capex: float = tunable(
        0.02, "fraction of CAPEX per year", "1.5-3%/yr including O&M, insurance and land",
        "Fixed annual operating cost; enters both NPV and LCOS.",
    )
    availability: float = tunable(
        0.97, "-", "95-98% for a mature BESS",
        "Fraction of the year the asset can trade, after outages and maintenance. It scales annual "
        "revenue directly, so it is a first-order input despite usually appearing as a footnote.",
    )
    residual_value_frac_of_capex: float = tunable(
        0.0, "fraction of CAPEX", "0-10%; second-life value remains speculative",
        "Value recovered at end of life. Left at zero by default so the investment case is not "
        "propped up by an assumption nobody can underwrite today.",
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.availability <= 1.0:
            raise ValueError("availability must lie in (0, 1].")
        if self.lifetime_years <= 0:
            raise ValueError("lifetime_years must be strictly positive.")
        if not 0.0 <= self.residual_value_frac_of_capex < 1.0:
            raise ValueError("residual_value_frac_of_capex must lie in [0, 1).")


# --------------------------------------------------------------------------------------
# Simulation setup
# --------------------------------------------------------------------------------------
@dataclass
class SimulationParams:
    """How the experiment itself is run: resolution, horizon, sample of days, random seeds."""

    dt_hours: float = tunable(
        0.25, "h", "0.25 h - the European day-ahead resolution since 2025",
        "Market time unit. Finer resolution exposes more short-lived spreads (and more variables); "
        "coarsening to 1 h systematically understates arbitrage value.",
    )
    horizon_hours: float = tunable(
        24.0, "h", "24 h for day-ahead; 36-48 h with a rolling horizon",
        "Length of one optimisation. Beyond 24 h the terminal-SoC artefact matters less, but the "
        "price is no longer known past gate closure.",
    )
    n_backtest_days: int = tunable(
        30, "days", "30-365; more days narrow the confidence interval on the mean",
        "Number of independent days solved in the backtest. Daily profit is heavy-tailed, so a "
        "small sample gives a noisy mean - which is why the notebook reports the standard error.",
    )
    day_sample_seed: int = tunable(
        32, "-", "any value; fixed for reproducibility",
        "Seed selecting which calendar days enter the backtest. Fixed so that two strategies are "
        "always compared on the *same* days.",
    )
    pv_seed: int = tunable(
        42, "-", "any value; fixed for reproducibility",
        "Seed for the synthetic PV generator on the reference day.",
    )
    scenario_seed: int = tunable(
        7, "-", "any value; fixed for reproducibility",
        "Seed for the stochastic-scheduling scenario generator.",
    )

    @property
    def n_intervals(self) -> int:
        return round(self.horizon_hours / self.dt_hours)

    def __post_init__(self) -> None:
        if self.dt_hours <= 0 or self.horizon_hours <= 0:
            raise ValueError("dt_hours and horizon_hours must be strictly positive.")
        ratio = self.horizon_hours / self.dt_hours
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("horizon_hours must be an integer multiple of dt_hours.")


# --------------------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------------------
_SECTIONS = ("battery", "plant", "market", "degradation", "economics", "simulation")


@dataclass
class ProjectConfig:
    """Everything the model, the backtest and the economics need, in one auditable object.

    A named ``ProjectConfig`` *is* a trading strategy in this project: the operating policy
    follows from the parameters, so comparing policies means comparing configurations on
    identical input data. ``experiments.compare_strategies`` takes a list of these.
    """

    name: str = "Base case"
    battery: BatteryParams = field(default_factory=BatteryParams)
    plant: PlantParams = field(default_factory=PlantParams)
    market: MarketParams = field(default_factory=MarketParams)
    degradation: DegradationParams = field(default_factory=DegradationParams)
    economics: EconomicParams = field(default_factory=EconomicParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)

    # -- Derived, cross-section quantities ---------------------------------------------
    def degradation_cost_eur_per_mwh_throughput(self) -> float:
        """The single number linking the operating problem to the investment case."""
        return self.degradation.marginal_throughput_cost_eur_per_mwh()

    def breakeven_spread_eur_per_mwh(self, buy_price_eur_per_mwh: float = 0.0) -> float:
        """Sale price at which one arbitrage cycle exactly breaks even.

        Buying 1 MWh (AC) at ``p_buy`` costs ``p_buy + tariff_ch + c_degr`` and returns
        ``eta_ch * eta_dis`` MWh to the grid, each earning ``p_sell - tariff_dis - c_degr``.
        Setting profit to zero and solving for the sale price:

            p_sell = (p_buy + tariff_ch + c_degr) / eta_rt + tariff_dis + c_degr

        This closed form is the trader's version of the LP. Every charge/discharge decision the
        optimiser makes can be read off it, and it shows why round-trip efficiency, network
        tariffs and degradation are not second-order details but the definition of when the
        battery is allowed to move at all.
        """
        c_degr = self.degradation_cost_eur_per_mwh_throughput()
        gross_cost = buy_price_eur_per_mwh + self.market.grid_tariff_charge_eur_per_mwh + c_degr
        return (
            gross_cost / self.battery.eta_roundtrip
            + self.market.grid_tariff_discharge_eur_per_mwh
            + c_degr
        )

    # -- Ergonomics --------------------------------------------------------------------
    def variant(self, name: str, **section_overrides: dict) -> "ProjectConfig":
        """Return a copy under a new name with selected fields overridden.

        Keyword arguments are section names mapping to the fields to change::

            cfg.variant("No SoC window", battery=dict(soc_min_frac=0.0, soc_max_frac=1.0))

        Unknown sections or fields raise immediately, so a typo in a sensitivity sweep cannot
        silently leave a parameter at its default and produce a duplicated result.
        """
        unknown = set(section_overrides) - set(_SECTIONS)
        if unknown:
            raise KeyError(f"Unknown configuration section(s): {sorted(unknown)}. Valid: {_SECTIONS}")

        new_sections = {}
        for section in _SECTIONS:
            current = getattr(self, section)
            overrides = section_overrides.get(section)
            if not overrides:
                new_sections[section] = replace(current)
                continue
            valid = {f.name for f in fields(current)}
            bad = set(overrides) - valid
            if bad:
                raise KeyError(f"Unknown field(s) for '{section}': {sorted(bad)}. Valid: {sorted(valid)}")
            new_sections[section] = replace(current, **overrides)
        return ProjectConfig(name=name, **new_sections)

    def summary(self, section: str | None = None) -> pd.DataFrame:
        """Every parameter with its value, unit, typical range and role - the audit table.

        Rendering this in the notebook means no figure rests on an assumption the reader cannot
        see. Pass ``section`` to restrict it to one block.
        """
        sections = _SECTIONS if section is None else (section,)
        rows = []
        for sec in sections:
            obj = getattr(self, sec)
            for f in fields(obj):
                rows.append(
                    {
                        "section": sec,
                        "parameter": f.name,
                        "value": getattr(obj, f.name),
                        "unit": f.metadata.get("unit", ""),
                        "typical range": f.metadata.get("typical", ""),
                        "role / effect of increasing it": f.metadata.get("why", ""),
                    }
                )
        return pd.DataFrame(rows).set_index(["section", "parameter"])

    def key_figures(self) -> pd.Series:
        """The handful of derived numbers worth checking before trusting any result."""
        b = self.battery
        return pd.Series(
            {
                "duration [h]": b.duration_h,
                "C-rate, charge [1/h]": b.c_rate_ch,
                "usable energy [MWh]": b.e_usable_mwh,
                "usable fraction of nameplate [-]": b.e_usable_mwh / b.e_nominal_mwh,
                "round-trip efficiency [-]": b.eta_roundtrip,
                "degradation cost [EUR/MWh throughput]": self.degradation_cost_eur_per_mwh_throughput(),
                "break-even spread from 0 EUR/MWh [EUR/MWh]": self.breakeven_spread_eur_per_mwh(0.0),
                "installed cost of storage [EUR]": self.degradation.capex_eur_per_mwh * b.e_nominal_mwh,
            },
            name=self.name,
        )

    def to_dict(self) -> dict:
        """Flat ``{'section.field': value}`` mapping, for logging a run's exact assumptions."""
        out: dict = {"name": self.name}
        for sec in _SECTIONS:
            for f in fields(getattr(self, sec)):
                out[f"{sec}.{f.name}"] = getattr(getattr(self, sec), f.name)
        return out


def diff_configs(*configs: ProjectConfig) -> pd.DataFrame:
    """Show only the parameters on which the given configurations disagree.

    When a comparison of five strategies returns five different margins, this answers the only
    question that matters next: what, exactly, was different between them?
    """
    frame = pd.DataFrame([c.to_dict() for c in configs]).set_index("name").T
    return frame[frame.nunique(axis=1) > 1]
