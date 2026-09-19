# BESS Optimal Scheduling — Arbitrage, Degradation & Bankability

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Solver](https://img.shields.io/badge/solver-Gurobi%20%7C%20CBC-orange.svg)](https://www.gurobi.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC--BY%204.0-lightgrey.svg)](data/README.md)

Optimal day-ahead scheduling of a grid-connected battery + PV plant, framed as a linear program
and solved with Gurobi — then pushed past *"does the LP solve"* into the question that actually
decides whether a battery project gets built: **which operating strategy is worth adopting, and
how would you know?**

> Grew out of an optimisation exercise in DTU course **46765 – Machine Learning for Energy
> Systems**. The baseline LP follows that exercise and is reproduced unmodified. Everything from
> the extended formulation onward is my own work — see [Provenance](#provenance) and
> [Use of AI](#use-of-ai).

---

## The argument in one paragraph

A battery does not earn money by being optimally scheduled. It earns money by being optimally
scheduled *and still being alive in year eight*. The textbook formulation maximises market
revenue, which means it operates the cells from 0% to 100% state of charge, empties the battery
at midnight because stored energy is worthless after the last interval, pushes 2.6 MW through a
2 MW inverter, and cycles as hard as the spread allows because cycling is free. Repair those
assumptions one at a time, add the cost of a cycle to the objective, and link that cost to the
asset's remaining life, and the ranking of strategies inverts: **the schedule with the highest
daily margin turns out to have less than half the project NPV.**

## Headline results

From the run committed in the notebook — 4 MWh / 2 MW LFP system, 1 MW co-located PV behind a
2 MW connection, 30 random days of DK1 15-minute prices, 6% discount rate. These numbers move
with the day sample and the price window; the notebook regenerates all of them.

| Finding | Number |
|---|---|
| Gross margin, base case | **777 ± 83 EUR/day** at 0.85 equivalent cycles/day |
| Cost of a cycle, derived from CAPEX and warranty | **23.4 EUR/MWh** throughput ⇒ a **48 EUR/MWh** break-even spread |
| Naive "maximise revenue" strategy | +50% daily margin, **2.6-year** asset life, **NPV 240 k€ vs 539 k€** |
| NPV-maximising price to put on a cycle | **≈ 50 EUR/MWh**, ~2× the textbook value — worth **+17% NPV** |
| Break-even CAPEX (arbitrage only) | **273 k€/MWh** fixed schedule · **291 k€/MWh** re-optimised |
| The course baseline's charging schedule | exceeds the inverter rating by **28%** in 17% of intervals |
| Value of a perfect price forecast (EVPI) | **0.4%** with amplitude uncertainty · **11%** with timing uncertainty |
| Gurobi vs. open-source CBC on the same LP | agree to **1.6 × 10⁻⁹** relative |

Four of these are results the baseline formulation cannot express at all, because it contains
neither a lifetime nor a cost of capital.

## What is actually in here

**A model with no hidden constants.** 42 parameters, each documented in `config.py` with its
unit, the range seen in real utility-scale projects, and which way the result moves when it is
increased. `cfg.summary()` prints the lot. Every realism term is an independent switch, so the
notebook measures each one's contribution instead of asserting that it matters.

**An LP whose behaviour is traceable to a rule.** The dual of the energy-balance constraint is
the marginal value of stored energy. Stationarity gives a closed-form trigger condition —

```
p_sell  ≥  (p_buy + τ_ch + c_degr) / (η_ch·η_dis)  +  τ_dis  +  c_degr
```

— and the notebook verifies the solver's duals match it to machine precision (1.4 × 10⁻¹⁴). The
optimiser is not a black box; it is executing a price rule that fits on one line.

**Degradation counted consistently.** Throughput-based and depth-of-discharge-aware costs are
both expressed in *warranted reference cycles*, which gives a testable property: at an ageing
exponent of 1 the two are identical by construction, so any divergence at k > 1 is genuinely the
depth effect rather than a mismatch of units. (A comparison that skips this step reports a
spurious ~+25% "depth effect" that is pure unit confusion.)

**A fine-tuning workbench.** A strategy *is* a configuration, so comparing strategies means
solving identical days under different assumptions — same dates, same PV seeds, standard errors
reported. `experiments.sensitivity()` sweeps any single parameter; `compare_strategies()`
tabulates any set of them.

**Verification, not assertion.** Every structural claim in the notebook is a machine-checked
assertion: the generalisation check between baseline and extended model, the duality identities,
the k = 1 degradation identity, the P&L reconciliation against the solver's own objective, and
the two-solver cross-check.

## Repository structure

```
BESS-Optimal-Scheduling/
├── bess_optimal_scheduling.ipynb   # the project: narrative, mathematics, results
├── src/bess_opt/
│   ├── config.py           # THE CONTROL PANEL - every parameter, documented and validated
│   ├── data.py             # Energinet price fetch + cache; synthetic PV; scenario generation
│   ├── model.py            # Gurobi: baseline / extended / stochastic / fixed-schedule evaluation
│   ├── model_opensource.py # PuLP + CBC replica of the extended LP - no licence needed
│   ├── degradation.py      # cycle counting, ageing law, cost of a cycle
│   ├── economics.py        # cash flows, NPV/IRR/LCOS, lifetime, break-even CAPEX
│   ├── backtest.py         # solving many days
│   ├── experiments.py      # strategy comparison and parameter sweeps
│   └── plotting.py         # figure styling and the recurring plots
├── data/
│   ├── spot_prices_dk1.csv # cached Energinet prices (CC-BY 4.0 - see data/README.md)
│   └── README.md
├── requirements.txt
└── LICENSE
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook bess_optimal_scheduling.ipynb
```

**No Gurobi licence?** `src/bess_opt/model_opensource.py` solves the identical extended LP with
the open-source CBC solver bundled with PuLP. The notebook cross-checks the two and asserts they
agree, so no result here depends on holding a licence.

**Changing assumptions** means editing the control-panel cell near the top of the notebook and
re-running. Nothing downstream hard-codes a parameter, and inconsistent configurations raise on
construction rather than quietly producing an optimistic schedule.

## Methodological points worth flagging

Three corrections made during this work, recorded because they are easy mistakes and each one
moves the answer materially:

- **Unit anchoring in degradation.** Comparing a throughput-based cost against a depth-aware one
  without expressing both in the same unit cycle manufactures a ~25% "depth effect" out of the
  reference DoD alone.
- **Double counting in the DCF.** Degradation is an opportunity cost in the *operating* problem
  and is the CAPEX in the *investment* problem. Deducting an annual degradation charge from the
  operating margin *and* booking the CAPEX at year zero charges the same asset twice. Here it
  appears once, and shows up in the cash flows as a shortened life.
- **Integer asset life.** Rounding a degradation-limited lifetime to whole years makes NPV jump
  discontinuously as a parameter is swept — by enough to invert the ranking of two strategies.
  The final period is fractional.

## Limitations

Arbitrage revenue only (no reserve markets, which for a real BESS are often the larger share);
perfect foresight within each day, so every revenue figure is an upper bound; a price-taking
assumption; an ageing exponent that is swept rather than calibrated; peak-valley rather than full
rainflow cycle counting; synthetic PV uncorrelated with the price series; no mid-life
augmentation; and a 30-day sample of one bidding zone. Section 14 of the notebook states each of
these with its direction of bias, and the roadmap orders the fixes by how much they would change
the conclusions.

## Provenance

The problem statement and the baseline LP follow the DTU 46765 BESS day-ahead scheduling
exercise, reproduced unmodified in `bess_opt.model.build_bess_lp`. Everything else — the extended
formulation and its switchable realism terms, the duality analysis, the degradation model, the
experiment framework and the cycle-price tuning result, the lifetime-linked economics, the
open-source cross-check, and the stochastic implementation with EVPI and VSS — was developed
independently for this project.

Day-ahead price data: © **Energinet**, [Energi Data Service](https://www.energidataservice.dk/),
CC-BY 4.0. PV production is synthetic; `data/README.md` explains why.

## Use of AI

In line with DTU's rules on the use of generative AI in coursework, stated explicitly rather than
left to inference.

I used **Claude (Anthropic)** as a pair-programming and editing assistant while building this
project: refactoring the codebase into documented modules, drafting and revising explanatory
prose, and reviewing the implementation for errors. The three methodological corrections listed
above were identified during that review.

What AI did not do is decide what this project should investigate or what its results mean. The
modelling choices, the parameter values, the interpretation of every result, and the judgement
about which simplifications are acceptable are mine, and I can defend each of them. Every
numerical claim is produced by the code in this repository and reproduced by re-running it.

## License

Code: MIT (see `LICENSE`). Price data: CC-BY 4.0, © Energinet — attribution as required in
`data/README.md`.
