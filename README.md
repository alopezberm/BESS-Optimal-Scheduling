# BESS Optimal Scheduling — Arbitrage, Degradation & Bankability

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Gurobi](https://img.shields.io/badge/optimizer-Gurobi%20%7C%20CBC-orange.svg)](https://www.gurobi.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC--BY%204.0-lightgrey.svg)](data/README.md)

Optimal day-ahead scheduling of a grid-connected battery + PV plant, framed as a linear
program and solved with Gurobi (with an open-source fallback so anyone can reproduce it
without a license) — then pushed past "does the LP solve" into the questions that actually
decide whether a battery project gets built: how fast does it wear out, and at what
hardware price does it pay for itself?

> Grew out of an optimization exercise in DTU course **46765 – Machine Learning for Energy
> Systems**. The base linear program (charge/discharge/state-of-charge arbitrage) follows
> that exercise's formulation; everything from the extended formulation onward — direct PV
> sale, degradation modelling, project economics, stochastic scheduling, the open-source
> reproducibility check — is my own extension. See [Acknowledgements](#acknowledgements).

## Why this project

Battery arbitrage notebooks usually stop at "the LP finds a profitable schedule." That's the
easy 80%. The interesting questions for an actual investment decision are:

- Cycling the battery harder for a few more euros of arbitrage profit — is it worth the wear?
  A shallow 20→40% cycle and a deep 0→100% cycle move the same energy but do **not** age
  the cell the same amount.
- Nobody agrees on what a battery costs installed. So instead of guessing a number, **sweep
  it**: at what CAPEX (EUR/MWh) does the arbitrage revenue this battery actually earns stop
  covering its cost?
- Day-ahead prices and PV are not known in advance. How much is that uncertainty worth?

This project builds toward answering all three, on top of a solid, correct LP.

## Roadmap

| # | Section | Status |
|---|---|---|
| 1 | Problem & motivation | ✅ |
| 2 | Data: real day-ahead prices (CC-BY, cached) + synthetic PV generator | ✅ |
| 3 | Mathematical formulation — baseline + extended | ✅ |
| 4 | Baseline model (faithful to the course formulation) | ✅ |
| 5 | Extended model: direct PV sale, curtailment, degradation cost in the objective | ✅ |
| 6 | Multi-day backtest | ✅ |
| 7 | Battery cycling & degradation analysis (throughput **and** depth-of-discharge aware) | ✅ (v1, simplified stress curve) |
| 8 | Project economics: CAPEX sweep → NPV, IRR, LCOS, break-even CAPEX | ✅ (v1, flat-revenue assumption) |
| 9 | Reproducibility: identical LP solved with an open-source solver (PuLP/CBC) | ✅ |
| 10 | Stochastic scheduling under price & PV scenarios (the course's optional, unsolved extension) | ✅ (3-scenario, here-and-now schedule) |
| 11 | Rolling-horizon backtest with realistic forecast error | 📋 planned |
| 12 | Revenue stacking (arbitrage + frequency reserves) | 📋 planned |
| 13 | Degradation curves calibrated to real cell datasheets, embedded endogenously (convex piecewise-linear) in the LP | 📋 planned |
| 14 | Price-forecasting model feeding the rolling-horizon strategy | 📋 planned |

✅ implemented · 🚧 partial/simplified · 📋 future work — see the notebook's closing section
for the full reasoning behind each planned item.

## Repository structure

```
bess-optimal-scheduling/
├── 02_gurobi_exercise.ipynb   # the project notebook - narrative, math, results
├── src/bess_opt/
│   ├── data.py                # Energinet price fetch+cache, synthetic PV, scenario generator
│   ├── model.py                # Gurobi: baseline / extended / stochastic LP formulations
│   ├── model_opensource.py    # PuLP/CBC replica of the extended LP (no license needed)
│   ├── degradation.py          # equivalent cycles, DoD half-cycle counting, degradation cost
│   ├── economics.py            # NPV, IRR, LCOS, CAPEX break-even sweep
│   ├── backtest.py             # multi-day backtesting utilities
│   └── plotting.py             # shared matplotlib styling
├── data/
│   ├── spot_prices_dk1.csv    # cached Energinet prices (CC-BY 4.0, see data/README.md)
│   └── README.md
├── requirements.txt
└── LICENSE
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook 02_gurobi_exercise.ipynb
```

No Gurobi license? Run the notebook as-is up to the reproducibility section, or use
`src/bess_opt/model_opensource.py` directly — it solves the identical extended LP with the
open-source CBC solver PuLP ships with.

## Formulation at a glance

15-minute resolution, one day, `T = 96` intervals. Decision variables per interval `t`:
charge power `P_ch[t]`, discharge power `P_dis[t]`, PV routed directly to the grid
`P_pv_direct[t]`, PV curtailed `P_pv_curt[t]`, and state of charge `E[t]`. Maximizing

```
Σ_t  price[t] · (P_dis[t] + P_pv_direct[t] − P_ch[t]) · Δt   −   c_degr · Σ_t (P_ch[t] + P_dis[t]) · Δt
```

subject to the battery's energy balance, power/energy limits, and the PV split
`P_pv_direct[t] + P_pv_to_batt[t] + P_pv_curt[t] = P_pv[t]`. Full derivation, the baseline
(course) formulation it extends, and the stochastic variant are in the notebook.

## Example results

From the run this project ships with (30 random days, DK1 Oct-2025–Sep-2026 prices,
`E_max = 4 MWh`, `P_max = 2 MW`, `η = 0.97`, 6% discount rate, 12-year lifetime — see the
notebook outputs; these numbers move with the random day sample and the price data window):

- Mean daily arbitrage profit ≈ **1,154 EUR/day** at ~2.9 equivalent full cycles/day
  (correlation between daily profit and price volatility: **0.73**, as expected for an
  arbitrage strategy).
- Break-even battery CAPEX for arbitrage-only revenue: **≈ 263,000 EUR/MWh** — in the right
  ballpark of real utility-scale BESS installed costs, a reassuring sanity check on the whole
  model chain, not just a number to report. At an illustrative 150,000 EUR/MWh CAPEX: **44.9%
  IRR**, **2.2-year** simple payback, **18.2 EUR/MWh** LCOS.
- The depth-of-discharge-aware degradation cost was **+34%** relative to the naive
  throughput-only estimate on the reference day — i.e. *how* a given amount of energy is
  cycled matters, not just how much.
- Value of perfect day-ahead information (EVPI, Section 10): **≈ 158 EUR/day (15%)** above
  the best single here-and-now schedule — a concrete price on the uncertainty this project
  otherwise optimizes under.

## Acknowledgements

The problem statement and the baseline LP (arbitrage-only, PV forced through the battery)
follow an exercise from DTU's course *46765 – Machine Learning for Energy Systems*. Day-ahead
price data is © Energinet, CC-BY 4.0 (see `data/README.md`). Everything else — the extended
formulation, degradation and economics modules, the open-source reproducibility check, and
the stochastic scheduling implementation — was built independently for this project.
