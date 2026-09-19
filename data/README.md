# Data sources and licensing

## Day-ahead prices — `spot_prices_dk1.csv` (committed)

- **Source:** [Energi Data Service](https://www.energidataservice.dk/) (Energinet), dataset
  [`DayAheadPrices`](https://www.energidataservice.dk/tso-electricity/DayAheadPrices), price
  area `DK1`, 15-minute resolution.
- **License:** CC-BY 4.0 ([terms](https://www.energidataservice.dk/terms-and-conditions)) —
  free to copy, modify and redistribute, including commercially, with attribution.
- **Attribution:** *Source: Energinet ([www.energidataservice.dk](https://www.energidataservice.dk))*
- **How it was obtained:** `src/bess_opt/data.py::fetch_spot_prices`, which calls the public
  Energi Data Service REST API and caches the result here as CSV so the project runs offline
  and reproducibly. Re-run with `force_refresh=True` to pull the latest data.

## PV production — synthetic, not a static file

The exercise this project grew out of pointed to the DTU **EnergyDataDK** platform
(`energydata.dk`) for real measured PV production series. Unlike Energinet's price data,
EnergyDataDK's general terms of service state datasets **cannot be used for commercial or
redistribution purposes unless the specific dataset's own license explicitly allows it**, and
the license attached to the particular "Production - Solar" datasets used while developing
this exercise was not established. To stay on the safe side, **no measured PV file is
committed to this repository**.

Instead, `src/bess_opt/data.py::synthetic_pv_profile` generates a physically-motivated PV
profile from a clear-sky solar-geometry model (latitude-and-day-of-year-dependent solar
elevation) modulated by randomised daily cloudiness and short-term cloud-transient noise. It
is deterministic given a seed, and — usefully — can generate arbitrarily many years of data,
which is exactly what the multi-day backtest and scenario-generation sections need.

**If you have your own (legally obtained) measured PV CSV** you'd like to use instead, drop it
in `data/raw_pv/` (already git-ignored — see `.gitignore`) and load it in place of the call to
`synthetic_pv_profile` in the notebook's data section. That call appears in exactly two places (the
reference day and `bess_opt.backtest.solve_day`), and both expect a `pandas.Series` of MW indexed
by the same 15-minute timestamps.
