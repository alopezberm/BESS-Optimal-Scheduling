"""Data access: real day-ahead prices (Energinet API, cached) and a synthetic PV generator.

Real PV measurements from the DTU EnergyDataDK portal are *not* redistributed here because
that platform's terms of service restrict commercial/redistribution use unless the specific
dataset's own licence says otherwise (unlike Energinet's price data, which is CC-BY 4.0 and
explicitly reusable). See ``data/README.md`` for the full reasoning. Instead this module ships
a small, physically-motivated synthetic PV generator, which has the added benefit of being able
to produce as much data as needed for backtesting and scenario analysis.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ENERGINET_API_URL = "https://api.energidataservice.dk/dataset/{dataset}"


def fetch_energinet_dataset(dataset: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Query an Energi Data Service dataset over ``[start_date, end_date)``.

    Dates are ``'YYYY-MM-DD'`` strings. Published by Energinet under CC-BY 4.0
    (https://www.energidataservice.dk/terms-and-conditions): free to reuse, modify and
    redistribute, including commercially, with attribution.
    """
    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    params = {"limit": 0, "start": start_date, "end": end_date}
    response = requests.get(ENERGINET_API_URL.format(dataset=dataset), params=params, timeout=30)
    response.raise_for_status()
    records = response.json().get("records", [])
    if not records:
        raise ValueError(f"No records returned for dataset={dataset!r} in [{start_date}, {end_date})")
    return pd.DataFrame(records)


def fetch_spot_prices(
    start_date: str,
    end_date: str,
    price_area: str = "DK1",
    cache_path: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Day-ahead spot prices [EUR/MWh] for ``price_area``, indexed by *local* Danish time.

    Indexing by local time (rather than UTC) keeps calendar-day slicing correct around
    midnight and across the CET/CEST clock change, which matters once we start selecting
    "day 1..96" windows by local date further down the notebook.

    The result is cached to ``cache_path`` as CSV so the whole project stays reproducible
    offline and we don't hit the API on every notebook run.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path, parse_dates=["time"]).set_index("time")["price"]
        cached.index.name = "time"
        return cached

    raw = fetch_energinet_dataset("DayAheadPrices", start_date, end_date)
    raw = raw.loc[raw["PriceArea"] == price_area].copy()
    raw["TimeDK"] = pd.to_datetime(raw["TimeDK"])
    raw = raw.sort_values("TimeDK").drop_duplicates(subset="TimeDK")

    prices = raw.set_index("TimeDK")["DayAheadPriceEUR"].rename("price")
    prices.index.name = "time"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_frame().to_csv(cache_path)

    return prices


def synthetic_clear_sky_shape(timestamps: pd.DatetimeIndex, latitude_deg: float = 55.7) -> np.ndarray:
    """Relative clear-sky irradiance shape (0-1) for local timestamps.

    Simplified solar-geometry model: Cooper's approximation for the solar declination plus
    the standard cosine-of-zenith-angle formula. This is a shape, not a calibrated irradiance
    model - good enough to drive a synthetic PV profile, not a substitute for measured data.
    """
    day_of_year = timestamps.dayofyear.to_numpy()
    hour_local = (timestamps.hour + timestamps.minute / 60).to_numpy()

    declination = np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + day_of_year))))
    hour_angle = np.radians(15 * (hour_local - 12))
    lat = np.radians(latitude_deg)

    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, 0.0, None)

    peak = cos_zenith.max()
    return cos_zenith / peak if peak > 0 else cos_zenith


def synthetic_pv_profile(
    timestamps: pd.DatetimeIndex,
    capacity_mw: float = 1.0,
    latitude_deg: float = 55.7,
    seed: int | None = 0,
) -> pd.Series:
    """A physically-motivated *synthetic* PV production profile [MW].

    Clear-sky solar-geometry shape, modulated by:
      - a random daily "clearness index" (Beta-distributed, skewed towards overcast -
        broadly representative of the Danish climate), and
      - small per-interval multiplicative noise for cloud transients.

    Deterministic given ``seed``, and able to generate an arbitrary number of years of
    data - useful both as a drop-in replacement for the (license-encumbered) measured PV
    export and as a generator of independent scenarios for stochastic optimization.
    """
    rng = np.random.default_rng(seed)
    shape = synthetic_clear_sky_shape(timestamps, latitude_deg=latitude_deg)

    calendar_day = timestamps.normalize()
    unique_days = calendar_day.unique()
    # Beta(2.2, 2.0) rescaled to [0.15, 1.0]: mean clearness ~0.6, skewed towards partly cloudy.
    daily_clearness = pd.Series(0.15 + 0.85 * rng.beta(2.2, 2.0, size=len(unique_days)), index=unique_days)
    clearness = calendar_day.map(daily_clearness).to_numpy()

    cloud_noise = np.clip(rng.normal(loc=1.0, scale=0.06, size=len(timestamps)), 0.7, 1.15)

    pv_mw = np.clip(capacity_mw * shape * clearness * cloud_noise, 0.0, capacity_mw)
    return pd.Series(pv_mw, index=timestamps, name="pv_mw")


def make_price_scenarios(
    base_prices: np.ndarray,
    n_scenarios: int = 3,
    scale_range: tuple[float, float] = (0.7, 1.3),
    noise_std_frac: float = 0.10,
    seed: int | None = 0,
) -> dict[int, dict]:
    """Build simple day-ahead price scenarios around one observed/base price curve.

    Each scenario keeps the *shape* of ``base_prices`` but rescales its volatility around
    the daily mean and adds i.i.d. noise - giving qualitatively different (low/typical/high
    volatility) days to feed the stochastic-scheduling extension. Equal probability by default.
    """
    rng = np.random.default_rng(seed)
    scales = np.linspace(scale_range[0], scale_range[1], n_scenarios)
    mean_price = base_prices.mean()
    std_price = base_prices.std()

    scenarios = {}
    for s, scale in enumerate(scales):
        noise = rng.normal(0.0, noise_std_frac * std_price, size=base_prices.shape)
        prices_s = mean_price + scale * (base_prices - mean_price) + noise
        scenarios[s] = {"probability": 1.0 / n_scenarios, "prices": prices_s}
    return scenarios
