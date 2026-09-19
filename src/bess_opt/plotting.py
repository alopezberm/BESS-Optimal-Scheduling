"""Figures: one visual language for the whole project.

Three rules are applied consistently, and each is a decision rather than a preference.

**No dual-axis charts.** Plotting two quantities of different units against a left and a
right y-axis lets the author place the crossings wherever they like simply by rescaling one
side, which is why the practice manufactures correlations that are not in the data. Wherever
two different units have to be read together - power and state of charge, margin and cycle
count - this module stacks aligned subplots sharing the x-axis instead. The comparison stays
available and the reader can no longer be misled by a choice of scale.

**Colour carries identity, not decoration.** The categorical hues below are a palette
validated for colour-vision deficiency separation and for contrast against the plot surface,
assigned in a fixed order so that a given quantity keeps its colour across every figure:
price is always blue, PV always orange, state of charge always aqua. Signed quantities -
charging against discharging, positive against negative NPV - use the diverging blue/red
pair, where the two poles read as opposites rather than as two arbitrary categories.

**Every series is labelled.** Identity is never conveyed by colour alone, so a legend is
present whenever more than one series is drawn, and single-series plots carry the identity in
the title rather than in a legend box.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots, assigned by role and never cycled.
PALETTE = {
    "price": "#2a78d6",       # slot 1, blue
    "pv": "#eb6834",          # slot 2, orange
    "soc": "#1baf7a",         # slot 3, aqua
    "shadow": "#eda100",      # slot 4, yellow - the marginal value of stored energy
    "charge": "#2a78d6",      # diverging: cool pole
    "discharge": "#e34948",   # diverging: warm pole
    "positive": "#2a78d6",
    "negative": "#e34948",
    "muted": "#8a8a85",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
}

_SERIES_ORDER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def set_style() -> None:
    """Apply the shared figure style. Call once, at the top of the notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "figure.dpi": 110,
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["ink_soft"],
            "axes.labelcolor": PALETTE["ink_soft"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink_soft"],
            "ytick.color": PALETTE["ink_soft"],
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def _hours(schedule: pd.DataFrame) -> np.ndarray:
    """Hour of day as a float, so every intraday figure shares one x-axis."""
    time = pd.to_datetime(schedule["time"])
    return (time.dt.hour + time.dt.minute / 60).to_numpy()


# --------------------------------------------------------------------------------------
# One day
# --------------------------------------------------------------------------------------
def plot_dispatch(schedule: pd.DataFrame, cfg, title: str = "Dispatch") -> plt.Figure:
    """The operator's view of one day: price, power flows and state of charge, aligned.

    Three panels rather than three curves on two axes. The alignment is the whole point -
    reading downwards from a price peak to the discharge bar beneath it to the falling SoC
    below that is how one checks the schedule makes sense, and it requires the panels to share
    an x-axis, not a y-axis.
    """
    hours = _hours(schedule)
    b = cfg.battery
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 1]})

    ax = axes[0]
    ax.plot(hours, schedule["price"], color=PALETTE["price"])
    ax.set_ylabel("Price\n[EUR/MWh]")
    ax.set_title(title)
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.8)

    ax = axes[1]
    ax.step(hours, schedule["P_ch"], where="post", color=PALETTE["charge"], label="Charge from grid")
    ax.step(hours, -schedule["P_dis"], where="post", color=PALETTE["discharge"], label="Discharge to grid")
    if "P_pv_to_batt" in schedule:
        ax.step(hours, schedule["P_pv_to_batt"], where="post", color=PALETTE["pv"],
                linestyle="--", label="PV into battery")
    if "P_pv_curt" in schedule and schedule["P_pv_curt"].abs().max() > 1e-6:
        ax.step(hours, schedule["P_pv_curt"], where="post", color=PALETTE["muted"],
                linestyle=":", label="PV curtailed")
    ax.axhline(0, color=PALETTE["ink_soft"], linewidth=0.8)
    ax.set_ylabel("Power\n[MW]")
    ax.legend(ncols=2, fontsize=8.5, loc="upper left")

    ax = axes[2]
    ax.plot(hours, schedule["E"], color=PALETTE["soc"])
    ax.axhline(b.e_max_mwh, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.axhline(b.e_min_mwh, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.annotate(f"operating window  {b.soc_min_frac:.0%}-{b.soc_max_frac:.0%} SoC",
                xy=(0.5, b.e_max_mwh), fontsize=8, color=PALETTE["ink_soft"], va="bottom")
    ax.set_ylim(-0.05 * b.e_nominal_mwh, 1.05 * b.e_nominal_mwh)
    ax.set_ylabel("Stored energy\n[MWh]")
    ax.set_xlabel("Hour of day")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))

    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def plot_shadow_price(schedule: pd.DataFrame, cfg, title: str = "What the optimiser is really trading on") -> plt.Figure:
    """Market price against the marginal value of stored energy - both in EUR/MWh.

    These two quantities share a unit, so they belong on one axis, and putting them there is
    what makes the decision rule visible: the battery charges while the price sits below the
    value of storing it and discharges while it sits above. The shaded band is the region in
    which neither is worth doing, whose width is set by round-trip efficiency, network tariffs
    and the cost of a cycle - the dead band every real trading desk works around.
    """
    if "soc_shadow_price" not in schedule:
        raise ValueError(
            "This schedule carries no dual variables. They exist only for a continuous model - "
            "set battery.forbid_simultaneous_ch_dis=False to recover them."
        )
    hours = _hours(schedule)
    b, market = cfg.battery, cfg.market
    c_degr = cfg.degradation_cost_eur_per_mwh_throughput()
    lam = schedule["soc_shadow_price"].to_numpy()

    # Price levels at which charging and discharging become individually worthwhile, given the
    # current marginal value of stored energy - i.e. the two edges of the no-trade band.
    charge_trigger = b.eta_ch * lam - market.grid_tariff_charge_eur_per_mwh - c_degr
    discharge_trigger = lam / b.eta_dis + market.grid_tariff_discharge_eur_per_mwh + c_degr

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.fill_between(hours, charge_trigger, discharge_trigger, color=PALETTE["muted"], alpha=0.18,
                    label="No-trade band (losses + tariffs + cycle cost)")
    ax.plot(hours, schedule["price"], color=PALETTE["price"], label="Day-ahead price")
    ax.plot(hours, lam, color=PALETTE["shadow"], linestyle="--", label="Marginal value of stored energy")

    charging = schedule["P_ch"].to_numpy() > 1e-6
    discharging = schedule["P_dis"].to_numpy() > 1e-6
    ax.scatter(hours[charging], schedule["price"].to_numpy()[charging], s=26,
               color=PALETTE["charge"], zorder=3, label="Charging")
    ax.scatter(hours[discharging], schedule["price"].to_numpy()[discharging], s=26,
               color=PALETTE["discharge"], marker="D", zorder=3, label="Discharging")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(title)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=8.5, ncols=2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_depth_histogram(half_cycles: pd.DataFrame, cfg, title: str = "How the energy was cycled") -> plt.Figure:
    """Distribution of half-cycle depths, with the warranty's reference depth marked.

    The reference line is what makes the histogram interpretable rather than merely
    descriptive: swings to its left are gentler than the datasheet assumes and swings to its
    right harsher, and under a convex ageing law that is the difference between a warranty
    that holds and one that does not.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if half_cycles.empty:
        ax.text(0.5, 0.5, "The battery did not cycle", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["ink_soft"])
    else:
        ax.hist(half_cycles["depth_of_discharge"] * 100, bins=12,
                color=PALETTE["soc"], edgecolor=PALETTE["surface"], linewidth=1.5)
        ref = cfg.degradation.reference_dod * 100
        ax.axvline(ref, color=PALETTE["discharge"], linestyle="--", linewidth=1.5)
        ax.annotate(f"warranty reference\n{ref:.0f}% DoD", xy=(ref, ax.get_ylim()[1] * 0.85),
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                    color=PALETTE["ink_soft"])
    ax.set_xlabel("Depth of each half-cycle [% of nominal energy]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Many days
# --------------------------------------------------------------------------------------
def plot_daily_margin(daily: pd.DataFrame, title: str = "Daily gross margin") -> plt.Figure:
    """Margin per day, with the mean marked.

    Drawn as bars in date order rather than as a distribution because the asymmetry is the
    finding: a small number of volatile days carry a disproportionate share of the annual
    result, and a histogram hides which days those were.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    values = daily["gross_margin_eur"].to_numpy()
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in values]
    ax.bar(range(len(values)), values, color=colors, width=0.75)
    mean = float(values.mean())
    ax.axhline(mean, color=PALETTE["ink_soft"], linestyle="--", linewidth=1.2)
    ax.annotate(f"mean {mean:,.0f} EUR/day", xy=(len(values) * 0.98, mean),
                xytext=(0, 5), textcoords="offset points", ha="right",
                fontsize=9, color=PALETTE["ink_soft"])
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(d) for d in daily.index], rotation=90, fontsize=7)
    ax.set_ylabel("Gross margin [EUR]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_margin_vs_volatility(daily: pd.DataFrame) -> plt.Figure:
    """Daily margin against that day's price dispersion, with a fitted line.

    An arbitrage asset is paid for price *variation*, not price level: a flat expensive day
    offers nothing to shift. This scatter is the test of that claim, and it is worth running
    because a weak relationship here would mean the schedule is earning its money somewhere
    other than where the model says it is.
    """
    x = daily["price_std"].to_numpy()
    y = daily["gross_margin_eur"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, color=PALETTE["price"], s=34, alpha=0.85, edgecolor=PALETTE["surface"], linewidth=1)
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 50)
        ax.plot(grid, slope * grid + intercept, color=PALETTE["muted"], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Within-day price standard deviation [EUR/MWh]")
    ax.set_ylabel("Gross margin [EUR/day]")
    ax.set_title(f"Margin tracks volatility, not price level  (r = {corr:.2f})")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Comparisons and sweeps
# --------------------------------------------------------------------------------------
def plot_strategy_comparison(
    table: pd.DataFrame,
    metrics: tuple[str, ...] = ("gross_margin_eur_per_day", "equivalent_full_cycles_per_day",
                                "effective_lifetime_years", "npv_eur"),
    labels: tuple[str, ...] | None = None,
) -> plt.Figure:
    """One panel per metric, strategies on a shared y-axis.

    Small multiples rather than a grouped bar chart or a twin axis: the metrics have
    incompatible units - EUR, cycles, years - and any attempt to place them on one scale would
    turn a legitimate comparison into an artefact of normalisation. Reading across a row gives
    a strategy's full profile, which is the question a comparison is asked to answer.
    """
    labels = labels or metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.5 * len(table) + 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    names = list(table.index)
    y = np.arange(len(names))

    for ax, metric, label in zip(axes, metrics, labels):
        values = table[metric].to_numpy(dtype=float)
        colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in values]
        ax.barh(y, values, color=colors, height=0.6)
        ax.axvline(0, color=PALETTE["ink_soft"], linewidth=0.8)
        ax.set_title(label, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        span = max(abs(values.max()), abs(values.min())) or 1.0
        for yi, v in zip(y, values):
            ax.annotate(f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}",
                        xy=(v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=8, color=PALETTE["ink_soft"])
        ax.set_xlim(min(0, values.min()) - 0.18 * span, max(0, values.max()) + 0.25 * span)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def plot_sensitivity(
    sweep: pd.DataFrame,
    parameter: str,
    metrics: tuple[str, ...] = ("gross_margin_eur_per_day", "equivalent_full_cycles_per_day", "npv_eur"),
    labels: tuple[str, ...] | None = None,
    title: str | None = None,
    mark_best: str | None = "npv_eur",
) -> plt.Figure:
    """Stacked panels showing how each metric responds to one swept parameter.

    Stacked and x-aligned rather than overlaid on twin axes, for the reason given in the module
    docstring. ``mark_best`` draws a vertical line at the parameter value maximising the named
    metric, which is usually the point of running the sweep in the first place.
    """
    labels = labels or metrics
    x = sweep[parameter].to_numpy(dtype=float)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.1 * len(metrics) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)

    best_x = None
    if mark_best and mark_best in sweep:
        best_x = float(x[int(np.nanargmax(sweep[mark_best].to_numpy(dtype=float)))])

    for ax, metric, label, color in zip(axes, metrics, labels, _SERIES_ORDER):
        ax.plot(x, sweep[metric].to_numpy(dtype=float), "o-", color=color)
        ax.set_ylabel(label, fontsize=9)
        if best_x is not None:
            ax.axvline(best_x, color=PALETTE["muted"], linestyle="--", linewidth=1.1)

    if best_x is not None:
        axes[0].annotate(f"best {mark_best}\nat {parameter} = {best_x:g}",
                         xy=(best_x, axes[0].get_ylim()[1]), xytext=(5, -12),
                         textcoords="offset points", fontsize=8.5, color=PALETTE["ink_soft"], va="top")
    axes[-1].set_xlabel(parameter)
    axes[0].set_title(title or f"Sensitivity to {parameter}")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def plot_capex_breakeven(
    sweeps: dict[str, pd.DataFrame],
    breakevens: dict[str, float] | None = None,
    title: str = "Project NPV against installed cost",
) -> plt.Figure:
    """NPV as a function of CAPEX, for one or more sweeps drawn on the same axes.

    All series share a unit here, so one axis is correct. Drawing the fixed-schedule and
    re-optimised sweeps together is the point: the vertical gap between them is the value the
    optimiser recovers by adapting its cycling to the price of the asset, which a single curve
    cannot show.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for (label, sweep), color in zip(sweeps.items(), _SERIES_ORDER):
        ax.plot(sweep["capex_eur_per_mwh"], sweep["npv_eur"], "o-", color=color, label=label, markersize=4)
        if breakevens and label in breakevens and np.isfinite(breakevens[label]):
            ax.axvline(breakevens[label], color=color, linestyle="--", linewidth=1.1, alpha=0.8)
            ax.annotate(f"{breakevens[label]:,.0f}", xy=(breakevens[label], 0),
                        xytext=(3, 8), textcoords="offset points", fontsize=8.5, color=color)
    ax.axhline(0, color=PALETTE["ink_soft"], linewidth=0.9)
    ax.set_xlabel("Battery CAPEX [EUR/MWh of nominal energy]")
    ax.set_ylabel("Project NPV [EUR]")
    ax.set_title(title)
    if len(sweeps) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_cashflows(flows: pd.DataFrame, title: str = "Project cash flows") -> plt.Figure:
    """Cash flow per period with the discounted cumulative line beneath it."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    years = flows["year"].to_numpy(dtype=float)
    net = flows["net_cash_flow_eur"].to_numpy(dtype=float)

    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in net]
    axes[0].bar(years, net, color=colors, width=0.7)
    axes[0].axhline(0, color=PALETTE["ink_soft"], linewidth=0.8)
    axes[0].set_ylabel("Net cash flow\n[EUR]")
    axes[0].set_title(title)

    axes[1].plot(years, np.cumsum(net), "o-", color=PALETTE["price"], markersize=4)
    axes[1].axhline(0, color=PALETTE["ink_soft"], linewidth=0.8)
    axes[1].set_ylabel("Cumulative\n[EUR]")
    axes[1].set_xlabel("Year")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig
