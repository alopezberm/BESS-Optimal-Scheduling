"""Small, consistent matplotlib styling shared by every figure in the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = {
    "price": "#2661D8",
    "pv": "#F2A20C",
    "soc": "#2CA05A",
    "charge": "#2661D8",
    "discharge": "#D8365B",
    "curtailment": "#9A9CA5",
    "profit": "#2CA05A",
    "npv_pos": "#2CA05A",
    "npv_neg": "#D8365B",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )


def plot_price_and_pv(schedule: pd.DataFrame, title: str) -> plt.Figure:
    hours = schedule["time"].dt.hour + schedule["time"].dt.minute / 60
    fig, ax = plt.subplots()
    ax.plot(hours, schedule["price"], color=PALETTE["price"], label="Day-ahead price [EUR/MWh]")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Price [EUR/MWh]")
    ax2 = ax.twinx()
    ax2.plot(hours, schedule["P_pv"], color=PALETTE["pv"], label="PV [MW]")
    ax2.set_ylabel("PV power [MW]")
    ax2.grid(False)
    ax.set_title(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88), frameon=False)
    return fig


def plot_soc(schedule: pd.DataFrame, e_max_mwh: float, title: str) -> plt.Figure:
    hours = schedule["time"].dt.hour + schedule["time"].dt.minute / 60
    fig, ax = plt.subplots()
    ax.plot(hours, schedule["E"], color=PALETTE["soc"], label="State of charge [MWh]")
    ax.axhline(e_max_mwh, color=PALETTE["curtailment"], linestyle="--", linewidth=1, label="Capacity")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy [MWh]")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def plot_charge_discharge(schedule: pd.DataFrame, title: str) -> plt.Figure:
    hours = schedule["time"].dt.hour + schedule["time"].dt.minute / 60
    fig, ax = plt.subplots()
    ax.step(hours, schedule["P_ch"], where="post", color=PALETTE["charge"], label="Charge [MW]")
    ax.step(hours, -schedule["P_dis"], where="post", color=PALETTE["discharge"], label="Discharge [MW]")
    if "P_pv_curt" in schedule:
        ax.step(hours, schedule["P_pv_curt"], where="post", color=PALETTE["curtailment"], label="PV curtailed [MW]")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power [MW]")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def plot_profit_by_day(results: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(results)), results["profit_eur"], color=PALETTE["profit"])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([str(d) for d in results["date"]], rotation=90)
    ax.set_xlabel("Day")
    ax.set_ylabel("Profit [EUR]")
    ax.set_title(title)
    return fig


def plot_capex_breakeven(sweep: pd.DataFrame, breakeven_capex: float) -> plt.Figure:
    fig, ax = plt.subplots()
    colors = [PALETTE["npv_pos"] if v >= 0 else PALETTE["npv_neg"] for v in sweep["npv_eur"]]
    ax.bar(sweep["capex_eur_per_mwh"], sweep["npv_eur"], width=sweep["capex_eur_per_mwh"].diff().median() * 0.9, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    if breakeven_capex == breakeven_capex:  # not NaN
        ax.axvline(breakeven_capex, color="black", linestyle="--", linewidth=1)
        ax.annotate(
            f"break-even\n~{breakeven_capex:,.0f} EUR/MWh",
            xy=(breakeven_capex, 0),
            xytext=(breakeven_capex, sweep["npv_eur"].max() * 0.5),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Battery CAPEX [EUR/MWh]")
    ax.set_ylabel("Project NPV [EUR]")
    ax.set_title("NPV vs. battery CAPEX - arbitrage-only revenue")
    return fig
