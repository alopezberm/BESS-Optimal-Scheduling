"""bess_opt - optimal scheduling, degradation and economics of a grid-connected battery + PV plant.

Module map, in the order the notebook uses them:

``config``            every tunable parameter, documented with units and typical ranges
``data``              day-ahead prices (Energinet, cached) and the synthetic PV generator
``model``             the Gurobi formulations: baseline, extended, stochastic
``model_opensource``  the same extended model in PuLP/CBC, as an independent cross-check
``degradation``       cycle counting and the EUR cost of ageing
``economics``         discounted cash flow, LCOS, break-even CAPEX
``backtest``          solving many days
``experiments``       strategy comparison and parameter sweeps - the fine-tuning workbench
``plotting``          shared figure styling and the recurring plots
"""

__version__ = "0.2.0"
