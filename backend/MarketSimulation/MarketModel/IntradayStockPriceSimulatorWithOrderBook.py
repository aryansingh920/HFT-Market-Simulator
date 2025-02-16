"""
Created on 16/02/2025

@author: Aryan

Filename: IntradayStockPriceSimulatorWithOrderBook.py

Relative Path: backend/MarketSimulation/MarketModel/IntradayStockPriceSimulatorWithOrderBook.py
"""

"""
IntradayStockPriceSimulatorWithOrderBook.py

Example script for simulating intraday price movements (e.g., a single trading day)
using the same underlying logic as StockPriceSimulatorWithOrderBook.py.

Core logic (order book updates, regime shifting, GARCH, etc.) is unchanged.
We merely reduce the duration to 1 day and increase the number of steps to
represent intraday intervals (e.g., minutes).
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from MarketModel.StockPriceSimulatorWithOrderBook import  StockPriceSimulatorWithOrderBook
from MarketModel.OrderBook import OrderBook

# -------------------------------------------------------
# Import the same classes/logic used in StockPriceSimulatorWithOrderBook.
# (Adjust these import paths as needed to match your project structure.)
# -------------------------------------------------------

# -------------------------------------------------------
# Intraday simulator class
# -------------------------------------------------------


class IntradayStockPriceSimulatorWithOrderBook(StockPriceSimulatorWithOrderBook):
    """
    A subclass or specialized simulator for intraday movements (one trading day).
    Inherits the same logic from StockPriceSimulatorWithOrderBook but narrows the
    time horizon to a single session (e.g., 390 one-minute steps).
    """

    def __init__(
        self,
        initial_price=100.0,
        fundamental_value=100.0,
        initial_liquidity=1e6,
        steps_per_day=390,          # e.g. 390 one-minute intervals in a 6.5-hour trading day
        base_volatility=0.01,
        regimes=None,
        transition_probabilities=None,
        # ... pass through any other relevant parameters used in StockPriceSimulatorWithOrderBook ...
        random_seed=42
    ):
        # We set 'duration' effectively to 1 trading day, but the real driver is 'steps_per_day'.
        # The rest of the constructor remains the same as in StockPriceSimulatorWithOrderBook.

        super().__init__(
            initial_price=initial_price,
            fundamental_value=fundamental_value,
            initial_liquidity=initial_liquidity,
            duration=1.0,  # conceptually 1 day (not 1 year)
            steps=int(steps_per_day),
            base_volatility=base_volatility,
            regimes=regimes,
            transition_probabilities=transition_probabilities,
            # pass other parameters...
            random_seed=random_seed
        )

    # The rest of your core logic remains the same because we inherit it.
    # You do not override the update logic, order book logic, or regime transitions.


# -------------------------------------------------------
# Example usage / main execution
# -------------------------------------------------------
