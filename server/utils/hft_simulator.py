"""
Created on 15/01/2025

@author: Aryan

Filename: hft_simulator.py

Relative Path: server/utils/hft_simulator.py
"""

from Market.MarketSimulator import MarketSimulator
from Market.Graph import plot_symbol_candlestick

if __name__ == "__main__":
    # simulator = MarketSimulator(
    #     lambda_rate=5,            # Higher means more frequent orders
    #     initial_liquidity=10,      # Some initial orders
    #     symbols=["AAPL", "GOOG"],  # or ["AAPL", "GOOG", "AMZN"]
    #     heat_duration_minutes=1,  # 10-minute heat
    #     mu=0.0,                   # GBM drift
    #     sigma=0.02                # GBM volatility
    # )
    # simulator.run(steps=1000)

    plot_symbol_candlestick(
        symbol_csv_path="./simulation_logs/heat_2/symbol_AAPL.csv", freq='1ms')

    # Check your "simulation_logs" directory for the new CSV files.
