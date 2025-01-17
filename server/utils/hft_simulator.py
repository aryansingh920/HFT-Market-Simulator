"""
Created on 15/01/2025

@author: Aryan

Filename: hft_simulator.py

Relative Path: server/utils/hft_simulator.py
"""

from Market.MarketSimulator import MarketSimulator
from Market.Graph import create_dash_app, plot_symbol_candlestick

if __name__ == "__main__":
    simulation_config = {
        "name": "MyHFTSimulation",
        "lambda_rate": 10,
        "initial_liquidity": 5,
        "symbols": ["AAPL", "GOOG"],
        "heat_duration_minutes": 0.5,
        "mu": 0.0,
        "sigma": 0.05,
        "initial_stock_prices": {
                "AAPL": 120.0,
                "GOOG": 1500.0
        }
    }
    # simulator.run(steps=1000)
    simulator = MarketSimulator(config=simulation_config)
    # simulator.run(steps=1000)

    symbol_csv_file = "./simulation_logs/heat_1/symbol_GOOG.csv"
    frequency = '1ms'  # or '1S', '100ms', etc.
    plot_symbol_candlestick(
        symbol_csv_path=symbol_csv_file, freq=frequency)

    # dash_app = create_dash_app(symbol_csv_file, freq=frequency)
    # dash_app.run_server(debug=True, port=8051)


    # Check your "simulation_logs" directory for the new CSV files.
