"""
Created on 15/01/2025

@author: Aryan

Filename: hft_simulator.py

Relative Path: server/utils/hft_simulator.py
"""

from Market.MarketSimulator import MarketSimulator
from Market.Graph import create_dash_app, plot_symbol_candlestick

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

    symbol_csv_file = "./simulation_logs/heat_1/symbol_AAPL.csv"
    frequency = '1ms'  # or '1S', '100ms', etc.
    plot_symbol_candlestick(
        symbol_csv_path=symbol_csv_file, freq='1ms')

    dash_app = create_dash_app(symbol_csv_file, freq=frequency)
    dash_app.run_server(debug=True, port=8051)


    # Check your "simulation_logs" directory for the new CSV files.
