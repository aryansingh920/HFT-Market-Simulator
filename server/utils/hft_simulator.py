"""
hft_simulator.py

Main script to run a dynamic HFT simulation.
"""

from Market.MarketSimulator import MarketSimulator
from Market.dynamic_config import build_simulation_config
from Market.Graph import plot_symbol_candlestick

if __name__ == "__main__":
    # 1) Build the simulation config (dynamically):
    simulation_config = build_simulation_config(
        sim_name="MyDynamicHFTSimulation")

    # 2) Initialize the MarketSimulator with this dynamic config
    simulator = MarketSimulator(config=simulation_config)

    # 3) Run the simulation for a certain number of steps
    simulator.run(steps=100000)

    # 4) Afterwards, pick a symbol CSV and visualize the candlesticks
    #    For example, assume we pick "GOOG" from the logs of heat_1:
    symbol_csv_file = "./simulation_logs/heat_1/symbol_GOOG.csv"
    frequency = '1S'  # or '1S', '100ms', etc.

    plot_symbol_candlestick(symbol_csv_path=symbol_csv_file, freq=frequency)

    # If you want a Dash app:
    # from Market.Graph import create_dash_app
    # dash_app = create_dash_app(symbol_csv_file, freq=frequency)
    # dash_app.run_server(debug=True, port=8051)
