"""
Created on 03/02/2025

@author: Aryan

Filename: main.py

Relative Path: server/main.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from MarketModel.StockPriceSimulatorWithOrderBook import StockPriceSimulatorWithOrderBook
from MarketModel.IntradayStockPriceSimulatorWithOrderBook import IntradayStockPriceSimulatorWithOrderBook
from MarketModel.Dashboard import Dashboard
from config import configs_nvidia, configs_pure_gbm, configs_test, intraday_config
from MarketModel.DataLogger import save_simulation_steps_csv, save_orderbook_snapshots_csv, save_config_as_json
import os

# if __name__ == "__main__":
#     results = []
#     config_names = []
#     config_details = []

#     simulator = IntradayStockPriceSimulator(**intraday_config)

#     # Run the simulation
#     simulator.run_simulation()

#     # Retrieve results as a pandas DataFrame
#     results = simulator.get_results()

#     # For each configuration in configs_nvidia, instantiate and run the new simulator.
#     for idx, config in enumerate(configs_test, 1):
#         name = config.get("name", f"Simulation_{idx}")
#         print(f"Running simulation with order book: {name}")

#         simulator = StockPriceSimulatorWithOrderBook(**config)
#         result = simulator.simulate()
#         results.append(result)
#         config_names.append(name)
#         config_details.append(config)

#         # Create an output folder if needed
#         output_folder = "backend/simulation_output"
#         os.makedirs(output_folder, exist_ok=True)

#         # Save simulation step data
#         steps_filename = os.path.join(output_folder, f"{name}_steps.csv")
#         save_simulation_steps_csv(result, steps_filename)

#         # Save order book snapshots
#         orderbook_filename = os.path.join(
#             output_folder, f"{name}_orderbook.csv")
#         save_orderbook_snapshots_csv(result, orderbook_filename)

#         # Optionally, save the configuration once per simulation.
#         config_filename = os.path.join(output_folder, f"{name}_config.json")
#         save_config_as_json(config, config_filename)

#     # Optionally, launch the dashboard to visualize the simulation.
#     dashboard = Dashboard(results, config_names, config_details)
#     dashboard.run()


if __name__ == "__main__":
    # Instantiate intraday simulator with preloaded configs
    simulator = IntradayStockPriceSimulatorWithOrderBook(
        initial_price=intraday_config["intraday_config"]["initial_price"],
        fundamental_value=intraday_config["intraday_config"]["fundamental_value"],
        steps_per_day=intraday_config["intraday_config"]["steps_per_day"],
        base_volatility=intraday_config["intraday_config"]["base_volatility"],
        regimes=intraday_config["intraday_regimes"],
        transition_probabilities=intraday_config["intraday_transition_probabilities"],
        random_seed=intraday_config["intraday_config"]["random_seed"]
    )

    # Run the simulation
    results = simulator.simulate()

    # Extract values
    prices = results["prices"]
    # Changed "times" to "time" to match the dictionary
    times = results["time"]

    # Visualization
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=times, y=prices, mode='lines+markers',
                   name='Intraday Price'),
        row=1, col=1
    )
    fig.update_layout(
        title="Intraday Price Simulation (Single Trading Day)",
        xaxis_title="Intraday Steps",
        yaxis_title="Price",
        template="plotly_dark"
    )
    fig.show()
