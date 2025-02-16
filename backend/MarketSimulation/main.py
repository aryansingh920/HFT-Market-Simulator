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
    # Example: define intraday regime logic similar to your original code
    # Just as an illustration—use the same or a simplified set of regimes as in your daily simulator.
    intraday_regimes = [
        {
            'name': 'morning_session',
            'drift': 0.00,
            'vol_scale': 1.5,
        },
        {
            'name': 'midday_lull',
            'drift': 0.00,
            'vol_scale': 0.8,
        },
        {
            'name': 'afternoon_ramp',
            'drift': 0.00,
            'vol_scale': 1.2,
        },
    ]

    # Example transition matrix for the single day. Adjust as desired:
    # With 390 steps in a day, you might have a few transitions to different intraday "phases."
    intraday_transition_probabilities = {
        'morning_session': {'morning_session': 0.90, 'midday_lull': 0.10},
        'midday_lull':     {'midday_lull': 0.95, 'afternoon_ramp': 0.05},
        'afternoon_ramp':  {'afternoon_ramp': 0.98}
    }

    # Instantiate our intraday simulator
    simulator = IntradayStockPriceSimulatorWithOrderBook(
        initial_price=100.0,
        fundamental_value=100.0,
        steps_per_day=390,  # 1-minute intervals
        base_volatility=0.005,
        regimes=intraday_regimes,
        transition_probabilities=intraday_transition_probabilities,
        random_seed=2025
    )

    # Run the simulation
    results = simulator.simulate()

    # results is typically a dictionary with keys like "prices", "times", etc.
    prices = results["prices"]
    # The times might be a range [0..1], effectively representing a single day
    times = results["time"]

    # Visualize with Plotly
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
        template="plotly_dark"  # or any other layout
    )
    fig.show()
