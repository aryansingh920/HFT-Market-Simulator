"""
Created on 03/02/2025

@author: Aryan

Filename: main.py

Relative Path: server/main.py
"""


from MarketModel.StockPriceSimulatorWithOrderBook import StockPriceSimulatorWithOrderBook
from MarketModel.Dashboard import Dashboard
from config import configs_nvidia
from MarketModel.DataLogger import save_simulation_steps_csv, save_orderbook_snapshots_csv, save_config_as_json
import os

if __name__ == "__main__":
    results = []
    config_names = []
    config_details = []

    # For each configuration in configs_nvidia, instantiate and run the new simulator.
    for idx, config in enumerate(configs_nvidia, 1):
        name = config.get("name", f"Simulation_{idx}")
        print(f"Running simulation with order book: {name}")

        simulator = StockPriceSimulatorWithOrderBook(**config)
        result = simulator.simulate()
        results.append(result)
        config_names.append(name)
        config_details.append(config)

        # Create an output folder if needed
        output_folder = "simulation_output"
        os.makedirs(output_folder, exist_ok=True)

        # Save simulation step data
        steps_filename = os.path.join(output_folder, f"{name}_steps.csv")
        save_simulation_steps_csv(result, steps_filename)

        # Save order book snapshots
        orderbook_filename = os.path.join(
            output_folder, f"{name}_orderbook.csv")
        save_orderbook_snapshots_csv(result, orderbook_filename)

        # Optionally, save the configuration once per simulation.
        config_filename = os.path.join(output_folder, f"{name}_config.json")
        save_config_as_json(config, config_filename)

    # Optionally, launch the dashboard to visualize the simulation.
    dashboard = Dashboard(results, config_names, config_details)
    dashboard.run()


