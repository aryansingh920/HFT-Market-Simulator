"""
Created on 03/02/2025

@author: Aryan

Filename: MarketSimulator.py

Relative Path: server/MarketModel/MarketSimulator.py
"""


# ============================================================================
# Super Class that ties Simulation & Dashboard together
# ============================================================================
from MarketModel import StockPriceSimulator
from MarketModel.Dashboard import Dashboard


class MarketSimulator:
    def __init__(self, configs):
        """
        configs: list of configuration dictionaries (each with a "name" key, etc.)
        """
        self.configs = configs
        self.results = []
        self.config_names = []
        self.config_details = []  # Keep track of each simulation config

    def run_simulations(self):
        for idx, config in enumerate(self.configs, 1):
            name = config.get("name", f"Simulation #{idx}")
            print(f"Running simulation: {name}")
            simulator = StockPriceSimulator.StockPriceSimulator(**config)
            result = simulator.simulate()
            self.results.append(result)
            self.config_names.append(name)
            self.config_details.append(config)  # Store config

    def launch_dashboard(self):
        dashboard = Dashboard(
            self.results, self.config_names, self.config_details)
        dashboard.run()
