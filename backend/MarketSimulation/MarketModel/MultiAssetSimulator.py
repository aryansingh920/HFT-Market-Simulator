"""
Created on 03/02/2025

@author: Aryan

Filename: MultiAssetSimulator.py

Relative Path: server/MarketModel/MultiAssetSimulator.py
"""

# ============================================================================
# Multi-Asset Simulator (for two or more stocks with correlations)
# ============================================================================


import numpy as np

from MarketModel import StockPriceSimulator


class MultiAssetSimulator:
    def __init__(self, configs, correlation_matrix):
        """
        configs: list of configuration dictionaries (one per asset)
        correlation_matrix: symmetric matrix defining correlation between assets
        """
        self.configs = configs
        self.correlation_matrix = np.array(correlation_matrix)
        self.num_assets = len(configs)
        self.simulators = [StockPriceSimulator(**cfg) for cfg in configs]

    def simulate(self):
        # Run the base simulation for each asset
        base_results = [sim.simulate() for sim in self.simulators]
        times = base_results[0]["time"]
        num_steps = len(times)

        # Create correlated random draws
        market_shocks = np.random.multivariate_normal(np.zeros(self.num_assets),
                                                      self.correlation_matrix,
                                                      size=num_steps)
        for asset_idx, res in enumerate(base_results):
            adjustment = market_shocks[:, asset_idx]
            # Normalize so the factor is 1.0 at t=0
            if adjustment[0] != 0:
                norm_factor = adjustment / adjustment[0]
            else:
                norm_factor = np.ones_like(adjustment)
            new_prices = [p * factor for p,
                          factor in zip(res["prices"], norm_factor)]
            res["prices"] = new_prices
            res["market_adjustment"] = norm_factor.tolist()

        return base_results
