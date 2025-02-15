"""
Created on 03/02/2025

@author: Aryan

Filename: AdvancedStockSimulator.py

Relative Path: server/MarketModel/AdvancedStockSimulator.py
"""

import numpy as np
from MarketModel import StockPriceSimulator


class AdvancedStockSimulator:
    def __init__(self, config):
        """
        Expects a configuration dict that may include:
          - sentiment_seed, news_flow_intensity, seasonality_params,
          - heston_params, refined_jump_params, etc.
        """
        self.config = config
        self.base_simulator = StockPriceSimulator(**config)
        self.sentiment_seed = config.get("sentiment_seed", 0.0)
        self.news_flow_intensity = config.get("news_flow_intensity", 0.05)
        self.seasonality_params = config.get(
            "seasonality_params", {"day_of_week": [1.0, 1.01, 0.99, 1.02, 0.98]})
        self.heston_params = config.get(
            "heston_params", {"kappa": 1.0, "theta": 0.04, "eta": 0.1})
        self.refined_jump_params = config.get(
            "refined_jump_params", {"intensity": 0.05, "df": 3})

    def simulate(self):
        results = self.base_simulator.simulate()
        results = self.apply_news_flow(results)
        results = self.apply_seasonality(results)
        results = self.apply_advanced_volatility(results)
        results = self.apply_refined_jump_models(results)
        results = self.apply_liquidity_transaction_costs(results)
        return results

    def apply_news_flow(self, results):
        """Inject random news-driven sentiment shocks into the simulation."""
        times = results["time"]
        sentiment_adjustments = np.zeros(len(times))
        for i in range(len(times)):
            if np.random.rand() < self.news_flow_intensity:
                shock = np.random.normal(self.sentiment_seed, 0.1)
                sentiment_adjustments[i] = shock
        results["news_sentiment"] = sentiment_adjustments.tolist()
        return results

    def apply_seasonality(self, results):
        """Apply a basic day-of-week effect (or any other pattern)."""
        times = results["time"]
        seasonal_factor = []
        day_factors = self.seasonality_params.get(
            "day_of_week", [1.0, 1.01, 0.99, 1.02, 0.98])
        for i, t in enumerate(times):
            day_index = i % len(day_factors)
            seasonal_factor.append(day_factors[day_index])
        results["seasonality_factor"] = seasonal_factor
        return results

    def apply_advanced_volatility(self, results):
        """Recalculate a volatility series using a Heston–style model."""
        times = results["time"]
        if len(times) < 2:
            return results  # Not enough points
        dt = times[1] - times[0]

        kappa = self.heston_params.get("kappa", 1.0)
        theta = self.heston_params.get("theta", 0.04)
        eta = self.heston_params.get("eta", 0.1)

        v = theta
        heston_vol = []
        for _ in times:
            heston_vol.append(np.sqrt(v))
            dW = np.random.normal(0, np.sqrt(dt))
            v = max(v + kappa * (theta - v) * dt +
                    eta * np.sqrt(max(v, 0)) * dW, 1e-8)
        results["advanced_volatilities"] = heston_vol
        return results

    def apply_refined_jump_models(self, results):
        """Adds extra jump events using a heavy–tailed (t–distribution) model."""
        prices = np.array(results["prices"])
        times = results["time"]
        jump_intensity = self.refined_jump_params.get("intensity", 0.05)
        df = self.refined_jump_params.get("df", 3)

        refined_jumps = np.zeros(len(prices))
        for i in range(len(prices)):
            if np.random.rand() < jump_intensity:
                jump = np.random.standard_t(df)
                refined_jumps[i] = jump
                # For demonstration, scale price by 5% of the jump
                prices[i] = prices[i] * (1 + 0.05 * jump)

        results["refined_jumps"] = refined_jumps.tolist()
        results["prices"] = prices.tolist()
        return results

    def apply_liquidity_transaction_costs(self, results):
        """Enhance transaction cost estimates based on volatility."""
        volatilities = np.array(results["volatilities"])
        base_tc = self.config.get("transaction_cost", 0.0005)
        if len(volatilities) == 0:
            # no data
            results["effective_transaction_costs"] = []
            return results
        vol_ratio = volatilities / np.mean(volatilities)
        effective_costs = base_tc * (1 + vol_ratio)
        results["effective_transaction_costs"] = effective_costs.tolist()
        return results

    def calibrate_parameters(self, historical_data):
        """Placeholder: Calibrate model parameters from real data."""
        calibrated_params = {}
        calibrated_params["garch_params"] = (0.01, 0.15, 0.80)
        calibrated_params["macro_impact"] = {'interest_rate': (0.03, 0.008),
                                             'inflation': (0.02, 0.004)}
        self.config.update(calibrated_params)
        return calibrated_params

    def stress_test(self, extreme_shock=0.3):
        """Apply an extreme shock for stress testing."""
        stressed_config = self.config.copy()
        stressed_config["base_volatility"] = self.config.get(
            "base_volatility", 0.2) * 2
        stressed_config["initial_liquidity"] = self.config.get(
            "initial_liquidity", 1e6) * 0.5
        stressed_simulator = StockPriceSimulator(**stressed_config)
        stressed_results = stressed_simulator.simulate()
        # Example: apply a uniform drop to all prices
        stressed_results["prices"] = [
            p * (1 - extreme_shock) for p in stressed_results["prices"]]
        return stressed_results
