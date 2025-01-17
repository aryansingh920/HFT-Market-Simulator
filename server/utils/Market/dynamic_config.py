"""
dynamic_config.py

Demonstrates how to build a dynamic simulation config using the param_utils.
"""

from Market.param_utils import (
    dynamic_lambda_rate,
    dynamic_initial_liquidity,
    dynamic_mu,
    dynamic_sigma
)


def build_symbol_config(symbol, initial_price, heat_duration_minutes=0.5):
    """
    Build a dictionary of parameters for a single symbol,
    based on that symbol's initial_price and the heat_duration_minutes.
    """
    return {
        "initial_price": initial_price,
        "lambda_rate": dynamic_lambda_rate(
            initial_price=initial_price,
            heat_duration_minutes=heat_duration_minutes
        ),
        "initial_liquidity": dynamic_initial_liquidity(
            initial_price=initial_price,
            heat_duration_minutes=heat_duration_minutes
        ),
        "mu": dynamic_mu(
            initial_price=initial_price
        ),
        "sigma": dynamic_sigma(
            initial_price=initial_price
        ),
        # If you want each symbol to have its own per-heat time, you can store it here too
        "heat_duration_minutes": heat_duration_minutes
    }


def build_simulation_config(sim_name="MyHFTSimulation"):
    """
    Build the top-level simulation config referencing symbol-specific configs.
    Feel free to add more symbols or logic as desired.
    """
    # Suppose we define some initial prices here:
    symbol_prices = {
        "AAPL": 120.0,
        "GOOG": 1500.0,
        "TSLA": 700.0
    }
    heat_time = 0.5  # 30 seconds per heat

    # Build symbol configs
    symbols_config = {}
    for sym, price in symbol_prices.items():
        symbols_config[sym] = build_symbol_config(
            symbol=sym,
            initial_price=price,
            heat_duration_minutes=heat_time
        )

    # The main simulation config
    simulation_config = {
        "name": sim_name,
        "symbols_config": symbols_config,
        # If you want to handle a 'global' heat_duration_minutes, you can do so here:
        "global_heat_duration": heat_time
    }

    return simulation_config
