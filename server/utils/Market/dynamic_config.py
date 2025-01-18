"""
dynamic_config.py
"""

from Market.param_utils import (
    dynamic_lambda_rate,
    dynamic_initial_liquidity,
    dynamic_mu,
    dynamic_sigma
)


def build_symbol_config(symbol, initial_price, heat_duration_minutes=0.5):
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
        "heat_duration_minutes": heat_duration_minutes
    }


def build_simulation_config(sim_name="MyHFTSimulation"):
    symbol_prices = {
        "AAPL": 120.0,
        "GOOG": 1500.0,
        "TSLA": 700.0
    }
    heat_time = 0.25  # 30 seconds per heat

    symbols_config = {}
    for sym, price in symbol_prices.items():
        symbols_config[sym] = build_symbol_config(
            symbol=sym,
            initial_price=price,
            heat_duration_minutes=heat_time
        )

    simulation_config = {
        "name": sim_name,
        "symbols_config": symbols_config,
        "global_heat_duration": heat_time
    }
    return simulation_config
