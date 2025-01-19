"""
Created on 19/01/2025

@author: Aryan

Filename: param_utils.py

Relative Path: server/Market/param_utils.py
"""


import random


def dynamic_lambda_rate(initial_price, heat_duration_minutes,
                        base_lambda=10.0, scaling_factor=0.1):
    """
    Compute order arrival rate (orders/sec) as a function of:
      - initial_price
      - heat_duration_minutes
      - base_lambda and scaling_factor (tunable constants)
    """
    # Example logic:
    #   - scale by initial_price / 100
    #   - inversely scale by heat_duration_minutes (shorter heat => more intense trading?)
    rate = base_lambda + scaling_factor * \
        (initial_price / 100.0) * (1.0 / max(0.1, heat_duration_minutes))
    # Clip to a reasonable range, e.g. [1, 1000]
    return max(1, min(rate, 1000))


def dynamic_initial_liquidity(initial_price, heat_duration_minutes,
                              base_liquidity=5, price_factor=0.05):
    """
    Compute how many limit orders to seed the book with at the start.
      - Possibly scale with initial_price
      - Possibly scale with heat_duration_minutes
    """
    # Example logic:
    #   - base_liquidity + (some fraction of initial_price)
    #   - if heat_duration is short, maybe we reduce initial liquidity slightly, etc.
    liquidity = base_liquidity + int(price_factor * (initial_price / 10.0))
    return max(1, liquidity)


def dynamic_mu(initial_price, base_mu=0.01):
    """
    Compute drift (annualized or short-term) based on initial_price or other factors.
    For HFT, mu is often quite small, so the logic can be minimal.
    """
    # Example: if price is very large, we might assume a slightly higher/lower drift
    # purely for demonstration.
    if initial_price > 1000:
        return base_mu * 1.5  # 1.5% if a higher-priced stock
    else:
        return base_mu


def dynamic_sigma(initial_price, base_sigma=0.05):
    """
    Compute volatility based on initial_price.
    Higher-priced symbols might be less volatile in percentage terms, or vice versa.
    """
    # Example: slightly reduce volatility if the stock is very high-priced
    if initial_price > 1000:
        return max(0.01, base_sigma * 0.8)  # reduce 20% if big cap
    else:
        return base_sigma
