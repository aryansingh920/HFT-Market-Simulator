"""
Created on 01/02/2025

@author: Aryan

Filename: dynamic_regime_shift.py

Relative Path: server/dynamic_regime_shift.py
"""

import numpy as np
import matplotlib.pyplot as plt
from config import configs, configs_historical

def simulate_dynamic_stock_price(
    duration=1,                     # Total simulation time in years
    steps=252,                      # Number of time steps (e.g. trading days)
    initial_price=100,              # Starting price
    fundamental_value=100,          # Initial “true” value
    initial_liquidity=1e6,          # Starting liquidity level
    base_volatility=0.2,            # Annualized base volatility
    # List of regime dictionaries (for Markov switching)
    regimes=None,
    # Optional list of regime–switch tuples (mu, sigma, duration)
    regime_switch=None,
    garch_params=(0.02, 0.1, 0.85),   # (omega, alpha, beta) for GARCH update
    macro_impact={'interest_rate': (0.03, 0.01),
                  'inflation': (0.02, 0.005)},  # (level, volatility)
    sentiment_params=(0.5, 0.2),      # (mean-reversion speed, volatility)
    # (price drop threshold, liquidity threshold)
    flash_crash_threshold=(-0.15, 3),
    # How strongly the market maker counteracts deviations
    market_maker_power=0.1,
    transaction_cost=0.0005,          # Base transaction cost factor
    jump_params=(0.1, 0.02, 0.1),       # (jump_intensity, jump_mean, jump_std)
    # Speed for mean-reversion (when low liquidity)
    mean_reversion_speed=0.1,
    # Long-term mean price for mean-reverting models (default to fundamental)
    long_term_mean=None,
    # Chance of a random market shock (jump diffusion)
    market_shock_prob=0.01,
    # Optionally apply a global shock ("bullish" or "bearish")
    market_shock=None,
    random_seed=None
):
    """
    A simulation that combines many market features:
      - Dynamic regime switching via a Markov chain over provided regimes.
      - Multiple simulation “modes” (GBM, mean–reversion, jump–diffusion, stochastic volatility, extra regime–switching).
      - Macroeconomic factor evolution (interest rate, inflation) affecting drift.
      - GARCH–style volatility updating.
      - Sentiment feedback and liquidity dynamics.
      - Flash–crash conditions and market maker stabilization.
      - Global market shocks.
    
    Adjust parameters to simulate different market environments.
    """
    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = duration / steps
    prices = [initial_price]
    fundamentals = [fundamental_value]
    volatilities = [base_volatility]

    # Set long-term mean to fundamental value if not provided
    if long_term_mean is None:
        long_term_mean = fundamental_value

    # Initialize liquidity and sentiment
    liquidity = initial_liquidity
    sentiment = 0.0
    liquidity_ema = 0.0  # Exponential moving average for volume/liquidity effect

    # Initialize macroeconomic factors from provided tuples
    interest_rate, interest_vol = macro_impact.get(
        'interest_rate', (0.03, 0.01))
    inflation, inflation_vol = macro_impact.get('inflation', (0.02, 0.005))

    # Unpack GARCH parameters (omega, alpha, beta)
    garch_omega, garch_alpha, garch_beta = garch_params

    # --- Set up regime switching via a Markov chain ---
    if regimes is None or len(regimes) == 0:
        regimes = [{'name': 'normal', 'drift': 0.05, 'vol_scale': 1.0,
                    'transitions': {'normal': 1.0}}]
    regime_names = [r['name'] for r in regimes]

    # Build a transition matrix from the regimes’ "transitions" dictionaries.
    transition_matrix = np.zeros((len(regimes), len(regimes)))
    for i, r in enumerate(regimes):
        for j, name in enumerate(regime_names):
            transition_matrix[i, j] = r.get('transitions', {}).get(name, 0.0)
        row_sum = np.sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] /= row_sum
        else:
            transition_matrix[i, i] = 1.0
    current_regime = regime_names[0]

    # --- Optional extra regime switching (from second code sample) ---
    if regime_switch is not None and len(regime_switch) > 0:
        regime_switch_idx = 0
        regime_switch_mu, regime_switch_sigma, regime_switch_duration = regime_switch[
            regime_switch_idx]
        time_in_regime_switch = 0.0

    # For stochastic volatility (Heston–like) model, initialize variance v.
    v = base_volatility**2

    # --- Helper function: choose which simulation model to use this step ---
    def choose_model(t, current_price, liquidity):
        # With a small probability, trigger a jump–diffusion (market shock)
        if np.random.rand() < market_shock_prob:
            return "jump_diffusion"
        # If liquidity is low, use a mean–reverting model
        if liquidity < 0.5 * initial_liquidity:
            return "mean_reverting"
        # If an alternate regime switch parameter is provided, sometimes use it
        if regime_switch is not None and (t % max(1, int(steps/10)) == 0):
            return "regime_switching"
        # In the middle period, use stochastic volatility
        if steps//3 < t < 2*steps//3:
            return "stochastic_volatility"
        # Default: standard GBM model
        return "standard"

    # --- Begin simulation loop ---
    for t in range(1, steps):
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price

        # Update macro factors (interest rate and inflation follow a small random walk)
        interest_rate += np.random.normal(0, interest_vol * np.sqrt(dt))
        inflation += np.random.normal(0, inflation_vol * np.sqrt(dt))

        # Update the fundamental value (assumed to drift slowly)
        new_fundamental = fundamentals[-1] * np.exp(0.02 * dt +
                                                    0.02 * np.random.normal(0, np.sqrt(dt)))
        fundamentals.append(new_fundamental)

        # Compute deviation from the fundamental value
        deviation = (current_price - new_fundamental) / new_fundamental

        # --- Regime switching using Markov transition ---
        regime_idx = regime_names.index(current_regime)
        current_regime = np.random.choice(
            regime_names, p=transition_matrix[regime_idx])
        regime_params = next(r for r in regimes if r['name'] == current_regime)

        # Update sentiment: a mean–reverting process toward 0
        sentiment += sentiment_params[0] * (0 - sentiment) * dt + \
            sentiment_params[1] * np.random.normal(0, np.sqrt(dt))

        # --- GARCH–style volatility update ---
        if len(prices) > 1:
            ret = np.log(current_price / prev_price)
        else:
            ret = 0
        new_variance = garch_omega + garch_alpha * \
            (ret ** 2) + garch_beta * (volatilities[-1] ** 2)
        garch_vol = np.sqrt(new_variance * dt)
        # Adjust volatility by the regime’s scaling factor (if defined)
        model_base_vol = garch_vol * regime_params.get('vol_scale', 1.0)

        # --- Liquidity and market–maker impact ---
        volume = np.abs(ret) * liquidity * (1 + 3 * sentiment)
        liquidity_ema = 0.9 * liquidity_ema + 0.1 * volume
        liquidity_ratio = liquidity_ema / initial_liquidity if initial_liquidity != 0 else 0
        liquidity_impact = 1 / (1 + liquidity_ratio)
        mm_force = market_maker_power * deviation * liquidity_impact

        # --- Flash Crash Check ---
        if len(prices) > 5:
            recent_max = np.max(prices[-5:])
            recent_drop = (current_price - recent_max) / recent_max
            if recent_drop < flash_crash_threshold[0] and liquidity < flash_crash_threshold[1]:
                new_price = current_price * 0.95
                prices.append(new_price)
                volatilities.append(model_base_vol)
                continue

        # --- Choose simulation model for this step ---
        model = choose_model(t, current_price, liquidity)

        # Compute a drift effect that includes the regime’s drift, macro effects, and market–maker force.
        drift_effect = regime_params.get(
            'drift', 0.05) + 0.5 * interest_rate - 0.8 * inflation + mm_force
        shock = 0  # to be computed below

        # --- Model-specific simulation ---
        if model == "standard":
            # Standard geometric Brownian motion
            dW = np.random.normal(0, np.sqrt(dt))
            shock = model_base_vol * dW + sentiment * model_base_vol / 2
            new_price = current_price * np.exp(drift_effect * dt + shock)

        elif model == "mean_reverting":
            # Price reverts toward the long–term mean (here taken as the fundamental)
            dW = np.random.normal(0, np.sqrt(dt))
            dS = mean_reversion_speed * (long_term_mean - current_price) * dt + \
                model_base_vol * current_price * dW
            new_price = current_price + dS

        elif model == "jump_diffusion":
            # Include a jump component (simulate jump event with probability proportional to jump intensity)
            dW = np.random.normal(0, np.sqrt(dt))
            jump_intensity, jump_mean, jump_std = jump_params
            jump = 0
            if np.random.rand() < jump_intensity * dt:
                jump = np.random.normal(jump_mean, jump_std)
            shock = model_base_vol * dW + jump + sentiment * model_base_vol / 2
            new_price = current_price * np.exp(drift_effect * dt + shock)

        elif model == "stochastic_volatility":
            # A simple Heston–like model for stochastic volatility
            kappa = mean_reversion_speed
            theta = base_volatility ** 2
            eta = 0.1  # volatility of volatility
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))
            dv = kappa * (theta - v) * dt + eta * np.sqrt(max(v, 0)) * dW2
            v = max(v + dv, 1e-8)  # ensure variance stays positive
            shock = np.sqrt(v) * dW1 + sentiment * np.sqrt(v) / 2
            new_price = current_price * np.exp(drift_effect * dt + shock)

        elif model == "regime_switching" and regime_switch is not None:
            # Use an alternative regime switch (as in the second code sample)
            dW = np.random.normal(0, np.sqrt(dt))
            shock = regime_switch_mu * dt + regime_switch_sigma * dW
            new_price = current_price * np.exp(shock)
            time_in_regime_switch += dt
            if time_in_regime_switch >= regime_switch_duration:
                regime_switch_idx = (regime_switch_idx +
                                     1) % len(regime_switch)
                regime_switch_mu, regime_switch_sigma, regime_switch_duration = regime_switch[
                    regime_switch_idx]
                time_in_regime_switch = 0.0
        else:
            # Fallback to standard GBM if none of the above applies.
            dW = np.random.normal(0, np.sqrt(dt))
            shock = model_base_vol * dW + sentiment * model_base_vol / 2
            new_price = current_price * np.exp(drift_effect * dt + shock)

        # --- Transaction Cost Adjustments ---
        effective_cost = transaction_cost * (1 + 2 * liquidity_ratio)
        # Apply a friction: if the shock was positive, reduce gains; if negative, dampen losses.
        if shock > 0:
            new_price *= (1 - effective_cost)
        else:
            new_price *= (1 + effective_cost)

        # --- Update liquidity ---
        liquidity *= np.exp(0.01 * dt + 0.002 *
                            np.random.normal(0, np.sqrt(dt)))
        liquidity *= (1 - 0.2 * np.abs(shock))

        # Record the new price and volatility for this step.
        prices.append(new_price)
        volatilities.append(model_base_vol)

    # --- Apply a global market shock if specified ---
    if market_shock == "bullish":
        prices = [p * 1.1 for p in prices]
    elif market_shock == "bearish":
        prices = [p * 0.9 for p in prices]

    # --- Plot the results ---
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Market Price')
    plt.plot(fundamentals, '--', label='Fundamental Value')
    plt.title(
        f"Dynamic Stock Price Simulation - Final Regime: {current_regime}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    return prices


if __name__ == "__main__":

    # 1) Run the modern configs
    # for i, config in enumerate(configs, 1):
    #     print(f"Running simulation for configuration {i}")
    #     simulate_dynamic_stock_price(**config)
    # Optionally, remove plt.show() in the function and do something like:
    # plt.savefig(f"modern_config_{i}.png")

    # 2) Now run the historical ones
    for idx, config in enumerate(configs_historical, 1):
        run_name = f"Historical Simulation #{idx}"
        print(f"Running {run_name}")
        simulate_dynamic_stock_price(**config)
        # Optionally, remove plt.show() in the function and do something like:
        # plt.savefig(f"historical_config_{idx}.png")

    # If you removed plt.show() in the function, do it once here:
    # plt.show()
