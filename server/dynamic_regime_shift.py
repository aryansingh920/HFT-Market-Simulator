import numpy as np
import matplotlib.pyplot as plt


def simulate_dynamic_stock_price(
    duration=1,  # Total time duration in years
    steps=252,  # Number of steps (e.g., trading days in a year)
    initial_price=100,  # Initial stock price
    initial_liquidity=1_000_000,  # Initial liquidity
    drift=0.05,  # Mean return (annualized)
    volatility=0.2,  # Base volatility (annualized)
    mean_reversion_speed=0.1,  # For mean-reverting GBM
    long_term_mean=100,  # Long-term mean price for mean-reverting GBM
    jump_intensity=0.1,  # Jump intensity for jump-diffusion
    jump_mean=0.02,  # Mean of the jump size
    jump_std=0.1,  # Stddev of the jump size
    transaction_cost=0.001,  # Transaction cost as a fraction of price
    # Optional: Regime-switching parameters [(mu, sigma, duration)]
    regime_switch=None,
    hurst=0.5,  # Hurst parameter for fractional GBM
    market_shock_prob=0.01,  # Probability of a random market shock
    market_shock=None,  # "bullish" or "bearish"
    title="Stock Price Simulation"
):
    dt = duration / steps  # Time step
    prices = [initial_price]  # Initialize price list
    np.random.seed(42)  # For reproducibility

    # Initial regime-switching parameters
    if regime_switch:
        regime_idx = 0
        regime_mu, regime_sigma, regime_duration = regime_switch[regime_idx]
        time_in_regime = 0

    def choose_model(t, current_price, liquidity):
        """Dynamically selects the GBM model based on time, liquidity, and shocks."""
        if np.random.rand() < market_shock_prob:  # Simulate a market shock
            return "jump_diffusion"
        elif liquidity < 500_000:  # Low liquidity leads to more mean-reverting behavior
            return "mean_reverting"
        elif t < steps // 3:  # Early phase uses standard GBM
            return "standard"
        elif t < 2 * steps // 3:  # Mid-phase uses stochastic volatility
            return "stochastic_volatility"
        elif t % 50 == 0:  # Occasionally simulate regime-switching
            return "regime_switching"
        else:  # Default fallback
            return "standard"

    for t in range(steps):
        current_price = prices[-1]
        model = choose_model(t, current_price, initial_liquidity)

        # Standard GBM
        if model == "standard":
            dW = np.random.normal(0, np.sqrt(dt))
            dS = drift * current_price * dt + volatility * current_price * dW
            prices.append(current_price + dS)

        # Mean-Reverting GBM
        elif model == "mean_reverting":
            dW = np.random.normal(0, np.sqrt(dt))
            dS = mean_reversion_speed * \
                (long_term_mean - current_price) * \
                dt + volatility * current_price * dW
            prices.append(current_price + dS)

        # Jump-Diffusion GBM
        elif model == "jump_diffusion":
            dW = np.random.normal(0, np.sqrt(dt))
            jump = np.random.poisson(
                jump_intensity * dt) * np.random.normal(jump_mean, jump_std)
            dS = drift * current_price * dt + volatility * \
                current_price * dW + jump * current_price
            prices.append(current_price + dS)

        # Stochastic Volatility GBM (Heston model)
        elif model == "stochastic_volatility":
            v = volatility**2  # Initial variance
            kappa = mean_reversion_speed
            theta = volatility**2  # Long-term variance
            eta = 0.1  # Volatility of volatility
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))
            dv = kappa * (theta - v) * dt + eta * np.sqrt(v) * dW2
            v = max(v + dv, 0)  # Ensure variance is non-negative
            dS = drift * current_price * dt + np.sqrt(v) * current_price * dW1
            prices.append(current_price + dS)

        # Regime-Switching GBM
        elif model == "regime_switching" and regime_switch:
            if time_in_regime >= regime_duration:
                regime_idx = (regime_idx + 1) % len(regime_switch)
                regime_mu, regime_sigma, regime_duration = regime_switch[regime_idx]
                time_in_regime = 0
            dW = np.random.normal(0, np.sqrt(dt))
            dS = regime_mu * current_price * dt + regime_sigma * current_price * dW
            prices.append(current_price + dS)
            time_in_regime += dt

        # Transaction Costs (Apply globally)
        prices[-1] -= transaction_cost * prices[-1]

        # Update Liquidity (Random Walk)
        initial_liquidity *= 1 + np.random.normal(0, 0.0001)

    # Apply Bullish or Bearish Shock (10% Up or Down)
    if market_shock == "bullish":
        prices = [p * 1.1 for p in prices]
    elif market_shock == "bearish":
        prices = [p * 0.9 for p in prices]

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label="Dynamic GBM Simulation")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()

    return prices


configs = [
    {"duration": 3, "steps": 756, "initial_price": 100, "drift": 0.1, "volatility": 0.3,
     "regime_switch": [(0.1, 0.3, 500), (-0.2, 0.5, 256)], "jump_intensity": 0.05,
     "jump_mean": -0.2, "jump_std": 0.1, "market_shock_prob": 0.02, "title": "Dot-Com Bubble"},  # Dot-Com Bubble

    {"duration": 1, "steps": 252, "initial_price": 150, "drift": 0.02, "volatility": 0.2,
     "regime_switch": [(-0.3, 0.6, 150), (0.05, 0.3, 102)], "jump_intensity": 0.1,
     "jump_mean": -0.3, "jump_std": 0.15, "market_shock_prob": 0.05, "title": "2008 Crisis"},  # 2008 Crisis

    {"duration": 1, "steps": 252, "initial_price": 300, "drift": 0.05, "volatility": 0.4,
     "regime_switch": [(-0.4, 0.6, 50), (0.1, 0.3, 202)], "jump_intensity": 0.15,
     "jump_mean": -0.2, "jump_std": 0.25, "market_shock_prob": 0.1, "title": "COVID-19 Crash"},  # COVID-19 Crash

    {"duration": 5, "steps": 1260, "initial_price": 200, "drift": 0.08, "volatility": 0.15,
     "jump_intensity": 0.01, "jump_mean": -0.05, "jump_std": 0.03, "market_shock_prob": 0.005, "title": "Prolonged Bull Market"},  # Prolonged Bull Market

    {"duration": 1, "steps": 252, "initial_price": 50, "drift": 0.5, "volatility": 0.5,
     "jump_intensity": 0.2, "jump_mean": 0.3, "jump_std": 0.15, "market_shock_prob": 0.1, "title": "Hyperinflation"},  # Hyperinflation

    {"duration": 1, "steps": 252, "initial_price": 100, "drift": 0.02, "volatility": 0.4,
     "mean_reversion_speed": 0.2, "long_term_mean": 90, "jump_intensity": 0.08,
     "jump_mean": -0.1, "jump_std": 0.1, "market_shock_prob": 0.05, "title": "Post Liquidity Crisis"},  # Post-Liquidity Crisis

    {"duration": 1 / 252, "steps": 10_000, "initial_price": 100, "drift": 0.001, "volatility": 0.05,
     "transaction_cost": 0.0001, "market_shock_prob": 0.0001, "jump_intensity": 0.01,
     "jump_mean": -0.01, "jump_std": 0.02, "title": "HFT Simulation"},  # HFT Simulation
]


if __name__ == "__main__":
    for config in configs:
        simulate_dynamic_stock_price(**config)
