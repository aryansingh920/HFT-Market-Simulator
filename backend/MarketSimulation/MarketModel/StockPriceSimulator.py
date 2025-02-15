"""
Created on 03/02/2025

@author: Aryan

Filename: StockPriceSimulator.py

Relative Path: server/MarketModel/StockPriceSimulator.py
"""
import numpy as np


# ============================================================================
# Simulation Logic Class (unchanged except for recording extra output)
# ============================================================================
class StockPriceSimulator:
    def __init__(self, **params):
        """
        Accepts the same parameters as your original simulate_dynamic_stock_price function.
        """
        self.params = params

    def simulate(self):
        # Unpack parameters (using the same defaults as before)
        params = self.params
        duration = params.get("duration", 1)
        steps = params.get("steps", 252)
        initial_price = params.get("initial_price", 100)
        fundamental_value = params.get("fundamental_value", 100)
        initial_liquidity = params.get("initial_liquidity", 1e6)
        base_volatility = params.get("base_volatility", 0.2)
        regimes = params.get("regimes", None)
        regime_switch = params.get("regime_switch", None)
        garch_params = params.get("garch_params", (0.02, 0.1, 0.85))
        macro_impact = params.get("macro_impact", {'interest_rate': (0.03, 0.01),
                                                   'inflation': (0.02, 0.005)})
        sentiment_params = params.get("sentiment_params", (0.5, 0.2))
        flash_crash_threshold = params.get("flash_crash_threshold", (-0.15, 3))
        market_maker_power = params.get("market_maker_power", 0.1)
        transaction_cost = params.get("transaction_cost", 0.0005)
        jump_params = params.get("jump_params", (0.1, 0.02, 0.1))
        mean_reversion_speed = params.get("mean_reversion_speed", 0.1)
        long_term_mean = params.get("long_term_mean", None)
        market_shock_prob = params.get("market_shock_prob", 0.01)
        market_shock = params.get("market_shock", None)
        random_seed = params.get("random_seed", None)

        if random_seed is not None:
            np.random.seed(random_seed)

        dt = duration / steps
        prices = [initial_price]
        fundamentals = [fundamental_value]
        # This will store the scaled GARCH volatility
        volatilities = [base_volatility]
        times = [0]  # This will be our x-axis in "time" (years)
        regime_history = []  # Record the regime at each simulation step

        if long_term_mean is None:
            long_term_mean = fundamental_value

        liquidity = initial_liquidity
        sentiment = 0.0
        liquidity_ema = 0.0

        interest_rate, interest_vol = macro_impact.get(
            'interest_rate', (0.03, 0.01))
        inflation, inflation_vol = macro_impact.get('inflation', (0.02, 0.005))

        garch_omega, garch_alpha, garch_beta = garch_params

        # --- Set up regime switching via a Markov chain ---
        if regimes is None or len(regimes) == 0:
            regimes = [{'name': 'normal', 'drift': 0.05, 'vol_scale': 1.0,
                        'transitions': {'normal': 1.0}}]
        regime_names = [r['name'] for r in regimes]

        transition_matrix = np.zeros((len(regimes), len(regimes)))
        for i, r in enumerate(regimes):
            for j, name in enumerate(regime_names):
                transition_matrix[i, j] = r.get(
                    'transitions', {}).get(name, 0.0)
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum
            else:
                transition_matrix[i, i] = 1.0
        current_regime = regime_names[0]

        # --- Optional extra regime switching ---
        if regime_switch is not None and len(regime_switch) > 0:
            regime_switch_idx = 0
            regime_switch_mu, regime_switch_sigma, regime_switch_duration = regime_switch[
                regime_switch_idx]
            time_in_regime_switch = 0.0

        # Initialize GARCH conditional variance h (for the period, not annualized)
        h = (base_volatility ** 2) * dt

        # --- Main simulation loop ---
        for t in range(1, steps):
            current_price = prices[-1]
            prev_price = prices[-2] if len(prices) > 1 else current_price

            # Update macro factors (random walk)
            interest_rate += np.random.normal(0, interest_vol * np.sqrt(dt))
            inflation += np.random.normal(0, inflation_vol * np.sqrt(dt))

            # Update the fundamental value
            new_fundamental = fundamentals[-1] * np.exp(
                0.02 * dt + 0.02 * np.random.normal(0, np.sqrt(dt)))
            fundamentals.append(new_fundamental)

            deviation = (current_price - new_fundamental) / new_fundamental

            # --- Regime switching using Markov transition ---
            regime_idx = regime_names.index(current_regime)
            current_regime = np.random.choice(
                regime_names, p=transition_matrix[regime_idx])
            regime_history.append(current_regime)
            regime_params = next(
                r for r in regimes if r['name'] == current_regime)

            # Update sentiment
            sentiment += sentiment_params[0] * (
                0 - sentiment) * dt + sentiment_params[1] * np.random.normal(0, np.sqrt(dt))

            if len(prices) > 1:
                ret = np.log(current_price / prev_price)
            else:
                ret = 0

            # --- GARCH volatility update ---
            # Update conditional variance h using GARCH(1,1) formulation:
            # h = (garch_omega * dt) + garch_alpha * (ret**2) + garch_beta * h
            h = (garch_omega * dt) + garch_alpha * (ret ** 2) + garch_beta * h
            garch_vol = np.sqrt(h)
            model_base_vol = garch_vol * regime_params.get('vol_scale', 1.0)

            volume = np.abs(ret) * liquidity * (1 + 3 * sentiment)
            liquidity_ema = 0.9 * liquidity_ema + 0.1 * volume
            liquidity_ratio = liquidity_ema / initial_liquidity if initial_liquidity != 0 else 0
            liquidity_impact = 1 / (1 + liquidity_ratio)
            mm_force = market_maker_power * deviation * liquidity_impact

            # --- Flash Crash Check ---
            if len(prices) > 5:
                recent_max = np.max(prices[-5:])
                recent_drop = (current_price - recent_max) / recent_max
                if (recent_drop < flash_crash_threshold[0]) and (liquidity < flash_crash_threshold[1]):
                    new_price = current_price * 0.95
                    prices.append(new_price)
                    volatilities.append(model_base_vol)
                    times.append(t * dt)
                    continue

            # --- Choose simulation model ---
            def choose_model(t, current_price, liquidity):
                if np.random.rand() < market_shock_prob:
                    return "jump_diffusion"
                if liquidity < 0.5 * initial_liquidity:
                    return "mean_reverting"
                if regime_switch is not None and (t % max(1, int(steps/10)) == 0):
                    return "regime_switching"
                if steps // 3 < t < 2 * steps // 3:
                    return "stochastic_volatility"
                return "standard"

            model = choose_model(t, current_price, liquidity)
            drift_effect = (regime_params.get('drift', 0.05) +
                            0.5 * interest_rate - 0.8 * inflation + mm_force)
            shock = 0

            if model == "standard":
                dW = np.random.normal(0, np.sqrt(dt))
                shock = model_base_vol * dW + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            elif model == "mean_reverting":
                dW = np.random.normal(0, np.sqrt(dt))
                dS = (mean_reversion_speed * (long_term_mean - current_price)
                      * dt + model_base_vol * current_price * dW)
                new_price = current_price + dS

            elif model == "jump_diffusion":
                dW = np.random.normal(0, np.sqrt(dt))
                jump_intensity, jump_mean, jump_std = jump_params
                jump = 0
                if np.random.rand() < jump_intensity * dt:
                    jump = np.random.normal(jump_mean, jump_std)
                shock = model_base_vol * dW + jump + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            elif model == "stochastic_volatility":
                kappa = mean_reversion_speed
                theta = base_volatility ** 2
                eta = 0.1  # volatility of volatility
                dW1 = np.random.normal(0, np.sqrt(dt))
                dW2 = np.random.normal(0, np.sqrt(dt))
                dv = kappa * (theta - h) * dt + eta * np.sqrt(max(h, 0)) * dW2
                h = max(h + dv, 1e-8)
                shock = np.sqrt(h) * dW1 + sentiment * np.sqrt(h) / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            elif model == "regime_switching" and regime_switch is not None:
                dW = np.random.normal(0, np.sqrt(dt))
                shock = regime_switch_mu * dt + regime_switch_sigma * dW
                new_price = current_price * np.exp(shock)
                time_in_regime_switch += dt
                if time_in_regime_switch >= regime_switch_duration:
                    regime_switch_idx = (
                        regime_switch_idx + 1) % len(regime_switch)
                    regime_switch_mu, regime_switch_sigma, regime_switch_duration = regime_switch[
                        regime_switch_idx]
                    time_in_regime_switch = 0.0

            else:
                # Default fallback is basically "standard"
                dW = np.random.normal(0, np.sqrt(dt))
                shock = model_base_vol * dW + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            # Transaction cost effect
            effective_cost = transaction_cost * (1 + 2 * liquidity_ratio)
            if shock > 0:
                new_price *= (1 - effective_cost)
            else:
                new_price *= (1 + effective_cost)

            # Liquidity adjustments
            liquidity *= np.exp(0.01 * dt + 0.002 *
                                np.random.normal(0, np.sqrt(dt)))
            liquidity *= (1 - 0.2 * np.abs(shock))

            prices.append(new_price)
            volatilities.append(model_base_vol)
            times.append(t * dt)

        if market_shock == "bullish":
            prices = [p * 1.1 for p in prices]
        elif market_shock == "bearish":
            prices = [p * 0.9 for p in prices]

        results = {
            "time": times,
            "prices": prices,
            "fundamentals": fundamentals,
            "volatilities": volatilities,
            "regime_history": regime_history
        }
        return results
