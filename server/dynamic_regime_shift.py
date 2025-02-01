"""
Created on 01/02/2025

@author: Aryan

Filename: dynamic_regime_shift.py
Relative Path: server/dynamic_regime_shift.py
"""

import numpy as np
import json  # For pretty–printing config details
import dash
from dash import dcc, html
import plotly.graph_objs as go

# Import all simulation configurations from config.py
from config import configs_historical


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

        # For stochastic volatility
        v = base_volatility ** 2

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

        # --- Main simulation loop ---
        for t in range(1, steps):
            current_price = prices[-1]
            prev_price = prices[-2] if len(prices) > 1 else current_price

            # Update macro factors (random walk)
            interest_rate += np.random.normal(0, interest_vol * np.sqrt(dt))
            inflation += np.random.normal(0, inflation_vol * np.sqrt(dt))

            # Update the fundamental value
            new_fundamental = fundamentals[-1] * np.exp(0.02 * dt +
                                                        0.02 * np.random.normal(0, np.sqrt(dt)))
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
            sentiment += sentiment_params[0] * (0 - sentiment) * dt + \
                sentiment_params[1] * np.random.normal(0, np.sqrt(dt))

            if len(prices) > 1:
                ret = np.log(current_price / prev_price)
            else:
                ret = 0

            new_variance = garch_omega + garch_alpha * \
                (ret ** 2) + garch_beta * (volatilities[-1] ** 2)
            garch_vol = np.sqrt(new_variance * dt)
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
            model = choose_model(t, current_price, liquidity)
            drift_effect = (regime_params.get('drift', 0.05) +
                            0.5 * interest_rate - 0.8 * inflation +
                            mm_force)
            shock = 0

            if model == "standard":
                dW = np.random.normal(0, np.sqrt(dt))
                shock = model_base_vol * dW + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            elif model == "mean_reverting":
                dW = np.random.normal(0, np.sqrt(dt))
                dS = (mean_reversion_speed * (long_term_mean - current_price) * dt +
                      model_base_vol * current_price * dW)
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
                dv = kappa * (theta - v) * dt + eta * np.sqrt(max(v, 0)) * dW2
                v = max(v + dv, 1e-8)
                shock = np.sqrt(v) * dW1 + sentiment * np.sqrt(v) / 2
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


# ============================================================================
# Dashboard / Graph Generation Class (using Dash and Plotly)
# ============================================================================
class Dashboard:
    def __init__(self, simulation_results_list, config_names, config_details):
        """
        simulation_results_list: list of simulation results (dicts from simulate())
        config_names: list of simulation names
        config_details: list of the configuration dictionaries used for each simulation
        """
        self.results_list = simulation_results_list
        self.config_names = config_names
        self.config_details = config_details
        self.app = dash.Dash(__name__)
        self.build_layout()

    def segment_data_by_regime(self, time, prices, regimes):
        segments = []
        start_idx = 0
        current_regime = regimes[0]
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                segments.append(
                    (current_regime, time[start_idx:i+1], prices[start_idx:i+1]))
                start_idx = i
                current_regime = regimes[i]
        segments.append((current_regime, time[start_idx:], prices[start_idx:]))
        return segments

    def create_figure(self, result, name):
        segments = self.segment_data_by_regime(
            result['time'], result['prices'], result['regime_history'])
        fig = go.Figure()
        for regime, t_seg, p_seg in segments:
            fig.add_trace(go.Scatter(
                x=t_seg, y=p_seg,
                mode='lines+markers',
                name=f"{regime} regime",
                hovertemplate="Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>"
            ))
        fig.add_trace(go.Scatter(
            x=result['time'], y=result['fundamentals'],
            mode='lines',
            name='Fundamental Value',
            line=dict(dash='dash', color='black'),
            hovertemplate="Time: %{x:.2f}<br>Fundamental: %{y:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"{name} Simulation",
            xaxis_title="Time (Years)",
            yaxis_title="Price",
            hovermode="closest"
        )
        return fig

    def build_layout(self):
        tabs = []
        # For each simulation, create a Tab with the graph + JSON config
        for result, name, cfg in zip(self.results_list, self.config_names, self.config_details):
            fig = self.create_figure(result, name)
            tab = dcc.Tab(label=name, children=[
                html.Div([
                    dcc.Graph(figure=fig),
                    html.H2("Simulation Configuration"),
                    # Show config as formatted JSON
                    html.Pre(json.dumps(cfg, indent=2))
                ])
            ])
            tabs.append(tab)
        self.app.layout = html.Div([
            html.H1("Market Simulation Dashboard"),
            dcc.Tabs(children=tabs)
        ])

    def run(self):
        self.app.run_server(debug=True)


# ============================================================================
# Super Class that ties Simulation & Dashboard together
# ============================================================================
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
            simulator = StockPriceSimulator(**config)
            result = simulator.simulate()
            self.results.append(result)
            self.config_names.append(name)
            self.config_details.append(config)  # Store config

    def launch_dashboard(self):
        dashboard = Dashboard(
            self.results, self.config_names, self.config_details)
        dashboard.run()


# ============================================================================
# Advanced Single Stock Simulator with Additional Features
# ============================================================================
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


# ============================================================================
# Multi-Asset Simulator (for two or more stocks with correlations)
# ============================================================================
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


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    # 1) Run the historical configs
    historical_simulator = MarketSimulator(configs_historical)
    historical_simulator.run_simulations()
    # These are your historical results, config details, and names
    hist_results = historical_simulator.results
    hist_config_names = historical_simulator.config_names
    hist_config_details = historical_simulator.config_details

    # 2) Run the advanced single-stock simulation
    advanced_config = {
        "name": "Advanced Single Stock",
        "duration": 1,
        "steps": 252,
        "initial_price": 100,
        "fundamental_value": 100,
        "initial_liquidity": 1e6,
        "base_volatility": 0.2,
        "transaction_cost": 0.0005,
        "sentiment_seed": 0.1,
        "news_flow_intensity": 0.1,
        "seasonality_params": {"day_of_week": [1.0, 1.02, 0.98, 1.03, 0.97]},
        "heston_params": {"kappa": 1.2, "theta": 0.04, "eta": 0.15},
        "refined_jump_params": {"intensity": 0.05, "df": 3},
        # ... other advanced parameters
    }
    advanced_sim = AdvancedStockSimulator(advanced_config)
    advanced_result = advanced_sim.simulate()

    # We'll store it in a list so we can unify it with the others
    advanced_results_list = [advanced_result]
    advanced_config_names = [advanced_config["name"]]
    advanced_config_details = [advanced_config]

    # 3) Run the multi-asset simulation
    multi_asset_configs = [
        {
            "name": "Stock A",
            "duration": 1,
            "steps": 252,
            "initial_price": 100,
            "fundamental_value": 100,
            "initial_liquidity": 1e6,
            "base_volatility": 0.2,
            "transaction_cost": 0.0005,
        },
        {
            "name": "Stock B",
            "duration": 1,
            "steps": 252,
            "initial_price": 150,
            "fundamental_value": 150,
            "initial_liquidity": 1e6,
            "base_volatility": 0.25,
            "transaction_cost": 0.0007,
        }
    ]
    correlation_matrix = [
        [1.0, 0.7],
        [0.7, 1.0]
    ]
    multi_asset_sim = MultiAssetSimulator(
        multi_asset_configs, correlation_matrix)
    multi_asset_results = multi_asset_sim.simulate()
    # `multi_asset_results` is a list of dicts (one result per stock)

    # We'll build parallel lists for the multi-asset run:
    ma_results_list = []
    ma_names = []
    ma_details = []
    for cfg, res in zip(multi_asset_configs, multi_asset_results):
        ma_results_list.append(res)
        ma_names.append(cfg["name"])  # "Stock A", "Stock B"
        ma_details.append(cfg)

    # 4) Combine everything into one set of lists
    all_results = hist_results + advanced_results_list + ma_results_list
    all_names = hist_config_names + advanced_config_names + ma_names
    all_details = hist_config_details + advanced_config_details + ma_details

    # 5) Launch ONE dashboard with everything
    from dash import Dash
    from dynamic_regime_shift import Dashboard  # or wherever your Dashboard is

    combined_dashboard = Dashboard(all_results, all_names, all_details)
    combined_dashboard.run()
