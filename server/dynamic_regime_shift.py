import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objs as go
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
        v = base_volatility**2

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
            # record regime for this step
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
                if recent_drop < flash_crash_threshold[0] and liquidity < flash_crash_threshold[1]:
                    new_price = current_price * 0.95
                    prices.append(new_price)
                    volatilities.append(model_base_vol)
                    times.append(t * dt)
                    continue

            # --- Choose simulation model ---
            model = choose_model(t, current_price, liquidity)
            drift_effect = regime_params.get(
                'drift', 0.05) + 0.5 * interest_rate - 0.8 * inflation + mm_force
            shock = 0

            if model == "standard":
                dW = np.random.normal(0, np.sqrt(dt))
                shock = model_base_vol * dW + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)
            elif model == "mean_reverting":
                dW = np.random.normal(0, np.sqrt(dt))
                dS = mean_reversion_speed * (long_term_mean - current_price) * dt + \
                    model_base_vol * current_price * dW
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
                dW = np.random.normal(0, np.sqrt(dt))
                shock = model_base_vol * dW + sentiment * model_base_vol / 2
                new_price = current_price * np.exp(drift_effect * dt + shock)

            effective_cost = transaction_cost * (1 + 2 * liquidity_ratio)
            if shock > 0:
                new_price *= (1 - effective_cost)
            else:
                new_price *= (1 + effective_cost)

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
    def __init__(self, simulation_results_list, config_names):
        """
        Accepts a list of simulation results (each a dict from simulate())
        and a corresponding list of configuration names.
        """
        self.results_list = simulation_results_list
        self.config_names = config_names
        self.app = dash.Dash(__name__)
        self.build_layout()

    def segment_data_by_regime(self, time, prices, regimes):
        """
        Segments the time and price data into contiguous pieces where
        the regime is constant.
        """
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
        """
        Create an interactive Plotly figure that displays:
          - The price data segmented by regime (each segment gets its own trace).
          - The fundamental value as a dashed line.
          - Correct x–axis labeling (“Time (Years)”).
        """
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
        # Create a tab for each simulation configuration.
        tabs = []
        for result, name in zip(self.results_list, self.config_names):
            fig = self.create_figure(result, name)
            tab = dcc.Tab(label=name, children=[
                dcc.Graph(figure=fig)
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
        configs is a list of dictionaries (each one is a configuration).
        Each configuration must include a "name" key if you want a custom label.
        """
        self.configs = configs
        self.results = []
        self.config_names = []

    def run_simulations(self):
        for idx, config in enumerate(self.configs, 1):
            name = config.get("name", f"Simulation #{idx}")
            print(f"Running simulation: {name}")
            simulator = StockPriceSimulator(**config)
            result = simulator.simulate()
            self.results.append(result)
            self.config_names.append(name)

    def launch_dashboard(self):
        dashboard = Dashboard(self.results, self.config_names)
        dashboard.run()


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    # You can either import your configurations from a separate config file
    # or define them here. Below we use the "configs_historical" as an example.

    # Create the super class instance with all the configurations.
    market_simulator = MarketSimulator(configs_historical)
    market_simulator.run_simulations()
    market_simulator.launch_dashboard()
