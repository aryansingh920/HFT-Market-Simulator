"""
Created on 15/02/2025

@author: Aryan

Filename: config.py

Relative Path: backend/MarketSimulation/config.py
"""

import numpy as np
import random
# Global simulation parameters
# Default: 1 step per day if not modified. (For high resolution, total steps will be higher.)
default_steps_per_day = 1

# -------------------------
# Configuration Dictionaries
# -------------------------

intraday_config = {
    "intraday_config": {
        "initial_price": 100.0,
        "fundamental_value": 100.0,
        "initial_liquidity": 1_000_000,    # liquidity for the order book
        "steps_per_day": 390,             # 1 step per minute in a 6.5-hour trading day
        "base_volatility": 0.005,         # baseline intraday volatility
        "drift": 0.001,                   # small intraday drift
        # how strongly the MM moves prices based on order imbalance
        "market_maker_power": 0.05,
        # transaction cost rate (30 bps per trade)
        "transaction_cost": 0.0003,
        "random_seed": random.randint(1, 10)                 # reproducibility
    },

    "intraday_regimes": [
        {
            'name': 'morning_session',
            'drift': 0.04,
            'vol_scale': 1.5,
        },
        {
            'name': 'midday_lull',
            'drift': 0.00,
            'vol_scale': 0.8,
        },
        {
            'name': 'afternoon_ramp',
            'drift': 0.00,
            'vol_scale': 1.2,
        },
    ],

    "intraday_transition_probabilities": {
        'morning_session': {'morning_session': 0.90, 'midday_lull': 0.10},
        'midday_lull':     {'midday_lull': 0.95, 'afternoon_ramp': 0.05},
        'afternoon_ramp':  {'afternoon_ramp': 0.98}
    }
}




configs_test = [
    {
        'name': 'Complex Regime Example',
        'duration': 10,               # 1 year
        # For 1 step per day, use 252; for higher resolution, increase this value.
        'steps': 252,
        'initial_price': 100,
        'fundamental_value': 100,
        'initial_liquidity': 1e6,
        'base_volatility': 0.9,

        # Daily transition probabilities (to be adjusted per step)
        'original_transitions': {
            'pre_crisis': {'pre_crisis': 0.85, 'collapse': 0.15},
            'collapse': {'collapse': 0.3, 'rebound': 0.3},
            'rebound': {'rebound': 0.9, 'collapse': 0.1, 'post_crisis': 0.3},
            'post_crisis': {'post_crisis': 0.2}
        },

        # Base regime parameters (without transitions)
        'regimes': [
            {'name': 'pre_crisis', 'drift': 0.55, 'vol_scale': 1.0},
            {'name': 'collapse', 'drift': -0.1, 'vol_scale': 3.5},
            {'name': 'rebound', 'drift': 0.35, 'vol_scale': 2.0},
            {'name': 'post_crisis', 'drift': 0.58, 'vol_scale': 5.2}
        ],

        # Other parameters
        'garch_params': (0.005, 0.08, 0.85),
        'macro_impact': {
            'interest_rate': (0.03, 0.005),
            'inflation': (0.02, 0.002)
        },
        'sentiment_params': (0.3, 0.1),
        'flash_crash_threshold': (-0.15, 2),
        'market_maker_power': 0.1,
        'transaction_cost': 0.0005,
        'jump_params': (0.05, 0.02, 0.1),
        'mean_reversion_speed': 0.1,
        'long_term_mean': 100,
        'market_shock_prob': 0.01,
        'random_seed': 2025
    }
]

configs_pure_gbm = [
    {
        'name': 'PureGBMTest',
        'duration': 1,
        'steps': 252,
        'initial_price': 100,
        'fundamental_value': 100,
        'initial_liquidity': 1e6,
        'base_volatility': 0.2,
        'garch_params': (0.0, 0.0, 1.0),
        'macro_impact': {
            'interest_rate': (0.0, 0.0),
            'inflation': (0.0, 0.0)
        },
        'sentiment_params': (0.0, 0.0),
        'flash_crash_threshold': (-999, 999),
        'market_maker_power': 0.0,
        'transaction_cost': 0.0,
        'jump_params': (0.0, 0.0, 0.0),
        'mean_reversion_speed': 0.0,
        'long_term_mean': 100,
        'market_shock_prob': 0.0,
        'regimes': [
            {
                'name': 'normal',
                'drift': 0.07,
                'vol_scale': 1.0,
                'transitions': {'normal': 1.0}
            }
        ]
    }
]

configs_nvidia = [
    {
        'name': 'NVIDIA_HypeCycle_Stable',
        'duration': 1,
        # For high resolution, e.g. 78 steps per day: 252 * 78 steps per year
        'steps': 252 * 78,
        'initial_price': 150,
        'fundamental_value': 150,
        'initial_liquidity': 5e9,
        'base_volatility': 0.35,

        # Revised daily transition probabilities
        'original_transitions': {
            'steady_growth': {'steady_growth': 0.88, 'earnings_surge': 0.10, 'market_correction': 0.02},
            'earnings_surge': {'earnings_surge': 0.65, 'hypergrowth': 0.30, 'market_correction': 0.05},
            'hypergrowth': {'hypergrowth': 0.55, 'peak_frenzy': 0.40, 'market_correction': 0.05},
            'peak_frenzy': {'peak_frenzy': 0.40, 'market_correction': 0.60},
            'market_correction': {'market_correction': 0.70, 'steady_growth': 0.30}
        },

        # Base regime parameters
        'regimes': [
            {'name': 'steady_growth', 'drift': 0.25, 'vol_scale': 1.2},
            {'name': 'earnings_surge', 'drift': 0.80, 'vol_scale': 1.5},
            {'name': 'hypergrowth', 'drift': 1.20, 'vol_scale': 2.0},
            {'name': 'peak_frenzy', 'drift': 1.80, 'vol_scale': 2.5},
            {'name': 'market_correction', 'drift': -0.40, 'vol_scale': 2.0}
        ],

        'garch_params': (0.005, 0.10, 0.85),
        'jump_params': (0.05, 0.10, 0.15),
        'sentiment_params': (0.35, 0.15),
        'max_price': 1e6,
        'max_volatility': 5.0,
        'min_liquidity': 1e-5,
        'flash_crash_threshold': (-0.20, 1.5),
        'market_maker_power': 0.15,
        'transaction_cost': 0.0002,
        'mean_reversion_speed': 0.05,
        'long_term_mean': 150,
        'market_shock_prob': 0.05,
        'random_seed': 2025,
        'special_events': {
            'gpu_breakthrough': {
                'probability': 0.10,
                'impact': 0.25
            },
            'export_restrictions': {
                'probability': 0.08,
                'impact': -0.20
            }
        }
    }
]

# --------------------------------------
# Adjust Transition Probabilities Function
# --------------------------------------


def adjust_transition_probabilities(config, steps_per_day):
    """
    Converts daily transition probabilities in 'original_transitions'
    to per–step probabilities using the formula:
        step_probability = daily_probability / steps_per_day
    It then builds the 'transitions' dictionary for each regime.
    """
    if 'original_transitions' not in config:
        return

    adjusted_regimes = []
    original_transitions = config['original_transitions']

    for regime_name in original_transitions:
        daily_transitions = original_transitions[regime_name]
        step_transitions = {}
        for target, daily_prob in daily_transitions.items():
            step_prob = daily_prob / steps_per_day
            step_transitions[target] = round(step_prob, 6)
        # Compute staying probability so the row sums to 1
        total_leave = sum(step_transitions.values())
        step_transitions[regime_name] = max(0, 1 - total_leave)

        base_regime = next(
            r for r in config['regimes'] if r['name'] == regime_name)
        adjusted_regimes.append({
            'name': regime_name,
            'drift': base_regime['drift'],
            'vol_scale': base_regime['vol_scale'],
            'transitions': step_transitions
        })

    # Replace the original 'regimes' with the adjusted ones.
    config['regimes'] = adjusted_regimes

# --------------------------------------
# Apply Transition Adjustments to Configs
# --------------------------------------


def apply_adjustments(configs):
    for config in configs:
        if 'original_transitions' in config:
            # Compute effective steps per day from total steps and duration.
            duration = config.get('duration', 1)
            steps = config.get('steps', 252)
            steps_per_day = steps / (252 * duration)
            adjust_transition_probabilities(config, steps_per_day)


# Apply adjustments for configs that use regime transitions.
apply_adjustments(configs_test)
apply_adjustments(configs_nvidia)

# Now, configs_test, configs_pure_gbm, and configs_nvidia are adjusted accordingly.
