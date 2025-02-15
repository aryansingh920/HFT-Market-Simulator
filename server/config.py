# """
# Market Simulation Configuration File
# Updated with proper regime transition scaling
# """

import numpy as np

# # Global simulation parameters
steps_per_day = 10  # Default value, can be modified as needed

configs_test = [
    {
        'name': 'Complex Regime Example',
        'duration': 1,               # 1 year
        'steps': 252 * steps_per_day,
        'initial_price': 100,
        'fundamental_value': 100,
        'initial_liquidity': 1e6,
        'base_volatility': 50,

        # Original daily transition probabilities
        'original_transitions': {
            'pre_crisis': {'pre_crisis': 0.85, 'collapse': 0.15},
            'collapse': {'collapse': 0.7, 'rebound': 0.3},
            'rebound': {'rebound': 0.6, 'collapse': 0.1, 'post_crisis': 0.3},
            'post_crisis': {'post_crisis': 1.0}
        },

        # Base regime parameters (without transitions)
        'regimes': [
            {'name': 'pre_crisis', 'drift': 0.05, 'vol_scale': 1.0},
            {'name': 'collapse', 'drift': -0.5, 'vol_scale': 3.5},
            {'name': 'rebound', 'drift': 0.15, 'vol_scale': 2.0},
            {'name': 'post_crisis', 'drift': 0.08, 'vol_scale': 1.2}
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


# def adjust_transition_probabilities(config, steps_per_day):
#     """
#     Properly converts daily transition probabilities to per-step probabilities
#     while maintaining Markov chain validity
#     """
#     if 'original_transitions' not in config:
#         return

#     adjusted_regimes = []
#     original_transitions = config['original_transitions']

#     # Process each regime
#     for regime_name in original_transitions:
#         # Get original daily transitions
#         daily_transitions = original_transitions[regime_name]

#         # Convert to per-step transitions
#         step_transitions = {}
#         for target, daily_prob in daily_transitions.items():
#             step_prob = daily_prob / steps_per_day
#             step_transitions[target] = round(step_prob, 6)

#         # Calculate remaining probability for staying in current regime
#         total_leave = sum(step_transitions.values())
#         step_transitions[regime_name] = max(0, 1 - total_leave)

#         # Find original regime parameters
#         base_regime = next(
#             r for r in config['regimes'] if r['name'] == regime_name)

#         # Create adjusted regime
#         adjusted_regimes.append({
#             'name': regime_name,
#             'drift': base_regime['drift'],
#             'vol_scale': base_regime['vol_scale'],
#             'transitions': step_transitions
#         })

#     # Update config with adjusted regimes
#     config['regimes'] = adjusted_regimes


# # Apply transitions adjustment when file is loaded
# if __name__ == "__main__":
#     # Apply to all test configs that have original_transitions
#     for config in configs_test:
#         if 'original_transitions' in config:
#             adjust_transition_probabilities(config, steps_per_day)


configs_nvidia = [
    {
        'name': 'NVIDIA_HypeCycle_Stable',
        'duration': 1,
        'steps': 252 * 78,
        'initial_price': 150,
        'fundamental_value': 150,
        'initial_liquidity': 5e9,
        'base_volatility': 0.35,

        # Revised regime transitions with numerical safeguards
        'original_transitions': {
            'steady_growth': {'steady_growth': 0.88, 'earnings_surge': 0.10, 'market_correction': 0.02},
            'earnings_surge': {'earnings_surge': 0.65, 'hypergrowth': 0.30, 'market_correction': 0.05},
            'hypergrowth': {'hypergrowth': 0.55, 'peak_frenzy': 0.40, 'market_correction': 0.05},
            'peak_frenzy': {'peak_frenzy': 0.40, 'market_correction': 0.60},
            'market_correction': {'market_correction': 0.70, 'steady_growth': 0.30}
        },

        # Capped regime parameters with volatility limits
        'regimes': [
            {'name': 'steady_growth', 'drift': 0.25,
                'vol_scale': 1.2},  # Reduced from 0.35
            {'name': 'earnings_surge', 'drift': 0.80,
                'vol_scale': 1.5},  # Reduced from 1.20
            {'name': 'hypergrowth', 'drift': 1.20,
                'vol_scale': 2.0},  # Reduced from 2.50
            {'name': 'peak_frenzy', 'drift': 1.80,
                'vol_scale': 2.5},  # Reduced from 4.00
            {'name': 'market_correction', 'drift': -0.40,
                'vol_scale': 2.0}  # Reduced from -0.60
        ],

        # Stabilized GARCH parameters
        'garch_params': (0.005, 0.10, 0.85),

        # Controlled jump parameters
        'jump_params': (0.05, 0.10, 0.15),  # Reduced jump magnitudes

        # Bounded sentiment parameters
        'sentiment_params': (0.35, 0.15),  # Reduced feedback

        # Additional numerical safeguards
        'max_price': 1e6,  # Absolute price ceiling
        'max_volatility': 5.0,  # Volatility cap
        'min_liquidity': 1e-5,  # Prevent division by zero

        # Updated flash crash protection
        'flash_crash_threshold': (-0.20, 1.5),  # More conservative

        # Other parameters with stability checks
        'market_maker_power': 0.15,
        'transaction_cost': 0.0002,
        'mean_reversion_speed': 0.05,
        'long_term_mean': 150,
        'market_shock_prob': 0.05,
        'random_seed': 2025,

        # Event parameters with bounds
        'special_events': {
            'gpu_breakthrough': {
                'probability': 0.10,
                'impact': 0.25  # Reduced from 0.35
            },
            'export_restrictions': {
                'probability': 0.08,
                'impact': -0.20  # Reduced from -0.25
            }
        }
    }
]
# Apply transition probability adjustments


def adjust_transition_probabilities(config, steps_per_day):
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

    config['regimes'] = adjusted_regimes


# Initialize with 78 steps/day (3.5 minute bars)
for config in configs_nvidia:
    adjust_transition_probabilities(config, 78)
    # Adjust total steps for duration
    config['steps'] = 252 * 78 * config['duration']
