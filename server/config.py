"""
Created on 01/02/2025

@author: Aryan

Filename: config.py
Relative Path: server/config.py
"""


configs_nvidia = [
    {
        # Basic simulation settings
        'name': 'Nvidia: Tech Giant - Innovation & Volatility',
        'duration': 5,             # Simulation duration in years
        'steps': 1260,             # Approx. 252 trading days per year * 5 years
        'initial_price': 250,      # Starting stock price
        'fundamental_value': 250,  # Fundamental value (used in mean–reversion)
        'initial_liquidity': 1e7,  # High liquidity typical for a large-cap tech stock
        'base_volatility': 0.3,    # Base annualized volatility

        # Market regimes represented as a Markov chain:
        # Each regime has a specific drift (annualized return), a volatility scaling factor,
        # and a set of transition probabilities to other regimes.
        'regimes': [
            {
                'name': 'steady_growth',
                'drift': 0.15,       # 15% annual drift in a stable, growing market
                'vol_scale': 1.0,
                'transitions': {
                    'steady_growth': 0.80,
                    'innovation_boom': 0.10,
                    'regulatory_pressure': 0.05,
                    'tech_correction': 0.03,
                    'market_boom': 0.02
                }
            },
            {
                'name': 'innovation_boom',
                'drift': 0.30,       # Strong growth during major innovation breakthroughs
                'vol_scale': 1.2,
                'transitions': {
                    'innovation_boom': 0.70,
                    'steady_growth': 0.15,
                    'market_boom': 0.10,
                    'tech_correction': 0.05
                }
            },
            {
                'name': 'regulatory_pressure',
                'drift': 0.05,       # Low growth due to regulatory or legal pressures
                'vol_scale': 1.5,
                'transitions': {
                    'regulatory_pressure': 0.60,
                    'steady_growth': 0.25,
                    'tech_correction': 0.15
                }
            },
            {
                'name': 'tech_correction',
                'drift': -0.25,      # Negative drift during market corrections
                'vol_scale': 0.5,
                'transitions': {
                    'tech_correction': 0.75,
                    'steady_growth': 0.20,
                    'innovation_boom': 0.05
                }
            },
            {
                'name': 'market_boom',
                'drift': 0.40,       # Exuberant market environment with high valuations
                'vol_scale': 22.1,
                'transitions': {
                    'market_boom': 0.80,
                    'steady_growth': 0.15,
                    'innovation_boom': 0.05
                }
            },
            {
                'name': 'crash',
                # Severe downturn (e.g., during a tech bubble burst)
                'drift': 10.50,
                'vol_scale': 2.0,
                'transitions': {
                    'crash': 0.90,
                    'recovery': 0.10
                }
            },
            {
                'name': 'recovery',
                'drift': 40.20,       # Recovery phase following a crash
                'vol_scale': 5.3,
                'transitions': {
                    'recovery': 5.85,
                    'steady_growth': 100.15
                }
            }
        ],

        # GARCH parameters (omega, alpha, beta) for updating variance
        'garch_params': (0.015, 0.12, 0.82),

        # Macro-economic factors affecting the drift:
        # Each is defined as a tuple: (base value, volatility)
        'macro_impact': {
            'interest_rate': (0.03, 0.01),
            'inflation': (0.02, 0.005)
        },

        # Sentiment parameters: (mean reversion speed, volatility of sentiment shocks)
        'sentiment_params': (0.5, 0.2),

        # Flash crash settings: (price drop threshold, liquidity threshold)
        'flash_crash_threshold': (-0.2, 2),

        # Market maker influence (affects how liquidity impacts price)
        'market_maker_power': 0.1,

        # Transaction cost applied to trades (affects effective price)
        'transaction_cost': 0.0005,

        # Jump diffusion parameters: (intensity, mean jump size, jump standard deviation)
        'jump_params': (0.12, -0.2, 0.3),

        # Mean reversion parameters for alternative price dynamics
        'mean_reversion_speed': 0.1,
        'long_term_mean': 250,

        # Probability of a market shock event and shock type (e.g., bullish, bearish)
        'market_shock_prob': 0.02,
        'market_shock': None,

        # Advanced simulation parameters for the AdvancedStockSimulator
        'sentiment_seed': 0.2,
        'news_flow_intensity': 0.1,
        'seasonality_params': {
            # Example day-of-week multipliers
            'day_of_week': [1.0, 0.98, 1.02, 1.01, 0.99]
        },
        'heston_params': {
            'kappa': 1.2,    # Speed of mean reversion for volatility
            'theta': 0.04,   # Long-term variance
            'eta': 0.2       # Volatility of volatility
        },
        'refined_jump_params': {
            'intensity': 0.05,  # Intensity of additional jump events
            'df': 3             # Degrees of freedom for the t-distribution jump model
        },

        # Optional external regime switching events (list of tuples: (mu, sigma, duration))
        'regime_switch': [
            (0.0, 0.05, 0.5),
            (0.05, 0.1, 0.25)
        ],

        # Random seed for reproducibility
        'random_seed': 2025
    }
]


# configs_historical = [
#     {
#         'name': 'Dot-Com Bubble (1995–2002)',
#         'duration': 7,
#         'steps': 1764,  # 7 years of trading days
#         'initial_price': 500,
#         'fundamental_value': 200,
#         'regimes': [
#             {'name': 'boom', 'drift': 0.25, 'vol_scale': 1.2,
#              'transitions': {'boom': 0.85, 'peak': 0.1, 'burst': 0.05}},
#             {'name': 'peak', 'drift': 0.10, 'vol_scale': 1.5,
#              'transitions': {'peak': 0.6, 'burst': 0.4}},
#             {'name': 'burst', 'drift': -0.3, 'vol_scale': 2.2,
#              'transitions': {'burst': 0.9, 'recovery': 0.1}},
#             {'name': 'recovery', 'drift': 0.07, 'vol_scale': 1.0,
#              'transitions': {'recovery': 0.9, 'burst': 0.1}}
#         ],
#         'garch_params': (0.02, 0.08, 0.88),
#         'jump_params': (0.1, -0.15, 0.2),
#         'market_maker_power': 0.05,
#         'transaction_cost': 0.0005,
#         'random_seed': 2000
#     },
#     {
#         'name': '2008 Global Financial Crisis',
#         'duration': 3,
#         'steps': 756,
#         'initial_price': 1500,
#         'fundamental_value': 1200,
#         'regimes': [
#             {'name': 'pre_crisis', 'drift': 0.05, 'vol_scale': 1.0,
#              'transitions': {'pre_crisis': 0.85, 'collapse': 0.15}},
#             {'name': 'collapse', 'drift': -0.5, 'vol_scale': 3.5,
#              'transitions': {'collapse': 0.7, 'rebound': 0.3}},
#             {'name': 'rebound', 'drift': 0.15, 'vol_scale': 2.0,
#              'transitions': {'rebound': 0.6, 'collapse': 0.1, 'post_crisis': 0.3}},
#             {'name': 'post_crisis', 'drift': 0.08, 'vol_scale': 1.2,
#              'transitions': {'post_crisis': 1.0}}
#         ],
#         'garch_params': (0.03, 0.12, 0.78),
#         'jump_params': (0.2, -0.25, 0.4),  # Frequent large drops
#         'flash_crash_threshold': (-0.3, 2),
#         'transaction_cost': 0.001,
#         'random_seed': 2008
#     },
#     {
#         'name': 'COVID-19 Market Crash & Recovery (2020)',
#         'duration': 2,
#         'steps': 504,
#         'initial_price': 3200,
#         'fundamental_value': 3000,
#         'regimes': [
#             {'name': 'normal', 'drift': 0.08, 'vol_scale': 0.9,
#              'transitions': {'normal': 0.85, 'crash': 0.15}},
#             {'name': 'crash', 'drift': -0.35, 'vol_scale': 4.5,
#              'transitions': {'crash': 0.6, 'rebound': 0.4}},
#             {'name': 'rebound', 'drift': 0.3, 'vol_scale': 2.0,
#              'transitions': {'rebound': 0.5, 'crash': 0.1, 'bull_market': 0.4}},
#             {'name': 'bull_market', 'drift': 0.15, 'vol_scale': 0.8,
#              'transitions': {'bull_market': 0.95, 'correction': 0.05}},
#             {'name': 'correction', 'drift': -0.1, 'vol_scale': 1.5,
#              'transitions': {'correction': 0.3, 'bull_market': 0.7}}
#         ],
#         'garch_params': (0.05, 0.2, 0.6),
#         'jump_params': (0.1, -0.4, 0.3),  # Sudden large drawdowns
#         'market_shock_prob': 0.03,
#         'transaction_cost': 0.0008,
#         'random_seed': 2020
#     },
#     {
#         'name': 'Roaring Twenties & 1929 Great Depression',
#         'duration': 10,
#         'steps': 2520,
#         'initial_price': 100,
#         'fundamental_value': 120,
#         'regimes': [
#             {'name': 'boom', 'drift': 0.2, 'vol_scale': 0.9,
#              'transitions': {'boom': 0.88, 'peak': 0.1, 'crash': 0.02}},
#             {'name': 'peak', 'drift': 0.1, 'vol_scale': 1.5,
#              'transitions': {'peak': 0.5, 'crash': 0.5}},
#             {'name': 'crash', 'drift': -0.5, 'vol_scale': 3.0,
#              'transitions': {'crash': 0.8, 'recovery': 0.2}},
#             {'name': 'recovery', 'drift': 0.08, 'vol_scale': 1.2,
#              'transitions': {'recovery': 0.7, 'boom': 0.3}}
#         ],
#         'garch_params': (0.02, 0.1, 0.85),
#         'jump_params': (0.15, -0.3, 0.25),  # Large downward jumps
#         'flash_crash_threshold': (-0.4, 1.5),
#         'transaction_cost': 0.0003,
#         'random_seed': 1929
#     },
#     {
#         'name': 'Hyperinflation in Weimar Germany (1921–1923)',
#         'duration': 3,
#         'steps': 756,
#         'initial_price': 1,
#         'fundamental_value': 1,
#         'regimes': [
#             {'name': 'slow_rise', 'drift': 0.2, 'vol_scale': 1.2,
#              'transitions': {'slow_rise': 0.6, 'accelerating': 0.4}},
#             {'name': 'accelerating', 'drift': 0.5, 'vol_scale': 2.5,
#              'transitions': {'accelerating': 0.7, 'hyperinflation': 0.3}},
#             {'name': 'hyperinflation', 'drift': 2.0, 'vol_scale': 5.0,
#              'transitions': {'hyperinflation': 0.8, 'collapse': 0.2}},
#             {'name': 'collapse', 'drift': -0.9, 'vol_scale': 3.5,
#              'transitions': {'collapse': 1.0}}
#         ],
#         'garch_params': (0.1, 0.4, 0.4),
#         'jump_params': (0.3, 1.0, 0.5),  # Massive upward jumps
#         'transaction_cost': 0.002,
#         'random_seed': 1921
#     },
#     {
#         'name': 'Black Monday (1987)',
#         'duration': 1,
#         'steps': 252,
#         'initial_price': 2000,
#         'fundamental_value': 1950,
#         'regimes': [
#             {'name': 'normal', 'drift': 0.06, 'vol_scale': 1.0,
#              'transitions': {'normal': 0.99, 'crash': 0.01}},
#             {'name': 'crash', 'drift': -0.3, 'vol_scale': 6.0,
#              'transitions': {'crash': 0.5, 'recovery': 0.5}},
#             {'name': 'recovery', 'drift': 0.12, 'vol_scale': 1.5,
#              'transitions': {'recovery': 0.9, 'normal': 0.1}}
#         ],
#         'garch_params': (0.02, 0.15, 0.75),
#         'jump_params': (0.25, -0.5, 0.3),  # Large downward jumps
#         'flash_crash_threshold': (-0.25, 3),
#         'transaction_cost': 0.0007,
#         'random_seed': 1987
#     }
# ]
