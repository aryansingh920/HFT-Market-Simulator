"""
Created on 01/02/2025

@author: Aryan

Filename: config.py

Relative Path: server/config.py
"""

# configs = [
#     {   # Configuration 1: Bull Market with Corrections
#         'duration': 3,
#         'steps': 756,
#         'initial_price': 100,
#         'fundamental_value': 100,
#         'regimes': [
#             {'name': 'bull', 'drift': 0.12, 'vol_scale': 0.7,
#              'transitions': {'bull': 0.92, 'correction': 0.06, 'bear': 0.02}},
#             {'name': 'correction', 'drift': -0.05, 'vol_scale': 1.3,
#              'transitions': {'correction': 0.7, 'bull': 0.2, 'bear': 0.1}},
#             {'name': 'bear', 'drift': -0.18, 'vol_scale': 1.8,
#              'transitions': {'bear': 0.85, 'bull': 0.1, 'correction': 0.05}}
#         ],
#         'garch_params': (0.01, 0.08, 0.89),
#         'market_maker_power': 0.15,
#         'transaction_cost': 0.0003,
#         'jump_params': (0.05, -0.2, 0.1),
#         'random_seed': 42
#     },
#     {   # Configuration 2: Crisis Period with High Volatility and Market Shock
#         'duration': 2,
#         'steps': 504,
#         'initial_price': 150,
#         'fundamental_value': 150,
#         'regimes': [
#             {'name': 'panic', 'drift': -0.35, 'vol_scale': 2.5,
#              'transitions': {'panic': 0.6, 'rebound': 0.4, 'recovery': 0.0}},
#             {'name': 'rebound', 'drift': 0.25, 'vol_scale': 1.8,
#              'transitions': {'rebound': 0.5, 'recovery': 0.5, 'panic': 0.0}},
#             {'name': 'recovery', 'drift': 0.08, 'vol_scale': 1.2,
#              'transitions': {'recovery': 0.85, 'rebound': 0.1, 'panic': 0.05}}
#         ],
#         'garch_params': (0.05, 0.15, 0.75),
#         'flash_crash_threshold': (-0.2, 2),
#         'transaction_cost': 0.001,
#         'jump_params': (0.1, -0.3, 0.15),
#         'random_seed': 2023,
#         'market_shock_prob': 0.05,
#         'market_shock': 'bearish'
#     },
#     {   # Configuration 3: Prolonged Bull Market (Low Volatility Growth)
#         'duration': 5,
#         'steps': 1260,
#         'initial_price': 200,
#         'fundamental_value': 200,
#         'regimes': [
#             {'name': 'growth', 'drift': 0.08, 'vol_scale': 0.5,
#              'transitions': {'growth': 0.95, 'shock': 0.05}},
#             {'name': 'shock', 'drift': -0.1, 'vol_scale': 1.5,
#              'transitions': {'shock': 0.3, 'growth': 0.7}}
#         ],
#         'macro_impact': {'interest_rate': (0.01, 0.005),
#                          'inflation': (0.01, 0.003)},
#         'sentiment_params': (0.3, 0.15),
#         'market_maker_power': 0.2,
#         'random_seed': 101,
#         'jump_params': (0.02, -0.05, 0.03)
#     }
# ]


configs_historical = [
    {   # 📈 **Dot-Com Bubble (1995–2002)**
        'duration': 7,
        'steps': 1764,  # 7 years of trading
        'initial_price': 500,
        'fundamental_value': 200,
        'regimes': [
            {'name': 'boom', 'drift': 0.25, 'vol_scale': 1.2,
             'transitions': {'boom': 0.85, 'peak': 0.1, 'burst': 0.05}},
            {'name': 'peak', 'drift': 0.10, 'vol_scale': 1.5,
             'transitions': {'peak': 0.6, 'burst': 0.4}},
            {'name': 'burst', 'drift': -0.3, 'vol_scale': 2.2,
             'transitions': {'burst': 0.9, 'recovery': 0.1}},
            {'name': 'recovery', 'drift': 0.07, 'vol_scale': 1.0,
             'transitions': {'recovery': 0.9, 'burst': 0.1}}
        ],
        'garch_params': (0.02, 0.08, 0.88),
        'jump_params': (0.1, -0.15, 0.2),  # High chance of sudden drops
        'market_maker_power': 0.05,
        'transaction_cost': 0.0005,
        'random_seed': 2000
    },

    {   # 📉 **2008 Global Financial Crisis**
        'duration': 3,
        'steps': 756,  # 3 years of trading
        'initial_price': 1500,
        'fundamental_value': 1200,
        'regimes': [
            {'name': 'pre_crisis', 'drift': 0.05, 'vol_scale': 1.0,
             'transitions': {'pre_crisis': 0.85, 'collapse': 0.15}},
            {'name': 'collapse', 'drift': -0.5, 'vol_scale': 3.5,
             'transitions': {'collapse': 0.7, 'rebound': 0.3}},
            {'name': 'rebound', 'drift': 0.15, 'vol_scale': 2.0,
             'transitions': {'rebound': 0.6, 'collapse': 0.1, 'post_crisis': 0.3}},
            {'name': 'post_crisis', 'drift': 0.08, 'vol_scale': 1.2,
             'transitions': {'post_crisis': 1.0}}
        ],
        'garch_params': (0.03, 0.12, 0.78),
        'jump_params': (0.2, -0.25, 0.4),  # Frequent large drops
        'flash_crash_threshold': (-0.3, 2),
        'transaction_cost': 0.001,
        'random_seed': 2008
    },

    {   # 🚀 **COVID-19 Market Crash & Recovery (2020)**
        'duration': 2,
        'steps': 504,
        'initial_price': 3200,
        'fundamental_value': 3000,
        'regimes': [
            {'name': 'normal', 'drift': 0.08, 'vol_scale': 0.9,
             'transitions': {'normal': 0.85, 'crash': 0.15}},
            {'name': 'crash', 'drift': -0.35, 'vol_scale': 4.5,
             'transitions': {'crash': 0.6, 'rebound': 0.4}},
            {'name': 'rebound', 'drift': 0.3, 'vol_scale': 2.0,
             'transitions': {'rebound': 0.5, 'crash': 0.1, 'bull_market': 0.4}},
            {'name': 'bull_market', 'drift': 0.15, 'vol_scale': 0.8,
             'transitions': {'bull_market': 0.95, 'correction': 0.05}},
            {'name': 'correction', 'drift': -0.1, 'vol_scale': 1.5,
             'transitions': {'correction': 0.3, 'bull_market': 0.7}}
        ],
        'garch_params': (0.05, 0.2, 0.6),
        'jump_params': (0.1, -0.4, 0.3),  # Sudden large drawdowns
        'market_shock_prob': 0.03,
        'transaction_cost': 0.0008,
        'random_seed': 2020
    },

    {   # 🏆 **The Roaring Twenties & 1929 Great Depression**
        'duration': 10,
        'steps': 2520,
        'initial_price': 100,
        'fundamental_value': 120,
        'regimes': [
            {'name': 'boom', 'drift': 0.2, 'vol_scale': 0.9,
             'transitions': {'boom': 0.88, 'peak': 0.1, 'crash': 0.02}},
            {'name': 'peak', 'drift': 0.1, 'vol_scale': 1.5,
             'transitions': {'peak': 0.5, 'crash': 0.5}},
            {'name': 'crash', 'drift': -0.5, 'vol_scale': 3.0,
             'transitions': {'crash': 0.8, 'recovery': 0.2}},
            {'name': 'recovery', 'drift': 0.08, 'vol_scale': 1.2,
             'transitions': {'recovery': 0.7, 'boom': 0.3}}
        ],
        'garch_params': (0.02, 0.1, 0.85),
        'jump_params': (0.15, -0.3, 0.25),  # Large downward jumps
        'flash_crash_threshold': (-0.4, 1.5),
        'transaction_cost': 0.0003,
        'random_seed': 1929
    },

    {   # 🔥 **Hyperinflation in Weimar Germany (1921–1923)**
        'duration': 3,
        'steps': 756,
        'initial_price': 1,
        'fundamental_value': 1,
        'regimes': [
            {'name': 'slow_rise', 'drift': 0.2, 'vol_scale': 1.2,
             'transitions': {'slow_rise': 0.6, 'accelerating': 0.4}},
            {'name': 'accelerating', 'drift': 0.5, 'vol_scale': 2.5,
             'transitions': {'accelerating': 0.7, 'hyperinflation': 0.3}},
            {'name': 'hyperinflation', 'drift': 2.0, 'vol_scale': 5.0,
             'transitions': {'hyperinflation': 0.8, 'collapse': 0.2}},
            {'name': 'collapse', 'drift': -0.9, 'vol_scale': 3.5,
             'transitions': {'collapse': 1.0}}
        ],
        'garch_params': (0.1, 0.4, 0.4),
        'jump_params': (0.3, 1.0, 0.5),  # Massive upward jumps in price
        'transaction_cost': 0.002,
        'random_seed': 1921
    },

    {   # 📊 **Black Monday (1987)**
        'duration': 1,
        'steps': 252,
        'initial_price': 2000,
        'fundamental_value': 1950,
        'regimes': [
            {'name': 'normal', 'drift': 0.06, 'vol_scale': 1.0,
             'transitions': {'normal': 0.99, 'crash': 0.01}},
            {'name': 'crash', 'drift': -0.3, 'vol_scale': 6.0,
             'transitions': {'crash': 0.5, 'recovery': 0.5}},
            {'name': 'recovery', 'drift': 0.12, 'vol_scale': 1.5,
             'transitions': {'recovery': 0.9, 'normal': 0.1}}
        ],
        'garch_params': (0.02, 0.15, 0.75),
        'jump_params': (0.25, -0.5, 0.3),  # Large downward jumps
        'flash_crash_threshold': (-0.25, 3),
        'transaction_cost': 0.0007,
        'random_seed': 1987
    }
]
