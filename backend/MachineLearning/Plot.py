"""
Created on 15/02/2025

@author: Aryan

Filename: Plot.py

Relative Path: backend/MachineLearning/Plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from constants import ORDERBOOK_FEATURES, STEPS_FEATURES
# Load the processed data
orderbook_df = pd.read_csv(ORDERBOOK_FEATURES)
steps_df = pd.read_csv(STEPS_FEATURES)

# Plot Bid-Ask Spread
plt.figure(figsize=(12, 6))
plt.plot(orderbook_df['bid_ask_spread'],
         label="Bid-Ask Spread", color='blue', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Bid-Ask Spread")
plt.title("Bid-Ask Spread Over Time")
plt.legend()
plt.show()

# Plot Order Depth
plt.figure(figsize=(12, 6))
plt.plot(orderbook_df['order_depth'],
         label="Order Depth", color='green', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Order Depth")
plt.title("Order Depth Over Time")
plt.legend()
plt.show()

# Plot VWAP
plt.figure(figsize=(12, 6))
plt.plot(orderbook_df['vwap'], label="VWAP", color='red', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Volume Weighted Average Price (VWAP)")
plt.title("VWAP Over Time")
plt.legend()
plt.show()

# Plot Market Impact
plt.figure(figsize=(12, 6))
plt.plot(orderbook_df['market_impact'],
         label="Market Impact", color='purple', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Market Impact")
plt.title("Market Impact Over Time")
plt.legend()
plt.show()

# Plot Moving Averages from Steps Data
plt.figure(figsize=(12, 6))
plt.plot(steps_df['moving_avg_5'],
         label="5-Period Moving Avg", color='orange', alpha=0.7)
plt.plot(steps_df['moving_avg_10'],
         label="10-Period Moving Avg", color='brown', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Moving Averages")
plt.title("Moving Averages Over Time")
plt.legend()
plt.show()

# Plot Momentum
plt.figure(figsize=(12, 6))
plt.plot(steps_df['momentum'], label="Momentum", color='cyan', alpha=0.7)
plt.xlabel("Time (Index)")
plt.ylabel("Momentum Indicator")
plt.title("Momentum Over Time")
plt.legend()
plt.show()
