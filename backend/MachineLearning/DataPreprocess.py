"""
Created on 15/02/2025

@author: Aryan

Filename: DataPreprocess.py
Relative Path: backend/MachineLearning/DataPreprocess.py
"""

import pandas as pd
import numpy as np
from constants import ORDERBOOK_SNAPSHOT, SIMULATION_STEPS, ORDERBOOK_FEATURES, STEPS_FEATURES, column_mapping

# ---------------------------
# 1. Load Data
# ---------------------------
orderbook_path = ORDERBOOK_SNAPSHOT
steps_path = SIMULATION_STEPS

orderbook_df = pd.read_csv(orderbook_path)
steps_df = pd.read_csv(steps_path)

# ---------------------------
# 2. Standardize Column Names
# ---------------------------
orderbook_df.columns = [col.strip().lower() for col in orderbook_df.columns]
steps_df.columns = [col.strip().lower() for col in steps_df.columns]

print("Orderbook Columns:", orderbook_df.columns)
print("Steps Columns:", steps_df.columns)

# ---------------------------
# 3. Process Order Book Data
# ---------------------------
# Apply column mapping
orderbook_df.rename(columns=column_mapping, inplace=True)

# Convert relevant columns to float (ensuring correct dtype)
numeric_columns = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume']
for col in numeric_columns:
    orderbook_df[col] = pd.to_numeric(orderbook_df[col], errors='coerce')

print("\nMissing Values in Orderbook Data (Before Handling):")
print(orderbook_df.isna().sum())

# Fix missing data:
# - Drop rows missing key price data
orderbook_df = orderbook_df.dropna(subset=['bid_price', 'ask_price'])
# - Replace missing volumes with a small constant (0.01)
orderbook_df['bid_volume'] = orderbook_df['bid_volume'].fillna(0.01)
orderbook_df['ask_volume'] = orderbook_df['ask_volume'].fillna(0.01)

print("\nMissing Values in Orderbook Data (After Handling):")
print(orderbook_df.isna().sum())

if orderbook_df.empty:
    print("⚠️ ERROR: Orderbook dataframe is still empty after handling missing values! Check your dataset.")

# ---------------------------
# 4. Feature Extraction for Order Book Data
# ---------------------------
orderbook_df['bid_ask_spread'] = orderbook_df['ask_price'] - \
    orderbook_df['bid_price']
orderbook_df['order_depth'] = orderbook_df['bid_volume'] + \
    orderbook_df['ask_volume']
orderbook_df['vwap'] = (
    (orderbook_df['bid_price'] * orderbook_df['bid_volume']) +
    (orderbook_df['ask_price'] * orderbook_df['ask_volume'])
) / (orderbook_df['bid_volume'] + orderbook_df['ask_volume'])
orderbook_df['market_impact'] = orderbook_df['bid_ask_spread'] * \
    orderbook_df['order_depth']

# ---------------------------
# 5. Process Step Data
# ---------------------------
# Convert timestamp to datetime
steps_df['timestamp'] = pd.to_datetime(steps_df['timestamp'])

# Set timestamp as index temporarily for rolling operations
steps_df.set_index('timestamp', inplace=True)

# Convert relevant columns to numeric
steps_numeric_columns = ['price', 'fundamental', 'volatility']
for col in steps_numeric_columns:
    steps_df[col] = pd.to_numeric(steps_df[col], errors='coerce')

# Compute moving averages (market trends)
steps_df['moving_avg_5'] = steps_df['price'].rolling(window=5).mean()
steps_df['moving_avg_10'] = steps_df['price'].rolling(window=10).mean()
steps_df['momentum'] = steps_df['price'].diff()

# Compute trade execution patterns
steps_df['trade_frequency'] = steps_df['price'].rolling(window=5).count()
steps_df['avg_trade_size'] = steps_df['price'].rolling(window=5).mean()

# Drop rows with NaNs introduced by rolling or diff operations
steps_df.dropna(inplace=True)

# Reset index so that 'timestamp' becomes a regular column again
steps_df.reset_index(inplace=True)

# ---------------------------
# 6. Save Processed Data
# ---------------------------
orderbook_df.to_csv(ORDERBOOK_FEATURES, index=False)
steps_df.to_csv(STEPS_FEATURES, index=False)

print("\n✅ Feature extraction complete! Processed files saved.")
