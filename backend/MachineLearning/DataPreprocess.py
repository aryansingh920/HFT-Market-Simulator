import pandas as pd
import numpy as np

# Load Order Book Data
orderbook_path = "backend/simulation_output/Complex Regime Example_orderbook.csv"
steps_path = "backend/simulation_output/Complex Regime Example_steps.csv"

orderbook_df = pd.read_csv(orderbook_path)
steps_df = pd.read_csv(steps_path)

# Standardize column names (strip spaces and lowercase)
orderbook_df.columns = [col.strip().lower() for col in orderbook_df.columns]
steps_df.columns = [col.strip().lower() for col in steps_df.columns]

# Print column names to verify
print("Orderbook Columns:", orderbook_df.columns)
print("Steps Columns:", steps_df.columns)

# Column Mapping
column_mapping = {
    'best_bid': 'bid_price',
    'best_ask': 'ask_price',
    'bid_orders': 'bid_volume',
    'ask_orders': 'ask_volume'
}
orderbook_df.rename(columns=column_mapping, inplace=True)

# Convert relevant columns to float (ensuring correct dtype)
numeric_columns = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume']

for col in numeric_columns:
    # Convert to float, setting errors to NaN
    orderbook_df[col] = pd.to_numeric(orderbook_df[col], errors='coerce')

# Drop rows with NaN (optional, if needed)
orderbook_df.dropna(subset=numeric_columns, inplace=True)

### Feature Extraction: Order Book ###

# Compute Bid-Ask Spread
orderbook_df['bid_ask_spread'] = orderbook_df['ask_price'] - \
    orderbook_df['bid_price']

# Compute Order Depth (Liquidity Measure)
orderbook_df['order_depth'] = orderbook_df['bid_volume'] + \
    orderbook_df['ask_volume']

# Compute VWAP (Volume-Weighted Average Price)
orderbook_df['vwap'] = (orderbook_df['bid_price'] * orderbook_df['bid_volume'] +
                        orderbook_df['ask_price'] * orderbook_df['ask_volume']) / \
    (orderbook_df['bid_volume'] + orderbook_df['ask_volume'])

# Compute Market Impact (Price Movement with Large Orders)
orderbook_df['market_impact'] = orderbook_df['bid_ask_spread'] * \
    orderbook_df['order_depth']

### Feature Extraction: Step Data ###

# Convert Timestamp to Datetime
steps_df['timestamp'] = pd.to_datetime(steps_df['timestamp'])
steps_df.set_index('timestamp', inplace=True)

# Convert Step Data Columns to Numeric
steps_numeric_columns = ['price', 'fundamental', 'volatility']
for col in steps_numeric_columns:
    steps_df[col] = pd.to_numeric(steps_df[col], errors='coerce')

# Compute Moving Averages (Market Trends)
steps_df['moving_avg_5'] = steps_df['price'].rolling(window=5).mean()
steps_df['moving_avg_10'] = steps_df['price'].rolling(window=10).mean()
steps_df['momentum'] = steps_df['price'].diff()

# Compute Trade Execution Patterns
steps_df['trade_frequency'] = steps_df['price'].rolling(window=5).count()
steps_df['avg_trade_size'] = steps_df['price'].rolling(window=5).mean()

# Save Processed Data
orderbook_df.to_csv(
    "backend/processed_output/orderbook_features.csv", index=False)
steps_df.to_csv("backend/processed_output/steps_features.csv", index=False)

print("Feature extraction complete! Processed files saved.")
