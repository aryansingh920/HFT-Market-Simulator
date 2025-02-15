SIMULATION_OUTPUT = 'simulation_output'
REGIME_CONFIG = f"{SIMULATION_OUTPUT}/Complex Regime Example_config.json"
ORDERBOOK_SNAPSHOT = f"{SIMULATION_OUTPUT}/Complex Regime Example_orderbook.csv"
SIMULATION_STEPS = f"{SIMULATION_OUTPUT}/Complex Regime Example_steps.csv"

PROCESSED_DATA = 'processed_data'
ORDERBOOK_FEATURES = f"{PROCESSED_DATA}/orderbook_features.csv"
STEPS_FEATURES = f"{PROCESSED_DATA}/steps_features.csv"


column_mapping = {
    'best_bid': 'bid_price',
    'best_ask': 'ask_price',
    'bid_orders': 'bid_volume',
    'ask_orders': 'ask_volume'
}
