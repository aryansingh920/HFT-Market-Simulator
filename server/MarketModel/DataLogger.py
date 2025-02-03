"""
Created on 03/02/2025

@author: Aryan

Filename: DataLogger.py

Relative Path: server/MarketModel/DataLogger.py
"""

import csv
import json


def save_simulation_data_to_csv(simulation_result, config, filename):
    """
    Saves simulation data along with order book snapshots and hyperparameters to a CSV file.
    
    Each row in the CSV corresponds to one simulation step (timestamp) and contains:
      - timestamp, price, fundamental value, volatility, regime,
      - from the order book (if available): best_bid, best_ask, bid_orders, ask_orders,
      - a JSON dump of the hyper–parameters from config.
    
    Parameters:
      simulation_result: dict with keys "time", "prices", "fundamentals", "volatilities",
                         "regime_history", and optionally "order_book_history".
      config: The configuration dictionary used in the simulation.
      filename: The name of the CSV file to be saved.
    """
    times = simulation_result.get("time", [])
    prices = simulation_result.get("prices", [])
    fundamentals = simulation_result.get("fundamentals", [])
    volatilities = simulation_result.get("volatilities", [])
    regimes = simulation_result.get("regime_history", [])
    order_book_history = simulation_result.get("order_book_history", None)

    # Define the columns/field names.
    fieldnames = [
        "timestamp",
        "price",
        "fundamental",
        "volatility",
        "regime",
        "best_bid",
        "best_ask",
        "bid_orders",
        "ask_orders",
        "hyperparameters"
    ]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Assume that the lists for simulation data are aligned (i.e. same length)
        for i in range(len(times)):
            row = {
                "timestamp": times[i],
                "price": prices[i],
                "fundamental": fundamentals[i],
                "volatility": volatilities[i],
                "regime": regimes[i] if i < len(regimes) else regimes[-1] if regimes else "",
                "hyperparameters": json.dumps(config)
            }
            # If order book snapshots were recorded, add the corresponding values.
            if order_book_history and i < len(order_book_history):
                snapshot = order_book_history[i]
                row["best_bid"] = snapshot.get("best_bid", "")
                row["best_ask"] = snapshot.get("best_ask", "")
                # Save the full lists of orders as JSON strings.
                row["bid_orders"] = json.dumps(snapshot.get("bid_orders", ""))
                row["ask_orders"] = json.dumps(snapshot.get("ask_orders", ""))
            else:
                row["best_bid"] = ""
                row["best_ask"] = ""
                row["bid_orders"] = ""
                row["ask_orders"] = ""

            writer.writerow(row)


def save_simulation_data(simulation_result, config, stock_name):
    """
    Saves stock price data, order book data, and hyperparameters into separate CSVs.
    
    - stock_prices.csv -> Price, volatility, regime per timestamp.
    - order_book.csv -> Best bid, best ask, and full order book at each timestamp.
    - simulation_config.csv -> Hyperparameters (key-value).
    - regime_transitions.csv -> Tracks regime changes over time.
    """

    # 1. Save stock prices
    with open(f"stock_prices.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stock_name", "price",
                        "fundamental", "volatility", "regime"])
        for i, t in enumerate(simulation_result["time"]):
            writer.writerow([t, stock_name, simulation_result["prices"][i],
                             simulation_result["fundamentals"][i],
                             simulation_result["volatilities"][i],
                             simulation_result["regime_history"][i]])

    # 2. Save order book data
    with open(f"order_book.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stock_name", "best_bid",
                        "best_ask", "bid_orders", "ask_orders"])
        for i, t in enumerate(simulation_result["time"]):
            best_bid = simulation_result.get("order_book_history", [{}])[
                i].get("best_bid", "")
            best_ask = simulation_result.get("order_book_history", [{}])[
                i].get("best_ask", "")
            bid_orders = json.dumps(simulation_result.get(
                "order_book_history", [{}])[i].get("bid_orders", ""))
            ask_orders = json.dumps(simulation_result.get(
                "order_book_history", [{}])[i].get("ask_orders", ""))
            writer.writerow(
                [t, stock_name, best_bid, best_ask, bid_orders, ask_orders])

    # 3. Save simulation config
    with open(f"simulation_config.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "stock_name",
                        "param_name", "param_value"])
        config_id = 1  # If multiple configs, increment this.
        for key, value in config.items():
            writer.writerow([config_id, stock_name, key, json.dumps(value)])

    # 4. Save regime transitions
    with open(f"regime_transitions.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stock_name",
                        "previous_regime", "new_regime"])
        prev_regime = simulation_result["regime_history"][0]
        for i, t in enumerate(simulation_result["time"]):
            new_regime = simulation_result["regime_history"][i]
            if new_regime != prev_regime:
                writer.writerow([t, stock_name, prev_regime, new_regime])
                prev_regime = new_regime
