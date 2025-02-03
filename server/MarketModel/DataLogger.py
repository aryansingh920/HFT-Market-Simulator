import csv
import json


def save_simulation_steps_csv(simulation_result, filename):
    """
    Saves the main simulation step data to a CSV file.
    
    The CSV will contain the following columns:
      - timestamp, price, fundamental, volatility, regime
    """
    times = simulation_result.get("time", [])
    prices = simulation_result.get("prices", [])
    fundamentals = simulation_result.get("fundamentals", [])
    volatilities = simulation_result.get("volatilities", [])
    regimes = simulation_result.get("regime_history", [])

    fieldnames = ["timestamp", "price", "fundamental", "volatility", "regime"]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(times)):
            row = {
                "timestamp": times[i],
                "price": prices[i],
                "fundamental": fundamentals[i],
                "volatility": volatilities[i],
                "regime": regimes[i] if i < len(regimes) else ""
            }
            writer.writerow(row)
    print(f"Simulation steps saved to {filename}")


def save_orderbook_snapshots_csv(simulation_result, filename):
    """
    Saves the order book snapshots from the simulation result to a CSV file.
    
    The CSV will contain the following columns:
      - timestamp, best_bid, best_ask, bid_orders, ask_orders
      
    The bid_orders and ask_orders are stored as JSON strings.
    """
    order_book_history = simulation_result.get("order_book_history", [])

    fieldnames = ["timestamp", "best_bid",
                  "best_ask", "bid_orders", "ask_orders"]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in order_book_history:
            row = {
                "timestamp": snapshot.get("timestamp", ""),
                "best_bid": snapshot.get("best_bid", ""),
                "best_ask": snapshot.get("best_ask", ""),
                "bid_orders": json.dumps(snapshot.get("bid_orders", [])),
                "ask_orders": json.dumps(snapshot.get("ask_orders", []))
            }
            writer.writerow(row)
    print(f"Order book snapshots saved to {filename}")


def save_config_as_json(config, filename):
    """
    Saves the configuration (hyperparameters) as a JSON file.
    """
    with open(filename, mode="w") as jsonfile:
        json.dump(config, jsonfile, indent=2)
    print(f"Configuration saved to {filename}")
