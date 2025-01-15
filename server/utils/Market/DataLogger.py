"""
Created on 15/01/2025

@author: Aryan

Filename: DataLogger.py

Relative Path: server/utils/Market/DataLogger.py
"""

import csv
import os
import time


class DataLogger:
    """
    Handles all data logging: trades, order book snapshots, etc.
    For simplicity, logs to a single CSV file per heat, with a single 
    table that includes both trade events and snapshot events.
    """

    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = base_log_dir

        # Will be set each time a heat starts/ends
        self.current_heat_id = None
        self.current_heat_dir = None

        # In-memory storage for the current heat (all events!)
        self.events = []

    def start_new_heat(self, heat_id):
        """Called by HeatManager (or simulator) to start a new heat."""
        self.current_heat_id = heat_id
        self.current_heat_dir = os.path.join(
            self.base_log_dir, f"heat_{heat_id}"
        )
        os.makedirs(self.current_heat_dir, exist_ok=True)

        # Clear any old data from memory
        self.events.clear()

    def log_trade(self, symbol, trade_type, trade_size, trade_price):
        """
        Called whenever a trade happens. Store as an event of type 'TRADE'.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "TRADE",
            "step": None,
            "symbol": symbol,
            "trade_type": trade_type,
            "trade_size": trade_size,
            "trade_price": trade_price,
            "best_bid": None,
            "best_ask": None,
            "mid_price": None,
        }
        self.events.append(record)

    def log_snapshot(self, step, symbol, best_bid, best_ask, mid_price):
        """
        Called at each simulation step to record a 'SNAPSHOT'.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "SNAPSHOT",
            "step": step,
            "symbol": symbol,
            "trade_type": None,
            "trade_size": None,
            "trade_price": None,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
        }
        self.events.append(record)

    def end_heat(self):
        """
        Called when the heat ends to actually write the logs to disk,
        then clear them from memory.
        """
        # Sort by timestamp so it's continuous in time
        self.events.sort(key=lambda x: x["timestamp"])

        log_file = os.path.join(self.current_heat_dir, "all_events.csv")
        fieldnames = [
            "timestamp",
            "event_type",
            "step",
            "symbol",
            "trade_type",
            "trade_size",
            "trade_price",
            "best_bid",
            "best_ask",
            "mid_price"
        ]
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.events:
                writer.writerow(row)

        print(
            f"Heat {self.current_heat_id} data saved to {self.current_heat_dir}"
        )

        # Clear from memory after saving
        self.events.clear()
