"""
Created on 15/01/2025

@author: Aryan

Filename: DataLogger.py

Relative Path: server/utils/Market/DataLogger.py
"""

import csv
import json
import os
import time


class DataLogger:
    """
    Handles all data logging: orders, trades, order book snapshots, etc.
    For clarity, logs to multiple CSV files per heat:
      - orders.csv        (all orders)
      - trades.csv        (all trades)
      - snapshots.csv     (all symbols, all snapshots)
      - symbol_<SYM>.csv  (per symbol snapshots/trades)
    """

    def __init__(self, base_log_dir="logs", symbols=None):
        self.base_log_dir = base_log_dir
        self.symbols = symbols if symbols else []

        # Will be set each time a heat starts/ends
        self.current_heat_id = None
        self.current_heat_dir = None

        # In-memory storage for the current heat
        self.order_events = []     # All orders
        self.trade_events = []     # All trades
        self.snapshot_events = []  # All snapshots
        self.symbol_events = {}    # symbol -> list of events

    def start_new_heat(self, heat_id):
        """Called by HeatManager (or simulator) to start a new heat."""
        self.current_heat_id = heat_id
        self.current_heat_dir = os.path.join(
            self.base_log_dir, f"heat_{heat_id}"
        )
        os.makedirs(self.current_heat_dir, exist_ok=True)

        # Initialize or clear in-memory logs
        self.order_events.clear()
        self.trade_events.clear()
        self.snapshot_events.clear()

        self.symbol_events = {sym: [] for sym in self.symbols}

    def log_order(self, order):
        """Log each new order event in memory."""
        record = {
            "timestamp": time.time(),
            "event_type": "ORDER",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "trader_type": order.trader_type,
            "order_type": order.order_type,
            "side": order.side,
            "size": order.size,
            "price": order.price
        }
        self.order_events.append(record)
        # Also store in symbol-specific log (for reference)
        self.symbol_events[order.symbol].append(record)

    def log_order_book(self, symbol, bids, asks, order_id):
        """
        Log the state of the order book for a given symbol right after processing an order.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "ORDER_BOOK",
            "symbol": symbol,
            "order_id": order_id,
            # Serialize bids and asks as JSON strings for readability in CSV
            "bids": json.dumps(bids),
            "asks": json.dumps(asks)
        }
        # Append the order book log to in-memory sym

    def log_trade(self, symbol, trade_type, trade_size, trade_price):
        """
        Called whenever a trade happens. Store as an event of type 'TRADE'.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "TRADE",
            "symbol": symbol,
            "trade_type": trade_type,
            "trade_size": trade_size,
            "trade_price": trade_price
        }
        self.trade_events.append(record)
        # Also store in symbol-specific log
        self.symbol_events[symbol].append(record)

    def log_snapshot(self, step, symbol, best_bid, best_ask, mid_price):
        """
        Called at each simulation step to record a 'SNAPSHOT'.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "SNAPSHOT",
            "step": step,
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price
        }
        self.snapshot_events.append(record)
        # Also store in symbol-specific log
        self.symbol_events[symbol].append(record)

    def end_heat(self):
        """
        Called when the heat ends to write the logs to disk,
        then clear them from memory.
        """
        # Sort each type of event by timestamp for continuity
        self.order_events.sort(key=lambda x: x["timestamp"])
        self.trade_events.sort(key=lambda x: x["timestamp"])
        self.snapshot_events.sort(key=lambda x: x["timestamp"])

        # Write orders
        orders_file = os.path.join(self.current_heat_dir, "orders.csv")
        order_fieldnames = [
            "timestamp",
            "event_type",
            "order_id",
            "symbol",
            "trader_type",
            "order_type",
            "side",
            "size",
            "price"
        ]
        self._write_csv(orders_file, order_fieldnames, self.order_events)

        # Write trades
        trades_file = os.path.join(self.current_heat_dir, "trades.csv")
        trade_fieldnames = [
            "timestamp",
            "event_type",
            "symbol",
            "trade_type",
            "trade_size",
            "trade_price"
        ]
        self._write_csv(trades_file, trade_fieldnames, self.trade_events)

        # Write snapshots (for all symbols)
        snapshots_file = os.path.join(self.current_heat_dir, "snapshots.csv")
        snapshot_fieldnames = [
            "timestamp",
            "event_type",
            "step",
            "symbol",
            "best_bid",
            "best_ask",
            "mid_price"
        ]
        self._write_csv(snapshots_file, snapshot_fieldnames,
                        self.snapshot_events)

        # Write per-symbol CSV
        for sym, events in self.symbol_events.items():
            events.sort(key=lambda x: x["timestamp"])
            # Identify the fieldnames that might appear for each event type
            fieldnames = set()
            for e in events:
                fieldnames.update(e.keys())
            fieldnames = list(fieldnames)

            symbol_file = os.path.join(
                self.current_heat_dir, f"symbol_{sym}.csv")
            self._write_csv(symbol_file, fieldnames, events)

        print(
            f"Heat {self.current_heat_id} data saved to {self.current_heat_dir}"
        )

        # Clear in-memory data after saving
        self.order_events.clear()
        self.trade_events.clear()
        self.snapshot_events.clear()
        self.symbol_events.clear()

    @staticmethod
    def _write_csv(filepath, fieldnames, rows):
        """Utility method to write rows to CSV with given fieldnames."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

