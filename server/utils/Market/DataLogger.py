"""
DataLogger.py

Handles all data logging: orders, trades, order book snapshots, etc.
Also broadcasts each event in real-time via Socket.IO.
"""

import csv
import json
import os
import time

# NEW IMPORT for real-time broadcasting:
from Market.SocketManager import broadcast_event


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
        """Log each new order event in memory and broadcast."""
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
        self.symbol_events[order.symbol].append(record)

        # NEW: broadcast the order event in real-time
        broadcast_event("new_order", record)

    def log_order_book(self, symbol, bids, asks, order_id):
        """
        Log the state of the order book for a given symbol right after processing an order
        and broadcast.
        """
        record = {
            "timestamp": time.time(),
            "event_type": "ORDER_BOOK",
            "symbol": symbol,
            "order_id": order_id,
            "bids": json.dumps(bids),  # store as JSON for CSV readability
            "asks": json.dumps(asks)
        }
        # We don't store this in an array above (unless you want to track all snapshots).
        # Optionally, you could store in self.snapshot_events if you want:
        #   self.snapshot_events.append(record)
        #   self.symbol_events[symbol].append(record)

        # NEW: broadcast the order-book update
        broadcast_event("order_book", record)

    def log_trade(self, symbol, trade_type, trade_size, trade_price):
        """
        Called whenever a trade happens. Store as an event of type 'TRADE'
        and broadcast.
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
        self.symbol_events[symbol].append(record)

        # NEW: broadcast the trade
        broadcast_event("trade", record)

    def log_snapshot(self, step, symbol, best_bid, best_ask, mid_price):
        """
        Called at each simulation step to record a 'SNAPSHOT' and broadcast.
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
        self.symbol_events[symbol].append(record)

        # NEW: broadcast the snapshot
        broadcast_event("snapshot", record)

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
