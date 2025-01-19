"""
Created on 19/01/2025

@author: Aryan

Filename: DataLogger.py

Relative Path: server/Market/DataLogger.py
"""


import csv
import json
import os
import time
import asyncio

# Import only the broadcast_event function
from Market.WebSocketManager import broadcast_event
# Import the module to dynamically access global_event_loop
import Market.WebSocketManager as ws_manager


class DataLogger:
    """
    Handles all data logging: orders, trades, order book snapshots, etc.
    For clarity, logs to multiple CSV files per heat:
      - orders.csv        (all orders)
      - trades.csv        (all trades)
      - snapshots.csv     (all symbols, all snapshots)
      - symbol_<SYM>.csv  (per symbol, includes orders/trades/snapshots)
    """

    def __init__(self, base_log_dir="logs", symbols=None, loop=None):
        self.base_log_dir = base_log_dir
        self.symbols = symbols if symbols else []
        self.current_heat_id = None
        self.current_heat_dir = None
        self.order_events = []
        self.trade_events = []
        self.snapshot_events = []
        self.symbol_events = {}
        self.loop = loop

    def start_new_heat(self, heat_id):
        self.current_heat_id = heat_id
        self.current_heat_dir = os.path.join(
            self.base_log_dir, f"heat_{heat_id}")
        os.makedirs(self.current_heat_dir, exist_ok=True)
        self.order_events.clear()
        self.trade_events.clear()
        self.snapshot_events.clear()
        self.symbol_events = {sym: [] for sym in self.symbols}

    def log_order(self, order):
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
        self._safe_broadcast("new_order", record)

    def log_order_book(self, symbol, bids, asks, order_id):
        record = {
            "timestamp": time.time(),
            "event_type": "ORDER_BOOK",
            "symbol": symbol,
            "order_id": order_id,
            "bids": json.dumps(bids),
            "asks": json.dumps(asks)
        }
        self._safe_broadcast("order_book", record)

    def log_trade(self, symbol, trade_type, trade_size, trade_price):
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
        self._safe_broadcast("trade", record)

    def log_snapshot(self, step, symbol, best_bid, best_ask, mid_price):
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
        self._safe_broadcast("snapshot", record)

    def end_heat(self):
        self.order_events.sort(key=lambda x: x["timestamp"])
        self.trade_events.sort(key=lambda x: x["timestamp"])
        self.snapshot_events.sort(key=lambda x: x["timestamp"])

        orders_file = os.path.join(self.current_heat_dir, "orders.csv")
        order_fieldnames = [
            "timestamp", "event_type", "order_id", "symbol",
            "trader_type", "order_type", "side", "size", "price"
        ]
        self._write_csv(orders_file, order_fieldnames, self.order_events)

        trades_file = os.path.join(self.current_heat_dir, "trades.csv")
        trade_fieldnames = [
            "timestamp", "event_type", "symbol",
            "trade_type", "trade_size", "trade_price"
        ]
        self._write_csv(trades_file, trade_fieldnames, self.trade_events)

        snapshots_file = os.path.join(self.current_heat_dir, "snapshots.csv")
        snapshot_fieldnames = [
            "timestamp", "event_type", "step", "symbol",
            "best_bid", "best_ask", "mid_price"
        ]
        self._write_csv(snapshots_file, snapshot_fieldnames,
                        self.snapshot_events)

        for sym, events in self.symbol_events.items():
            events.sort(key=lambda x: x["timestamp"])
            fieldnames = list({key for e in events for key in e})
            symbol_file = os.path.join(
                self.current_heat_dir, f"symbol_{sym}.csv")
            self._write_csv(symbol_file, fieldnames, events)

        print(
            f"Heat {self.current_heat_id} data saved to {self.current_heat_dir}")

        self.order_events.clear()
        self.trade_events.clear()
        self.snapshot_events.clear()
        self.symbol_events.clear()

    @staticmethod
    def _write_csv(filepath, fieldnames, rows):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _safe_broadcast(self, event_name, data):
        """
        Schedule the broadcast_event on the stored event loop.
        """
        if self.loop is not None:
            # print(f"[DataLogger] Scheduling broadcast for event: {event_name}")
            self.loop.call_soon_threadsafe(
                asyncio.create_task,
                broadcast_event(event_name, data)
            )
        else:
            print(
                f"[DataLogger] No event loop available for event: {event_name}")
