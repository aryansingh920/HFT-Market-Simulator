"""
Created on 03/02/2025

@author: Aryan

Filename: OrderBook.py

Relative Path: server/MarketModel/OrderBook.py
"""


import numpy as np


class OrderBook:
    """
    This class implements a simple order book that keeps track of two sides:
      - Bids (buy orders): sorted by descending price and then ascending time.
      - Asks (sell orders): sorted by ascending price and then ascending time.
      
    It supports limit orders (which add liquidity) and market orders (which immediately
    match against the best available price). Matching orders are executed if the highest bid
    meets or exceeds the lowest ask. In this simplified implementation, each order is assumed
    to have a fixed quantity.
    """

    def __init__(self):
        self.bids = []  # List of bid orders (each a dict)
        self.asks = []  # List of ask orders (each a dict)
        self.order_id_counter = 0
        self.history = []  # To record snapshots of the order book state

    def add_order(self, side, order_type, price, quantity, timestamp):
        """
        Adds a new order to the book.
        - side: "bid" or "ask"
        - order_type: "limit" or "market"
        - price: For a limit order, the limit price. For a market order, this will be ignored.
        - quantity: The order size (here assumed to be a positive number).
        - timestamp: The simulation time at which the order arrives.
        """
        self.order_id_counter += 1
        order = {
            "order_id": self.order_id_counter,
            "side": side,
            "order_type": order_type,
            "price": price if order_type == "limit" else None,
            "quantity": quantity,
            "timestamp": timestamp
        }
        if side == "bid":
            self.bids.append(order)
            # Sort bids: highest price first; if equal, earlier time first.
            self.bids.sort(
                key=lambda o: (-o["price"] if o["price"] is not None else -np.inf, o["timestamp"]))
        elif side == "ask":
            self.asks.append(order)
            # Sort asks: lowest price first; if equal, earlier time first.
            self.asks.sort(key=lambda o: (
                o["price"] if o["price"] is not None else np.inf, o["timestamp"]))
        # Attempt to match orders after the new order is added.
        self.match_orders()
        return order

    def match_orders(self):
        """
        Continuously matches the best bid and best ask if their prices overlap.
        For a match, the trade price is taken to be the best ask price.
        If orders are only partially filled, their remaining quantity is kept.
        Fully filled orders are removed.
        """
        trades = []
        while self.bids and self.asks:
            # For market orders, treat them as having an effective price equal to the opposing side.
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            bid_price = best_bid["price"] if best_bid["order_type"] == "limit" else np.inf
            ask_price = best_ask["price"] if best_ask["order_type"] == "limit" else -np.inf

            # Determine if a trade can occur. For market orders, we assume they immediately cross.
            if (bid_price >= ask_price) or (best_bid["order_type"] == "market") or (best_ask["order_type"] == "market"):
                trade_quantity = min(
                    best_bid["quantity"], best_ask["quantity"])
                # Use best ask price if available; if one side is a market order, default to a notional price.
                trade_price = best_ask["price"] if best_ask["price"] is not None else best_bid["price"]
                trades.append({
                    "bid_order_id": best_bid["order_id"],
                    "ask_order_id": best_ask["order_id"],
                    "quantity": trade_quantity,
                    "price": trade_price
                })
                best_bid["quantity"] -= trade_quantity
                best_ask["quantity"] -= trade_quantity
                if best_bid["quantity"] <= 0:
                    self.bids.pop(0)
                if best_ask["quantity"] <= 0:
                    self.asks.pop(0)
            else:
                # No more matching orders
                break
        return trades

    def record_state(self, timestamp):
        """
        Saves a snapshot of the current order book state, including:
          - Best bid and best ask.
          - The current lists of bid and ask orders.
        """
        best_bid = self.bids[0]["price"] if self.bids and self.bids[0]["price"] is not None else None
        best_ask = self.asks[0]["price"] if self.asks and self.asks[0]["price"] is not None else None
        snapshot = {
            "timestamp": timestamp,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_orders": [order.copy() for order in self.bids],
            "ask_orders": [order.copy() for order in self.asks]
        }
        self.history.append(snapshot)

    def get_history(self):
        """Returns the recorded history of order book snapshots."""
        return self.history
