import csv
import os
import time
import random
import numpy as np
from collections import defaultdict
from datetime import datetime


class Order:
    def __init__(self, order_id, symbol, trader_type, order_type, side, size, price=None):
        """
        :param order_id: Unique ID for each order
        :param symbol: Stock symbol (e.g., 'AAPL', 'GOOG')
        :param trader_type: "Market Maker" or "Trader"
        :param order_type: "market" or "limit"
        :param side: "buy" or "sell"
        :param size: Size/quantity of the order
        :param price: Price for limit orders (None for market orders)
        """
        self.order_id = order_id
        self.symbol = symbol
        self.trader_type = trader_type
        self.order_type = order_type
        self.side = side
        self.size = size
        self.price = price

    def __repr__(self):
        return (f"Order(id={self.order_id}, symbol={self.symbol}, "
                f"type={self.order_type}, side={self.side}, "
                f"size={self.size}, price={self.price})")


class OrderBook:
    """
    Stores and matches orders for a single symbol.
    Bids are kept in descending order by price.
    Asks are kept in ascending order by price.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        # Each entry in bids/asks: [price, size, order_id, trader_type]
        self.bids = []
        self.asks = []

    def add_limit_order(self, order):
        if order.side == "buy":
            self.bids.append(
                [order.price, order.size, order.order_id, order.trader_type])
            self.bids.sort(key=lambda x: x[0], reverse=True)
        else:  # sell
            self.asks.append(
                [order.price, order.size, order.order_id, order.trader_type])
            self.asks.sort(key=lambda x: x[0])

    def add_market_order(self, order, data_logger=None):
        if order.side == "buy":
            self.match_incoming_buy(order, data_logger)
        else:
            self.match_incoming_sell(order, data_logger)

    def match_incoming_buy(self, order, data_logger=None):
        """
        Executes trades for a buy market order against the order book's asks.
        """
        while order.size > 0 and self.asks:
            best_ask = self.asks[0]
            ask_price, ask_size, ask_order_id, ask_trader_type = best_ask

            if ask_size <= 0:
                self.asks.pop(0)
                continue

            trade_size = min(order.size, ask_size)
            trade_price = ask_price

            # Log the trade if data_logger is provided
            if data_logger:
                data_logger.log_trade(
                    symbol=self.symbol,
                    trade_type="BUY",
                    trade_size=trade_size,
                    trade_price=trade_price
                )

            # Decrement sizes
            order.size -= trade_size
            best_ask[1] -= trade_size

            # Remove the ask if fully filled
            if best_ask[1] <= 0:
                self.asks.pop(0)

    def match_incoming_sell(self, order, data_logger=None):
        """
        Executes trades for a sell market order against the order book's bids.
        """
        while order.size > 0 and self.bids:
            best_bid = self.bids[0]
            bid_price, bid_size, bid_order_id, bid_trader_type = best_bid

            if bid_size <= 0:
                self.bids.pop(0)
                continue

            trade_size = min(order.size, bid_size)
            trade_price = bid_price

            # Log the trade if data_logger is provided
            if data_logger:
                data_logger.log_trade(
                    symbol=self.symbol,
                    trade_type="SELL",
                    trade_size=trade_size,
                    trade_price=trade_price
                )

            order.size -= trade_size
            best_bid[1] -= trade_size

            if best_bid[1] <= 0:
                self.bids.pop(0)

    def match_limit_orders(self, data_logger=None):
        """
        Match top of the book if there's a cross: best_bid.price >= best_ask.price
        """
        while self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            bid_price, bid_size, _, _ = best_bid
            ask_price, ask_size, _, _ = best_ask

            trade_price = ask_price
            trade_size = min(bid_size, ask_size)

            # Log the limit cross trade
            if data_logger:
                data_logger.log_trade(
                    symbol=self.symbol,
                    trade_type="BID_CROSS",  # or "LIMIT_CROSS"
                    trade_size=trade_size,
                    trade_price=trade_price
                )

            best_bid[1] -= trade_size
            best_ask[1] -= trade_size

            if best_bid[1] <= 0:
                self.bids.pop(0)
            if best_ask[1] <= 0:
                self.asks.pop(0)

    def get_best_bid(self):
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0][0] if self.asks else None

    def get_mid_price(self):
        """
        Returns the midpoint of the best bid and best ask if both exist,
        otherwise returns None.
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None


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


class HeatManager:
    """
    Manages heat durations and notifies the DataLogger when a new heat starts
    and when one ends.
    """

    def __init__(self, heat_duration_minutes=1, data_logger=None):
        self.heat_duration_seconds = heat_duration_minutes * 60
        self.heat_start_time = None
        self.heat_count = 0
        self.data_logger = data_logger

    def start_heat(self):
        """
        Called once at the beginning or whenever we rotate into a new heat.
        """
        self.heat_count += 1
        self.heat_start_time = time.time()
        if self.data_logger:
            self.data_logger.start_new_heat(self.heat_count)
        print(f"--- Starting Heat #{self.heat_count} at {datetime.now()} ---")

    def check_and_rotate_heat(self):
        """
        Checks if current heat exceeded duration; if yes, ends it and starts a new one.
        """
        if self.heat_start_time is None:
            # No heat running yet
            self.start_heat()
            return

        elapsed = time.time() - self.heat_start_time
        if elapsed >= self.heat_duration_seconds:
            # End current heat
            if self.data_logger:
                self.data_logger.end_heat()

            # Start a new heat
            self.start_heat()


class MarketSimulator:
    def __init__(
        self,
        lambda_rate=10,
        initial_liquidity=10,
        symbols=None,
        heat_duration_minutes=1
    ):
        """
        :param lambda_rate: Poisson rate for order arrivals
        :param initial_liquidity: Number of initial limit orders per symbol
        :param symbols: List of symbols to simulate
        :param heat_duration_minutes: Duration of each 'heat' in minutes
        """

        if symbols is None:
            symbols = ["AAPL", "GOOG"]
        self.symbols = symbols

        # Create an order book for each symbol
        self.order_books = {symbol: OrderBook(symbol) for symbol in symbols}

        self.lambda_rate = lambda_rate
        self.order_id_counter = 0
        self.initial_liquidity = initial_liquidity

        # Create a DataLogger and HeatManager
        self.data_logger = DataLogger(base_log_dir="simulation_logs")
        self.heat_manager = HeatManager(
            heat_duration_minutes=heat_duration_minutes,
            data_logger=self.data_logger
        )

    def initialize_order_books(self):
        """
        Create some random initial liquidity for each symbol.
        """
        for symbol in self.symbols:
            for _ in range(self.initial_liquidity):
                # Random bids
                self.order_id_counter += 1
                bid_price = random.uniform(95, 99)
                bid_size = random.randint(1, 10)
                bid_order = Order(
                    self.order_id_counter, symbol, "Market Maker", "limit", "buy", bid_size, bid_price
                )
                self.order_books[symbol].add_limit_order(bid_order)

                # Random asks
                self.order_id_counter += 1
                ask_price = random.uniform(101, 105)
                ask_size = random.randint(1, 10)
                ask_order = Order(
                    self.order_id_counter, symbol, "Market Maker", "limit", "sell", ask_size, ask_price
                )
                self.order_books[symbol].add_limit_order(ask_order)

    def generate_order(self):
        """
        Generate a random order (market/limit, buy/sell) for a random symbol.
        """
        symbol = random.choice(self.symbols)
        order_type = "market" if random.random() < 0.5 else "limit"
        trader_type = "Trader" if random.random() < 0.8 else "Market Maker"
        side = "buy" if random.random() < 0.5 else "sell"
        size = random.randint(1, 10)

        price = None
        if order_type == "limit":
            mid_price = self.order_books[symbol].get_mid_price() or 100
            delta = random.uniform(-5, 5)
            price = mid_price + delta
            price = max(1, price)  # ensure price > 0

        self.order_id_counter += 1
        order = Order(
            self.order_id_counter,
            symbol,
            trader_type,
            order_type,
            side,
            size,
            price
        )
        return order

    def simulate_price_movement(self, symbol):
        """
        Dummy price movement simulation around the mid price.
        """
        mid_price = self.order_books[symbol].get_mid_price()
        if mid_price:
            noise = np.random.normal(0, 0.5)
            return mid_price + noise
        return None

    def run(self, steps=50):
        """
        Run the simulation for a given number of steps.
        """
        print("Initializing order books with liquidity...")
        self.initialize_order_books()

        # Start the first heat explicitly
        self.heat_manager.start_heat()

        for step in range(steps):
            # Random delay ~ Exp(1/lambda_rate)
            delay = np.random.exponential(1 / self.lambda_rate)
            time.sleep(delay)

            # Check if we need to rotate into a new heat
            self.heat_manager.check_and_rotate_heat()

            # Generate a new order
            new_order = self.generate_order()
            print(f"Step={step+1}, New order: {new_order}")

            # Process the new order
            order_book = self.order_books[new_order.symbol]
            if new_order.order_type == "market":
                order_book.add_market_order(
                    new_order, data_logger=self.data_logger)
            else:
                order_book.add_limit_order(new_order)
                order_book.match_limit_orders(data_logger=self.data_logger)

            # Simulate (dummy) price movement for this symbol
            new_price = self.simulate_price_movement(new_order.symbol)
            if new_price:
                print(f"{new_order.symbol} price ~ {new_price:.2f}")

            # Log a snapshot of the market state
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            mid_price = order_book.get_mid_price()
            self.data_logger.log_snapshot(
                step=step+1,
                symbol=new_order.symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price
            )

        # End the last heat after finishing steps
        self.data_logger.end_heat()
        print("Simulation complete.")


if __name__ == "__main__":
    simulator = MarketSimulator(
        lambda_rate=5,            # Higher means more frequent orders
        initial_liquidity=5,      # Some initial orders
        symbols=["AAPL"],
        # symbols=["AAPL", "GOOG", "AMZN"],
        heat_duration_minutes=0.5  # e.g., 30 seconds for quick demo
    )
    simulator.run(steps=1000)

    # Check your "simulation_logs" directory for per-heat folders with CSV data.
