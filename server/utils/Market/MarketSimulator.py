"""
Created on 15/01/2025

@author: Aryan

Filename: MarketSimulator.py

Relative Path: server/utils/Market/MarketSimulator.py
"""

import random
import time
import math
import numpy as np

from Market.HeatManager import HeatManager
from Market.Order import Order
from Market.OrderBook import OrderBook
from Market.DataLogger import DataLogger


class MarketSimulator:
    """
    A market simulation environment that generates and processes orders,
    tracks price movements via GBM, and logs data via DataLogger.
    """

    def __init__(self, config):
        """
        Initialize the simulator with a configuration dictionary.

        The `config` dictionary can include the following keys:
            - name (str): A label for the simulation, e.g., "MyHFTSimulation".
            - lambda_rate (float): Poisson rate for order arrivals.
            - initial_liquidity (int): Number of initial limit orders per symbol.
            - symbols (list of str): List of stock symbols to simulate.
            - heat_duration_minutes (float): Duration (in minutes) for each heat.
            - mu (float): Drift for the Geometric Brownian Motion (GBM).
            - sigma (float): Volatility for the GBM.
            - initial_stock_prices (dict): Initial prices per symbol, e.g.,
                  { "AAPL": 120.0, "GOOG": 1500.0 }
                If not provided for a symbol, defaults to 100.0.
        """

        # Pull parameters from config with sensible defaults
        self.name = config.get("name", "DefaultSimulation")
        self.lambda_rate = config.get("lambda_rate", 10)
        self.initial_liquidity = config.get("initial_liquidity", 10)
        self.symbols = config.get("symbols", ["AAPL", "GOOG"])
        self.heat_duration_minutes = config.get("heat_duration_minutes", 1.0)
        self.mu = config.get("mu", 0.0)       # GBM drift
        self.sigma = config.get("sigma", 0.02)  # GBM volatility

        # Create an OrderBook for each symbol
        self.order_books = {symbol: OrderBook(
            symbol) for symbol in self.symbols}

        # Initialize current prices from config or default to 100.0
        initial_prices = config.get("initial_stock_prices", {})
        self.current_price = {
            sym: initial_prices.get(sym, 100.0) for sym in self.symbols
        }

        # Order ID counter
        self.order_id_counter = 0

        # Create a DataLogger
        self.data_logger = DataLogger(
            base_log_dir="simulation_logs",
            symbols=self.symbols
        )

        # Create a HeatManager
        self.heat_manager = HeatManager(
            heat_duration_minutes=self.heat_duration_minutes,
            data_logger=self.data_logger
        )

    def initialize_order_books(self):
        """
        Create some random initial liquidity for each symbol.
        The reference price for each symbol is self.current_price[symbol].
        """
        for symbol in self.symbols:
            # For variety, allow a small random offset from the initial price
            self.current_price[symbol] = self.current_price[symbol] + \
                random.uniform(-5, 5)

            for _ in range(self.initial_liquidity):
                # Random BUY limit orders
                self.order_id_counter += 1
                bid_price = max(
                    1.0, self.current_price[symbol] - random.uniform(1, 5))
                bid_size = random.randint(1, 10)
                bid_order = Order(
                    order_id=self.order_id_counter,
                    symbol=symbol,
                    trader_type="Market Maker",
                    order_type="limit",
                    side="buy",
                    size=bid_size,
                    price=bid_price
                )
                self.order_books[symbol].add_limit_order(bid_order)

                # Random SELL limit orders
                self.order_id_counter += 1
                ask_price = self.current_price[symbol] + random.uniform(1, 5)
                ask_size = random.randint(1, 10)
                ask_order = Order(
                    order_id=self.order_id_counter,
                    symbol=symbol,
                    trader_type="Market Maker",
                    order_type="limit",
                    side="sell",
                    size=ask_size,
                    price=ask_price
                )
                self.order_books[symbol].add_limit_order(ask_order)

    def generate_random_order(self):
        """
        Generate a random order (market or limit) for a random symbol.
        Price (for limit) is within ±5% of the current price.
        """
        symbol = random.choice(self.symbols)
        order_type = "market" if random.random() < 0.5 else "limit"
        trader_type = "Trader" if random.random() < 0.8 else "Market Maker"
        side = "buy" if random.random() < 0.5 else "sell"
        size = random.randint(1, 10)

        price = None
        if order_type == "limit":
            mid_price = self.current_price[symbol]
            delta = mid_price * 0.05  # ±5%
            price = mid_price + random.uniform(-delta, delta)
            price = max(1.0, price)

        self.order_id_counter += 1
        order = Order(
            order_id=self.order_id_counter,
            symbol=symbol,
            trader_type=trader_type,
            order_type=order_type,
            side=side,
            size=size,
            price=price
        )
        return order

    def simulate_price_movement(self, symbol, dt=1.0):
        """
        Update self.current_price[symbol] using Geometric Brownian Motion (GBM).
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z).
        """
        S_t = self.current_price[symbol]
        drift = (self.mu - 0.5 * (self.sigma ** 2)) * dt
        diffusion = self.sigma * math.sqrt(dt) * np.random.normal()
        S_tplus = S_t * math.exp(drift + diffusion)
        # Ensure price doesn't go below 0.01
        self.current_price[symbol] = max(0.01, S_tplus)

    def run(self, steps=50):
        """
        Run the simulation for a given number of steps.
        By default, we tie it to exactly one "heat" duration.

        If you want the simulation to be purely step-based (fast as possible),
        remove or adjust the time.sleep logic below.
        """

        print(f"Starting simulation '{self.name}'...")
        print("Initializing order books with liquidity...")
        self.initialize_order_books()

        # Start the first (and only) heat
        self.heat_manager.start_heat()

        # Total real time for one heat (in seconds)
        total_sim_time = self.heat_manager.heat_duration_seconds
        # Time per step (seconds), so steps fill the heat duration
        time_per_step = total_sim_time / steps

        for step in range(steps):
            step_start_time = time.time()

            # 1) Generate a random order
            new_order = self.generate_random_order()
            self.data_logger.log_order(new_order)
            print(f"Step {step + 1}: New order: {new_order}")

            # 2) Process the new order
            order_book = self.order_books[new_order.symbol]
            if new_order.order_type == "market":
                order_book.add_market_order(
                    new_order, data_logger=self.data_logger)
            else:
                order_book.add_limit_order(new_order)
                order_book.match_limit_orders(data_logger=self.data_logger)

            # 3) Simulate price movement for each symbol
            for sym in self.symbols:
                self.simulate_price_movement(sym, dt=1.0)

            # 4) Log snapshots for each symbol
            for sym in self.symbols:
                ob = self.order_books[sym]
                best_bid = ob.get_best_bid()
                best_ask = ob.get_best_ask()
                mid_price = ob.get_mid_price()
                self.data_logger.log_snapshot(
                    step=step + 1,
                    symbol=sym,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price
                )

            # 5) (Optional) Sleep so that total steps fill the heat duration in real time
            elapsed = time.time() - step_start_time
            remainder = time_per_step - elapsed
            if remainder > 0:
                time.sleep(remainder)

        # End the heat after finishing all steps
        self.data_logger.end_heat()
        print(f"Simulation '{self.name}' complete.")
