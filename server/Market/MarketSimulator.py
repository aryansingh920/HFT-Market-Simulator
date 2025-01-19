"""
Created on 19/01/2025

@author: Aryan

Filename: MarketSimulator.py

Relative Path: server/Market/MarketSimulator.py
"""


import random
import time
import math
import numpy as np
from tqdm import trange
from Market.HeatManager import HeatManager
from Market.Order import Order
from Market.OrderBook import OrderBook
from Market.DataLogger import DataLogger


class MarketSimulator:
    """
    A market simulation environment that can handle symbol-specific parameters
    and retains complete logging of orders, trades, and snapshots.
    """

    def __init__(self, config, loop=None):
        self.name = config.get("name", "DefaultSimulation")
        self.symbols_config = config.get("symbols_config", {})
        self.heat_duration = config.get("global_heat_duration", 1.0)

        self.symbols = list(self.symbols_config.keys())

        # Data logger for logging + real-time broadcasting
        self.data_logger = DataLogger(
            base_log_dir="simulation_logs",
            symbols=self.symbols,
            loop=loop

        )

        # HeatManager to handle rotating logs each "heat"
        self.heat_manager = HeatManager(
            heat_duration_minutes=self.heat_duration,
            data_logger=self.data_logger
        )

        # Create an OrderBook and track current_price for each symbol
        self.order_books = {}
        self.current_price = {}
        self.order_id_counter = 0

        for sym, sym_conf in self.symbols_config.items():
            self.order_books[sym] = OrderBook(sym)
            self.current_price[sym] = sym_conf.get("initial_price", 100.0)

        # Pre-seed each order book with initial liquidity
        self.initialize_order_books()

    def initialize_order_books(self):
        """
        Seed initial limit orders in each symbol's order book.
        """
        for sym, sym_conf in self.symbols_config.items():
            liquidity_count = sym_conf.get("initial_liquidity", 5)
            price = self.current_price[sym]

            # Optionally nudge the start price a bit
            self.current_price[sym] = price + random.uniform(-5, 5)

            for _ in range(liquidity_count):
                # BUY side
                self.order_id_counter += 1
                bid_price = max(
                    1.0,
                    self.current_price[sym] - random.uniform(1, 5)
                )
                bid_size = random.randint(1, 10)
                bid_order = Order(
                    order_id=self.order_id_counter,
                    symbol=sym,
                    trader_type="Market Maker",
                    order_type="limit",
                    side="buy",
                    size=bid_size,
                    price=bid_price
                )
                self.order_books[sym].add_limit_order(bid_order)

                # SELL side
                self.order_id_counter += 1
                ask_price = self.current_price[sym] + random.uniform(1, 5)
                ask_size = random.randint(1, 10)
                ask_order = Order(
                    order_id=self.order_id_counter,
                    symbol=sym,
                    trader_type="Market Maker",
                    order_type="limit",
                    side="sell",
                    size=ask_size,
                    price=ask_price
                )
                self.order_books[sym].add_limit_order(ask_order)

    def generate_random_order(self, symbol):
        """
        Generate a random order (market or limit) for the given symbol.
        """
        # 50% chance market, 50% chance limit
        order_type = "market" if random.random() < 0.5 else "limit"
        # 80% chance "Trader", 20% chance "Market Maker"
        trader_type = "Trader" if random.random() < 0.8 else "Market Maker"
        side = "buy" if random.random() < 0.5 else "sell"
        size = random.randint(1, 10)

        price = None
        if order_type == "limit":
            mid_price = self.current_price[symbol]
            delta = mid_price * 0.05
            price = mid_price + random.uniform(-delta, delta)
            price = max(1.0, price)

        self.order_id_counter += 1
        order = Order(
            order_id=self.order_id_counter,
            symbol=symbol,  # <-- use 'symbol' here
            trader_type=trader_type,
            order_type=order_type,
            side=side,
            size=size,
            price=price
        )
        return order

    def simulate_price_movement(self, symbol, dt=1.0):
        """
        Simulate price movement using an Ornstein–Uhlenbeck process.
        """
        sym_conf = self.symbols_config[symbol]

        # Defaulting the long-run mean to initial price if not specified
        theta = sym_conf.get("long_run_mean", sym_conf["initial_price"])
        kappa = sym_conf.get("kappa", 0.2)
        sigma = sym_conf.get("sigma", 0.05)

        S_t = self.current_price[symbol]
        dS = kappa * (theta - S_t) * dt
        dW = sigma * math.sqrt(dt) * np.random.normal()
        S_next = S_t + dS + dW

        S_next = max(1.0, S_next)
        self.current_price[symbol] = S_next

    def run(self, steps=50):
        """
        Run the simulation for a certain number of steps in a single "heat".
        """
        print(f"Starting simulation '{self.name}'...")
        self.heat_manager.start_heat()

        total_sim_time = self.heat_manager.heat_duration_seconds
        time_per_step = total_sim_time / steps if steps else 0.0

        # Use trange for a progress bar in the console
        for step in trange(steps, desc="Simulation Progress"):
            step_start_time = time.time()

            # For each symbol, generate and process random orders
            for sym, sym_conf in self.symbols_config.items():
                lam_rate = sym_conf.get("lambda_rate", 10.0)
                num_orders = np.random.poisson(lam_rate)

                for _ in range(num_orders):
                    new_order = self.generate_random_order(sym)

                    # Log the order
                    self.data_logger.log_order(new_order)

                    # Process
                    ob = self.order_books[sym]
                    if new_order.order_type == "market":
                        ob.add_market_order(
                            new_order, data_logger=self.data_logger)
                    else:
                        ob.add_limit_order(new_order)
                        ob.match_limit_orders(data_logger=self.data_logger)

                    # Log the updated order book state
                    self.data_logger.log_order_book(
                        symbol=sym,
                        bids=ob.bids,
                        asks=ob.asks,
                        order_id=new_order.order_id
                    )

            # After orders are processed, update prices
            for sym in self.symbols:
                self.simulate_price_movement(sym, dt=1.0)

            # Log snapshots
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

            elapsed = time.time() - step_start_time
            remainder = time_per_step - elapsed
            if remainder > 0:
                time.sleep(remainder)

        # End the heat once steps are finished
        self.data_logger.end_heat()
        print(f"Simulation '{self.name}' complete.")
