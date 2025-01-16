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
    def __init__(
        self,
        lambda_rate=10,
        initial_liquidity=10,
        symbols=None,
        heat_duration_minutes=1,
        mu=0.0,     # GBM drift
        sigma=0.02  # GBM volatility
    ):
        """
        :param lambda_rate: Poisson rate for order arrivals
        :param initial_liquidity: Number of initial limit orders per symbol
        :param symbols: List of symbols to simulate
        :param heat_duration_minutes: Duration of each 'heat' in minutes
        :param mu: Drift (GBM)
        :param sigma: Volatility (GBM)
        """
        if symbols is None:
            symbols = ["AAPL", "GOOG"]
        self.symbols = symbols

        # Create an order book for each symbol
        self.order_books = {symbol: OrderBook(symbol) for symbol in symbols}

        self.lambda_rate = lambda_rate
        self.order_id_counter = 0
        self.initial_liquidity = initial_liquidity

        # GBM parameters
        self.mu = mu
        self.sigma = sigma
        # We will track a "current_price" for each symbol using GBM
        self.current_price = {symbol: 100.0 for symbol in symbols}  # default

        # Create a DataLogger and HeatManager
        self.data_logger = DataLogger(
            base_log_dir="simulation_logs",
            symbols=self.symbols
        )
        self.heat_manager = HeatManager(
            heat_duration_minutes=heat_duration_minutes,
            data_logger=self.data_logger
        )

    def initialize_order_books(self):
        """
        Create some random initial liquidity for each symbol.
        Also, initialize current_price[symbol] around 100.
        """
        for symbol in self.symbols:
            # Slight random offset for the "current" price
            self.current_price[symbol] = 100.0 + random.uniform(-5, 5)

            for _ in range(self.initial_liquidity):
                # Random bids
                self.order_id_counter += 1
                bid_price = self.current_price[symbol] - random.uniform(1, 5)
                bid_price = max(1, bid_price)
                bid_size = random.randint(1, 10)
                bid_order = Order(
                    self.order_id_counter, symbol, "Market Maker",
                    "limit", "buy", bid_size, bid_price
                )
                self.order_books[symbol].add_limit_order(bid_order)

                # Random asks
                self.order_id_counter += 1
                ask_price = self.current_price[symbol] + random.uniform(1, 5)
                ask_size = random.randint(1, 10)
                ask_order = Order(
                    self.order_id_counter, symbol, "Market Maker",
                    "limit", "sell", ask_size, ask_price
                )
                self.order_books[symbol].add_limit_order(ask_order)

    def generate_order(self):
        """
        Generate a random order (market/limit, buy/sell) for a random symbol.
        The limit price is based on a reference mid-price = self.current_price[symbol].
        """
        symbol = random.choice(self.symbols)
        order_type = "market" if random.random() < 0.5 else "limit"
        trader_type = "Trader" if random.random() < 0.8 else "Market Maker"
        side = "buy" if random.random() < 0.5 else "sell"
        size = random.randint(1, 10)

        price = None
        if order_type == "limit":
            # Use current_price[symbol] as reference mid
            mid_price = self.current_price[symbol]
            # Price range within ±5% of mid
            delta = mid_price * 0.05
            price = mid_price + random.uniform(-delta, delta)
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

    def simulate_price_movement(self, symbol, dt=1.0):
        """
        Uses Geometric Brownian Motion (GBM) to update self.current_price[symbol].
        S(t+dt) = S(t)*exp((mu-0.5*sigma^2)*dt + sigma*sqrt(dt)*Z).
        """
        S_t = self.current_price[symbol]
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * np.random.normal()
        S_tplus = S_t * math.exp(drift + diffusion)
        self.current_price[symbol] = max(0.01, S_tplus)  # avoid zero price

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
            # Log the order
            self.data_logger.log_order(new_order)

            print(f"Step={step+1}, New order: {new_order}")

            # Process the new order
            order_book = self.order_books[new_order.symbol]
            if new_order.order_type == "market":
                order_book.add_market_order(
                    new_order, data_logger=self.data_logger
                )
            else:
                order_book.add_limit_order(new_order)
                order_book.match_limit_orders(data_logger=self.data_logger)

            # Simulate GBM price movement for **all** symbols
            for sym in self.symbols:
                self.simulate_price_movement(sym)

            # Log snapshots for **all** symbols to ensure continuous time series
            for sym in self.symbols:
                ob = self.order_books[sym]
                best_bid = ob.get_best_bid()
                best_ask = ob.get_best_ask()
                mid_price = ob.get_mid_price()
                # Or use the updated GBM price as "mid_price" if desired.
                # We'll keep the order book mid as is for consistent comparison.
                self.data_logger.log_snapshot(
                    step=step + 1,
                    symbol=sym,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price
                )

        # End the last heat after finishing steps
        self.data_logger.end_heat()
        print("Simulation complete.")
