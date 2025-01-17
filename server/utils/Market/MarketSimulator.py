"""
MarketSimulator.py

An updated MarketSimulator that:
  - Accepts a "symbols_config" dictionary with per-symbol parameters
    (lambda_rate, mu, sigma, etc.).
  - Retains the data logging logic for orders, trades, and snapshots.
  - Uses a HeatManager to manage session-based logging.
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
    A market simulation environment that can handle symbol-specific parameters
    and retains complete logging of orders, trades, and snapshots.

    Example of expected config structure:
    {
        "name": "MyHFTSimulation",

        "symbols_config": {
            "AAPL": {
                "initial_price": 120.0,
                "lambda_rate": 25,
                "initial_liquidity": 10,
                "mu": 0.01,
                "sigma": 0.05,
                "heat_duration_minutes": 0.5
            },
            "GOOG": {
                "initial_price": 1500.0,
                "lambda_rate": 50,
                "initial_liquidity": 20,
                "mu": 0.01,
                "sigma": 0.05,
                "heat_duration_minutes": 0.5
            }
            ...
        },

        # Optionally, a global override for heat duration, or fallback:
        "global_heat_duration": 0.5
    }
    """

    def __init__(self, config):
        """
        Initialize the simulator with a configuration dict.
        """
        self.name = config.get("name", "DefaultSimulation")

        # "symbols_config" is a dictionary of per-symbol parameters.
        # e.g.: config["symbols_config"]["AAPL"]["lambda_rate"] = 25
        self.symbols_config = config.get("symbols_config", {})

        # If you want a single global heat duration, we can default to that here:
        self.heat_duration = config.get("global_heat_duration", 1.0)

        # Prepare a list of symbols from the keys of symbols_config
        self.symbols = list(self.symbols_config.keys())

        # Data logger to track all events for these symbols
        self.data_logger = DataLogger(
            base_log_dir="simulation_logs",
            symbols=self.symbols
        )

        # HeatManager to handle rotating logs each "heat"
        self.heat_manager = HeatManager(
            heat_duration_minutes=self.heat_duration,
            data_logger=self.data_logger
        )

        # Create an OrderBook for each symbol
        self.order_books = {}
        self.current_price = {}
        self.order_id_counter = 0

        for sym, sym_conf in self.symbols_config.items():
            self.order_books[sym] = OrderBook(sym)
            self.current_price[sym] = sym_conf.get("initial_price", 100.0)

        # After setting up, pre-seed each order book with some initial liquidity
        self.initialize_order_books()

    def initialize_order_books(self):
        """
        Seed initial limit orders in each symbol's order book.
        The number of such orders = symbol-specific 'initial_liquidity'.
        """
        for sym, sym_conf in self.symbols_config.items():
            liquidity_count = sym_conf.get("initial_liquidity", 5)
            price = self.current_price[sym]

            # Optionally nudge the start price a bit for variety
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
        Generate a random order (market or limit) for the given symbol,
        referencing the current_price for establishing a limit price.
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
            # ±5% around current price
            delta = mid_price * 0.05
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
        Example price movement using a basic GBM with optional random shock.

        We retrieve mu, sigma from the symbol's config:
           mu = symbol_config["mu"]
           sigma = symbol_config["sigma"]
        """
        sym_conf = self.symbols_config[symbol]
        mu_gbm = sym_conf.get("mu", 0.0)
        sigma_gbm = sym_conf.get("sigma", 0.02)

        S_t = self.current_price[symbol]

        # Geometric Brownian Motion drift/diffusion
        drift_gbm = (mu_gbm - 0.5 * sigma_gbm**2) * dt
        diffusion_gbm = sigma_gbm * math.sqrt(dt) * np.random.normal()
        S_tplus = S_t * math.exp(drift_gbm + diffusion_gbm)

        # Optional shock event (1% chance)
        if random.random() < 0.01:
            shock = np.random.choice([-1, 1]) * random.uniform(0.05, 0.1)
            S_tplus *= (1 + shock)
            print(f"[{symbol}] News shock applied: {shock:.2%}")

        # Ensure price doesn't drop below a minimum
        self.current_price[symbol] = max(1.0, S_tplus)

        # Optional debug:
        # print(f"[DEBUG] {symbol}: old={S_t:.2f}, new={self.current_price[symbol]:.2f}")

    def run(self, steps=50):
        """
        Run the simulation for the specified number of steps,
        filling one "heat" managed by HeatManager.

        Each step:
          - For each symbol, generate a Poisson(lambda_rate) # of orders
          - Process those orders (market or limit)
          - Update the price
          - Log snapshots
        """
        print(f"Starting simulation '{self.name}'...")
        self.heat_manager.start_heat()

        # total real-time duration for the heat
        total_sim_time = self.heat_manager.heat_duration_seconds
        # time allocated per step in real seconds
        time_per_step = total_sim_time / steps if steps else 0.0

        for step in range(steps):
            step_start_time = time.time()

            # For each symbol, generate and process orders
            for sym, sym_conf in self.symbols_config.items():
                lam_rate = sym_conf.get("lambda_rate", 10.0)
                # number of new orders ~ Poisson(lam_rate)
                num_orders = np.random.poisson(lam_rate)

                for _ in range(num_orders):
                    new_order = self.generate_random_order(sym)

                    # 1) Log the order
                    self.data_logger.log_order(new_order)

                    # 2) Process the order in the OrderBook
                    ob = self.order_books[sym]
                    if new_order.order_type == "market":
                        ob.add_market_order(
                            new_order, data_logger=self.data_logger)
                    else:
                        ob.add_limit_order(new_order)
                        ob.match_limit_orders(data_logger=self.data_logger)

            # After processing orders, we update the price for each symbol
            for sym in self.symbols:
                self.simulate_price_movement(sym, dt=1.0)

            # Now log a snapshot for each symbol
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

            # (Optional) Sleep so the total run time matches the heat duration
            elapsed = time.time() - step_start_time
            remainder = time_per_step - elapsed
            if remainder > 0:
                time.sleep(remainder)

        # End the heat after finishing all steps
        self.data_logger.end_heat()
        print(f"Simulation '{self.name}' complete.")
