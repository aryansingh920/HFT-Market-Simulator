import random
import time

import numpy as np

from server.utils.Market import DataLogger, HeatManager, Order, OrderBook


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
