import numpy as np
import random
import time
from collections import defaultdict


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
        # Each entry: (price, size, order_id, trader_type, etc.)
        self.bids = []
        self.asks = []

    def add_limit_order(self, order):
        """
        Insert the limit order in the appropriate list while maintaining
        sorted order:
          - Bids sorted descending by price
          - Asks sorted ascending by price
        """
        if order.side == "buy":
            self.bids.append(
                [order.price, order.size, order.order_id, order.trader_type])
            # Sort descending by price
            self.bids.sort(key=lambda x: x[0], reverse=True)
        else:  # order.side == "sell"
            self.asks.append(
                [order.price, order.size, order.order_id, order.trader_type])
            # Sort ascending by price
            self.asks.sort(key=lambda x: x[0])

    def add_market_order(self, order):
        """
        For a market order, immediately try to match it against
        the opposite side of the book until the order is fully filled
        or there is no more depth.
        """
        if order.side == "buy":
            # match against the best ask
            self.match_incoming_buy(order)
        else:
            # match against the best bid
            self.match_incoming_sell(order)

    def match_incoming_buy(self, order):
        """
        Executes trades for a buy market order against the order book's asks.
        """
        while order.size > 0 and self.asks:
            best_ask = self.asks[0]
            ask_price, ask_size, ask_order_id, ask_trader_type = best_ask

            # If we have no liquidity or the best ask is invalid, break
            if ask_size <= 0:
                self.asks.pop(0)
                continue

            # Trade executes at the ask price
            trade_size = min(order.size, ask_size)
            trade_price = ask_price

            print(
                f"[{self.symbol}] Trade executed: BUY {trade_size} @ {trade_price:.2f}")
            # Decrement both the market order size and the ask size
            order.size -= trade_size
            best_ask[1] -= trade_size  # reduce ask_size

            # If the ask is fully filled, remove it
            if best_ask[1] <= 0:
                self.asks.pop(0)

    def match_incoming_sell(self, order):
        """
        Executes trades for a sell market order against the order book's bids.
        """
        while order.size > 0 and self.bids:
            best_bid = self.bids[0]
            bid_price, bid_size, bid_order_id, bid_trader_type = best_bid

            if bid_size <= 0:
                self.bids.pop(0)
                continue

            # Trade executes at the bid price
            trade_size = min(order.size, bid_size)
            trade_price = bid_price

            print(
                f"[{self.symbol}] Trade executed: SELL {trade_size} @ {trade_price:.2f}")
            order.size -= trade_size
            best_bid[1] -= trade_size

            if best_bid[1] <= 0:
                self.bids.pop(0)

    def match_limit_orders(self):
        """
        After adding a limit order, we try matching the top of book
        if there's a cross: bid[0].price >= ask[0].price
        """
        while self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            bid_price, bid_size, _, _ = best_bid
            ask_price, ask_size, _, _ = best_ask

            trade_price = ask_price  # or bid_price; typically you'd do something like midpoint
            trade_size = min(bid_size, ask_size)

            print(f"[{self.symbol}] Limit match: {trade_size} @ {trade_price:.2f}")

            # Reduce sizes
            best_bid[1] -= trade_size
            best_ask[1] -= trade_size

            # Remove from the book if fully filled
            if best_bid[1] <= 0:
                self.bids.pop(0)
            if best_ask[1] <= 0:
                self.asks.pop(0)

    def get_best_bid(self):
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0][0] if self.asks else None

    def get_mid_price(self):
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None


class MarketSimulator:
    def __init__(self, lambda_rate=10, initial_liquidity=10, symbols=None):
        """
        :param lambda_rate: Poisson rate for order arrivals
        :param initial_liquidity: Number of initial limit orders per symbol
        :param symbols: List of symbols to simulate
        """
        if symbols is None:
            symbols = ["AAPL", "GOOG"]  # Default symbols
        self.symbols = symbols

        # Create an order book for each symbol
        self.order_books = {symbol: OrderBook(symbol) for symbol in symbols}

        self.lambda_rate = lambda_rate  # Poisson arrival rate
        self.order_id_counter = 0
        self.initial_liquidity = initial_liquidity

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
                bid_order = Order(self.order_id_counter, symbol, "Market Maker", "limit",
                                  "buy", bid_size, bid_price)
                self.order_books[symbol].add_limit_order(bid_order)

                # Random asks
                self.order_id_counter += 1
                ask_price = random.uniform(101, 105)
                ask_size = random.randint(1, 10)
                ask_order = Order(self.order_id_counter, symbol, "Market Maker", "limit",
                                  "sell", ask_size, ask_price)
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
            # Approximate around mid price or around 100 if the mid is None
            mid_price = self.order_books[symbol].get_mid_price() or 100
            delta = random.uniform(-5, 5)
            price = mid_price + delta
            # Ensure price > 0
            price = max(1, price)

        self.order_id_counter += 1
        order = Order(self.order_id_counter, symbol,
                      trader_type, order_type, side, size, price)
        return order

    def simulate_price_movement(self, symbol):
        """
        Just a dummy price movement simulation around the mid price.
        """
        mid_price = self.order_books[symbol].get_mid_price()
        if mid_price:
            noise = np.random.normal(0, 0.5)
            new_price = mid_price + noise
            return new_price
        return None

    def run(self, steps=50):
        """
        Run the simulation for a given number of steps.
        """
        print("Initializing order books with liquidity...")
        self.initialize_order_books()

        for step in range(steps):
            # Random delay ~ Exp(1/lambda_rate)
            delay = np.random.exponential(1 / self.lambda_rate)
            # In a real simulator you might accumulate "logical time" instead
            time.sleep(delay)

            new_order = self.generate_order()
            print(f"Step={step+1}, New order: {new_order}")

            # Process the new order
            order_book = self.order_books[new_order.symbol]
            if new_order.order_type == "market":
                # Match immediately
                order_book.add_market_order(new_order)
            else:
                # Add the limit order, then match if crossing
                order_book.add_limit_order(new_order)
                order_book.match_limit_orders()

            # Simulate price movement for this symbol
            new_price = self.simulate_price_movement(new_order.symbol)
            if new_price:
                print(
                    f"{new_order.symbol} Price moved to approximately {new_price:.2f}")


if __name__ == "__main__":
    simulator = MarketSimulator(
        lambda_rate=5,           # Higher means more frequent orders
        initial_liquidity=5,     # Fewer orders at initialization for brevity
        symbols=["AAPL", "GOOG", "AMZN"]  # Multiple symbols
    )
    simulator.run(steps=20)
