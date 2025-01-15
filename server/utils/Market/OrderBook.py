"""
Created on 15/01/2025

@author: Aryan

Filename: OrderBook.py

Relative Path: server/utils/Market/OrderBook.py
"""

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
