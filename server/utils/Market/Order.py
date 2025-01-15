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
