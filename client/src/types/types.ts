// Snapshot Event Type
interface SnapshotEvent {
  event_name: "snapshot";
  data: {
    timestamp: number;
    event_type: "SNAPSHOT";
    step: number;
    symbol: string;
    best_bid: number;
    best_ask: number;
    mid_price: number;
  };
}

// Order Book Event Type
interface OrderBookEvent {
  event_name: "order_book";
  data: {
    timestamp: number;
    event_type: "ORDER_BOOK";
    symbol: string;
    order_id: number;
    bids: string; // JSON string representation of bids
    asks: string; // JSON string representation of asks
  };
}

// Trade Event Type
interface TradeEvent {
  event_name: "trade";
  data: {
    timestamp: number;
    event_type: "TRADE";
    symbol: string;
    trade_type: "BUY" | "SELL";
    trade_size: number;
    trade_price: number;
  };
}

interface NewOrderEvent {
  event_name: "new_order";
  data: {
    timestamp: number;
    event_type: "ORDER";
    order_id: number;
    symbol: string;
    trader_type: string; // e.g., "Trader"
    order_type: string; // e.g., "market"
    side: "buy" | "sell";
    size: number;
    price: number | null; // Could be null for market orders
  };
}

// Combined Event Type
export type SocketEvent =
  | SnapshotEvent
  | OrderBookEvent
  | TradeEvent
  | NewOrderEvent;
