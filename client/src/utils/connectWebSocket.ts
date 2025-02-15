// src/utils/connectWebSocket.ts

/* eslint-disable @typescript-eslint/no-explicit-any */
import { SocketEvent } from "@/types/types";

// Type Guard to validate the structure of incoming messages
function isSocketEvent(event: any): event is SocketEvent {
  if (!event || typeof event !== "object") return false;

  const { event_name, data } = event;
  if (typeof event_name !== "string" || typeof data !== "object") return false;

  switch (event_name) {
    case "snapshot":
      return (
        data.event_type === "SNAPSHOT" &&
        typeof data.step === "number" &&
        typeof data.symbol === "string" &&
        typeof data.best_bid === "number" &&
        typeof data.best_ask === "number" &&
        typeof data.mid_price === "number"
      );
    case "order_book":
      return (
        data.event_type === "ORDER_BOOK" &&
        typeof data.symbol === "string" &&
        typeof data.order_id === "number" &&
        typeof data.bids === "string" &&
        typeof data.asks === "string"
      );
    case "trade":
      return (
        data.event_type === "TRADE" &&
        typeof data.symbol === "string" &&
        (data.trade_type === "BUY" || data.trade_type === "SELL") &&
        typeof data.trade_size === "number" &&
        typeof data.trade_price === "number"
      );
    case "new_order":
      return (
        data.event_type === "ORDER" &&
        typeof data.order_id === "number" &&
        typeof data.symbol === "string" &&
        typeof data.trader_type === "string" &&
        typeof data.order_type === "string" &&
        (data.side === "buy" || data.side === "sell") &&
        typeof data.size === "number" &&
        (typeof data.price === "number" || data.price === null)
      );
    default:
      return false;
  }
}

export const connectWebSocket = (
  url: string,
  onEvent: (event: SocketEvent) => void
) => {
  const socket = new WebSocket(url);

  socket.onopen = () => {
    console.log("WebSocket connection established.");
    // Send the start simulation command
    socket.send(JSON.stringify({ command: "start_simulation" }));
  };

  socket.onmessage = (message) => {
    try {
      const data = JSON.parse(message.data);

      // Validate and cast the data to a known event type
      if (isSocketEvent(data)) {
        onEvent(data);
      } else {
        console.warn("Unknown event type received:", data);
      }
    } catch (error) {
      console.error("Error parsing WebSocket message:", error);
    }
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onclose = () => {
    console.log("WebSocket connection closed.");
  };

  return socket;
};
