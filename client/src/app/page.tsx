// src/app/page.tsx

"use client";

import Canvas from "@/components/Canvas";
import { SocketEvent } from "@/types/types";
import { connectWebSocket } from "@/utils/connectWebSocket";
import { TradeProvider, useTradeContext } from "@/context/TradeContext";
import { useEffect } from "react";

export default function Home() {
  return (
    <TradeProvider>
      <WebSocketHandler />
      <Canvas />
    </TradeProvider>
  );
}

// Separate component to handle WebSocket connection
const WebSocketHandler: React.FC = () => {
  const { addTrade } = useTradeContext();

  useEffect(() => {
    const handleEvent = (event: SocketEvent) => {
      switch (event.event_name) {
        case "snapshot":
          console.log("Snapshot event received:", event.data);
          break;
        case "order_book":
          console.log("Order Book event received:", event.data);
          break;
        case "trade":
          console.log("Trade event received:", event.data);
          addTrade(event);
          break;
        case "new_order":
          console.log("New Order event received:", event.data);
          break;
        default:
          console.warn("Unknown event type:", event);
      }
    };

    const socketUrl = "ws://localhost:8888";
    const socket = connectWebSocket(socketUrl, handleEvent);

    return () => {
      socket.close();
    };
  }, [addTrade]);

  return null; // This component doesn't render anything
};
