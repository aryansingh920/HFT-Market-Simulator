"use client";
import Canvas from "@/components/Canvas";
import { SocketEvent } from "@/types/types";
import { connectWebSocket } from "@/utils/connectWebSocket";

export default function Home() {
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
        break;
      case "new_order":
        console.log("New Order event received:", event.data);
        break;
      default:
        console.warn("Unknown event type:", event);
    }
  };

  const socketUrl = "ws://localhost:8888";
  connectWebSocket(socketUrl, handleEvent);

  return <Canvas />;
}

// "use client";
// import React, { useState } from "react";
// import TradeGraph from "@/components/TradeGraph";
// import { connectWebSocket } from "@/utils/connectWebSocket";
// import { SocketEvent } from "@/types/types";

// export default function Home() {
//   const [trades, setTrades] = useState<
//     { timestamp: number; trade_price: number }[]
//   >([]);

//   const handleEvent = (event: SocketEvent) => {
//     if (event.event_name === "trade") {
//       setTrades((prev) => [
//         ...prev,
//         {
//           timestamp: event.data.timestamp,
//           trade_price: event.data.trade_price,
//         },
//       ]);
//     }
//   };

//   const socketUrl = "ws://localhost:8888";
//   connectWebSocket(socketUrl, handleEvent);

//   return (
//     <div className="p-4">
//       <h1 className="text-lg font-bold mb-4">Real-Time Trade Graph</h1>
//       <TradeGraph trades={trades} />
//     </div>
//   );
// }
