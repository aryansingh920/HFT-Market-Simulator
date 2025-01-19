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

