# WebSocketManager.py

import asyncio
import json
import websockets

connected_clients = set()

# Global reference to the WebSocket server's event loop
global_event_loop = None


async def handle_client(websocket, path=None):
    # print(f"[WebSocketManager] Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(
            f"[WebSocketManager] Client disconnected: {websocket.remote_address}")


async def broadcast_event(event_name, data):
    if not connected_clients:
        print(
            f"[WebSocketManager] No clients connected. Skipping broadcast for event: {event_name}")
        return
    message_dict = {
        "event_name": event_name,
        "data": data
    }
    message = json.dumps(message_dict)
    # print(
    # f"[WebSocketManager] Broadcasting event '{event_name}' to {len(connected_clients)} clients.")
    await asyncio.gather(
        *[client.send(message) for client in connected_clients],
        return_exceptions=True
    )
