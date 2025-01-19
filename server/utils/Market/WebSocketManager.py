# WebSocketManager.py

import asyncio
import json
import websockets

connected_clients = set()
global_event_loop = None
simulation_callback = None


def set_simulation_callback(callback):
    global simulation_callback
    simulation_callback = callback


async def handle_client(websocket, path=None):
    connected_clients.add(websocket)
    print(f"[WebSocketManager] Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                print("Data received from client: ", data)

                if data.get("command") == "start_simulation":
                    print("[WebSocketManager] Starting simulation...")
                    if simulation_callback:
                        # Run the simulation without awaiting it to prevent blocking
                        asyncio.create_task(simulation_callback())
            except json.JSONDecodeError:
                print(
                    f"[WebSocketManager] Received invalid JSON from client: {websocket.remote_address}")
                continue
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
    await asyncio.gather(
        *[client.send(message) for client in connected_clients],
        return_exceptions=True
    )
