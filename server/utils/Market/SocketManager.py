# """
# Created on 18/01/2025

# @author: Aryan

# Filename: SocketManager.py

# Relative Path: server/utils/Market/SocketManager.py
# """


# """
# SocketManager.py

# Creates a Socket.IO server to broadcast events in real-time.
# """


# # Create a Socket.IO server
# import eventlet
# import socketio
# sio = socketio.Server(cors_allowed_origins="*")
# # Wrap with a WSGI app
# app = socketio.WSGIApp(sio)


# @sio.event
# def connect(sid, environ):
#     print(f"[SocketManager] Client connected: {sid}")


# @sio.event
# def disconnect(sid):
#     print(f"[SocketManager] Client disconnected: {sid}")


# def broadcast_event(event_name, data):
#     """
#     Utility function to broadcast a given event (event_name)
#     with data (a dict) to all connected clients.
#     """
#     # print(f"[SocketManager] Emitting event: {event_name}, Data: {data}")
#     sio.emit(event_name, data)


# WebSocketManager.py

import asyncio
import json
import websockets

# Keep track of all connected client websockets
connected_clients = set()


async def handle_client(websocket, path):
    """
    This is called whenever a new client connects.
    We'll add the client to our set and remove when disconnected.
    """
    print(f"[WebSocketManager] Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            # If you want to handle incoming messages from the client,
            # you can process them here.
            # For now, we just ignore.
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(
            f"[WebSocketManager] Client disconnected: {websocket.remote_address}")


async def broadcast_event(event_name, data):
    """
    Broadcast an event to all connected clients.
    event_name: a string naming the event (e.g. "new_order")
    data: a Python dict with the payload to send
    """
    if not connected_clients:
        return  # No clients to send to, so just return

    message_dict = {
        "event_name": event_name,
        "data": data
    }
    message_str = json.dumps(message_dict)

    # Gather coroutines for sending to all clients
    await asyncio.gather(
        *[client.send(message_str) for client in connected_clients],
        return_exceptions=True
    )


def start_websocket_server(host="0.0.0.0", port=8765):
    """
    Starts the asyncio WebSocket server. You can call this function from your
    main entry point (e.g. in a separate thread or directly in an asyncio loop).
    """
    print(f"[WebSocketManager] Starting WebSocket server on {host}:{port}")
    return websockets.serve(handle_client, host, port)
