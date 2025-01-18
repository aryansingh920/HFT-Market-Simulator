"""
Created on 18/01/2025

@author: Aryan

Filename: SocketManager.py

Relative Path: server/utils/Market/SocketManager.py
"""


"""
SocketManager.py

Creates a Socket.IO server to broadcast events in real-time.
"""


# Create a Socket.IO server
import eventlet
import socketio
sio = socketio.Server(cors_allowed_origins="*")
# Wrap with a WSGI app
app = socketio.WSGIApp(sio)


@sio.event
def connect(sid, environ):
    print(f"[SocketManager] Client connected: {sid}")


@sio.event
def disconnect(sid):
    print(f"[SocketManager] Client disconnected: {sid}")


def broadcast_event(event_name, data):
    """
    Utility function to broadcast a given event (event_name)
    with data (a dict) to all connected clients.
    """
    # print(f"[SocketManager] Emitting event: {event_name}, Data: {data}")
    sio.emit(event_name, data)
