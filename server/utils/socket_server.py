"""
Created on 18/01/2025

@author: Aryan

Filename: socket_server.py

Relative Path: server/utils/socket_server.py
"""
"""
socket_server.py

Entry point to run the Socket.IO server with eventlet.
"""


import eventlet
from Market.SocketManager import app
if __name__ == "__main__":
    # Run the WSGI app (which includes the Socket.IO server) via eventlet
    PORT = 8000
    print(f"[SocketManager] Starting Socket.IO server on port {PORT}...")
    eventlet.wsgi.server(eventlet.listen(('', PORT)), app)
