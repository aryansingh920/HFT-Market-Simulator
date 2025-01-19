"""
Created on 19/01/2025

@author: Aryan

Filename: main.py

Relative Path: server/main.py
"""

import time
import asyncio
import threading
import websockets
import argparse

from Market.MarketSimulator import MarketSimulator
from Market.dynamic_config import build_simulation_config
from Market.WebSocketManager import handle_client, global_event_loop, set_simulation_callback


def run_websocket_server(port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global global_event_loop
    global_event_loop = loop

    async def start_server():
        return await websockets.serve(handle_client, "0.0.0.0", port)

    server = loop.run_until_complete(start_server())
    print(
        f"[Main] WebSocket server started on ws://0.0.0.0:{port}, waiting for connections...")

    try:
        loop.run_forever()
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the WebSocket server and HFT simulator.")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port for the WebSocket server (default: 8765)")
    args = parser.parse_args()

    # Pass the port as a tuple
    ws_thread = threading.Thread(
        target=run_websocket_server, args=(args.port,), daemon=True)
    ws_thread.start()

    while global_event_loop is None:
        time.sleep(0.1)

    simulation_config = build_simulation_config(
        sim_name="MyDynamicHFTSimulation", duration_mins=5
    )
    # Pass the global event loop to MarketSimulator
    simulator = MarketSimulator(
        config=simulation_config, loop=global_event_loop
    )

    async def run_simulation():
        print("[Main] Running simulation...")
        # Run the synchronous simulation in a separate thread
        loop = asyncio.get_event_loop()
        # Adjust steps as needed
        await loop.run_in_executor(None, simulator.run, 50)
        print("[Main] Simulation finished.")

    set_simulation_callback(run_simulation)

    print("[Main] Simulation callback set. Waiting for commands...")

    while True:
        # Keep the main thread alive or perform other tasks here
        time.sleep(1)
