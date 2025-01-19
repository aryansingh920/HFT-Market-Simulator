# hft_simulator.py

import time
import asyncio
import threading
import websockets

from Market.MarketSimulator import MarketSimulator
from Market.dynamic_config import build_simulation_config
from Market.WebSocketManager import handle_client, global_event_loop, set_simulation_callback


def run_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global global_event_loop
    global_event_loop = loop

    async def start_server():
        return await websockets.serve(handle_client, "0.0.0.0", 8765)

    server = loop.run_until_complete(start_server())
    print("[Main] WebSocket server started on ws://0.0.0.0:8765, waiting for connections...")

    try:
        loop.run_forever()
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()


if __name__ == "__main__":
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
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
        await loop.run_in_executor(None, simulator.run, 50000)
        print("[Main] Simulation finished.")

    set_simulation_callback(run_simulation)

    print("[Main] Simulation callback set. Waiting for commands...")

    while True:
        # Keep the main thread alive or perform other tasks here
        time.sleep(1)
