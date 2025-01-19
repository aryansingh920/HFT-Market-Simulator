

from Market.WebSocketManager import handle_client, global_event_loop, set_simulation_callback
from Market.dynamic_config import build_simulation_config
from Market.MarketSimulator import MarketSimulator
import threading
import time
import asyncio
import websockets


# hft_simulator.py


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
    # Create a thread for the WebSocket server
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()

    while global_event_loop is None:
        time.sleep(0.1)

    # Simulation setup
    simulation_config = build_simulation_config(
        sim_name="MyDynamicHFTSimulation", duration_mins=5)

    simulator = MarketSimulator(
        config=simulation_config, loop=global_event_loop)

    # Define a coroutine for running the simulation
    async def run_simulation():
        print("[Main] Running simulation...")
        await simulator.run(steps=50000)  # Adjust steps as needed
        print("[Main] Simulation finished.")

    # Set the simulation callback in WebSocketManager
    set_simulation_callback(run_simulation)

    print("[Main] Waiting for start_simulation command from WebSocket client...")

    while True:
        # Keep the main thread alive or perform other tasks here
        time.sleep(1)
