"""
hft_simulator.py

Main script to run a dynamic HFT simulation, now updated to broadcast
real-time events (orders, trades, snapshots, etc.) via Socket.IO.
"""

from Market.MarketSimulator import MarketSimulator
from Market.dynamic_config import build_simulation_config

if __name__ == "__main__":
    # 1) Build the simulation config (dynamically):
    simulation_config = build_simulation_config(
        sim_name="MyDynamicHFTSimulation")

    # 2) Initialize the MarketSimulator with this dynamic config
    simulator = MarketSimulator(config=simulation_config)

    # 3) Run the simulation for a certain number of steps
    #    As it runs, the DataLogger inside will broadcast all events via Socket.IO
    simulator.run(steps=1000)
