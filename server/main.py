"""
Created on 03/02/2025

@author: Aryan

Filename: main.py

Relative Path: server/main.py
"""


from MarketModel.StockPriceSimulatorWithOrderBook import StockPriceSimulatorWithOrderBook
from MarketModel.Dashboard import Dashboard
from config import configs_nvidia
from MarketModel.DataLogger import save_simulation_steps_csv, save_orderbook_snapshots_csv, save_config_as_json
import os

if __name__ == "__main__":
    results = []
    config_names = []
    config_details = []

    # For each configuration in configs_nvidia, instantiate and run the new simulator.
    for idx, config in enumerate(configs_nvidia, 1):
        name = config.get("name", f"Simulation_{idx}")
        print(f"Running simulation with order book: {name}")

        simulator = StockPriceSimulatorWithOrderBook(**config)
        result = simulator.simulate()
        results.append(result)
        config_names.append(name)
        config_details.append(config)

        # Create an output folder if needed
        output_folder = "simulation_output"
        os.makedirs(output_folder, exist_ok=True)

        # Save simulation step data
        steps_filename = os.path.join(output_folder, f"{name}_steps.csv")
        save_simulation_steps_csv(result, steps_filename)

        # Save order book snapshots
        orderbook_filename = os.path.join(
            output_folder, f"{name}_orderbook.csv")
        save_orderbook_snapshots_csv(result, orderbook_filename)

        # Optionally, save the configuration once per simulation.
        config_filename = os.path.join(output_folder, f"{name}_config.json")
        save_config_as_json(config, config_filename)

    # Optionally, launch the dashboard to visualize the simulation.
    dashboard = Dashboard(results, config_names, config_details)
    dashboard.run()



# ============================================================================
# Example usage
# ============================================================================
# if __name__ == "__main__":
#     # 1) Run the historical configs
#     historical_simulator = MarketSimulator.MarketSimulator(configs_nvidia)
#     historical_simulator.run_simulations()
#     # These are your historical results, config details, and names
#     hist_results = historical_simulator.results
#     hist_config_names = historical_simulator.config_names
#     hist_config_details = historical_simulator.config_details

#     # 2) Run the advanced single-stock simulation
#     # advanced_config = {
#     #     "name": "Advanced Single Stock",
#     #     "duration": 1,
#     #     "steps": 252,
#     #     "initial_price": 100,
#     #     "fundamental_value": 100,
#     #     "initial_liquidity": 1e6,
#     #     "base_volatility": 0.2,
#     #     "transaction_cost": 0.0005,
#     #     "sentiment_seed": 0.1,
#     #     "news_flow_intensity": 0.1,
#     #     "seasonality_params": {"day_of_week": [1.0, 1.02, 0.98, 1.03, 0.97]},
#     #     "heston_params": {"kappa": 1.2, "theta": 0.04, "eta": 0.15},
#     #     "refined_jump_params": {"intensity": 0.05, "df": 3},
#     #     # ... other advanced parameters
#     # }
#     # advanced_sim = AdvancedStockSimulator(advanced_config)
#     # advanced_result = advanced_sim.simulate()

#     # We'll store it in a list so we can unify it with the others
#     # advanced_results_list = [advanced_result]
#     # advanced_config_names = [advanced_config["name"]]
#     # advanced_config_details = [advanced_config]

#     # 3) Run the multi-asset simulation
#     # multi_asset_configs = [
#     #     {
#     #         "name": "Stock A",
#     #         "duration": 1,
#     #         "steps": 252,
#     #         "initial_price": 100,
#     #         "fundamental_value": 100,
#     #         "initial_liquidity": 1e6,
#     #         "base_volatility": 0.2,
#     #         "transaction_cost": 0.0005,
#     #     },
#     #     {
#     #         "name": "Stock B",
#     #         "duration": 1,
#     #         "steps": 252,
#     #         "initial_price": 150,
#     #         "fundamental_value": 150,
#     #         "initial_liquidity": 1e6,
#     #         "base_volatility": 0.25,
#     #         "transaction_cost": 0.0007,
#     #     }
#     # ]
#     # correlation_matrix = [
#     #     [1.0, 0.7],
#     #     [0.7, 1.0]
#     # ]
#     # multi_asset_sim = MultiAssetSimulator(
#     #     multi_asset_configs, correlation_matrix)
#     # multi_asset_results = multi_asset_sim.simulate()
#     # `multi_asset_results` is a list of dicts (one result per stock)

#     # We'll build parallel lists for the multi-asset run:
#     # ma_results_list = []
#     # ma_names = []
#     # ma_details = []
#     # for cfg, res in zip(multi_asset_configs, multi_asset_results):
#     # ma_results_list.append(res)
#     # ma_names.append(cfg["name"])  # "Stock A", "Stock B"
#     # ma_details.append(cfg)

#     # 4) Combine everything into one set of lists
#     # all_results = hist_results + advanced_results_list + ma_results_list
#     # all_names = hist_config_names + advanced_config_names + ma_names
#     # all_details = hist_config_details + advanced_config_details + ma_details

#     # 5) Launch ONE dashboard with everything

#     # combined_dashboard = Dashboard(all_results, all_names, all_details)
#     combined_dashboard = Dashboard.Dashboard(
#         hist_results, hist_config_names, hist_config_details)
#     combined_dashboard.run()
