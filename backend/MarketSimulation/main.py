"""
Created on 03/02/2025

@author: Aryan

Filename: main.py

Relative Path: server/main.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from MarketModel.StockPriceSimulatorWithOrderBook import StockPriceSimulatorWithOrderBook
# from MarketModel.IntradayStockPriceSimulatorWithOrderBook import IntradayStockPriceSimulatorWithOrderBook
from MarketModel.Dashboard import Dashboard
# from config import configs_nvidia, configs_pure_gbm, configs_test, intraday_config, IntradayStockPriceSimulator
from config import configs_nvidia, configs_pure_gbm, configs_test, configs_apple
from MarketModel.DataLogger import save_simulation_steps_csv, save_orderbook_snapshots_csv, save_config_as_json
import os


import matplotlib.pyplot as plt


def plot_price_with_regimes(sim_result):
    prices = sim_result['prices']
    time = sim_result['time']
    regimes = sim_result['regime_history']

    unique_regimes = list(set(regimes))
    color_map = {r: plt.cm.tab10(i) for i, r in enumerate(unique_regimes)}

    plt.figure(figsize=(14, 6))
    for i in range(1, len(prices)):
        r = regimes[i - 1]
        plt.plot(time[i-1:i+1], prices[i-1:i+1], color=color_map[r])

    plt.title("Price Trajectory with Regime Coloring")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend(handles=[plt.Line2D([0], [0], color=color, label=reg)
               for reg, color in color_map.items()])

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: pd.to_datetime(x, unit='s').strftime('%H:%M:%S')))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    plt.show()


if __name__ == "__main__":
    config_used = configs_apple
    config_used = config_used[0]  # Use the first configuration for testing
    simulator = StockPriceSimulatorWithOrderBook(**config_used)


    # Run the simulation
    results = simulator.simulate()
    plot_price_with_regimes(results)

    # make directory for simulation output
    if not os.path.exists(f"simulation_output/{config_used['name']}"):
        os.makedirs(f"simulation_output/{config_used['name']}")

    # Save simulation results to CSV
    save_simulation_steps_csv(
        results, f"simulation_output/{config_used['name']}/simulation_steps.csv")
    save_orderbook_snapshots_csv(
        results, f"simulation_output/{config_used['name']}/order_book_snapshots.csv")

    # Optionally save the configuration as JSON
    save_config_as_json(
        config_used, f"simulation_output/{config_used['name']}/config_used.json")



    # Extract values
    prices = results["prices"]
    # Changed "times" to "time" to match the dictionary
    times = results["time"]

    # Visualization
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=times, y=prices, mode='lines+markers',
                   name='Intraday Price'),
        row=1, col=1
    )
    fig.update_layout(
        title=config_used["name"],
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_tickformat="%H:%M:%S",
        xaxis_tickangle=-45,
        yaxis_tickformat=".2f",
        yaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=40, b=40),
        # height=600,
        # width=800,
        # template="plotly_black",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=16,
            color="black"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            tickwidth=1,
            ticklen=5
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            tickwidth=1,
            ticklen=5
        )

    )
    # fig.show()
