"""
Created on 03/02/2025

@author: Aryan

Filename: Dashboard.py

Relative Path: server/MarketModel/Dashboard.py
"""
import json  # For pretty–printing config details
import dash
from dash import dcc, html
import plotly.graph_objs as go
# ============================================================================
# Dashboard / Graph Generation Class (using Dash and Plotly)
# ============================================================================


class Dashboard:
    def __init__(self, simulation_results_list, config_names, config_details):
        """
        simulation_results_list: list of simulation results (dicts from simulate())
        config_names: list of simulation names
        config_details: list of the configuration dictionaries used for each simulation
        """
        self.results_list = simulation_results_list
        self.config_names = config_names
        self.config_details = config_details
        self.app = dash.Dash(__name__)
        self.build_layout()

    def segment_data_by_regime(self, time, prices, regimes):
        segments = []
        start_idx = 0
        current_regime = regimes[0]
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                segments.append(
                    (current_regime, time[start_idx:i+1], prices[start_idx:i+1]))
                start_idx = i
                current_regime = regimes[i]
        segments.append((current_regime, time[start_idx:], prices[start_idx:]))
        return segments

    def create_figure(self, result, name):
        segments = self.segment_data_by_regime(
            result['time'], result['prices'], result['regime_history'])
        fig = go.Figure()
        for regime, t_seg, p_seg in segments:
            fig.add_trace(go.Scatter(
                x=t_seg, y=p_seg,
                mode='lines+markers',
                name=f"{regime} regime",
                hovertemplate="Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>"
            ))
        fig.add_trace(go.Scatter(
            x=result['time'], y=result['fundamentals'],
            mode='lines',
            name='Fundamental Value',
            line=dict(dash='dash', color='black'),
            hovertemplate="Time: %{x:.2f}<br>Fundamental: %{y:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"{name} Simulation",
            xaxis_title="Time (Years)",
            yaxis_title="Price",
            hovermode="closest"
        )
        return fig

    def build_layout(self):
        tabs = []
        # For each simulation, create a Tab with the graph + JSON config
        for result, name, cfg in zip(self.results_list, self.config_names, self.config_details):
            fig = self.create_figure(result, name)
            tab = dcc.Tab(label=name, children=[
                html.Div([
                    dcc.Graph(figure=fig),
                    html.H2("Simulation Configuration"),
                    # Show config as formatted JSON
                    html.Pre(json.dumps(cfg, indent=2))
                ])
            ])
            tabs.append(tab)
        self.app.layout = html.Div([
            html.H1("Market Simulation Dashboard"),
            dcc.Tabs(children=tabs)
        ])

    def run(self):
        self.app.run_server(debug=True)
