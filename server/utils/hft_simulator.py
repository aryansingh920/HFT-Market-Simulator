from server.utils.Market import MarketSimulator







if __name__ == "__main__":
    simulator = MarketSimulator(
        lambda_rate=5,            # Higher means more frequent orders
        initial_liquidity=5,      # Some initial orders
        symbols=["AAPL"],
        # symbols=["AAPL", "GOOG", "AMZN"],
        heat_duration_minutes=0.5  # e.g., 30 seconds for quick demo
    )
    simulator.run(steps=1000)

    # Check your "simulation_logs" directory for per-heat folders with CSV data.
