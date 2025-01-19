"""
Created on 19/01/2025

@author: Aryan

Filename: HeatManager.py

Relative Path: server/Market/HeatManager.py
"""

from datetime import datetime
import time


class HeatManager:
    """
    Manages heat durations and notifies the DataLogger when a new heat starts
    and when one ends.
    """

    def __init__(self, heat_duration_minutes=1, data_logger=None):
        self.heat_duration_seconds = heat_duration_minutes * 60
        self.heat_start_time = None
        self.heat_count = 0
        self.data_logger = data_logger

    def start_heat(self):
        """
        Called once at the beginning or whenever we rotate into a new heat.
        """
        self.heat_count += 1
        self.heat_start_time = time.time()
        if self.data_logger:
            self.data_logger.start_new_heat(self.heat_count)
        print(f"--- Starting Heat #{self.heat_count} at {datetime.now()} ---")

    def check_and_rotate_heat(self):
        """
        Checks if current heat exceeded duration; if yes, ends it and starts a new one.
        """
        if self.heat_start_time is None:
            # No heat running yet
            self.start_heat()
            return

        elapsed = time.time() - self.heat_start_time
        if elapsed >= self.heat_duration_seconds:
            # End current heat
            if self.data_logger:
                self.data_logger.end_heat()

            # Start a new heat
            self.start_heat()
