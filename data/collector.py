"""
Data collector for historical system metrics.
Provides a thread-safe circular buffer and CSV export.
"""
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
import threading
import time
import os


class DataCollector:
    """
    Thread-safe collector that samples system metrics at a fixed interval
    and stores them in a rolling pandas DataFrame.
    """

    def __init__(self, max_rows: int = 1000, sample_interval_s: float = 1.0):
        self.max_rows = max_rows
        self.sample_interval_s = sample_interval_s
        self._buffer = deque(maxlen=max_rows)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def add_row(self, cpu_pct: float, ram_pct: float,
                cpu_per_core: list = None, num_processes: int = 0,
                disk_pct: float = 0.0):
        """Add a single observation."""
        now = datetime.now()
        row = {
            "timestamp": now,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "cpu_pct": round(cpu_pct, 2),
            "ram_pct": round(ram_pct, 2),
            "num_processes": num_processes,
            "disk_pct": round(disk_pct, 2),
        }
        if cpu_per_core:
            for i, v in enumerate(cpu_per_core):
                row[f"core_{i}"] = round(v, 2)

        with self._lock:
            self._buffer.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        """Return current buffer as a DataFrame."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame()
            df = pd.DataFrame(list(self._buffer))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def to_csv(self, path: str):
        """Export buffer to CSV file."""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(path, index=False)

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()
