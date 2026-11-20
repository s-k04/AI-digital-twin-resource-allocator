"""
Digital Twin engine.
Maintains a virtual replica of the real system, enables what-if simulations,
tracks deviation between twin state and actual system state.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import random


class DigitalTwin:
    """
    A virtual replica of a multi-core computing system.
    Tracks historical state, simulates synthetic workloads,
    and measures deviation from the real system.
    """

    def __init__(self, num_cores: int = None, total_ram_gb: float = None, history_len: int = 300):
        import psutil
        self.num_cores = num_cores or psutil.cpu_count(logical=True)
        self.total_ram_gb = total_ram_gb or round(psutil.virtual_memory().total / (1024 ** 3), 2)
        self.history_len = history_len

        # Ring buffers for twin state
        self.twin_cpu_history = deque(maxlen=history_len)   # list of per-core values
        self.twin_ram_history = deque(maxlen=history_len)   # ram percent
        self.real_cpu_history = deque(maxlen=history_len)
        self.real_ram_history = deque(maxlen=history_len)
        self.timestamps = deque(maxlen=history_len)

        # Active synthetic workloads
        self.synthetic_workloads = []

        # Twin state
        self.twin_cpu = [5.0] * self.num_cores
        self.twin_ram = 30.0

        # Deviation metrics
        self.cpu_deviation_history = deque(maxlen=history_len)
        self.ram_deviation_history = deque(maxlen=history_len)

    def update(self, real_cpu_per_core: list, real_ram_percent: float):
        """Update twin with real system data and advance simulation."""
        now = datetime.now()

        # Advance synthetic workloads
        self._advance_workloads()

        # Smooth twin CPU toward real + synthetic influence
        synthetic_load = self._compute_synthetic_load()
        for i in range(self.num_cores):
            real_val = real_cpu_per_core[i] if i < len(real_cpu_per_core) else 0
            target = min(100.0, real_val + synthetic_load[i])
            # Smooth with exponential moving average
            self.twin_cpu[i] = 0.7 * self.twin_cpu[i] + 0.3 * target

        syn_ram = self._compute_synthetic_ram()
        self.twin_ram = min(99.0, 0.7 * self.twin_ram + 0.3 * (real_ram_percent + syn_ram))

        # Store histories
        self.twin_cpu_history.append(list(self.twin_cpu))
        self.twin_ram_history.append(self.twin_ram)
        self.real_cpu_history.append(list(real_cpu_per_core))
        self.real_ram_history.append(real_ram_percent)
        self.timestamps.append(now)

        # Compute deviation
        cpu_dev = abs(np.mean(self.twin_cpu) - np.mean(real_cpu_per_core))
        ram_dev = abs(self.twin_ram - real_ram_percent)
        self.cpu_deviation_history.append(cpu_dev)
        self.ram_deviation_history.append(ram_dev)

    def inject_workload(self, name: str, cpu_load_percent: float, ram_load_gb: float, duration_s: float):
        """Inject a synthetic workload into the digital twin."""
        workload = {
            "name": name,
            "cpu_load": cpu_load_percent,
            "ram_load_gb": ram_load_gb,
            "duration_s": duration_s,
            "elapsed_s": 0.0,
            "cores": random.sample(range(self.num_cores), min(2, self.num_cores)),
            "created_at": datetime.now().isoformat(),
        }
        self.synthetic_workloads.append(workload)
        return workload

    def _advance_workloads(self, dt: float = 1.0):
        """Advance all workload timers by dt seconds."""
        active = []
        for wl in self.synthetic_workloads:
            wl["elapsed_s"] += dt
            if wl["elapsed_s"] < wl["duration_s"]:
                active.append(wl)
        self.synthetic_workloads = active

    def _compute_synthetic_load(self) -> list:
        """Compute per-core additional CPU load from synthetic workloads."""
        extra = [0.0] * self.num_cores
        for wl in self.synthetic_workloads:
            for core in wl["cores"]:
                extra[core] += wl["cpu_load"]
        return [min(e, 95.0) for e in extra]

    def _compute_synthetic_ram(self) -> float:
        """Compute extra RAM % from synthetic workloads."""
        extra_gb = sum(wl["ram_load_gb"] for wl in self.synthetic_workloads)
        return min((extra_gb / self.total_ram_gb) * 100, 50.0)

    def get_deviation(self) -> dict:
        """Get current deviation between twin and real system."""
        if len(self.cpu_deviation_history) == 0:
            return {"cpu_deviation": 0.0, "ram_deviation": 0.0, "health_score": 100.0}
        cpu_dev = float(np.mean(list(self.cpu_deviation_history)[-10:]))
        ram_dev = float(np.mean(list(self.ram_deviation_history)[-10:]))
        health = max(0.0, 100.0 - (cpu_dev + ram_dev) / 2)
        return {
            "cpu_deviation": round(cpu_dev, 2),
            "ram_deviation": round(ram_dev, 2),
            "health_score": round(health, 1),
        }

    def get_history_df(self) -> pd.DataFrame:
        """Return twin & real history as a DataFrame."""
        if len(self.timestamps) == 0:
            return pd.DataFrame()
        rows = []
        for i, ts in enumerate(self.timestamps):
            twin_c = self.twin_cpu_history[i]
            real_c = self.real_cpu_history[i]
            rows.append({
                "timestamp": ts,
                "twin_cpu_avg": round(np.mean(twin_c), 2),
                "real_cpu_avg": round(np.mean(real_c), 2),
                "twin_ram_pct": round(self.twin_ram_history[i], 2),
                "real_ram_pct": round(self.real_ram_history[i], 2),
            })
        return pd.DataFrame(rows)

    def run_whatif_simulation(self, scenario: dict) -> dict:
        """
        Run a what-if scenario and return predicted twin state.
        scenario = {
          'extra_cpu_pct': float,  # additional CPU load to inject
          'extra_ram_gb': float,   # additional RAM to allocate
          'duration_s': float,     # how long the extra load runs
        }
        """
        base_cpu = np.mean(self.twin_cpu) if self.twin_cpu else 20.0
        base_ram = self.twin_ram if self.twin_ram else 40.0

        extra_cpu = scenario.get("extra_cpu_pct", 0)
        extra_ram_gb = scenario.get("extra_ram_gb", 0)
        duration = scenario.get("duration_s", 30)

        pred_cpu = min(100, base_cpu + extra_cpu)
        pred_ram = min(100, base_ram + (extra_ram_gb / self.total_ram_gb) * 100)

        # Estimate system saturation
        saturated = pred_cpu > 85 or pred_ram > 90

        return {
            "scenario": scenario,
            "predicted_cpu_avg_pct": round(pred_cpu, 2),
            "predicted_ram_pct": round(pred_ram, 2),
            "saturation_risk": saturated,
            "recommendation": (
                "⚠️ System may saturate — reduce workload or scale resources."
                if saturated
                else "✅ System can handle this workload safely."
            ),
        }
