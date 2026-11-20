"""
Intelligent Resource Allocator.
Priority-based task queue with ML-guided resource reservation.
Compares ML-guided allocation vs traditional (static) allocation.
"""
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import heapq


@dataclass(order=True)
class Task:
    """Represents a compute task submitted to the system."""
    priority: int
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    cpu_requested: float = field(compare=False)   # % CPU requested
    ram_requested_gb: float = field(compare=False)
    duration_s: float = field(compare=False)
    arrival_time: float = field(compare=False, default_factory=time.time)
    start_time: Optional[float] = field(compare=False, default=None)
    finish_time: Optional[float] = field(compare=False, default=None)
    assigned_cores: List[int] = field(compare=False, default_factory=list)
    status: str = field(compare=False, default="PENDING")  # PENDING, RUNNING, DONE, REJECTED

    @property
    def wait_time(self):
        if self.start_time:
            return round(self.start_time - self.arrival_time, 3)
        return None

    @property
    def turnaround_time(self):
        if self.finish_time:
            return round(self.finish_time - self.arrival_time, 3)
        return None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority,
            "cpu_requested_%": self.cpu_requested,
            "ram_requested_gb": self.ram_requested_gb,
            "duration_s": self.duration_s,
            "status": self.status,
            "assigned_cores": self.assigned_cores,
            "wait_time_s": self.wait_time,
            "turnaround_s": self.turnaround_time,
        }


class ResourceAllocator:
    """
    Intelligent resource allocator that uses predicted CPU/RAM headroom
    to make smarter allocation decisions vs static thresholds.
    """

    def __init__(self, num_cores: int, total_ram_gb: float):
        self.num_cores = num_cores
        self.total_ram_gb = total_ram_gb
        self.core_load = [0.0] * num_cores   # current % load per core
        self.ram_used_gb = 0.0
        self.task_queue: List[Task] = []  # min-heap by priority (lower = higher priority)
        self.completed_tasks: List[Task] = []
        self.rejected_tasks: List[Task] = []
        self.running_tasks: List[Task] = []
        self.task_counter = 0

        # Thresholds
        self.cpu_threshold_traditional = 80.0  # Traditional: reject if any core > 80%
        self.ram_threshold_traditional_pct = 85.0
        self.cpu_threshold_ml = 90.0           # ML-guided: use predicted headroom
        self.ram_threshold_ml_pct = 92.0

    def submit_task(self, name: str, cpu_pct: float, ram_gb: float,
                    duration_s: float, priority: int = 5) -> Task:
        """Submit a task to the queue."""
        self.task_counter += 1
        task = Task(
            priority=priority,
            task_id=f"T{self.task_counter:04d}",
            name=name,
            cpu_requested=cpu_pct,
            ram_requested_gb=ram_gb,
            duration_s=duration_s,
        )
        heapq.heappush(self.task_queue, task)
        return task

    def _find_best_cores(self, cpu_requested_pct: float, strategy: str = "ml") -> List[int]:
        """Find least-loaded cores for assignment."""
        sorted_cores = sorted(range(self.num_cores), key=lambda i: self.core_load[i])
        num_to_assign = max(1, min(self.num_cores, int(np.ceil(cpu_requested_pct / 25))))
        return sorted_cores[:num_to_assign]

    def _can_allocate_traditional(self, task: Task) -> bool:
        """Static threshold-based allocation check."""
        if self.ram_used_gb + task.ram_requested_gb > (self.total_ram_gb * self.ram_threshold_traditional_pct / 100):
            return False
        avg_core_load = np.mean(self.core_load)
        if avg_core_load + task.cpu_requested / self.num_cores > self.cpu_threshold_traditional:
            return False
        return True

    def _can_allocate_ml(self, task: Task, predicted_cpu: float = None, predicted_ram_pct: float = None) -> bool:
        """ML-guided allocation using predicted headroom."""
        pred_cpu = predicted_cpu if predicted_cpu is not None else np.mean(self.core_load)
        pred_ram = predicted_ram_pct if predicted_ram_pct is not None else (self.ram_used_gb / self.total_ram_gb * 100)
        headroom_cpu = self.cpu_threshold_ml - pred_cpu
        headroom_ram = (self.ram_threshold_ml_pct / 100 * self.total_ram_gb) - self.ram_used_gb
        return (task.cpu_requested / self.num_cores <= headroom_cpu and
                task.ram_requested_gb <= headroom_ram)

    def allocate_next_task(self, strategy: str = "ml",
                           predicted_cpu: float = None,
                           predicted_ram_pct: float = None) -> Optional[Task]:
        """Pop highest-priority task and attempt allocation."""
        if not self.task_queue:
            return None

        task = heapq.heappop(self.task_queue)
        task.start_time = time.time()

        if strategy == "traditional":
            can = self._can_allocate_traditional(task)
        else:
            can = self._can_allocate_ml(task, predicted_cpu, predicted_ram_pct)

        if can:
            best_cores = self._find_best_cores(task.cpu_requested, strategy)
            task.assigned_cores = best_cores
            for c in best_cores:
                self.core_load[c] = min(100.0, self.core_load[c] + task.cpu_requested / len(best_cores))
            self.ram_used_gb += task.ram_requested_gb
            task.status = "RUNNING"
            task.finish_time = task.start_time + task.duration_s
            self.running_tasks.append(task)
            self.completed_tasks.append(task)
        else:
            task.status = "REJECTED"
            self.rejected_tasks.append(task)

        return task

    def release_finished_tasks(self, current_time: float = None):
        """Release resources for finished tasks."""
        t = current_time or time.time()
        still_running = []
        for task in self.running_tasks:
            if task.finish_time and t >= task.finish_time:
                for c in task.assigned_cores:
                    self.core_load[c] = max(0.0, self.core_load[c] - task.cpu_requested / max(1, len(task.assigned_cores)))
                self.ram_used_gb = max(0.0, self.ram_used_gb - task.ram_requested_gb)
                task.status = "DONE"
            else:
                still_running.append(task)
        self.running_tasks = still_running

    def get_stats(self) -> Dict:
        """Return allocation statistics."""
        done = [t for t in self.completed_tasks if t.status == "DONE"]
        wait_times = [t.wait_time for t in done if t.wait_time is not None]
        turnarounds = [t.turnaround_time for t in done if t.turnaround_time is not None]
        cpu_util = np.mean(self.core_load) if self.core_load else 0
        return {
            "total_submitted": self.task_counter,
            "completed": len(done),
            "rejected": len(self.rejected_tasks),
            "pending": len(self.task_queue),
            "avg_wait_time_s": round(np.mean(wait_times), 3) if wait_times else 0,
            "avg_turnaround_s": round(np.mean(turnarounds), 3) if turnarounds else 0,
            "cpu_utilization_pct": round(cpu_util, 2),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "throughput_tasks": len(done),
        }

    get_allocation_stats = get_stats

    def reset(self):
        """Reset allocator state."""
        self.core_load = [0.0] * self.num_cores
        self.ram_used_gb = 0.0
        self.task_queue = []
        self.completed_tasks = []
        self.rejected_tasks = []
        self.running_tasks = []
        self.task_counter = 0


def generate_random_tasks(n: int = 20, seed: int = 42) -> List[dict]:
    """Generate a list of random task specs for simulation."""
    rng = random.Random(seed)
    task_names = [
        "Data ETL", "Model Train", "Video Encode", "DB Query", "API Server",
        "Image Render", "Log Parser", "Backup Job", "Network Scan", "File Compress",
        "ML Inference", "Cache Warm", "Report Gen", "Stream Process", "Crypto Hash",
    ]
    tasks = []
    for i in range(n):
        tasks.append({
            "name": rng.choice(task_names),
            "cpu_pct": rng.uniform(5, 60),
            "ram_gb": rng.uniform(0.1, 4.0),
            "duration_s": rng.uniform(2, 30),
            "priority": rng.randint(1, 10),
        })
    return tasks
