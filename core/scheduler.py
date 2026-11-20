"""
Scheduling algorithms for comparison.
Implements: Round-Robin, FCFS, SJF, Priority, Predictive (ML-guided).
Runs simulations and returns benchmark metrics + Gantt chart data.
"""
import numpy as np
import pandas as pd
import time
import random
from typing import List, Dict
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class SimTask:
    """Task used for scheduling simulation."""
    task_id: str
    name: str
    burst_time: float    # CPU time needed (seconds)
    arrival_time: float  # time when task arrives
    priority: int        # 1=highest, 10=lowest
    ram_mb: float
    start_time: float = None
    finish_time: float = None
    waiting_time: float = None
    turnaround_time: float = None
    response_time: float = None


def _clone_tasks(tasks: List[SimTask]) -> List[SimTask]:
    return [deepcopy(t) for t in tasks]


# ─── Individual Schedulers ────────────────────────────────────────────────────

def schedule_fcfs(tasks: List[SimTask], num_cores: int = 1) -> Dict:
    """First Come First Served scheduling."""
    tasks = _clone_tasks(tasks)
    tasks.sort(key=lambda t: t.arrival_time)
    gantt = []
    current_time = 0.0
    for task in tasks:
        start = max(current_time, task.arrival_time)
        finish = start + task.burst_time
        task.start_time = start
        task.finish_time = finish
        task.waiting_time = start - task.arrival_time
        task.turnaround_time = finish - task.arrival_time
        task.response_time = task.waiting_time
        current_time = finish
        gantt.append({"task_id": task.task_id, "name": task.name, "start": start, "finish": finish, "core": 0})
    return {"tasks": tasks, "gantt": gantt, "algorithm": "FCFS"}


def schedule_sjf(tasks: List[SimTask], num_cores: int = 1) -> Dict:
    """Shortest Job First (non-preemptive)."""
    tasks = _clone_tasks(tasks)
    remaining = list(tasks)
    remaining.sort(key=lambda t: t.arrival_time)
    gantt = []
    current_time = 0.0
    done = []
    while remaining:
        available = [t for t in remaining if t.arrival_time <= current_time]
        if not available:
            current_time = min(t.arrival_time for t in remaining)
            continue
        shortest = min(available, key=lambda t: t.burst_time)
        remaining.remove(shortest)
        start = current_time
        finish = start + shortest.burst_time
        shortest.start_time = start
        shortest.finish_time = finish
        shortest.waiting_time = start - shortest.arrival_time
        shortest.turnaround_time = finish - shortest.arrival_time
        shortest.response_time = shortest.waiting_time
        current_time = finish
        gantt.append({"task_id": shortest.task_id, "name": shortest.name, "start": start, "finish": finish, "core": 0})
        done.append(shortest)
    return {"tasks": done, "gantt": gantt, "algorithm": "SJF"}


def schedule_round_robin(tasks: List[SimTask], quantum: float = 2.0, num_cores: int = 1) -> Dict:
    """Round Robin scheduling with configurable time quantum."""
    tasks = _clone_tasks(tasks)
    tasks.sort(key=lambda t: t.arrival_time)
    queue = []
    gantt = []
    remaining_burst = {t.task_id: t.burst_time for t in tasks}
    first_start = {t.task_id: None for t in tasks}
    current_time = 0.0
    task_map = {t.task_id: t for t in tasks}
    pending = list(tasks)

    # Add initially available tasks
    arrived = [t for t in pending if t.arrival_time <= current_time]
    for t in arrived:
        queue.append(t)
        pending = [x for x in pending if x.task_id != t.task_id]

    done_ids = set()
    max_iter = len(tasks) * 1000  # safety limit
    iterations = 0

    while queue and iterations < max_iter:
        iterations += 1
        task = queue.pop(0)
        if remaining_burst[task.task_id] <= 0:
            continue
        exec_time = min(quantum, remaining_burst[task.task_id])
        start = current_time
        finish = start + exec_time
        if first_start[task.task_id] is None:
            first_start[task.task_id] = start
        remaining_burst[task.task_id] -= exec_time
        gantt.append({"task_id": task.task_id, "name": task.name, "start": start, "finish": finish, "core": 0})
        current_time = finish

        # Arrive new tasks
        new_arrivals = [t for t in pending if t.arrival_time <= current_time]
        for t in new_arrivals:
            queue.append(t)
        pending = [t for t in pending if t not in new_arrivals]

        if remaining_burst[task.task_id] > 0.001:
            queue.append(task)
        else:
            task.finish_time = finish
            task.start_time = first_start[task.task_id]
            task.waiting_time = max(0, task.finish_time - task.arrival_time - task.burst_time)
            task.turnaround_time = task.finish_time - task.arrival_time
            task.response_time = first_start[task.task_id] - task.arrival_time
            done_ids.add(task.task_id)

    # Fallback for any that didn't complete
    for task in tasks:
        if task.task_id not in done_ids:
            task.finish_time = task.finish_time or current_time
            task.start_time = first_start.get(task.task_id) or task.arrival_time
            task.waiting_time = 0
            task.turnaround_time = task.burst_time
            task.response_time = 0

    return {"tasks": tasks, "gantt": gantt, "algorithm": "Round Robin"}


def schedule_priority(tasks: List[SimTask], num_cores: int = 1) -> Dict:
    """Non-preemptive Priority scheduling (lower value = higher priority)."""
    tasks = _clone_tasks(tasks)
    remaining = list(tasks)
    gantt = []
    current_time = 0.0
    done = []
    while remaining:
        available = [t for t in remaining if t.arrival_time <= current_time]
        if not available:
            current_time = min(t.arrival_time for t in remaining)
            continue
        highest = min(available, key=lambda t: t.priority)
        remaining.remove(highest)
        start = current_time
        finish = start + highest.burst_time
        highest.start_time = start
        highest.finish_time = finish
        highest.waiting_time = start - highest.arrival_time
        highest.turnaround_time = finish - highest.arrival_time
        highest.response_time = highest.waiting_time
        current_time = finish
        gantt.append({"task_id": highest.task_id, "name": highest.name, "start": start, "finish": finish, "core": 0})
        done.append(highest)
    return {"tasks": done, "gantt": gantt, "algorithm": "Priority"}


def schedule_predictive(tasks: List[SimTask], predicted_load: float = 30.0, num_cores: int = 4) -> Dict:
    """
    ML-Guided Predictive Scheduling.
    Reorders tasks based on predicted system load, priority, and burst time.
    Uses predicted headroom to maximize throughput.
    """
    tasks = _clone_tasks(tasks)
    # Score = alpha*priority + beta*burst_time + gamma*predicted_headroom_fit
    headroom = max(0, 100 - predicted_load)
    for task in tasks:
        cpu_fit = 1.0 if (task.burst_time * 3) <= headroom else 0.5
        task._score = task.priority * 0.4 + (1 / max(task.burst_time, 0.1)) * 0.3 + cpu_fit * 0.3

    tasks.sort(key=lambda t: (t.arrival_time, getattr(t, "_score", 1)))

    gantt = []
    # Multi-core simulation
    core_free_at = [0.0] * num_cores
    done = []

    tasks_sorted = sorted(tasks, key=lambda t: (t.arrival_time, -getattr(t, "_score", 0)))
    for task in tasks_sorted:
        # Pick earliest free core
        core_idx = int(np.argmin(core_free_at))
        start = max(core_free_at[core_idx], task.arrival_time)
        finish = start + task.burst_time
        task.start_time = start
        task.finish_time = finish
        task.waiting_time = max(0, start - task.arrival_time)
        task.turnaround_time = finish - task.arrival_time
        task.response_time = task.waiting_time
        core_free_at[core_idx] = finish
        gantt.append({"task_id": task.task_id, "name": task.name, "start": start, "finish": finish, "core": core_idx})
        done.append(task)

    return {"tasks": done, "gantt": gantt, "algorithm": "Predictive (ML-Guided)"}


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

def run_all_schedulers(tasks: List[SimTask], quantum: float = 2.0,
                       num_cores: int = 4, predicted_load: float = 30.0) -> Dict:
    """Run all scheduling algorithms and return comparative metrics."""
    results = {}
    schedulers = {
        "FCFS": lambda: schedule_fcfs(tasks, num_cores),
        "SJF": lambda: schedule_sjf(tasks, num_cores),
        "Round Robin": lambda: schedule_round_robin(tasks, quantum, num_cores),
        "Priority": lambda: schedule_priority(tasks, num_cores),
        "Predictive": lambda: schedule_predictive(tasks, predicted_load, num_cores),
    }

    for name, fn in schedulers.items():
        try:
            res = fn()
            metrics = _compute_metrics(res["tasks"])
            results[name] = {**res, "metrics": metrics}
        except Exception as e:
            results[name] = {"error": str(e), "tasks": [], "gantt": [], "metrics": {}}

    return results


def _compute_metrics(tasks: List[SimTask]) -> Dict:
    """Compute scheduling performance metrics."""
    wait_times = [t.waiting_time for t in tasks if t.waiting_time is not None]
    turnarounds = [t.turnaround_time for t in tasks if t.turnaround_time is not None]
    response_times = [t.response_time for t in tasks if t.response_time is not None]

    if not tasks:
        return {}

    total_time = max((t.finish_time for t in tasks if t.finish_time), default=1)
    busy_time = sum(t.burst_time for t in tasks)
    cpu_util = round(min(100, (busy_time / total_time) * 100), 2) if total_time > 0 else 0

    return {
        "avg_wait_time": round(np.mean(wait_times), 3) if wait_times else 0,
        "avg_turnaround": round(np.mean(turnarounds), 3) if turnarounds else 0,
        "avg_response_time": round(np.mean(response_times), 3) if response_times else 0,
        "throughput": round(len(tasks) / max(total_time, 1), 3),
        "cpu_utilization": cpu_util,
        "total_tasks": len(tasks),
        "makespan": round(total_time, 3),
    }


def generate_sim_tasks(n: int = 15, seed: int = 42) -> List[SimTask]:
    """Generate synthetic tasks for scheduling simulation."""
    rng = random.Random(seed)
    task_names = [
        "ML Train", "Data ETL", "API Call", "Render", "Compile",
        "DB Write", "Cache Build", "Network IO", "Compress", "Analyze",
        "Report", "Stream", "Encrypt", "Index", "Score",
    ]
    tasks = []
    for i in range(n):
        tasks.append(SimTask(
            task_id=f"P{i+1:02d}",
            name=rng.choice(task_names),
            burst_time=round(rng.uniform(1, 15), 2),
            arrival_time=round(rng.uniform(0, 20), 2),
            priority=rng.randint(1, 10),
            ram_mb=rng.uniform(50, 1000),
        ))
    return tasks
