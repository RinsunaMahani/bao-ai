"""
Bao AI - Real-time System & Pipeline Telemetry
Monitors CPU, RAM, Disk usage, and pipeline phase latencies.
"""

import time

import psutil


def get_system_telemetry() -> dict:
    """Returns real-time host resource metrics."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2),
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "disk_percent": disk.percent
    }


class BenchmarkTimer:
    """Context manager to profile internal execution latencies."""

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)