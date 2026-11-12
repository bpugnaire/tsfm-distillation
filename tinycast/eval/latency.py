"""CPU latency: threads in {1, 8}, batch in {1, 32}, warm-up then timed runs -> p50/p95 (ms)."""

from __future__ import annotations

import os
import platform
import subprocess
import time

import numpy as np
import torch


def hardware() -> dict:
    cpu = platform.processor()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "system": platform.system(),
        "n_cpus": os.cpu_count(),
        "torch": torch.__version__,
    }


def measure(
    model,
    context_length: int,
    horizon: int,
    threads=(1, 8),
    batches=(1, 32),
    n_runs: int = 50,
    n_warmup: int = 3,
    seed: int = 0,
    max_seconds_per_config: float = 60.0,
) -> dict:
    """Times `model.forecast`. n_runs is capped by `max_seconds_per_config` (recorded)."""
    rng = np.random.default_rng(seed)
    base = 10 + np.sin(np.arange(context_length) * 2 * np.pi / 24)
    out = {}
    prev = torch.get_num_threads()
    try:
        for th in threads:
            torch.set_num_threads(th)
            for bs in batches:
                x = (base[None] + rng.normal(0, 0.1, (bs, context_length))).astype(np.float32)
                for _ in range(n_warmup):
                    model.forecast(x, horizon)
                times = []
                t_start = time.perf_counter()
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    model.forecast(x, horizon)
                    times.append((time.perf_counter() - t0) * 1e3)
                    if time.perf_counter() - t_start > max_seconds_per_config and len(times) >= 5:
                        break
                out[f"threads{th}_batch{bs}"] = {
                    "p50_ms": float(np.percentile(times, 50)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "n_runs": len(times),
                }
    finally:
        torch.set_num_threads(prev)
    return out
