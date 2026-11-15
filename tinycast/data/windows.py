"""(context, horizon) window sampling with stable ids, split-aware.

The split rule is on *targets*: a window belongs to a split iff its whole forecast target lies
inside that split's time range. Contexts may reach back into earlier periods (that is just the
past). Window ids are stable hashes of (source, series_id, target_start), so teacher caches can
be joined to windows across runs and machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tinycast.data.splits import Splits


@dataclass
class Window:
    window_id: str
    series_id: str
    context: np.ndarray  # (T,) float32, NaN = missing
    future: np.ndarray  # (H,) float32, NaN = missing
    target_start: pd.Timestamp
    target_end: pd.Timestamp


def window_id(source: str, series_id: str, target_start) -> str:
    return hashlib.sha1(f"{source}|{series_id}|{target_start}".encode()).hexdigest()[:16]


def _target_range(split: str, sp: Splits, last: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return pd.Timestamp.min.tz_localize("UTC"), sp.train_end
    if split == "val":
        return sp.val_start, sp.val_end
    if split == "fresh":
        return sp.fresh_start, last
    raise ValueError(split)


def candidate_origins(
    ts: pd.DatetimeIndex, split: str, sp: Splits, context_length: int, horizon: int
) -> np.ndarray:
    """Indices i such that target = [i, i+H) lies in the split and context = [i-T, i) exists."""
    lo, hi = _target_range(split, sp, ts[-1])
    i = np.arange(context_length, len(ts) - horizon + 1)
    ok = (ts[i] >= lo) & (ts[i + horizon - 1] <= hi)
    return i[ok]


def make_window(source, sid, ts, values, i, context_length, horizon) -> Window:
    return Window(
        window_id(source, sid, ts[i]),
        sid,
        values[i - context_length : i].astype(np.float32),
        values[i : i + horizon].astype(np.float32),
        ts[i],
        ts[i + horizon - 1],
    )


def sample_windows(
    series: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    split: str,
    sp: Splits,
    context_length: int = 512,
    horizon: int = 64,
    n_windows: int = 10000,
    min_observed: float = 0.9,
    seed: int = 0,
    source: str = "rte",
) -> list[Window]:
    """Uniformly sample `n_windows` windows over all series (deterministic given the seed)."""
    rng = np.random.default_rng(seed)
    pool = [
        (sid, i)
        for sid, (ts, _) in series.items()
        for i in candidate_origins(ts, split, sp, context_length, horizon)
    ]
    order = rng.permutation(len(pool))
    out: list[Window] = []
    for j in order:
        sid, i = pool[j]
        ts, v = series[sid]
        w = make_window(source, sid, ts, v, int(i), context_length, horizon)
        if (
            np.isfinite(w.context).mean() >= min_observed
            and np.isfinite(w.future).mean() >= min_observed
        ):
            out.append(w)
        if len(out) >= n_windows:
            break
    return out


def fresh_test_windows(
    series: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    sp: Splits,
    context_length: int = 512,
    horizon: int = 64,
    stride: int = 24,
    source: str = "rte",
    min_finite: float = 1.0,
) -> list[Window]:
    """Rolling origins every `stride` hours from fresh_start, aligned to 00:00 UTC.

    `min_finite`: share of future values that must be observed (metrics ignore missing targets). 1.0 for the headline
    suite; the long-horizon study uses 0.99 because one hour (2026-10-25 00:00 UTC, the DST night) is missing in every
    series and would otherwise remove every window longer than ~10 days."""
    out = []
    for sid, (ts, v) in sorted(series.items()):
        idx = candidate_origins(ts, "fresh", sp, context_length, horizon)
        idx = (
            idx[(ts[idx].hour == 0)][:: max(stride // 24, 1)] if stride % 24 == 0 else idx[::stride]
        )
        for i in idx:
            w = make_window(source, sid, ts, v, int(i), context_length, horizon)
            if np.isfinite(w.future).mean() >= min_finite:
                out.append(w)
    return out
