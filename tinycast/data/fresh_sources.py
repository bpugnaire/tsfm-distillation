"""Registry of fresh multi-series datasets (targets after splits.fresh_start, i.e. after every compared model's release).

Each entry: suite name -> (fine-tuning domain name, loader returning {series_id: (UTC hourly index, values)}).
TAGS: the window-id source tag of each suite (fresh_mel_ped keeps "melped", the tag its results were computed with).
Suites use the same rule as the RTE headline suite: daily origins at 00:00 UTC, targets entirely >= fresh_start,
every target hour observed, 512-step context, 64-step horizon, season 24. Fine-tuning histories end 1 h before
fresh_start (tinycast/data/target_domain.py).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _solar():
    from tinycast.data.sources.rte_production import series_arrays

    return series_arrays("solar")


def _wind():
    from tinycast.data.sources.rte_production import series_arrays

    return series_arrays("wind")


def _bike():
    from tinycast.data.sources.paris_bike import series_arrays

    return series_arrays()


def _mel():
    from tinycast.data.sources.melbourne_ped import series_arrays

    return series_arrays()


SOURCES: dict[str, tuple[str, Callable[[], dict[str, tuple[pd.DatetimeIndex, np.ndarray]]]]] = {
    "fresh_fr_solar": ("fr_solar", _solar),
    "fresh_fr_wind": ("fr_wind", _wind),
    "fresh_paris_bike": ("paris_bike", _bike),
    "fresh_mel_ped": ("mel_ped", _mel),
}
DOMAIN_LOADERS = {domain: loader for domain, loader in SOURCES.values()}
TAGS = {
    "fresh_fr_solar": "frsolar",
    "fresh_fr_wind": "frwind",
    "fresh_paris_bike": "parisbike",
    "fresh_mel_ped": "melped",
}


def fresh_suite(name: str, cfg):
    from tinycast.data.splits import load_splits
    from tinycast.data.windows import fresh_test_windows
    from tinycast.eval.benchmark import Suite

    _, loader = SOURCES[name]
    wins = fresh_test_windows(
        loader(), load_splits(), cfg.context_length, cfg.horizon, source=TAGS[name]
    )
    return Suite(
        name=name,
        ids=[w.window_id for w in wins],
        context=np.stack([w.context for w in wins]),
        future=np.stack([w.future for w in wins]),
        season=[24] * len(wins),
        meta=[dict(series_id=w.series_id, target_start=str(w.target_start)) for w in wins],
        flag=None,
    )
