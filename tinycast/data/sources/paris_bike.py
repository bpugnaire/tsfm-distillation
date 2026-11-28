"""Paris bike counters, hourly counts per counter (Ville de Paris open data, ODbL).

Source: https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs (rolling ~13 months; downloaded
2026-11-28: 2025-10-27 -> 2026-11-26, 113 counters). Timestamps are UTC from the API; full hourly grid, missing hours
NaN; counters with < 80% of the grid observed are dropped.

    uv run python -m tinycast.data.sources.paris_bike   # download (refresh)
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from tinycast.data.sources.rte_eco2mix import ROOT

URL = ("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/exports/parquet"
       "?select=id_compteur,sum_counts,date&timezone=UTC")  # fmt: skip
RAW = ROOT / "data" / "raw" / "paris_bike" / "counts_per_hour.parquet"
MIN_COVERAGE = 0.8


def download() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "tinycast-research"})
    with urllib.request.urlopen(req, timeout=900) as r:
        RAW.write_bytes(r.read())


def series_arrays() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    d = pd.read_parquet(RAW)
    d["t"] = pd.to_datetime(d["date"], utc=True).dt.floor("h")
    d = d.groupby(["id_compteur", "t"], as_index=False)["sum_counts"].sum()
    grid = pd.date_range(d["t"].min(), d["t"].max(), freq="h", tz="UTC")
    out = {}
    for sid, g in d.groupby("id_compteur"):
        v = g.set_index("t")["sum_counts"].reindex(grid).to_numpy(dtype=np.float32)
        if np.isfinite(v).mean() >= MIN_COVERAGE:
            out[f"bike:{sid}"] = (grid, v)
    return dict(sorted(out.items()))


if __name__ == "__main__":
    download()
    s = series_arrays()
    g = next(iter(s.values()))[0]
    print(f"{len(s)} counters kept, {g[0]} -> {g[-1]}")
