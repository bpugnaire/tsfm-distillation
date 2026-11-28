"""RTE éCO2mix production by source (solar, wind) for the 12 mainland regions, Licence Ouverte v2.0.

Same source, API and rules as rte_eco2mix.py (consumption), kept separate so the consumption pipeline behind the
headline results is untouched. Consolidated history ("cons-def") + real-time ("tr"); UTC; hourly means; gaps as NaN;
consolidated wins where both exist. Regions whose mean output is below 2% of the largest region's are dropped (e.g.
wind in Île-de-France: near-constant zeros carry no signal).

    uv run python -m tinycast.data.sources.rte_production   # download + process both variables
-> data/processed/fr_<solar|wind>_hourly.parquet  (series_id, timestamp, value)
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from tinycast.data.sources.rte_eco2mix import API, ROOT

RAW = ROOT / "data" / "raw" / "rte_production"
VARIABLES = {"solar": "solaire", "wind": "eolien"}
DATASETS = {"regional_def": "eco2mix-regional-cons-def", "regional_tr": "eco2mix-regional-tr"}
MIN_SHARE = 0.02


def processed(var: str):
    return ROOT / "data" / "processed" / f"fr_{var}_hourly.parquet"


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    fields = ",".join(VARIABLES.values())
    for key, ds in DATASETS.items():
        q = urllib.parse.urlencode({"select": f"date_heure,libelle_region,{fields}", "timezone": "UTC",
                                    "where": " or ".join(f"{f} is not null" for f in VARIABLES.values())})  # fmt: skip
        req = urllib.request.Request(
            f"{API}/{ds}/exports/parquet?{q}", headers={"User-Agent": "tinycast-research"}
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            (RAW / f"{key}.parquet").write_bytes(r.read())
        print(
            f"  {key}: {(RAW / f'{key}.parquet').stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s",
            flush=True,
        )


def process(var: str) -> pd.DataFrame:
    col = VARIABLES[var]
    frames = []
    for key in DATASETS:
        df = pd.read_parquet(RAW / f"{key}.parquet")
        df = df[df[col].notna()]
        frames.append(pd.DataFrame({"series_id": df["libelle_region"], "value": pd.to_numeric(df[col], errors="coerce"),
                                    "timestamp": pd.to_datetime(df["date_heure"], utc=True),
                                    "source": "def" if key.endswith("_def") else "tr"}))  # fmt: skip
    df = (
        pd.concat(frames)
        .sort_values("source")
        .drop_duplicates(["series_id", "timestamp"], keep="first")
    )
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = df.groupby(["series_id", "hour"], as_index=False)["value"].mean()
    means = hourly.groupby("series_id")["value"].mean()
    keep = means[means >= MIN_SHARE * means.max()].index
    out = []
    for sid, g in hourly[hourly["series_id"].isin(keep)].groupby("series_id"):
        idx = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h", tz="UTC")
        s = g.set_index("hour")["value"].reindex(idx)
        out.append(
            pd.DataFrame({"series_id": sid, "timestamp": idx, "value": s.to_numpy(np.float32)})
        )
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(processed(var))
    return res


def series_arrays(var: str) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    df = pd.read_parquet(processed(var))
    return {f"{var}:{sid}": (pd.DatetimeIndex(g["timestamp"]), g["value"].to_numpy(np.float32))
            for sid, g in df.groupby("series_id")}  # fmt: skip


if __name__ == "__main__":
    download()
    for v in VARIABLES:
        r = process(v)
        print(
            v, r["series_id"].nunique(), "regions", r["timestamp"].min(), "->", r["timestamp"].max()
        )
