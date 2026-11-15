"""RTE éCO2mix electricity consumption (France national + 12 regions), Licence Ouverte v2.0.

Source: ODRÉ (https://odre.opendatasoft.com). Consolidated history ("cons-def", 2012/13 -> ~2 months
ago) + real-time ("tr", last ~2.5 months). We keep UTC timestamps (no DST ambiguity), resample to
hourly means, and keep gaps as NaN. Output: data/processed/fr_power_hourly.parquet with columns
(series_id, timestamp, value) - one row per hour per series.
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from tinycast.config import BenchConfig

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "rte"
PROCESSED = ROOT / "data" / "processed" / "fr_power_hourly.parquet"
API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
DATASETS = {
    "national_def": ("eco2mix-national-cons-def", False),
    "national_tr": ("eco2mix-national-tr", False),
    "regional_def": ("eco2mix-regional-cons-def", True),
    "regional_tr": ("eco2mix-regional-tr", True),
}


def _export_url(dataset: str, regional: bool) -> str:
    select = "date_heure,consommation" + (",libelle_region" if regional else "")
    q = urllib.parse.urlencode(
        {"select": select, "where": "consommation is not null", "timezone": "UTC"}
    )
    return f"{API}/{dataset}/exports/parquet?{q}"


def download(force: bool = False) -> dict[str, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (ds, regional) in DATASETS.items():
        path = RAW / f"{key}.parquet"
        if force or not path.exists() or key.endswith("_tr"):  # real-time data always refreshed
            t0 = time.time()
            req = urllib.request.Request(
                _export_url(ds, regional), headers={"User-Agent": "tinycast-research"}
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                path.write_bytes(r.read())
            print(
                f"  {key}: {path.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s",
                flush=True,
            )
        out[key] = path
    return out


def process(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    paths = paths or {k: RAW / f"{k}.parquet" for k in DATASETS}
    frames = []
    for key, p in paths.items():
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["date_heure"], utc=True)
        df["series_id"] = df.get("libelle_region", "France")
        df["source"] = "def" if key.endswith("_def") else "tr"
        frames.append(df[["series_id", "timestamp", "consommation", "source"]])
    df = pd.concat(frames)
    # consolidated wins over real-time where both exist
    df = df.sort_values("source").drop_duplicates(["series_id", "timestamp"], keep="first")
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = df.groupby(["series_id", "hour"], as_index=False)["consommation"].mean()
    out = []
    for sid, g in hourly.groupby("series_id"):
        idx = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h", tz="UTC")
        s = g.set_index("hour")["consommation"].reindex(idx)
        out.append(
            pd.DataFrame({"series_id": sid, "timestamp": idx, "value": s.to_numpy(np.float32)})
        )
    res = pd.concat(out, ignore_index=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(PROCESSED)
    return res


def load_hourly() -> pd.DataFrame:
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} missing: run `make data`")
    return pd.read_parquet(PROCESSED)


def series_arrays(df: pd.DataFrame | None = None) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    df = load_hourly() if df is None else df
    return {
        sid: (pd.DatetimeIndex(g["timestamp"]), g["value"].to_numpy(np.float32))
        for sid, g in df.groupby("series_id")
    }


def read_csv_bytes(b: bytes) -> pd.DataFrame:  # helper kept for tests / debugging
    return pd.read_csv(io.BytesIO(b))


def fresh_suite(cfg: BenchConfig, min_finite: float = 1.0):
    """Rolling-origin windows whose *targets* lie entirely in the fresh test window."""
    from tinycast.data.splits import load_splits
    from tinycast.data.windows import fresh_test_windows
    from tinycast.eval.benchmark import Suite

    sp = load_splits()
    wins = fresh_test_windows(
        series_arrays(), sp, cfg.context_length, cfg.horizon, min_finite=min_finite
    )
    return Suite(
        name="fresh_fr_power",
        ids=[w.window_id for w in wins],
        context=np.stack([w.context for w in wins]),
        future=np.stack([w.future for w in wins]),
        season=[24] * len(wins),
        meta=[dict(series_id=w.series_id, target_start=str(w.target_start)) for w in wins],
        flag=None,
    )
