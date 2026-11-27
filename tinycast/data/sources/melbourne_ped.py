"""City of Melbourne pedestrian counts (hourly, per sensor location), the fresh version of our pedestrian domain.

Source: City of Melbourne Open Data, "Pedestrian Counting System (counts per hour)", CC BY
(https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour). The portal keeps a
rolling two years (downloaded 2026-11-27: 2024-11-27 -> 2026-11-26, 102 locations). Same sensors as the Monash
"pedestrian counts" dataset (2009-2020) that teachers may have seen in pretraining; targets after splits.fresh_start were
published after every model we compare.

Times are local (Australia/Melbourne, `sensing_date` + `hourday`); we convert to UTC on a full hourly grid: missing hours
are NaN, the repeated hour of the autumn DST change is averaged.

    uv run python -m tinycast.data.sources.melbourne_ped   # download (refresh) the raw parquet
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from tinycast.eval.benchmark import ROOT

URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-monthly-counts-per-hour/exports/parquet"
)
RAW = ROOT / "data" / "raw" / "melbourne_ped" / "counts_per_hour.parquet"
MIN_COVERAGE = 0.8  # series kept only if >= 80% of the grid hours are observed


def download() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "tinycast-research"})
    with urllib.request.urlopen(req, timeout=600) as r:
        RAW.write_bytes(r.read())


def series_arrays() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    d = pd.read_parquet(RAW, columns=["location_id", "sensing_date", "hourday", "pedestriancount"])
    local = pd.to_datetime(d["sensing_date"]) + pd.to_timedelta(d["hourday"], unit="h")
    utc = local.dt.tz_localize(
        "Australia/Melbourne", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    # ambiguous hour (DST fall-back): both readings map to NaT above; place them on the earlier UTC hour instead
    amb = utc.isna()
    if amb.any():
        utc[amb] = (
            local[amb]
            .dt.tz_localize("Australia/Melbourne", ambiguous=True, nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
    d = pd.DataFrame(
        {"sid": d["location_id"].astype(str), "t": utc, "v": d["pedestriancount"].astype(float)}
    )
    d = d.groupby(["sid", "t"], as_index=False)["v"].mean()
    grid = pd.date_range(d["t"].min().floor("h"), d["t"].max().ceil("h"), freq="h", tz="UTC")
    out = {}
    for sid, g in d.groupby("sid"):
        v = g.set_index("t")["v"].reindex(grid).to_numpy(dtype=np.float32)
        if np.isfinite(v).mean() >= MIN_COVERAGE:
            out[f"mel{sid}"] = (grid, v)
    return dict(sorted(out.items()))


def fresh_suite(cfg):
    """Rolling daily origins (00:00 UTC) whose targets lie entirely after splits.fresh_start (same rule as RTE)."""
    from tinycast.data.splits import load_splits
    from tinycast.data.windows import fresh_test_windows
    from tinycast.eval.benchmark import Suite

    wins = fresh_test_windows(
        series_arrays(), load_splits(), cfg.context_length, cfg.horizon, source="melped"
    )
    return Suite(
        name="fresh_mel_ped",
        ids=[w.window_id for w in wins],
        context=np.stack([w.context for w in wins]),
        future=np.stack([w.future for w in wins]),
        season=[24] * len(wins),
        meta=[dict(series_id=w.series_id, target_start=str(w.target_start)) for w in wins],
        flag=None,
    )


if __name__ == "__main__":
    download()
    s = series_arrays()
    print(
        f"{len(s)} series kept, grid {next(iter(s.values()))[0][0]} -> {next(iter(s.values()))[0][-1]}"
    )
