"""Multi-domain real series from autogluon/chronos_datasets (parquet), for the distillation study.

Selection rules (ADR 0005): explicit licence only; NO energy-domain data (electricity, smart meters, solar,
wind), so French electricity stays out-of-domain for the student; varied domains and frequencies.
`pedestrian_counts` is held out of distillation entirely: it is a second "your data" target domain.

Per series, targets are split by position (the datasets end at different dates, so a common calendar cutoff
is impossible): train <= 70% of the series, validation 70-85%, test 85-100%.
Teachers may have seen these public datasets in pretraining: fine for training the student; test numbers
on them are flagged "possibly seen by teachers".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "chronos_datasets"
REPO = "autogluon/chronos_datasets"

# name -> (licence, sampling frequency, season for baselines from the frequency, target column)
# Dropped after inspection (too short for a 64-step horizon): monash_hospital (84), monash_m3_monthly (68),
# monash_tourism_quarterly (63). monash_covid_deaths (212) yields train windows only.
DATASETS: dict[str, tuple[str, str, int, str]] = {
    "monash_traffic": ("CC BY 4.0", "1h", 24, "target"),
    "monash_kdd_cup_2018": ("CC BY 4.0", "1h", 24, "target"),
    "monash_rideshare": ("CC BY 4.0", "1h", 24, "price_mean"),
    "taxi_30min": ("Apache 2.0", "30min", 48, "target"),
    "monash_weather": ("CC BY 4.0", "1D", 7, "target"),
    "monash_temperature_rain": ("CC BY 4.0", "1D", 7, "t_mean"),
    "monash_saugeenday": ("CC BY 4.0", "1D", 7, "T1"),
    "monash_covid_deaths": ("CC BY 4.0", "1D", 7, "target"),
    "exchange_rate": ("MIT", "1D", 5, "target"),
    "monash_tourism_monthly": ("CC BY 4.0", "1M", 12, "target"),
    "monash_fred_md": ("CC BY 4.0", "1M", 12, "target"),
    "monash_pedestrian_counts": ("CC BY 4.0", "1h", 24, "target"),  # HELD OUT: target domain #2
}
HELD_OUT = {"monash_pedestrian_counts"}
SPLITS = {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}
MAX_SERIES_PER_DATASET = 400  # keeps any single large dataset from dominating


@dataclass
class Series:
    dataset: str
    series_id: str
    values: np.ndarray  # float32, NaN = missing
    season: int


def download(names: list[str] | None = None) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    out = []
    for name in names or list(DATASETS):
        files = [
            f
            for f in api.list_repo_files(REPO, repo_type="dataset")
            if f.startswith(f"{name}/") and f.endswith(".parquet")
        ]
        if not files:
            raise FileNotFoundError(f"{name}: no parquet files in {REPO}")
        for f in files:
            out.append(Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW)))
    return out


def load(name: str) -> list[Series]:
    files = sorted((RAW / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{name} not downloaded: run scripts/download_data.py --multidomain"
        )
    df = pd.concat(pd.read_parquet(f) for f in files)
    target_col = DATASETS[name][3]
    rng = np.random.default_rng(0)
    rows = df.iloc[rng.permutation(len(df))[:MAX_SERIES_PER_DATASET]]
    season = DATASETS[name][2]
    return [
        Series(name, str(r["id"]), np.asarray(r[target_col], dtype=np.float32), season)
        for _, r in rows.iterrows()
        if len(r[target_col]) >= 128
    ]


def windows(
    series: list[Series],
    split: str,
    n: int,
    seed: int,
    context_length: int = 512,
    horizon: int = 64,
    min_context: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Uniform random windows whose *targets* lie in the split's position range. Short histories are left-padded with NaN."""
    lo, hi = SPLITS[split]
    rng = np.random.default_rng([seed, 31, list(SPLITS).index(split)])
    # Balanced by dataset: pick a dataset uniformly, then a window uniformly inside it. Without this,
    # datasets with long series (traffic, weather) would dominate both training and evaluation.
    by_ds: dict[str, list[tuple[int, int]]] = {}
    for k, s in enumerate(series):
        L = len(s.values)
        first = max(min_context, int(lo * L))
        last = int(hi * L) - horizon  # target [i, i+H) must end inside the split
        if last >= first:
            by_ds.setdefault(s.dataset, []).append((k, first, last))  # type: ignore[arg-type]
    if not by_ds:
        raise ValueError(f"no {split} windows")
    names = sorted(by_ds)
    pick = []
    # Draw with a margin, then keep exactly n windows whose future is >= 90% observed: arms of a study
    # must have identical sizes, and downstream code relies on getting what it asked for.
    while len(pick) < n:
        for _ in range(int(1.3 * (n - len(pick))) + 10):
            spans = by_ds[names[rng.integers(len(names))]]
            k, first, last = spans[rng.integers(len(spans))]  # type: ignore[misc]
            i = int(rng.integers(first, last + 1))
            if np.isfinite(series[k].values[i : i + horizon]).mean() >= 0.9:
                pick.append((k, i))
    pick = pick[:n]
    ctx = np.full((len(pick), context_length), np.nan, dtype=np.float32)
    fut = np.empty((len(pick), horizon), dtype=np.float32)
    ids, seasons = [], []
    for j, (k, i) in enumerate(pick):
        v = series[k].values
        c = v[max(0, i - context_length) : i]
        ctx[j, context_length - len(c) :] = c
        fut[j] = v[i : i + horizon]
        ids.append(f"{series[k].dataset}/{series[k].series_id}@{i}")
        seasons.append(series[k].season)
    return ctx, fut, ids, seasons


def load_all(include_held_out: bool = False) -> list[Series]:
    return [s for name in DATASETS if include_held_out or name not in HELD_OUT for s in load(name)]
