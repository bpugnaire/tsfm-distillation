"""Training data v2 (Stage C2): more domains, energy included, contamination-filtered per series, fewer near-duplicates.

Pool = chronos_datasets training sets (minus the 4 GIFT-Eval test datasets and the held-out pedestrian counts)
     + the GiftEvalPretrain subsets listed in configs/data/gift_pretrain_v2.yaml.

Series-level filter: every 32-step chunk of the *train part* (first 70%) of a candidate series is hashed (z-normalised, as in
tinycast/eval/contamination.py) and looked up in a "forbidden" set built from GIFT-Eval + fev test data and our own target
series (RTE electricity, pedestrian counts). A series with ANY forbidden chunk is dropped. Result cached in
data/processed/data_v2_index.json (which series survive, and why the others did not).

Sampler: per dataset, origins on a grid of stride `horizon` (no two windows of a series share forecast steps), drawn without
replacement; equal quota per dataset, leftover quota of small datasets redistributed to the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import yaml

from tinycast.data.sources import hf_datasets as H
from tinycast.data.sources.hf_datasets import Series
from tinycast.eval.contamination import L, chunk_hashes

ROOT = Path(__file__).resolve().parents[3]
GP_DIR = ROOT / "data" / "raw" / "gift_pretrain"
INDEX = ROOT / "data" / "processed" / "data_v2_index.json"
DROP_CHRONOS = {
    "monash_covid_deaths",
    "monash_kdd_cup_2018",
    "monash_saugeenday",
    "monash_temperature_rain",
}  # GIFT test sets
MAX_SERIES = 1500
TRAIN_FRACTION = H.SPLITS["train"][1]


def season_of(freq: str) -> int:
    f = freq.upper().replace("MIN", "T")
    num = int("".join(c for c in f if c.isdigit()) or 1)
    unit = "".join(c for c in f if c.isalpha())[:1]
    per_day = {
        "T": 1440 // num,
        "H": 24 // max(num, 1),
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "A": 1,
        "Y": 1,
        "S": 86400 // num,
    }
    return max(1, per_day.get(unit, 1))


def _arrow_series(
    path: Path, name: str, rng: np.random.Generator, prefix: str = ""
) -> list[Series]:
    with ipc.open_stream(path) as r:
        t = r.read_all()
    freq = t.column("freq")[0].as_py() if "freq" in t.column_names else "H"
    out = []
    for i, row in enumerate(t.column("target").to_pylist()):
        rows = (
            row if row and isinstance(row[0], list) else [row]
        )  # multivariate: one series per variate
        for j, v in enumerate(rows):
            if v is None or len(v) < 128:
                continue
            out.append(
                Series(
                    name,
                    f"{prefix}{i}.{j}",
                    np.array([np.nan if x is None else x for x in v], np.float32),
                    season_of(freq),
                )
            )
    if len(out) > MAX_SERIES:
        out = [out[k] for k in sorted(rng.choice(len(out), MAX_SERIES, replace=False))]
    return out


def load_gift_pretrain(cfg_path: Path | None = None) -> list[Series]:
    cfg = yaml.safe_load(
        (cfg_path or ROOT / "configs" / "data" / "gift_pretrain_v2.yaml").read_text()
    )
    rng = np.random.default_rng(0)
    series = []
    for name in cfg["subsets"]:
        # a subset can span several arrow files: ids must stay unique across files (file 0 keeps its plain ids)
        for k, f in enumerate(sorted((GP_DIR / name).rglob("data-*.arrow"))):
            series += _arrow_series(f, name, rng, prefix=f"f{k}:" if k else "")
    return series


def _forbidden_hashes() -> np.ndarray:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc", ROOT / "scripts" / "contamination_check.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)  # type: ignore[union-attr]
    hs = []
    for loader in (cc.gift_eval, cc.fev):
        for series in loader().values():
            hs += [chunk_hashes(s, stride=L) for s in series]
    from tinycast.data.sources.rte_eco2mix import series_arrays

    hs += [chunk_hashes(v, stride=L) for _, v in series_arrays().values()]
    hs += [chunk_hashes(s.values, stride=L) for s in H.load("monash_pedestrian_counts")]
    return np.unique(np.concatenate(hs))


def _any_in_sorted(h: np.ndarray, sorted_set: np.ndarray) -> bool:
    """True if any element of h is in sorted_set (binary search; np.isin would re-sort the big set on every call)."""
    if len(h) == 0 or len(sorted_set) == 0:
        return False
    i = np.minimum(np.searchsorted(sorted_set, h), len(sorted_set) - 1)
    return bool((sorted_set[i] == h).any())


def build_index(log=print) -> dict:
    """Load the candidate pool, drop contaminated series, cache the list of surviving series ids."""
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    forbidden = _forbidden_hashes()
    log(f"forbidden chunk hashes: {len(forbidden):,}")
    pool = [s for name in H.DATASETS if name not in DROP_CHRONOS | H.HELD_OUT for s in H.load(name)]
    pool += load_gift_pretrain()
    log(f"candidate series: {len(pool):,}")
    keep, dropped = [], {}
    for n, s in enumerate(pool):
        if n % 5000 == 0:
            log(f"  checked {n:,}/{len(pool):,}")
        h = chunk_hashes(s.values[: int(TRAIN_FRACTION * len(s.values))], stride=1)
        hit = _any_in_sorted(h, forbidden)
        if hit:
            dropped[s.dataset] = dropped.get(s.dataset, 0) + 1
        else:
            keep.append(f"{s.dataset}/{s.series_id}")
    if len(set(keep)) != len(keep):
        raise ValueError("duplicate series ids in the data v2 pool")
    idx = {
        "kept": keep,
        "dropped_by_dataset": dropped,
        "n_candidates": len(pool),
        "n_forbidden_hashes": int(len(forbidden)),
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(idx))
    return idx


def load_v2() -> list[Series]:
    keep = set(build_index()["kept"])
    pool = [s for name in H.DATASETS if name not in DROP_CHRONOS | H.HELD_OUT for s in H.load(name)]
    pool += load_gift_pretrain()
    out = [s for s in pool if f"{s.dataset}/{s.series_id}" in keep]
    keys = [f"{s.dataset}/{s.series_id}" for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "duplicate series ids in data v2 (rebuild data/processed/data_v2_index.json)"
        )
    return out


def windows_v2(
    series: list[Series],
    split: str,
    n: int,
    seed: int,
    context_length: int = 512,
    horizon: int = 64,
):
    """Exactly n windows (or all available), targets inside the split, origins on a stride-`horizon` grid, no repeats."""
    lo, hi = H.SPLITS[split]
    rng = np.random.default_rng([seed, 47, list(H.SPLITS).index(split)])
    grids: dict[str, list[tuple[int, int]]] = {}
    for k, s in enumerate(series):
        Lx = len(s.values)
        first, last = max(64, int(lo * Lx)), int(hi * Lx) - horizon
        if last < first:
            continue
        offset = int(rng.integers(0, horizon))
        for i in range(first + offset, last + 1, horizon):
            if (
                np.isfinite(s.values[i : i + horizon]).mean() >= 0.9
                and np.isfinite(s.values[max(0, i - context_length) : i]).sum() >= 32
            ):
                grids.setdefault(s.dataset, []).append((k, i))
    names = sorted(grids)
    for nm in names:
        rng.shuffle(grids[nm])
    quota = {nm: 0 for nm in names}
    remaining, active = n, [nm for nm in names]
    while (
        remaining > 0 and active
    ):  # equal shares; small datasets give their leftover to the others
        share = max(1, remaining // len(active))
        for nm in list(active):
            take = min(share, len(grids[nm]) - quota[nm], remaining)
            quota[nm] += take
            remaining -= take
            if quota[nm] >= len(grids[nm]):
                active.remove(nm)
            if remaining == 0:
                break
    picks = [p for nm in names for p in grids[nm][: quota[nm]]]
    ctx = np.full((len(picks), context_length), np.nan, dtype=np.float32)
    fut = np.empty((len(picks), horizon), dtype=np.float32)
    ids, seasons = [], []
    for j, (k, i) in enumerate(picks):
        v = series[k].values
        c = v[max(0, i - context_length) : i]
        ctx[j, context_length - len(c) :] = c
        fut[j] = v[i : i + horizon]
        ids.append(f"{series[k].dataset}/{series[k].series_id}@{i}")
        seasons.append(series[k].season)
    return ctx, fut, ids, seasons, {nm: quota[nm] for nm in names}
