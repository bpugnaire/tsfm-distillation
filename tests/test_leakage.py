"""No evaluation leakage (CLAUDE.md ground rule 2). Runs on synthetic timestamps so it is fast and
does not need the downloaded data; a second test checks the real data when it is present."""

import numpy as np
import pandas as pd
import pytest

from tinycast.data import shapes as S
from tinycast.data.splits import load_splits
from tinycast.data.windows import fresh_test_windows, sample_windows

SP = load_splits()


def _fake_series(start="2025-06-01", end="2026-11-10"):
    ts = pd.date_range(start, end, freq="h", tz="UTC")
    v = (10 + np.sin(np.arange(len(ts)) * 2 * np.pi / 24)).astype(np.float32)
    return {"A": (ts, v), "B": (ts, v * 2)}


def test_fresh_start_after_every_model_release():
    for name, released in SP.model_releases.items():
        assert SP.fresh_start > released, f"fresh split starts before {name} was released"


def test_split_order():
    assert SP.train_end < SP.val_start <= SP.val_end < SP.fresh_start


def test_train_and_val_targets_never_reach_val_end_or_fresh():
    series = _fake_series()
    for split, end in (("train", SP.train_end), ("val", SP.val_end)):
        ws = sample_windows(series, split, SP, n_windows=500, seed=1)
        assert ws, split
        assert all(w.target_end <= end for w in ws)
        assert all(w.target_end < SP.fresh_start for w in ws)
    val = sample_windows(series, "val", SP, n_windows=500, seed=1)
    assert all(w.target_start >= SP.val_start for w in val)


def test_fresh_windows_disjoint_from_training_targets():
    series = _fake_series()
    fresh = fresh_test_windows(series, SP)
    assert fresh and all(w.target_start >= SP.fresh_start for w in fresh)
    train = sample_windows(series, "train", SP, n_windows=2000, seed=0)
    assert not ({w.window_id for w in fresh} & {w.window_id for w in train})
    assert max(w.target_end for w in train) < min(w.target_start for w in fresh)


def test_window_ids_stable():
    series = _fake_series()
    a = sample_windows(series, "train", SP, n_windows=50, seed=3)
    b = sample_windows(series, "train", SP, n_windows=50, seed=3)
    assert [w.window_id for w in a] == [w.window_id for w in b]


def test_gallery_seeds_disjoint_from_training_seeds():
    from tinycast.data.synthetic import TRAIN_SEED_MAX

    assert TRAIN_SEED_MAX <= SP.gallery_seed_offset == S.GALLERY_SEED_OFFSET
    assert all(s.meta["seed"] >= SP.gallery_seed_offset for s in S.gallery())


def test_real_data_splits_if_present():
    from tinycast.data.sources import rte_eco2mix

    if not rte_eco2mix.PROCESSED.exists():
        pytest.skip("real data not downloaded")
    series = rte_eco2mix.series_arrays()
    fresh = fresh_test_windows(series, SP)
    assert fresh and all(w.target_start >= SP.fresh_start for w in fresh)
    val = sample_windows(series, "val", SP, n_windows=200, seed=0)
    assert all(SP.val_start <= w.target_start and w.target_end <= SP.val_end for w in val)


def test_multidomain_windows_exact_count_and_split_positions():
    from tinycast.data.sources import hf_datasets as H

    try:
        series = H.load_all()
    except FileNotFoundError:
        pytest.skip("multidomain data not downloaded")
    for split in ("train", "val", "test"):
        ctx, fut, ids, _ = H.windows(series, split, 500, seed=1)
        assert len(ctx) == len(ids) == 500
        assert (np.isfinite(fut).mean(1) >= 0.9).all()
        lo, hi = H.SPLITS[split]
        lengths = {f"{s.dataset}/{s.series_id}": len(s.values) for s in series}
        for i in ids:
            key, start = i.rsplit("@", 1)
            L = lengths[key]
            assert int(start) >= int(lo * L) and int(start) + 64 <= int(
                hi * L
            )  # target inside the split
    assert not any(s.dataset in H.HELD_OUT for s in series)


@pytest.mark.parametrize("history", [100, 168, 720, 8760])
def test_finetune_history_origins_never_reach_the_test(history):
    from tinycast.data.target_domain import history_origins

    end = 20000
    start, origins = history_origins(end, history, min_context=32, horizon=64)
    assert start == end - history
    assert (origins + 64 <= end).all()  # every target ends before the test starts
    assert (origins - 32 >= start).all()  # contexts only use the user's history window


def test_finetune_history_real_domains():
    from tinycast.data import target_domain as T
    from tinycast.data.splits import load_splits

    try:
        tr, va = T.history_corpora("fr_power", history=720, max_windows=2000)
        series = T._domain_series("fr_power")
    except FileNotFoundError:
        pytest.skip("data not downloaded")
    assert len(tr.ids) and len(va.ids) and not set(tr.ids) & set(va.ids)
    from tinycast.data.sources.rte_eco2mix import series_arrays

    ts, _ = series_arrays()[series[0][0]]
    assert (
        ts[series[0][2] - 1] < load_splits().fresh_start <= ts[series[0][2]]
    )  # history ends right before fresh


def test_finetune_warm_mode_targets_in_history_contexts_from_past():
    from tinycast.data import target_domain as T

    try:
        tr, va = T.history_corpora("pedestrian", history=168, max_windows=5000, mode="warm")
        cold, _ = T.history_corpora("pedestrian", history=168, max_windows=5000, mode="cold")
    except FileNotFoundError:
        pytest.skip("data not downloaded")
    # warm contexts use the full past (512 observed steps); cold contexts only the 1-week window
    assert np.isfinite(tr.context).sum(1).mean() > 400
    assert np.isfinite(cold.context).sum(1).max() <= 168
    assert len(tr.ids) and len(va.ids) and not set(tr.ids) & set(va.ids)
