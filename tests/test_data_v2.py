import numpy as np
import pytest

from tinycast.data.sources import data_v2 as V
from tinycast.data.sources.hf_datasets import Series


@pytest.mark.parametrize(
    ("freq", "season"),
    [
        ("H", 24),
        ("1H", 24),
        ("15T", 96),
        ("15min", 96),
        ("30T", 48),
        ("5T", 288),
        ("D", 7),
        ("W-SUN", 52),
        ("M", 12),
        ("MS", 12),
        ("Q-DEC", 4),
        ("A-DEC", 1),
    ],
)
def test_season_from_frequency(freq, season):
    assert V.season_of(freq) == season


def _pool():
    rng = np.random.default_rng(0)
    big = [Series("big", str(k), rng.normal(size=5000).astype(np.float32), 24) for k in range(5)]
    small = [Series("small", "0", rng.normal(size=900).astype(np.float32), 24)]
    return big + small


def test_windows_v2_exact_count_no_repeats_and_redistribution():
    ctx, fut, ids, _, quota = V.windows_v2(_pool(), "train", 120, seed=0)
    assert len(ids) == len(set(ids)) == 120  # exact count, no repeated window
    assert (
        quota["small"] < 60 and quota["big"] == 120 - quota["small"]
    )  # small dataset exhausted, leftover moved
    by_series = {}
    for i in ids:
        key, pos = i.rsplit("@", 1)
        by_series.setdefault(key, []).append(int(pos))
    for pos in by_series.values():  # forecast parts of one series never overlap
        pos.sort()
        assert all(b - a >= 64 for a, b in zip(pos, pos[1:], strict=False))
    assert np.isfinite(fut).all()


def test_windows_v2_targets_inside_split():
    pool = _pool()
    lengths = {f"{s.dataset}/{s.series_id}": len(s.values) for s in pool}
    for split in ("train", "val"):
        _, _, ids, _, _ = V.windows_v2(pool, split, 50, seed=1)
        lo, hi = V.H.SPLITS[split]
        for i in ids:
            key, pos = i.rsplit("@", 1)
            assert int(pos) >= int(lo * lengths[key]) and int(pos) + 64 <= int(hi * lengths[key])
