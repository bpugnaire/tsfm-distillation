import numpy as np
import pytest

from tinycast.config import QUANTILE_LEVELS
from tinycast.distill.weighting import combine, parse


def test_parse_margin():
    assert parse("margin:timestral-5m+timestral-209m@0.1") == (
        "margin",
        ["timestral-5m", "timestral-209m"],
        0.1,
    )
    with pytest.raises(ValueError):
        parse("margin:a+b+c@0.1")
    with pytest.raises(ValueError):
        parse("margin:a+b@1.5")


def test_margin_picks_b_only_when_clearly_better():
    K, H = len(QUANTILE_LEVELS), 8
    y = np.full((3, H), 10.0, dtype=np.float32)
    a = np.full((3, K, H), 11.0, dtype=np.float32)  # teacher a: flat forecast, off by 1
    # teacher b off by 0.5 (50% better), 0.95 (5% better), 2 (worse): WQL is linear in the offset here
    b = np.stack([np.full((K, H), 10.0 + v, dtype=np.float32) for v in (0.5, 0.95, 2.0)])
    out, diag = combine("margin", [a, b], y, QUANTILE_LEVELS, 0.10)
    assert np.array_equal(out[0], b[0])  # >= 10% better -> b
    assert np.array_equal(out[1], a[1])  # only 5% better -> a
    assert np.array_equal(out[2], a[2])  # worse -> a
    assert diag["n_b"] == 1


def test_margin_without_ground_truth_keeps_a():
    K, H = len(QUANTILE_LEVELS), 4
    y = np.full((1, H), np.nan, dtype=np.float32)
    y2 = np.zeros((1, H), dtype=np.float32)
    a, b = np.ones((1, K, H), np.float32), np.zeros((1, K, H), np.float32)
    out, _ = combine(
        "margin",
        [
            a,
            a,
        ],
        y2,
        QUANTILE_LEVELS,
        0.1,
    )  # sanity: identical teachers -> a
    assert np.array_equal(out, a)
    with pytest.raises(ValueError):  # no ground truth at all: refuses, never silently
        combine("margin", [a, b], y, QUANTILE_LEVELS, 0.1)
