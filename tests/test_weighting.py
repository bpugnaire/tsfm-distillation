import numpy as np
import pytest

from tinycast.config import QUANTILE_LEVELS
from tinycast.distill import weighting as W


def test_parse():
    assert W.parse("t0-gamma") == ("single", ["t0-gamma"], None)
    assert W.parse("ensemble:a+b") == ("ensemble", ["a", "b"], None)
    assert W.parse("weighted:a+b+c@20") == ("weighted", ["a", "b", "c"], 20.0)
    with pytest.raises(ValueError):
        W.parse("weighted:a+b")  # needs a temperature


def test_ensemble_is_levelwise_mean():
    a, b = np.zeros((2, 13, 4)), np.full((2, 13, 4), 2.0)
    q, _ = W.combine("ensemble", [a, b], None, QUANTILE_LEVELS, None)
    np.testing.assert_allclose(q, 1.0)


def test_weighted_prefers_the_accurate_teacher_per_window():
    y = np.array([[10.0] * 4, [20.0] * 4])
    good = np.repeat(y[:, None, :], 13, 1)  # perfect on both windows
    bad = good + 5.0
    q, diag = W.combine("weighted", [bad, good], y, QUANTILE_LEVELS, tau=50.0)
    np.testing.assert_allclose(q, good, atol=1e-3)
    assert diag["best_teacher_share"] == [0, 2]
    q0, _ = W.combine(
        "weighted", [bad, good], y, QUANTILE_LEVELS, tau=0.0
    )  # tau = 0 -> plain average
    np.testing.assert_allclose(q0, (bad + good) / 2)


def test_weighted_needs_ground_truth():
    a = np.zeros((1, 13, 4))
    with pytest.raises(ValueError):
        W.combine("weighted", [a, a], np.full((1, 4), np.nan), QUANTILE_LEVELS, 1.0)


def test_outputs_are_float32():
    y = np.array([[1.0, 2.0]])
    a = np.zeros((1, 13, 2), dtype=np.float32)
    for kind, tau in (("ensemble", None), ("weighted", 3.0), ("single", None)):
        q, _ = W.combine(kind, [a, a + 1], y, QUANTILE_LEVELS, tau)
        assert q.dtype == np.float32, kind
