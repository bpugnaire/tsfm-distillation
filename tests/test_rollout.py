import numpy as np

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.rollout import RolloutForecaster


class Flat:
    """Forecasts the last value with fixed +-k spreads: easy to check by hand."""

    n_params = 0

    def forecast(self, context, horizon):
        last = np.asarray(context)[:, -1]
        offs = np.array([(lv - 0.5) * 10 for lv in QUANTILE_LEVELS], dtype=np.float32)
        return (last[:, None, None] + offs[None, :, None] + np.zeros((1, 1, horizon))).astype(
            np.float32
        )


def test_short_horizon_is_the_base_model():
    x = np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32)
    for mode in ("median", "paths"):
        r = RolloutForecaster(Flat(), "r", mode=mode)
        assert np.allclose(r.forecast(x, 64), Flat().forecast(x, 64))


def test_median_feedback_keeps_level_and_width():
    x = np.full((2, 512), 3.0, dtype=np.float32)
    q = RolloutForecaster(Flat(), "r", mode="median").forecast(x, 200)
    assert q.shape == (2, len(QUANTILE_LEVELS), 200)
    assert np.allclose(q[:, QUANTILE_LEVELS.index(0.5)], 3.0)
    assert np.allclose(q[:, -1, -1] - q[:, 0, -1], q[:, -1, 0] - q[:, 0, 0])  # bands never widen


def test_paths_widen_with_horizon_and_stay_monotone():
    x = np.zeros((2, 512), dtype=np.float32)
    q = RolloutForecaster(Flat(), "r", mode="paths", n_paths=400, seed=1).forecast(x, 256)
    assert (np.diff(q, axis=1) >= -1e-6).all()
    w = q[:, QUANTILE_LEVELS.index(0.9)] - q[:, QUANTILE_LEVELS.index(0.1)]
    assert w[:, 200:].mean() > 1.5 * w[:, :40].mean()  # uncertainty accumulates across blocks
