import numpy as np
import pytest
from scipy.stats import norm

from tinycast.config import CORE_LEVELS, QUANTILE_LEVELS
from tinycast.eval import metrics as M


def test_pinball_hand_computed():
    y = np.array([[10.0]])
    q = np.array([[[8.0], [12.0]]])  # (B=1, K=2, H=1)
    pl = M.pinball_loss(y, q, [0.9, 0.9])
    # under-prediction by 2 at tau=.9 costs .9*2; over-prediction by 2 costs .1*2
    assert pl[0, 0, 0] == pytest.approx(1.8)
    assert pl[0, 1, 0] == pytest.approx(0.2)


def test_wql_hand_computed():
    y = np.array([[1.0, 3.0]])
    levels = CORE_LEVELS
    # every quantile over-predicts by 1 -> rho_tau = (1 - tau) per step
    q = np.repeat((y + 1.0)[:, None, :], len(levels), axis=1)
    # sum_t rho = 2(1-tau); mean_tau of 2 * 2(1-tau) = 4 * 0.5 = 2; / sum|y| = 4 -> 0.5
    assert M.wql(y, q, levels)[0] == pytest.approx(0.5)


def test_wql_perfect_and_nan_handling():
    y = np.array([[2.0, np.nan, 2.0]])
    q = np.full((1, len(QUANTILE_LEVELS), 3), 2.0)
    assert M.wql(y, q, QUANTILE_LEVELS)[0] == pytest.approx(0.0)
    assert np.isnan(M.wql(np.zeros((1, 3)), q, QUANTILE_LEVELS)[0])  # sum|y|=0 -> undefined


def test_wql_selects_core_levels():
    y = np.array([[1.0]])
    q = np.ones((1, len(QUANTILE_LEVELS), 1))
    q[0, 0, 0] = -100.0  # wildly wrong 1% quantile must not affect the 9-level WQL
    assert M.wql(y, q, QUANTILE_LEVELS)[0] == pytest.approx(0.0)
    assert M.wql(y, q, QUANTILE_LEVELS, use_levels=None)[0] > 0


def _gaussian_crps(y, mu=0.0, sigma=1.0):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


@pytest.mark.parametrize("y_val", [-1.5, 0.0, 0.3, 2.0])
def test_quantile_loss_converges_to_gaussian_crps(y_val):
    """2 * mean_tau rho_tau over a dense uniform grid -> CRPS (closed form for a Gaussian)."""
    levels = np.linspace(0.0005, 0.9995, 1000)
    q = norm.ppf(levels)[None, :, None]
    y = np.array([[y_val]])
    approx = 2 * M.pinball_loss(y, q, levels).mean()
    assert approx == pytest.approx(_gaussian_crps(y_val), abs=2e-3)


def test_level_grid_bias_of_crps_estimator():
    """Equal-weight quantile-loss averages are biased, and the sign depends on the grid.

    For a calibrated Gaussian, E[rho_tau] = sigma * phi(z_tau), largest at the median. So:
    * GIFT-Eval's 9 levels (0.1..0.9) drop the *small* tail terms -> biased HIGH (~+9%);
    * our 13-level grid over-counts tail levels -> biased LOW (~-17%).
    See docs/decisions/0002-metrics.md.
    """
    exact = 1 / np.sqrt(np.pi)  # E_y CRPS(N(0,1), y) for y ~ N(0,1)
    rng = np.random.default_rng(0)
    ys = rng.standard_normal((20000, 1))
    assert _gaussian_crps(ys).mean() == pytest.approx(exact, abs=0.01)

    def estimator(levels):
        q = np.repeat(norm.ppf(levels)[None, :, None], len(ys), 0)
        return 2 * M.pinball_loss(ys, q, levels).mean()

    def analytic(levels):
        return 2 * np.mean(norm.pdf(norm.ppf(levels)))

    for levels in (CORE_LEVELS, QUANTILE_LEVELS):
        assert estimator(levels) == pytest.approx(analytic(levels), rel=0.02)
    assert analytic(CORE_LEVELS) / exact == pytest.approx(1.09, abs=0.01)
    assert analytic(QUANTILE_LEVELS) / exact == pytest.approx(0.825, abs=0.01)


def test_mase_hand_computed():
    ctx = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # lag-1 MAE = 1
    y = np.array([[6.0, 7.0]])
    q = np.zeros((1, len(QUANTILE_LEVELS), 2))
    q[0, QUANTILE_LEVELS.index(0.5)] = [6.0, 8.0]
    assert M.mase(y, q, QUANTILE_LEVELS, ctx, season=None)[0] == pytest.approx(0.5)


def test_mase_seasonal_scale_and_fallback():
    ctx = np.tile([0.0, 10.0], 10)[None]  # perfectly periodic with m=2 -> seasonal scale 0
    assert np.isnan(M.mase_scale(ctx, season=2)[0])
    assert M.mase_scale(ctx, season=None)[0] == pytest.approx(10.0)
    assert M.mase_scale(ctx, season=50)[0] == pytest.approx(10.0)  # too short: lag-1 fallback


def test_coverage_and_width():
    levels = QUANTILE_LEVELS
    y = np.array([[0.0, 5.0, 10.0, 20.0]])
    q = np.zeros((1, len(levels), 4))
    q[0, levels.index(0.1)] = 1.0
    q[0, levels.index(0.9)] = 11.0
    assert M.coverage(y, q, levels, 0.1, 0.9)[0] == pytest.approx(0.5)  # 5 and 10 inside
    assert M.interval_width(q, levels, 0.1, 0.9, np.array([2.0]))[0] == pytest.approx(5.0)


def test_fidelity():
    levels = QUANTILE_LEVELS
    qs = np.zeros((1, len(levels), 2))
    qt = np.ones((1, len(levels), 2))
    assert M.fidelity_quantile_loss(qs, qt, levels, np.array([2.0]))[0] == pytest.approx(0.5)
    assert M.fidelity_median_mae(qs, qt, levels, np.array([1.0]))[0] == pytest.approx(1.0)


def test_aggregation():
    assert M.geometric_mean(np.array([1.0, 4.0])) == pytest.approx(2.0)
    rel = M.relative_scores(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    np.testing.assert_allclose(rel, [0.5, 1.0])
    point, lo, hi = M.bootstrap_ci(np.full(50, 3.0))
    assert point == lo == hi == pytest.approx(3.0)
    point, lo, hi = M.bootstrap_ci(np.random.default_rng(0).lognormal(size=200))
    assert lo < point < hi


def test_cluster_bootstrap_is_wider_for_correlated_items():
    rng = np.random.default_rng(0)
    # 5 clusters, 40 near-identical items each: effective sample size is 5, not 200
    cluster_level = rng.lognormal(0, 0.5, 5)
    x = np.repeat(cluster_level, 40) * rng.lognormal(0, 0.01, 200)
    groups = np.repeat(np.arange(5), 40)
    _, lo_i, hi_i = M.bootstrap_ci(x)
    _, lo_c, hi_c = M.bootstrap_ci(x, groups=groups)
    assert (hi_c - lo_c) > 3 * (hi_i - lo_i)


def test_seasonal_naive_uses_most_recent_season():
    from tinycast.models.baselines import SeasonalNaive

    x = np.array([[0.0, 1, 2, 10, 11, 12, 20, 21, 22]])  # m=3: last season = [20, 21, 22]
    q = SeasonalNaive(season=3).forecast(x, 7)
    median = q[0, QUANTILE_LEVELS.index(0.5)]
    resid_med = np.median([10.0] * 6)  # every seasonal difference is 10
    np.testing.assert_allclose(
        median - resid_med * np.sqrt([1, 1, 1, 2, 2, 2, 3]), [20, 21, 22, 20, 21, 22, 20]
    )
    q1 = SeasonalNaive(season=1).forecast(x, 3)  # m=1: random-walk naive = last value
    assert q1[0, QUANTILE_LEVELS.index(0.5), 0] == pytest.approx(22 + np.median(np.diff(x[0])))
