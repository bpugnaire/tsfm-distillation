"""Forecast metrics. Pure numpy functions; every one has a hand-computed unit test.

Shapes used throughout:
    y        (B, H)      ground-truth future (NaN = missing, ignored)
    q        (B, K, H)   predicted quantiles at `levels` (K,)
    context  (B, T)      past values (NaN = missing), used for the MASE scale

Definitions (see docs/decisions/0002-metrics.md):

* pinball (quantile) loss   rho_tau(y, q) = max(tau (y - q), (tau - 1) (y - q))
* WQL  = mean_tau [ 2 * sum_t rho_tau(y_t, q_tau,t) / sum_t |y_t| ]
  This is GluonTS's `mean_weighted_sum_quantile_loss`, which GIFT-Eval reports as "CRPS".
  Why it approximates CRPS: CRPS(F, y) = 2 * integral_0^1 rho_tau(y, F^-1(tau)) dtau, so the
  mean of 2*rho over a uniform grid of levels is a Riemann sum of that integral. With only
  0.1..0.9 it ignores the tails, so it *underestimates* CRPS; tests/test_metrics.py checks
  the convergence against the closed-form Gaussian CRPS.
* MASE = mean_t |y_t - median_t| / mean |x_t - x_{t-m}| over the context (m = season, or 1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tinycast.config import CORE_LEVELS


def _level_index(levels: Sequence[float], wanted: Sequence[float]) -> list[int]:
    idx = []
    for w in wanted:
        hits = [i for i, lv in enumerate(levels) if abs(lv - w) < 1e-9]
        if not hits:
            raise ValueError(f"level {w} not in {tuple(levels)}")
        idx.append(hits[0])
    return idx


def pinball_loss(y: np.ndarray, q: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Element-wise pinball loss, shape (B, K, H). y: (B, H), q: (B, K, H)."""
    tau = np.asarray(levels, dtype=np.float64)[None, :, None]
    diff = y[:, None, :].astype(np.float64) - q.astype(np.float64)
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def wql(
    y: np.ndarray,
    q: np.ndarray,
    levels: Sequence[float],
    use_levels: Sequence[float] | None = CORE_LEVELS,
) -> np.ndarray:
    """Per-series weighted quantile loss, shape (B,). NaN in y is ignored.

    `use_levels` selects which levels enter the average (default: the 9 core levels shared
    by every model). Pass None to use all `levels`.
    """
    if use_levels is not None:
        idx = _level_index(levels, use_levels)
        q, levels = q[:, idx, :], [levels[i] for i in idx]
    mask = ~np.isnan(y)
    y0 = np.where(mask, y, 0.0)
    pl = pinball_loss(y0, q, levels) * mask[:, None, :]
    denom = np.abs(y0).sum(axis=-1)  # (B,)
    num = 2.0 * pl.sum(axis=-1).mean(axis=1)  # mean over levels of 2*sum_t rho
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, num / denom, np.nan)


def mase_scale(context: np.ndarray, season: int | None) -> np.ndarray:
    """In-sample mean absolute seasonal difference per series, shape (B,).

    Falls back to lag 1 when there is no seasonality or the context is shorter than 2 seasons.
    Returns NaN for series whose scale is 0 (e.g. a constant context): MASE is undefined there.
    """
    out = np.empty(context.shape[0])
    for b, x in enumerate(context):
        x = x[~np.isnan(x)]
        m = season if (season and len(x) > 2 * season) else 1
        d = np.abs(x[m:] - x[:-m]) if len(x) > m else np.array([])
        s = d.mean() if d.size else np.nan
        out[b] = s if s > 0 else np.nan
    return out


def mase(
    y: np.ndarray, q: np.ndarray, levels: Sequence[float], context: np.ndarray, season: int | None
) -> np.ndarray:
    """Per-series MASE of the median forecast, shape (B,)."""
    med = q[:, _level_index(levels, [0.5])[0], :]
    err = np.nanmean(np.abs(y - med), axis=-1)
    return err / mase_scale(context, season)


def coverage(
    y: np.ndarray, q: np.ndarray, levels: Sequence[float], lo: float, hi: float
) -> np.ndarray:
    """Per-series fraction of observed y inside [q_lo, q_hi], shape (B,)."""
    i_lo, i_hi = _level_index(levels, [lo, hi])
    inside = (y >= q[:, i_lo, :]) & (y <= q[:, i_hi, :])
    mask = ~np.isnan(y)
    return (inside & mask).sum(-1) / np.maximum(mask.sum(-1), 1)


def interval_width(
    q: np.ndarray, levels: Sequence[float], lo: float, hi: float, scale: np.ndarray
) -> np.ndarray:
    """Per-series mean width of [q_lo, q_hi] divided by `scale` (e.g. the MASE scale), (B,)."""
    i_lo, i_hi = _level_index(levels, [lo, hi])
    return (q[:, i_hi, :] - q[:, i_lo, :]).mean(-1) / scale


# --- fidelity to a teacher (Phases 4-5) -------------------------------------------------


def fidelity_quantile_loss(
    q_student: np.ndarray, q_teacher: np.ndarray, levels: Sequence[float], scale: np.ndarray
) -> np.ndarray:
    """Mean |q_s - q_t| over levels and horizon, divided by `scale`, shape (B,)."""
    return np.abs(q_student - q_teacher).mean(axis=(1, 2)) / scale


def fidelity_median_mae(
    q_student: np.ndarray, q_teacher: np.ndarray, levels: Sequence[float], scale: np.ndarray
) -> np.ndarray:
    i = _level_index(levels, [0.5])[0]
    return np.abs(q_student[:, i] - q_teacher[:, i]).mean(-1) / scale


# --- aggregation --------------------------------------------------------------------------


def geometric_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def relative_scores(model: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Per-series model/baseline ratio (GIFT-Eval style: aggregate with a geometric mean)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(model, dtype=np.float64) / np.asarray(baseline, dtype=np.float64)


def paired_ratio_ci(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Geometric mean of per-window ratios a/b (same windows) with a (cluster) bootstrap CI.

    Paired: both models are scored on the same windows, so window difficulty cancels out. This is far
    more sensitive than comparing two separately bootstrapped scores. < 1 means a is better.
    """
    return bootstrap_ci(relative_scores(a, b), "geomean", n_boot=n_boot, seed=seed, groups=groups)


def bootstrap_ci(
    x: np.ndarray,
    stat: str = "geomean",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """(point, lo, hi) of `stat` with a percentile bootstrap.

    Without `groups`, resamples items. With `groups` (e.g. the series id of each window),
    resamples whole groups (cluster bootstrap): windows of the same series are correlated
    (overlapping targets, shared regime), so resampling them independently gives CIs that
    are far too narrow.
    """
    x = np.asarray(x, dtype=np.float64)
    keep = np.isfinite(x) & ((x > 0) if stat == "geomean" else True)
    x = x[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    if stat == "geomean":
        f = geometric_mean
    elif stat == "mean":

        def f(v: np.ndarray) -> float:
            return float(np.mean(v))
    else:
        raise ValueError(stat)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    if g is None:
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        boots = np.array([f(x[i]) for i in idx])
    else:
        members = [np.flatnonzero(g == u) for u in np.unique(g)]
        k = len(members)
        boots = np.array(
            [
                f(x[np.concatenate([members[j] for j in rng.integers(0, k, k)])])
                for _ in range(n_boot)
            ]
        )
    return f(x), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
