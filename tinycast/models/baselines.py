"""Honest baselines: seasonal naive (empirical residual quantiles) and ETS (statsforecast).

Both estimate the season from the context (ACF) unless `season` is given, because the
student never sees the period either (drawn series have no time unit). NaNs in the context
are linearly interpolated for these classical models (they cannot take missing values).
"""

from __future__ import annotations

import warnings

import numpy as np

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.base import check_output


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    bad = np.isnan(x)
    if bad.all():
        raise ValueError("context is entirely missing")
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def estimate_season(x: np.ndarray, max_period: int | None = None, threshold: float = 0.3) -> int:
    """Seasonal period = lag of the highest local ACF peak above `threshold`, else 1."""
    x = interpolate_nans(x)
    x = x - np.linspace(x[0], x[-1], len(x))  # remove a linear trend first
    x = x - x.mean()
    n = len(x)
    max_period = max_period or n // 3
    denom = (x * x).sum()
    if denom <= 0 or max_period < 3:
        return 1
    acf = np.array([(x[: n - k] * x[k:]).sum() / denom for k in range(max_period + 1)])
    peaks = [k for k in range(2, max_period) if acf[k] > acf[k - 1] and acf[k] >= acf[k + 1]]
    peaks = [k for k in peaks if acf[k] > threshold]
    if not peaks:
        return 1
    best = max(acf[k] for k in peaks)
    # prefer the shortest lag that is nearly as good as the best (avoids picking 2x the period)
    return min(k for k in peaks if acf[k] >= 0.9 * best)


class SeasonalNaive:
    """y_{T+h} = y_{T+h-m*ceil(h/m)}; quantiles from in-sample seasonal-naive residuals,
    scaled by sqrt(ceil(h/m)) (random-walk-in-seasons error growth)."""

    native_levels = QUANTILE_LEVELS
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, name: str = "seasonal_naive", season: int | str = "auto"):
        self.name = name
        self.season = season

    def _season(self, i: int, x: np.ndarray) -> int:
        if isinstance(
            self.season, list | tuple
        ):  # per-series seasons (from each dataset's frequency)
            return int(self.season[i])
        return estimate_season(x) if self.season == "auto" else int(self.season)

    def _one(self, ctx: np.ndarray, h: int, i: int = 0) -> np.ndarray:
        x = interpolate_nans(ctx)
        m = self._season(i, x)
        m = m if len(x) > m else 1  # context shorter than one season: plain naive
        steps = np.arange(1, h + 1)
        k = np.ceil(steps / m).astype(int)  # seasons we must step back to reach an observed value
        # most recent observation with the same phase: x_{T-1+s-m*ceil(s/m)} (m=1 -> last value)
        point = x[len(x) - 1 + steps - m * k]
        resid = x[m:] - x[:-m]
        rq = np.quantile(resid, QUANTILE_LEVELS)  # (K,)
        return point[None, :] + rq[:, None] * np.sqrt(k)[None, :]

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        q = np.stack([self._one(c, horizon, i) for i, c in enumerate(context)])
        q = np.sort(q, axis=1)
        return check_output(q, len(context), horizon, self.name)


class ETS:
    """statsforecast AutoETS; its Gaussian prediction intervals -> quantiles."""

    native_levels = QUANTILE_LEVELS
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, name: str = "ets", season: int | str = "auto", max_season: int = 24):
        self.name = name
        self.season = season
        self.max_season = max_season
        self.fallbacks: list[str] = []  # written to the results JSON: never silent

    def _one(self, ctx: np.ndarray, h: int, i: int = 0) -> np.ndarray:
        from statsforecast.models import AutoETS

        x = interpolate_nans(ctx)
        m = SeasonalNaive._season(self, i, x)  # type: ignore[arg-type]
        if m > self.max_season:
            self.fallbacks.append(f"season {m} > {self.max_season}: non-seasonal ETS")
            m = 1
        # integer interval levels (statsforecast keys are 'lo-80', 'hi-98', ...)
        lvls = sorted({int(round(abs(2 * t - 1) * 100)) for t in QUANTILE_LEVELS if t != 0.5})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fc = AutoETS(season_length=m).forecast(y=x, h=h, level=lvls)
            except Exception as e:  # e.g. not enough data for the seasonal model
                self.fallbacks.append(f"seasonal fit failed ({type(e).__name__}): non-seasonal ETS")
                fc = AutoETS(season_length=1).forecast(y=x, h=h, level=lvls)
        out = []
        for t in QUANTILE_LEVELS:
            if t == 0.5:
                out.append(fc["mean"])
            else:
                lv = int(round(abs(2 * t - 1) * 100))
                key = f"lo-{lv}" if t < 0.5 else f"hi-{lv}"
                out.append(fc[key])
        return np.stack(out)  # non-finite values are rejected by check_output (no silent fix)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        q = np.stack([self._one(c, horizon, i) for i, c in enumerate(context)])
        return check_output(q, len(context), horizon, self.name)
