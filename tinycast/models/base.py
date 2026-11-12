"""The Forecaster protocol every model (baseline, teacher, comparator, student) implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tinycast.config import QUANTILE_LEVELS


@runtime_checkable
class Forecaster(Protocol):
    name: str
    quantile_levels: tuple[float, ...]
    # Levels the model produces itself; the others are interpolated/extrapolated by the adapter.
    native_levels: tuple[float, ...]

    def forecast(
        self,
        context: np.ndarray,  # (batch, T) float32, NaN = missing
        horizon: int,
        past_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> np.ndarray:  # (batch, len(quantile_levels), horizon)
        ...


def check_output(q: np.ndarray, batch: int, horizon: int, name: str) -> np.ndarray:
    """Validate an adapter's output; raise instead of silently fixing (CLAUDE.md section 11)."""
    expected = (batch, len(QUANTILE_LEVELS), horizon)
    if q.shape != expected:
        raise ValueError(f"{name}: output shape {q.shape} != {expected}")
    if not np.isfinite(q).all():
        raise ValueError(f"{name}: non-finite quantiles")
    if (np.diff(q, axis=1) < -1e-4 * (1 + np.abs(q[:, 1:]))).any():
        raise ValueError(f"{name}: quantiles are not monotone in the level axis")
    return q.astype(np.float32)


def extend_tails_exponential(
    knot_levels: tuple[float, ...], knot_values: np.ndarray, target_levels: tuple[float, ...]
) -> np.ndarray:
    """Map quantiles at `knot_levels` (..., K) to `target_levels` (..., L).

    Inside the knot range: linear interpolation in the level. Outside: the IQF exponential
    tails of Park et al. (arXiv 2111.06581), the same rule t0 uses: linear in log(q) on the
    left, in log(1-q) on the right, slopes from the two outermost knots, clamped at >= 0 so
    the result stays monotone. Used for models that do not emit 0.01/0.05/0.95/0.99 natively
    (Toto 2.0, Chronos-2 if needed). Always recorded in `native_levels`, never silent.
    """
    k = np.asarray(knot_levels, dtype=np.float64)
    v = np.asarray(knot_values, dtype=np.float64)
    out = []
    for lv in target_levels:
        if k[0] - 1e-9 <= lv <= k[-1] + 1e-9:
            j = int(np.clip(np.searchsorted(k, lv) - 1, 0, len(k) - 2))
            w = (lv - k[j]) / (k[j + 1] - k[j])
            out.append(v[..., j] * (1 - w) + v[..., j + 1] * w)
        elif lv < k[0]:
            s = np.clip((v[..., 1] - v[..., 0]) / np.log(k[1] / k[0]), 0, None)
            out.append(v[..., 0] + s * np.log(lv / k[0]))
        else:
            s = np.clip((v[..., -1] - v[..., -2]) / np.log((1 - k[-2]) / (1 - k[-1])), 0, None)
            out.append(v[..., -1] + s * np.log((1 - k[-1]) / (1 - lv)))
    return np.stack(out, axis=-1)
