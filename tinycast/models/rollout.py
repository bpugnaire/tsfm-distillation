"""Long horizons with a fixed-horizon model: forecast a block, append it to the context, repeat.

Two ways to append a block (see results/rollouts.json for how they behave on teachers):

* ``median``: append the median. One forward pass per block. This is what Timestral does past its native
  1024-step block (`TimestralForecaster._block_ar_decode`). The quantiles of later blocks are conditional on a
  "typical" past, so the bands are too narrow: the uncertainty of earlier blocks is never propagated.
* ``paths``: draw `n_paths` sample paths. Within a block, path values are read from the forecast quantiles with a
  Gaussian copula whose latent noise is AR(1) over time (rho = 0.9, carried across blocks), so a path that is high
  stays high for a while. Each path is appended to its own context; the final quantiles are taken across paths.
  Cost: n_paths forward passes per block.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.base import check_output


class RolloutForecaster:
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, base, name: str, mode: str = "median", n_paths: int = 100, rho: float = 0.9,
                 block: int = 64, context_length: int = 512, seed: int = 0, batch: int = 4096):  # fmt: skip
        if mode not in ("median", "paths"):
            raise ValueError(f"unknown rollout mode {mode!r}")
        self.base, self.name, self.mode = base, name, mode
        self.n_paths, self.rho, self.block, self.T = n_paths, rho, block, context_length
        self.seed, self.batch = seed, batch
        self.n_params = getattr(base, "n_params", 0)
        self.native_levels = getattr(base, "native_levels", QUANTILE_LEVELS)
        self.fallbacks = [
            f"rollout: {mode}" + (f", {n_paths} paths, rho {rho}" if mode == "paths" else "")
        ]

    def _base(self, ctx: np.ndarray, h: int) -> np.ndarray:
        out = [
            self.base.forecast(ctx[i : i + self.batch], h) for i in range(0, len(ctx), self.batch)
        ]
        return np.concatenate(out)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        x = np.asarray(context, dtype=np.float32)[:, -self.T :]
        if horizon <= self.block:
            return self._base(x, horizon)
        if self.mode == "median":
            return self._median(x, horizon)
        return self._paths(x, horizon)

    def _median(self, x: np.ndarray, horizon: int) -> np.ndarray:
        i50 = QUANTILE_LEVELS.index(0.5)
        blocks, ctx = [], x
        while sum(b.shape[-1] for b in blocks) < horizon:
            q = self._base(ctx[:, -self.T :], self.block)
            blocks.append(q)
            ctx = np.concatenate([ctx, q[:, i50]], axis=1)
        q = np.concatenate(blocks, axis=-1)[..., :horizon]
        return check_output(q, len(x), horizon, self.name)

    def _paths(self, x: np.ndarray, horizon: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n, P = len(x), self.n_paths
        lv = np.asarray(QUANTILE_LEVELS)
        ctx = np.repeat(x, P, axis=0)  # (n*P, T): path p of window i at row i*P + p
        z = rng.standard_normal(n * P)
        paths = []
        done = 0
        while done < horizon:
            q = self._base(ctx[:, -self.T :], self.block)  # (n*P, K, B)
            draws = np.empty((n * P, self.block), dtype=np.float32)
            for t in range(self.block):
                z = self.rho * z + np.sqrt(1 - self.rho**2) * rng.standard_normal(n * P)
                u = np.clip(
                    norm.cdf(z), lv[0], lv[-1]
                )  # beyond 1% / 99% we stay on the outer quantile
                k = np.clip(np.searchsorted(lv, u) - 1, 0, len(lv) - 2)
                w = (u - lv[k]) / (lv[k + 1] - lv[k])
                lo = q[np.arange(n * P), k, t]
                hi = q[np.arange(n * P), k + 1, t]
                draws[:, t] = lo + w * (hi - lo)
            paths.append(draws)
            ctx = np.concatenate([ctx, draws], axis=1)
            done += self.block
        sims = np.concatenate(paths, axis=1)[:, :horizon].reshape(n, P, horizon)
        q = np.quantile(sims, lv, axis=1).transpose(1, 0, 2)  # (n, K, H)
        return check_output(q.astype(np.float32), n, horizon, self.name)
