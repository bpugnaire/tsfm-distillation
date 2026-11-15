"""Forecasts computed in another environment, scored by our harness (e.g. IBM TTM, see scripts/ttm_forecast.py).

`checkpoint` is a folder holding one `<suite>.npz` (context as seen, point forecast) + `<suite>.json` (n_params,
latency, provenance) per suite. `forecast()` finds the file whose saved contexts equal the requested ones, so a
forecast can never be scored against windows it was not made for.

Point forecasts are scored as a degenerate distribution: all 13 quantiles = the point. Then WQL reduces to a
scaled absolute error, which is how GIFT-Eval scores point-only models too. Recorded in `fallbacks`, never silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.base import check_output


class PrecomputedForecaster:
    quantile_levels = QUANTILE_LEVELS
    native_levels = (0.5,)

    def __init__(self, name: str, checkpoint: str):
        self.name = name
        self.dir = Path(checkpoint)
        self.files = sorted(self.dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"{name}: no precomputed forecasts in {self.dir}")
        meta = json.loads(self.files[0].with_suffix(".json").read_text())
        self.n_params = int(meta["n_params"])
        self.size_mb = round(4 * self.n_params / 1e6, 3)  # fp32 weights
        self.meta = meta
        self.fallbacks = [
            "point forecast scored as a degenerate distribution (all quantiles = the point)"
        ]
        self._lat: dict | None = None

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        x = np.asarray(context, dtype=np.float32)
        for f in self.files:
            d = np.load(f)
            c = d["context"]
            if c.shape != x.shape:
                continue
            same_nan = np.array_equal(np.isnan(c), np.isnan(x))
            ok = np.isfinite(c)
            if same_nan and np.allclose(c[ok], x[ok], rtol=1e-4, atol=1e-4):
                self._lat = json.loads(f.with_suffix(".json").read_text()).get("latency")
                p = d["point"][:, :horizon]
                q = np.repeat(p[:, None, :], len(QUANTILE_LEVELS), axis=1)
                return check_output(q, len(x), horizon, self.name)
        raise ValueError(
            f"{self.name}: no precomputed forecasts for these {len(x)} contexts in {self.dir}"
        )

    @property
    def precomputed_latency(self) -> dict | None:
        """Latency measured in the model's own env with the same harness (tinycast/eval/latency.py)."""
        return self._lat
