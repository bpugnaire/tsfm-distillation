"""Thin adapters around teacher / comparator packages. No teacher-specific code elsewhere.

APIs verified against the installed sources; see docs/notes/teacher-apis.md.
"""

from __future__ import annotations

import numpy as np
import torch

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.base import check_output, extend_tails_exponential


def _device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


class TotoAdapter:
    """Datadog Toto 2.0 (`pip install toto-2`, `from toto2 import Toto2Model`).

    Native output: 9 quantiles 0.1..0.9 (fixed knots of a quantile head; no samples).
    0.01/0.05/0.95/0.99 are extended with IQF exponential tails -> flagged via native_levels.
    """

    quantile_levels = QUANTILE_LEVELS

    def __init__(self, name: str, checkpoint: str, device: str = "cpu", batch_size: int = 32):
        from toto2 import Toto2Model

        self.name = name
        self.device = _device(device)
        self.model = Toto2Model.from_pretrained(checkpoint).to(self.device).eval()
        self.native_levels = tuple(float(k) for k in self.model.output_head.knots)
        self.batch_size = batch_size
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        out = []
        for sl in _batches(len(context), self.batch_size):
            x = torch.as_tensor(context[sl], dtype=torch.float32, device=self.device)[:, None, :]
            mask = ~torch.isnan(x)
            inputs = {
                "target": torch.nan_to_num(x, nan=0.0),
                "target_mask": mask,
                "series_ids": torch.zeros(x.shape[0], 1, dtype=torch.long, device=self.device),
            }
            q = self.model.forecast(inputs, horizon=horizon, has_missing_values=bool((~mask).any()))
            out.append(q[:, :, 0, :].float().cpu().numpy())  # (9, b, H)
        q9 = np.concatenate(out, axis=1).transpose(1, 2, 0)  # (B, H, 9)
        q = extend_tails_exponential(self.native_levels, q9, QUANTILE_LEVELS).transpose(0, 2, 1)
        return check_output(q, len(context), horizon, self.name)


class T0Adapter:
    """The Forecasting Company t0 (`pip install 'tfc-t0[torch]'`, `from t0 import T0Forecaster`).

    t0-gamma natively emits 21 quantiles that contain our whole 13-level grid. We call
    `predict` with the vendor's post-processing heuristics OFF (raw model; see teacher-apis.md).
    """

    quantile_levels = QUANTILE_LEVELS

    def __init__(self, name: str, checkpoint: str, device: str = "cpu", batch_size: int = 32):
        from t0 import T0Forecaster

        self.name = name
        self.device = _device(device)
        self.model = T0Forecaster.from_pretrained(checkpoint).to(self.device).eval()
        trained = tuple(float(v) for v in self.model.config.quantile_levels)
        self.native_levels = tuple(
            lv for lv in QUANTILE_LEVELS if any(abs(lv - t) < 1e-9 for t in trained)
        )
        self.batch_size = batch_size
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        out = []
        for sl in _batches(len(context), self.batch_size):
            fc = self.model.predict(
                np.asarray(context[sl], dtype=np.float32), horizon, quantile_levels=QUANTILE_LEVELS
            )
            out.append(np.asarray(fc.quantiles.float().cpu()))  # (b, H, Q)
        q = np.concatenate(out).transpose(0, 2, 1)
        return check_output(q, len(context), horizon, self.name)


class Chronos2Adapter:
    """Amazon Chronos-2 (`chronos-forecasting`), comparator only.

    `predict_quantiles` silently *clamps* levels outside its trained range (log warning only),
    so we only request trained levels and extend the rest with IQF tails ourselves.
    Raw Chronos-2 quantiles occasionally cross (0.4-1.6% of cells on the gallery). We sort them
    (monotone rearrangement, never increases pinball loss) and RECORD the rate in `fallbacks`.
    """

    quantile_levels = QUANTILE_LEVELS

    def __init__(self, name: str, checkpoint: str, device: str = "cpu", batch_size: int = 32):
        from chronos import Chronos2Pipeline

        self.name = name
        self.device = _device(device)
        self.pipe = Chronos2Pipeline.from_pretrained(checkpoint, device_map=str(self.device))
        trained = [float(v) for v in self.pipe.quantiles]
        self.native_levels = tuple(
            lv for lv in QUANTILE_LEVELS if any(abs(lv - t) < 1e-9 for t in trained)
        )
        self.batch_size = batch_size
        self.n_params = sum(p.numel() for p in self.pipe.model.parameters())
        self.fallbacks: list[str] = []

    @torch.no_grad()
    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        x = torch.as_tensor(np.asarray(context, dtype=np.float32))[:, None, :]
        qs, _ = self.pipe.predict_quantiles(
            x, prediction_length=horizon, quantile_levels=list(self.native_levels),
            batch_size=self.batch_size,
        )  # fmt: skip
        qn = np.stack([q[0].float().cpu().numpy() for q in qs])  # (B, H, n_native)
        crossing = float((np.diff(qn, axis=-1) < 0).mean())
        if crossing > 0:
            self.fallbacks.append(f"sorted crossing quantiles: {100 * crossing:.2f}% of cells")
        qn = np.sort(qn, axis=-1)
        q = extend_tails_exponential(self.native_levels, qn, QUANTILE_LEVELS).transpose(0, 2, 1)
        return check_output(q, len(context), horizon, self.name)


class TimestralAdapter:
    """Comparator only. Disabled until the owner confirms the licence (revenue clause)."""

    def __init__(self, *args, **kwargs):
        raise PermissionError(
            "Timestral is disabled: its Modified MIT licence withdraws all rights if your "
            "company's (or employer's) monthly revenue exceeds $20M. The owner must confirm "
            "eligibility (see docs/decisions/0001-project-scope.md) before enabling it."
        )
