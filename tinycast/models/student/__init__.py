"""The student: a small patch-transformer (or MLP) quantile forecaster.

forward(raw context, horizon fixed) = unscale(net(scale(raw)))  -- the scaler is part of the
module, so the exported ONNX takes raw values and returns raw-space quantiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tinycast.config import QUANTILE_LEVELS, StudentConfig, from_dict

PRESETS: dict[str, dict] = {
    # ~0.3M / ~1M / ~3M params (printed at startup by train.py)
    "s": dict(d_model=64, n_layers=3, n_heads=4),
    "m": dict(d_model=128, n_layers=4, n_heads=4),
    "l": dict(d_model=192, n_layers=6, n_heads=6),
}


class RobustScaler(nn.Module):
    """Instance normalisation on the observed context.

    loc = median, scale = mean |x - median| (+ floor), then optional asinh.
    Median/MAD-style statistics ignore spikes; asinh compresses big excursions (as Toto 2.0
    and t0 do) while staying ~linear near 0. Both are ONNX-exportable.
    """

    def __init__(self, use_asinh: bool = True, eps: float = 1e-3):
        super().__init__()
        self.use_asinh = use_asinh
        self.eps = eps

    def stats(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # masked median via sorting: missing values pushed to +inf, take the middle observed one
        n_obs = mask.sum(-1, keepdim=True).clamp(min=1)
        filled = torch.where(mask, x, torch.full_like(x, float("inf")))
        srt = torch.sort(filled, dim=-1).values
        mid = ((n_obs - 1) // 2).long()
        loc = torch.gather(srt, -1, mid)
        dev = torch.where(mask, (x - loc).abs(), torch.zeros_like(x)).sum(-1, keepdim=True) / n_obs
        # floor relative to the level so a constant series does not explode
        scale = dev + self.eps * (loc.abs() + 1.0)
        return loc, scale

    def scale(self, x, loc, scale):
        z = (x - loc) / scale
        return torch.asinh(z) if self.use_asinh else z

    def unscale(self, z, loc, scale):
        return (torch.sinh(z) if self.use_asinh else z) * scale + loc


class MonotoneQuantileHead(nn.Module):
    """Predicts the median plus softplus gaps outward: crossing is impossible by construction."""

    def __init__(self, d_in: int, horizon: int, levels: tuple[float, ...]):
        super().__init__()
        self.levels = levels
        self.horizon = horizon
        self.i_med = levels.index(0.5)
        self.n_lo, self.n_hi = self.i_med, len(levels) - self.i_med - 1
        self.proj = nn.Linear(d_in, horizon * len(levels))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, d) -> (B, K, H)
        out = self.proj(h).view(h.shape[0], len(self.levels), self.horizon)
        med = out[:, self.i_med : self.i_med + 1]
        gaps = F.softplus(out) + 1e-4
        lo = med - torch.flip(torch.cumsum(torch.flip(gaps[:, : self.n_lo], [1]), 1), [1])
        hi = med + torch.cumsum(gaps[:, self.i_med + 1 :], 1)
        return torch.cat([lo, med, hi], dim=1)


class PatchTransformerNet(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        self.n_patches = c.context_length // c.patch_size
        self.embed = nn.Sequential(
            nn.Linear(2 * c.patch_size, c.d_model * 2),
            nn.GELU(),
            nn.Linear(c.d_model * 2, c.d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, c.d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            c.d_model, c.n_heads, c.d_model * c.ff_mult, c.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )  # fmt: skip
        self.encoder = nn.TransformerEncoder(layer, c.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.d_model)
        self.head = MonotoneQuantileHead(c.d_model, c.horizon, c.quantile_levels)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        p = self.c.patch_size
        zp = z.view(B, self.n_patches, p)
        mp = mask.view(B, self.n_patches, p).to(z.dtype)
        tok = self.embed(torch.cat([zp * mp, mp], dim=-1))
        tok = torch.cat([tok, self.query.expand(B, -1, -1)], dim=1) + self.pos
        h = self.encoder(tok)
        return self.head(self.norm(h[:, -1]))


class MLPNet(nn.Module):
    """N-HiTS-flavoured MLP ablation: flatten (values, mask) -> residual MLP -> quantile head."""

    def __init__(self, c: StudentConfig):
        super().__init__()
        d = c.d_model * 4
        self.inp = nn.Linear(2 * c.context_length, d)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(d, d)
            )
            for _ in range(c.n_layers)
        )
        self.head = MonotoneQuantileHead(d, c.horizon, c.quantile_levels)

    def forward(self, z, mask):
        m = mask.to(z.dtype)
        h = self.inp(torch.cat([z * m, m], dim=-1))
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


class Student(nn.Module):
    """raw (B, T) with NaN -> raw-space quantiles (B, K, H)."""

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        self.scaler = RobustScaler(c.use_asinh)
        self.net = PatchTransformerNet(c) if c.arch == "patch_transformer" else MLPNet(c)

    def prepare(self, x: torch.Tensor):
        """Left-pad/crop to context_length; returns scaled input, mask, loc, scale."""
        T = self.c.context_length
        if x.shape[-1] < T:
            x = F.pad(x, (T - x.shape[-1], 0), value=float("nan"))
        x = x[..., -T:]
        mask = ~torch.isnan(x)
        x0 = torch.nan_to_num(x, nan=0.0)
        loc, scale = self.scaler.stats(x0, mask)
        z = torch.where(mask, self.scaler.scale(x0, loc, scale), torch.zeros_like(x0))
        return z, mask, loc, scale

    def forward_scaled(self, x: torch.Tensor):
        """Returns (quantiles in scaled space, loc, scale): the losses work in scaled space."""
        z, mask, loc, scale = self.prepare(x)
        return self.net(z, mask), loc, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zq, loc, scale = self.forward_scaled(x)
        return self.scaler.unscale(zq, loc[..., None], scale[..., None])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_student(c: StudentConfig) -> Student:
    return Student(c)


class StudentForecaster:
    """Forecaster-protocol wrapper around a trained Student checkpoint (a run dir)."""

    quantile_levels = QUANTILE_LEVELS
    native_levels = QUANTILE_LEVELS

    def __init__(self, model: Student, name: str = "student", device: str = "cpu"):
        self.model = model.to(device).eval()
        self.name = name
        self.device = device
        self.n_params = model.n_params()

    @classmethod
    def from_run(
        cls, run_dir: str, name: str = "student", device: str = "cpu"
    ) -> StudentForecaster:
        run = Path(run_dir)
        cfg = json.loads((run / "student_config.json").read_text())
        c = from_dict(StudentConfig, cfg)
        m = Student(c)
        m.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
        return cls(m, name, device)

    @torch.no_grad()
    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        if horizon > self.model.c.horizon:
            raise ValueError(f"student horizon is {self.model.c.horizon}, asked {horizon}")
        x = torch.as_tensor(np.asarray(context, dtype=np.float32), device=self.device)
        q = self.model(x)[..., :horizon].cpu().numpy()
        from tinycast.models.base import check_output

        return check_output(q, len(context), horizon, self.name)


def level_weights(levels: tuple[float, ...]) -> torch.Tensor:
    """Probability mass each level represents (midpoint rule): makes sum_k w_k * 2 rho_k a
    consistent CRPS approximation on a non-uniform grid (see tests/test_metrics.py)."""
    lv = np.asarray(levels)
    edges = np.concatenate([[0.0], (lv[1:] + lv[:-1]) / 2, [1.0]])
    w = np.diff(edges)
    return torch.tensor(w / w.sum(), dtype=torch.float32)
