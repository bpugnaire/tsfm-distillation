"""Losses for probabilistic distillation. Pure functions on tensors; unit-tested.

Shapes: student quantiles q (B, K, H); ground truth y (B, H) with mask (B, H);
teacher quantiles qt (B, J, H) at teacher levels (J,). Everything is in the student's
*scaled* space (asinh((x - loc) / scale)). This is legitimate for quantile losses because
quantiles are equivariant under monotone transforms: the pinball minimiser in scaled space
is the scaled true quantile.

Why not classification-style KD (KL on softened logits)? A quantile head has no logits and
no categorical distribution; the natural "soft target" is the teacher's predictive
distribution, which we only see through its quantiles. Hence three options:

* L_gt                  pinball vs ground truth (the floor).
* L_kd_quantile         per-level Huber regression of q_k onto the teacher's q_k.
                        Simple, but it treats each level independently and copies the
                        teacher's errors at every level with the same weight.
* L_kd_pinball_teacher  pinball of each student level against the teacher's quantile
                        values used as weighted pseudo-observations. With weights = the
                        probability mass of each teacher level, this is the expected pinball
                        under the teacher's (discretised) distribution, i.e. the
                        "sample-pinball" loss, whose minimiser is the teacher's own quantile.
                        Toto 2.0 outputs no samples, so this replaces the plan's
                        sample-pinball loss exactly (journal 2026-11-12).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def level_mass(levels: Sequence[float], device=None) -> torch.Tensor:
    """Probability mass of each level (midpoint rule), sums to 1."""
    lv = torch.tensor(list(levels), dtype=torch.float32, device=device)
    edges = torch.cat([lv.new_zeros(1), (lv[1:] + lv[:-1]) / 2, lv.new_ones(1)])
    return edges.diff()


def horizon_weights(H: int, kind: str = "uniform", device=None) -> torch.Tensor:
    """w(h), mean 1. 'linear' up-weights long horizons (short ones dominate the raw loss)."""
    if kind == "uniform":
        w = torch.ones(H, device=device)
    elif kind == "linear":
        w = torch.linspace(0.5, 1.5, H, device=device)
    else:
        raise ValueError(kind)
    return w / w.mean()


def _rho(diff: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.maximum(tau * diff, (tau - 1) * diff)


def pinball_gt(
    q: torch.Tensor,
    y: torch.Tensor,
    levels: Sequence[float],
    mask: torch.Tensor | None = None,
    level_w: torch.Tensor | None = None,
    horizon_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """sum_k w_k * mean_{b,h} w(h) rho_tau_k(y - q_k). Default w_k = level mass (CRPS-consistent)."""
    tau = torch.tensor(list(levels), dtype=q.dtype, device=q.device)[None, :, None]
    level_w = level_mass(levels, q.device) if level_w is None else level_w
    mask = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
    hw = torch.ones(y.shape[-1], device=q.device) if horizon_w is None else horizon_w
    loss = _rho(y[:, None, :] - q, tau)  # (B, K, H)
    m = (mask[:, None, :].to(q.dtype)) * hw[None, None, :]
    per_level = (loss * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1e-8)
    return 2 * (per_level * level_w).sum()


def kd_pinball_teacher(
    q: torch.Tensor,
    qt: torch.Tensor,
    levels: Sequence[float],
    teacher_levels: Sequence[float],
    horizon_w: torch.Tensor | None = None,
    teacher_level_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected pinball of student levels under the teacher's discretised distribution.

    teacher_level_mask (J,) bool: which teacher levels may act as pseudo-observations
    (e.g. only Toto's *native* 0.1..0.9, not our extrapolated tails).
    """
    tau = torch.tensor(list(levels), dtype=q.dtype, device=q.device)[None, :, None, None]
    tw = level_mass(teacher_levels, q.device)
    if teacher_level_mask is not None:
        tw = tw * teacher_level_mask.to(tw.dtype)
    tw = tw / tw.sum()
    hw = torch.ones(q.shape[-1], device=q.device) if horizon_w is None else horizon_w
    diff = qt[:, None, :, :] - q[:, :, None, :]  # (B, K, J, H)
    loss = _rho(diff, tau) * tw[None, None, :, None] * hw[None, None, None, :]
    lw = level_mass(levels, q.device)
    return 2 * (loss.sum(2).mean((0, 2)) * lw).sum()


def kd_quantile_huber(
    q: torch.Tensor,
    qt: torch.Tensor,
    level_index: Sequence[int],
    delta: float = 1.0,
    horizon_w: torch.Tensor | None = None,
    level_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber(q_k, qt_k) on matching levels. level_index[k] = teacher column for student level k."""
    qt_m = qt[:, list(level_index), :]
    loss = F.huber_loss(q, qt_m, reduction="none", delta=delta)  # (B, K, H)
    hw = torch.ones(q.shape[-1], device=q.device) if horizon_w is None else horizon_w
    loss = loss * hw[None, None, :]
    if level_mask is not None:
        loss = loss[:, level_mask]
    return loss.mean()


def mixed_objective(l_gt: torch.Tensor, l_kd: torch.Tensor | None, alpha: float) -> torch.Tensor:
    """L = (1 - alpha) L_gt + alpha L_kd."""
    if alpha == 0 or l_kd is None:
        if alpha > 0:
            raise ValueError("alpha > 0 but no KD loss (no teacher cache?)")
        return l_gt
    return (1 - alpha) * l_gt + alpha * l_kd
