"""Combining several teachers into one distillation target (study Stage B, RQ3).

Composite teacher names (used as `data.teacher` in a TrainConfig):
  ensemble:a+b+c        mean of the teachers' quantiles, level by level ("quantile averaging" / Vincentisation)
  weighted:a+b+c@tau    per window, weights softmax(-tau * WQL_m) where WQL_m is teacher m's weighted quantile loss
                        on THAT training window's ground truth. Needs ground truth (real or generator futures);
                        it uses only training targets, never validation or test.
  margin:a+b@m          per window, teacher b (e.g. a bigger one) only where its WQL on the window's ground truth is at
                        least a fraction m lower than teacher a's (WQL_b < (1 - m) WQL_a); teacher a everywhere else.
                        Same ground-truth rule as `weighted`; m is fixed before training (owner 2026-11-23: m = 0.10).
Native levels of a composite = intersection of the members' native levels (Toto 2.0 has no native tails).
"""

from __future__ import annotations

import numpy as np

from tinycast.config import CORE_LEVELS


def parse(name: str) -> tuple[str, list[str], float | None]:
    """'weighted:a+b@20' -> ('weighted', ['a', 'b'], 20.0); 'x' -> ('single', ['x'], None)."""
    if ":" not in name:
        return "single", [name], None
    kind, rest = name.split(":", 1)
    tau = None
    if "@" in rest:
        rest, t = rest.rsplit("@", 1)
        tau = float(t)
    if kind not in ("ensemble", "weighted", "margin") or (
        kind in ("weighted", "margin") and tau is None
    ):
        raise ValueError(f"bad composite teacher {name!r}")
    if kind == "margin" and (len(rest.split("+")) != 2 or not 0 <= tau < 1):
        raise ValueError(f"bad composite teacher {name!r}")
    return kind, rest.split("+"), tau


def window_wql(q: np.ndarray, y: np.ndarray, levels) -> np.ndarray:
    """Per-window WQL on the core levels. q: (N, K, H), y: (N, H) with NaN allowed -> (N,)."""
    from tinycast.eval.metrics import wql

    return wql(y, q, levels, use_levels=CORE_LEVELS)


def combine(
    kind: str, qs: list[np.ndarray], y: np.ndarray | None, levels, tau: float | None
) -> tuple[np.ndarray, dict]:
    """qs: list of (N, K, H) teacher quantiles on the same windows. Returns (combined quantiles, diagnostics)."""
    if (
        kind == "margin"
    ):  # no stacking and chunked errors: 2 x 1M labels already take ~6.6 GB in float32
        if y is None or not np.isfinite(y).any():
            raise ValueError("margin teacher selection needs ground-truth futures")
        qa, qb = qs
        out = qa if qa.dtype == np.float32 else qa.astype(np.float32)  # in place: qa is overwritten
        use_b = np.zeros(len(qa), dtype=bool)
        for i in range(0, len(qa), 65536):
            sl = slice(i, i + 65536)
            a, b = window_wql(qa[sl], y[sl], levels), window_wql(qb[sl], y[sl], levels)
            use_b[sl] = (
                np.isfinite(a) & np.isfinite(b) & (b < (1 - tau) * a)
            )  # no ground truth -> a
        out[use_b] = qb[use_b]
        return out, {"share_b": float(use_b.mean()), "n_b": int(use_b.sum())}
    Q = np.stack(qs)  # (M, N, K, H)
    if kind == "single":
        return Q[0].astype(np.float32), {}
    if kind == "ensemble":
        return Q.mean(0).astype(np.float32), {}
    if y is None or not np.isfinite(y).any():
        raise ValueError("weighted teacher combination needs ground-truth futures")
    W = np.stack([window_wql(q, y, levels) for q in qs])  # (M, N)
    W = np.where(np.isfinite(W), W, np.nanmax(W[np.isfinite(W)]) if np.isfinite(W).any() else 1.0)
    logits = -tau * W
    logits -= logits.max(0, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(0, keepdims=True)  # (M, N)
    combined = (
        (w[:, :, None, None] * Q).sum(0).astype(np.float32)
    )  # float32: Apple MPS has no float64
    best = np.argmin(W, axis=0)
    return combined, {
        "mean_weight": w.mean(1).tolist(),
        "best_teacher_share": np.bincount(best, minlength=len(qs)).tolist(),
    }
