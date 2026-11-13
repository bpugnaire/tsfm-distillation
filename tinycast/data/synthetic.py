"""Synthetic training corpus, deterministic from seeds (never stored: regenerated on the fly).

Mix per series (by seed): KernelSynth-style GP kernel compositions (Chronos, Ansari et al. 2024),
randomised gallery shapes, and TSMixup convex combinations of both.
Training seeds are < TRAIN_SEED_MAX; the gallery uses seeds >= that (tests/test_leakage.py).
"""

from __future__ import annotations

import numpy as np

from tinycast.data import shapes as S

TRAIN_SEED_MAX = S.GALLERY_SEED_OFFSET
PERIODS = (12, 24, 48, 96, 168)


def _kernel(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    kind = rng.choice(
        ["linear", "rbf", "periodic", "rq", "white", "const"], p=[0.15, 0.2, 0.35, 0.1, 0.1, 0.1]
    )
    d = t[:, None] - t[None, :]
    if kind == "linear":
        c = rng.uniform(0, 1)
        return (t[:, None] - c) * (t[None, :] - c)
    if kind == "rbf":
        ell = rng.choice([0.02, 0.05, 0.1, 0.3])
        return np.exp(-0.5 * (d / ell) ** 2)
    if kind == "periodic":
        p = rng.choice(PERIODS) / len(t)
        ell = rng.choice([0.5, 1.0, 2.0])
        return np.exp(-2 * np.sin(np.pi * np.abs(d) / p) ** 2 / ell**2)
    if kind == "rq":
        a, ell = rng.choice([0.5, 1, 2]), rng.choice([0.05, 0.1, 0.3])
        return (1 + d**2 / (2 * a * ell**2)) ** (-a)
    if kind == "white":
        return np.eye(len(t)) * rng.uniform(0.01, 0.2)
    return np.full((len(t), len(t)), rng.uniform(0.1, 1.0))


def kernel_synth(rng: np.random.Generator, n: int, max_kernels: int = 4) -> np.ndarray:
    t = np.linspace(0, 1, n)
    K = _kernel(rng, t)
    for _ in range(rng.integers(0, max_kernels)):
        K2 = _kernel(rng, t)
        K = K + K2 if rng.uniform() < 0.5 else K * K2
    K = K + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K + 1e-4 * np.trace(K) / n * np.eye(n))
    x = L @ rng.standard_normal(n)
    return x * rng.uniform(0.5, 5) + rng.uniform(-2, 20)


def _shape(rng: np.random.Generator, n: int, context_length: int) -> np.ndarray:
    name = S.SHAPE_NAMES[rng.integers(len(S.SHAPE_NAMES))]
    sh = S.make_shape(
        name, int(rng.integers(TRAIN_SEED_MAX)), context_length, n - context_length, randomize=True
    )
    return np.concatenate([sh.context, sh.future]).astype(np.float64)


def synthetic_series(seed: int, n: int = 576, context_length: int = 512) -> np.ndarray:
    """One training series of length n (context + horizon). May contain NaN (gaps) in the context."""
    if not 0 <= seed < TRAIN_SEED_MAX:
        raise ValueError(f"training seeds must be in [0, {TRAIN_SEED_MAX})")
    rng = np.random.default_rng([seed, 7919])
    u = rng.uniform()
    if u < 0.4:
        return kernel_synth(rng, n).astype(np.float32)
    if u < 0.8:
        return _shape(rng, n, context_length).astype(np.float32)
    k = int(rng.integers(2, 4))
    parts = [
        kernel_synth(rng, n) if rng.uniform() < 0.5 else _shape(rng, n, context_length)
        for _ in range(k)
    ]
    w = rng.dirichlet(np.ones(k))
    mixed = sum(wi * p / (np.nanmean(np.abs(p)) + 1e-6) for wi, p in zip(w, parts, strict=True))
    return (mixed * rng.uniform(1, 20)).astype(np.float32)


def synthetic_batch(
    seeds: np.ndarray, context_length: int = 512, horizon: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    n = context_length + horizon
    if (
        len(seeds) >= 5000
    ):  # each series depends only on its seed: parallel generation gives identical results
        import os
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            rows = list(
                ex.map(
                    partial(synthetic_series, n=n, context_length=context_length),
                    [int(s) for s in seeds],
                    chunksize=256,
                )
            )
        arr = np.stack(rows)
    else:
        arr = np.stack([synthetic_series(int(s), n, context_length) for s in seeds])
    ctx, fut = arr[:, :context_length], arr[:, context_length:]
    fut = np.where(np.isnan(fut), np.nan, fut)  # shapes never put NaN in the future; keep explicit
    return ctx, fut
