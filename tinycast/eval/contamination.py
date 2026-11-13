"""Content-level contamination check: does a benchmark contain series (or pieces of series) we trained on?

Dataset names are not enough: the same series circulate under different names (Monash, Chronos, GIFT-Eval, fev...),
sometimes rescaled. So we compare *content*:

* every L-step chunk (L = 32) of every training series, at every position, is z-normalised (mean 0, std 1: robust to a
  change of units or offset), rounded to 0.01 and hashed (64-bit polynomial hash, vectorised);
* chunks that are nearly flat (std ~ 0: zeros, constants, saturated sensors) are skipped: they match everywhere and say
  nothing; chunks with missing values are skipped;
* benchmark series are cut into non-overlapping L-chunks; a chunk is "seen" if its hash is in the training set.
  We report the share of seen chunks per benchmark dataset, over the whole series and over its *tail* (the last 20%,
  where benchmarks place their test targets).

False positives need 32 consecutive z-normalised values to agree to 0.01: negligible for real series. Exactly periodic
patterns can collide across unrelated series; they are rare in real data and show up as small shares, not as ~100%.
Low-information chunks (fewer than MIN_DISTINCT distinct values: spikes on a flat line, steps) are skipped: on sparse /
intermittent series they matched unrelated data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

L = 32
_P = np.uint64(1_000_003)
_POW = np.array([pow(1_000_003, L - 1 - j, 2**64) for j in range(L)], dtype=np.uint64)


MIN_DISTINCT = 8  # a chunk must take >= 8 distinct values to count as evidence


def chunk_hashes(
    x: np.ndarray, stride: int = 1, min_std: float = 1e-6, min_distinct: int = MIN_DISTINCT
) -> np.ndarray:
    """Hashes of all z-normalised L-chunks of a 1-D series at the given stride (uint64, flat/NaN chunks dropped)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < L:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(x, L)[::stride]
    ok = np.isfinite(w).all(axis=1)
    w = w[ok]
    sd = w.std(axis=1)
    keep = sd > min_std * (1 + np.abs(w.mean(axis=1)))
    w, sd = w[keep], sd[keep]
    z = np.round((w - w.mean(axis=1, keepdims=True)) / sd[:, None], 2)
    # Low-information chunks (e.g. 31 equal values + one spike in intermittent or night-zero series) have the same
    # z-normalised shape in unrelated series: false matches. Require enough distinct values (found on GIFT-Eval:
    # car_parts 23% "seen" before this filter).
    zs = np.sort(z, axis=1)
    n_distinct = 1 + (np.diff(zs, axis=1) != 0).sum(axis=1)
    z = z[n_distinct >= min_distinct]
    q = (
        (z * 100).astype(np.int64).view(np.uint64)
    )  # two's complement view: stable integer code per value
    with np.errstate(over="ignore"):
        return (q * _POW).sum(axis=1, dtype=np.uint64)


@dataclass
class Fingerprint:
    """Sorted unique chunk hashes of a corpus, with the dataset each hash first came from."""

    hashes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))
    owner: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    names: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, datasets: dict[str, list[np.ndarray]]) -> Fingerprint:
        hs, ow, names = [], [], []
        for k, (name, series) in enumerate(datasets.items()):
            names.append(name)
            for s in series:
                h = chunk_hashes(s, stride=1)
                hs.append(h)
                ow.append(np.full(len(h), k, dtype=np.int32))
        if not hs:
            return cls(names=names)
        h, o = np.concatenate(hs), np.concatenate(ow)
        h, idx = np.unique(h, return_index=True)
        return cls(h, o[idx], names)

    def lookup(self, h: np.ndarray) -> np.ndarray:
        """Index of the owning training dataset for each hash, -1 if unseen."""
        if len(self.hashes) == 0 or len(h) == 0:
            return np.full(len(h), -1)
        pos = np.clip(np.searchsorted(self.hashes, h), 0, len(self.hashes) - 1)
        return np.where(self.hashes[pos] == h, self.owner[pos], -1)


def check_dataset(fp: Fingerprint, series: list[np.ndarray], tail: float = 0.2) -> dict:
    """Share of a benchmark dataset's chunks found in the training fingerprint (whole series and tail)."""
    tot = tot_tail = seen = seen_tail = 0
    by_owner: dict[str, int] = {}
    for s in series:
        s = np.asarray(s, dtype=np.float64)
        h = chunk_hashes(s, stride=L)
        own = fp.lookup(h)
        n = len(h)
        n_tail = max(1, int(round(tail * n))) if n else 0
        tot += n
        seen += int((own >= 0).sum())
        tot_tail += n_tail
        seen_tail += int((own[n - n_tail :] >= 0).sum()) if n_tail else 0
        for o in own[own >= 0]:
            by_owner[fp.names[o]] = by_owner.get(fp.names[o], 0) + 1
    return {
        "n_series": len(series),
        "n_chunks": tot,
        "seen_share": seen / tot if tot else 0.0,
        "seen_share_tail": seen_tail / tot_tail if tot_tail else 0.0,
        "matched_training_datasets": dict(sorted(by_owner.items(), key=lambda kv: -kv[1])),
    }
