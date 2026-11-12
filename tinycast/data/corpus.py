"""Deterministic training/validation corpora: real RTE windows + synthetic windows, joined by id.

A corpus is fully defined by (split, n_real, n_synth, seed, context, horizon). It is materialised
once into data/processed/corpus_<hash>.npz (regenerable, git-ignored) and joined to teacher caches
by `window_id`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tinycast.data.synthetic import TRAIN_SEED_MAX, synthetic_batch
from tinycast.data.windows import window_id

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorpusSpec:
    split: str  # train | val
    n_real: int
    n_synth: int
    seed: int = 0
    context_length: int = 512
    horizon: int = 64

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class Corpus:
    ids: np.ndarray  # (N,) str
    source: np.ndarray  # (N,) "rte" | "synth"
    context: np.ndarray  # (N, T) float32
    future: np.ndarray  # (N, H) float32


def _synth_seeds(spec: CorpusSpec) -> np.ndarray:
    # train and val use disjoint seed ranges inside [0, TRAIN_SEED_MAX)
    half = TRAIN_SEED_MAX // 2
    base = 0 if spec.split == "train" else half
    rng = np.random.default_rng([spec.seed, 17])
    return base + rng.choice(half, size=spec.n_synth, replace=False)


def build_corpus(spec: CorpusSpec, cache: bool = True) -> Corpus:
    path = ROOT / "data" / "processed" / f"corpus_{spec.split}_{spec.key()}.npz"
    if cache and path.exists():
        d = np.load(path, allow_pickle=False)
        return Corpus(d["ids"], d["source"], d["context"], d["future"])
    ids, src, ctx, fut = [], [], [], []
    if spec.n_real:
        from tinycast.data.sources.rte_eco2mix import series_arrays
        from tinycast.data.splits import load_splits
        from tinycast.data.windows import sample_windows

        ws = sample_windows(
            series_arrays(), spec.split, load_splits(), spec.context_length, spec.horizon,
            n_windows=spec.n_real, seed=spec.seed,
        )  # fmt: skip
        ids += [w.window_id for w in ws]
        src += ["rte"] * len(ws)
        ctx += [w.context for w in ws]
        fut += [w.future for w in ws]
    if spec.n_synth:
        seeds = _synth_seeds(spec)
        c, f = synthetic_batch(seeds, spec.context_length, spec.horizon)
        ids += [window_id("synth", str(int(s)), 0) for s in seeds]
        src += ["synth"] * len(seeds)
        ctx += list(c)
        fut += list(f)
    corpus = Corpus(
        np.array(ids),
        np.array(src),
        np.stack(ctx).astype(np.float32),
        np.stack(fut).astype(np.float32),
    )
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, ids=corpus.ids, source=corpus.source, context=corpus.context, future=corpus.future
        )
    return corpus
