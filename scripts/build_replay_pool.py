"""Replay pool for fine-tuning (Stage F): a fixed random sample of the distillation corpus with its teacher labels.

    uv run python scripts/build_replay_pool.py
-> data/processed/replay_pool_<teacher>_<corpus key>.npz  (ids, source, context, future, q: teacher quantiles)

Corpus: the 1M windows of C14 (500k synthetic + 500k data v2, key 48dd0113d4), a subset of the distribution the site
student (C15, 3M windows) was distilled on; labels: Timestral-5M, the site student's teacher. The cache is read shard by
shard (decoding all 1M labels at once takes several GB).
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from tinycast.config import TrainConfig, from_dict
from tinycast.data.cache import _decode, cache_dir
from tinycast.data.corpus import build_corpus
from tinycast.distill.train import specs
from tinycast.eval.benchmark import ROOT

TEACHER, RUN, N = "timestral-5m", "runs/2026-11-21_stage_c14_teacher5m_kd5m_1m", 20000


def main() -> None:
    import yaml

    cfg = from_dict(TrainConfig, yaml.safe_load((ROOT / RUN / "config.yaml").read_text()))
    spec = specs(cfg)[0]
    key = spec.key()
    corpus = build_corpus(spec)
    pick = np.sort(np.random.default_rng(0).choice(len(corpus.ids), N, replace=False))
    want = {w: j for j, w in enumerate(corpus.ids[pick])}
    d = cache_dir(TEACHER, key)
    man = json.loads((d / "manifest.json").read_text())
    K, H = len(man["levels"]), man["horizon"]
    q = np.full((N, K, H), np.nan, dtype=np.float32)
    for name in sorted(man["shards"]):
        t = pq.read_table(d / name)
        wid = t["window_id"].to_numpy(zero_copy_only=False)
        rows = [i for i, w in enumerate(wid) if w in want]
        if rows:
            blobs = t["q"].take(rows).to_pylist()
            q[[want[wid[i]] for i in rows]] = _decode(blobs, K, H)
    if not np.isfinite(q).all():
        raise ValueError("replay pool: some windows have no teacher labels")
    out = ROOT / "data" / "processed" / f"replay_pool_{TEACHER}_{key}.npz"
    np.savez(out, ids=corpus.ids[pick], source=corpus.source[pick], context=corpus.context[pick],
             future=corpus.future[pick], q=q, levels=np.array(man["levels"]))  # fmt: skip
    print(
        f"{N} windows ({dict(zip(*np.unique(corpus.source[pick], return_counts=True), strict=True))}) -> {out}"
    )


if __name__ == "__main__":
    main()
