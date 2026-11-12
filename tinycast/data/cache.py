"""Teacher-output cache: sharded parquet, resumable, integrity-checked.

Layout: data/cache/<teacher>/<corpus_key>/
    manifest.json                     teacher, checkpoint, package version, levels, config hash, shards
    shard_00000.parquet ...           window_id (str), q (fixed-size binary: float16 (K, H) row-major)

float16 quantiles: 13 x 64 x 2 bytes = 1.66 kB per window -> 300k windows ~ 500 MB per teacher.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def cache_dir(teacher: str, corpus_key: str) -> Path:
    return ROOT / "data" / "cache" / teacher / corpus_key


def _encode(q: np.ndarray) -> list[bytes]:
    return [row.astype(np.float16).tobytes() for row in q]


def _decode(blobs, K: int, H: int) -> np.ndarray:
    return np.stack([np.frombuffer(b, dtype=np.float16).reshape(K, H) for b in blobs])


def write_cache(
    teacher: str,
    corpus_key: str,
    ids: np.ndarray,
    contexts: np.ndarray,
    forecast_fn: Callable[[np.ndarray], np.ndarray],
    meta: dict,
    shard_size: int = 4096,
    log: Callable[[str], None] = print,
) -> Path:
    """Run `forecast_fn` (contexts -> (n, K, H)) shard by shard, skipping completed shards."""
    d = cache_dir(teacher, corpus_key)
    d.mkdir(parents=True, exist_ok=True)
    man_path = d / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {**meta, "shards": {}}
    if man_path.exists():
        for k in ("checkpoint", "levels", "horizon"):
            if man.get(k) != meta.get(k):
                raise ValueError(
                    f"cache {d} was built with a different {k}: {man.get(k)} != {meta.get(k)}"
                )
    n_shards = int(np.ceil(len(ids) / shard_size))
    t_start = time.time()
    done_now = 0
    for s in range(n_shards):
        name = f"shard_{s:05d}.parquet"
        if name in man["shards"] and (d / name).exists():
            continue
        sl = slice(s * shard_size, min((s + 1) * shard_size, len(ids)))
        q = forecast_fn(contexts[sl])
        if not np.isfinite(q).all():
            raise ValueError(f"non-finite teacher output in shard {s}")
        table = pa.table(
            {"window_id": pa.array(ids[sl].tolist()), "q": pa.array(_encode(q), pa.binary())}
        )
        pq.write_table(table, d / name)
        man["shards"][name] = int(sl.stop - sl.start)
        man_path.write_text(json.dumps(man, indent=1))
        done_now += sl.stop - sl.start
        rate = done_now / (time.time() - t_start)
        log(f"[{teacher}] shard {s + 1}/{n_shards}  {rate:.1f} windows/s")
    return d


def read_cache(
    teacher: str, corpus_key: str, ids: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Return (q float32 aligned to `ids`, manifest). Raises if any id is missing."""
    d = cache_dir(teacher, corpus_key)
    man = json.loads((d / "manifest.json").read_text())
    K, H = len(man["levels"]), man["horizon"]
    tables = [pq.read_table(d / name) for name in sorted(man["shards"])]
    wid = np.concatenate([t["window_id"].to_numpy(zero_copy_only=False) for t in tables])
    q = np.concatenate([_decode(t["q"].to_pylist(), K, H) for t in tables]).astype(np.float32)
    if ids is None:
        return q, man
    pos = {w: i for i, w in enumerate(wid)}
    missing = [w for w in ids if w not in pos]
    if missing:
        raise KeyError(f"{len(missing)} windows missing from cache {d} (e.g. {missing[0]})")
    return q[[pos[w] for w in ids]], man


def check_cache(teacher: str, corpus_key: str, expected_n: int) -> dict:
    q, man = read_cache(teacher, corpus_key)
    report = {
        "n": len(q),
        "expected": expected_n,
        "nan": int(np.isnan(q).sum()),
        "non_monotone_frac": float((np.diff(q, axis=1) < -1e-3).mean()),
    }
    if report["n"] != expected_n or report["nan"]:
        raise ValueError(f"cache integrity failed: {report}")
    return report
