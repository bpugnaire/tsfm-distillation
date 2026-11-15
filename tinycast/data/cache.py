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

from tinycast.config import QUANTILE_LEVELS

ROOT = Path(__file__).resolve().parents[2]


def cache_dir(teacher: str, corpus_key: str) -> Path:
    return ROOT / "data" / "cache" / teacher / corpus_key


ENCODING = "scaled_f16_v2"


def _encode(q: np.ndarray) -> list[bytes]:
    """Per window: float32 (loc, scale) + float16 of (q - loc) / scale, which lies in [-1, 1].

    v1 stored raw float16, which overflows above 65504 (FRED-MD, rideshare...): 3-5% of real-data labels became inf
    in Stage A (journal 2026-11-13). Normalising per window keeps float16's size with a range-relative precision.
    """
    q = q.astype(np.float64)
    loc = np.median(q.reshape(len(q), -1), axis=1)
    scale = np.abs(q - loc[:, None, None]).reshape(len(q), -1).max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    z = ((q - loc[:, None, None]) / scale[:, None, None]).astype(np.float16)
    head = np.stack([loc, scale], axis=1).astype(np.float32)
    return [head[i].tobytes() + z[i].tobytes() for i in range(len(q))]


def _decode(blobs, K: int, H: int) -> np.ndarray:
    out = np.empty((len(blobs), K, H), dtype=np.float32)
    for i, b in enumerate(blobs):
        loc, scale = np.frombuffer(b[:8], dtype=np.float32)
        out[i] = (
            np.frombuffer(b[8:], dtype=np.float16).reshape(K, H).astype(np.float32) * scale + loc
        )
    return out


def ids_fingerprint(ids: np.ndarray) -> str:
    """Hash of the window ids, in order: a cache is only valid for exactly this corpus content."""
    import hashlib

    return hashlib.sha1("\n".join(map(str, ids)).encode()).hexdigest()[:16]


class GuardedMPS:
    """Teacher on Apple MPS, with any sub-batch whose output is non-finite recomputed on CPU.

    Opt-in (`device: mps_guarded`) after measuring that MPS and CPU agree on the corpus (runs/mps_check.py, journal
    2026-11-16). The number of CPU fallbacks is logged; the agreement check is stored in the cache manifest.
    """

    def __init__(self, entry, sub: int = 64):
        import dataclasses

        from tinycast.models import build_model

        self.mps = build_model(dataclasses.replace(entry, device="mps"))
        self._cpu_entry = dataclasses.replace(entry, device="cpu")
        self._cpu = None
        self.sub = sub
        self.native_levels = self.mps.native_levels
        self.fallback_rows = 0
        self.check = None

    def _cpu_model(self):
        if self._cpu is None:
            from tinycast.models import build_model

            self._cpu = build_model(self._cpu_entry)
        return self._cpu

    @staticmethod
    def low_information(context: np.ndarray, min_distinct: int = 64) -> np.ndarray:
        """Rows with few distinct values (counts, intermittent, near-constant): where MPS and CPU disagree
        (Toto-313m: 3/1024 windows off by > 10%, all with 9-16 distinct values)."""
        return np.array([len(np.unique(r[np.isfinite(r)])) < min_distinct for r in context])

    def forecast(self, context, horizon):
        cpu_rows = self.low_information(context)
        out = np.empty((len(context), len(QUANTILE_LEVELS), horizon), dtype=np.float32)
        if cpu_rows.any():
            out[cpu_rows] = self._cpu_model().forecast(context[cpu_rows], horizon)
            self.fallback_rows += int(cpu_rows.sum())
        rows = np.where(~cpu_rows)[0]
        for i in range(0, len(rows), self.sub):
            r = rows[i : i + self.sub]
            try:
                q = self.mps.forecast(context[r], horizon)
                if not np.isfinite(q).all():
                    raise ValueError("non-finite")
            except ValueError:
                q = self._cpu_model().forecast(context[r], horizon)
                self.fallback_rows += len(r)
            out[r] = q
        if self.check is None and len(rows):  # agreement on MPS rows, stored in the manifest
            r = rows[: self.sub]
            a, b = out[r], self._cpu_model().forecast(context[r], horizon)
            x = context[r]
            rel = np.abs(a - b) / (np.abs(b).mean(axis=(1, 2), keepdims=True) + 1e-6)
            self.check = {
                "n": len(x),
                "median_rel_diff": float(np.median(rel)),
                "max_rel_diff": float(rel.max()),
            }
        return out


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
    meta = {**meta, "ids_sha": ids_fingerprint(ids), "encoding": ENCODING}
    man = json.loads(man_path.read_text()) if man_path.exists() else {**meta, "shards": {}}
    if man_path.exists():
        for k in ("checkpoint", "levels", "horizon", "ids_sha", "encoding"):
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
        "nan": int(
            (~np.isfinite(q)).sum()
        ),  # NaN *and* inf (v1 only checked NaN and missed overflows)
        "non_monotone_frac": float((np.diff(q, axis=1) < -1e-3).mean()),
    }
    if report["n"] != expected_n or report["nan"]:
        raise ValueError(f"cache integrity failed: {report}")
    return report


def cache_teacher_on_corpus(entry, spec, corpus, log: Callable[[str], None] = print) -> Path:
    """Run teacher `entry` (a ModelEntry) over `corpus` (built from CorpusSpec `spec`); resumable."""
    import importlib.metadata as md

    from tinycast.config import QUANTILE_LEVELS
    from tinycast.eval.benchmark import git_sha
    from tinycast.models import build_model

    pkg = {
        "toto": "toto-2",
        "t0": "tfc-t0",
        "chronos2": "chronos-forecasting",
        "timestral": "timestral",
    }[entry.kind]
    d = cache_dir(entry.name, spec.key())
    man = d / "manifest.json"
    if (
        man.exists()
        and json.loads(man.read_text()).get("encoding") == ENCODING
        and json.loads(man.read_text()).get("ids_sha") == ids_fingerprint(corpus.ids)
        and sum(json.loads(man.read_text())["shards"].values()) == len(corpus.ids)
    ):
        log(f"[{entry.name}] cache complete: {d}")
        return d
    import os

    import torch

    if entry.device == "auto" and os.environ.get("TINYCAST_CACHE_DEVICE") == "mps_guarded":
        entry.device = "mps_guarded"  # opt-in per run (Stage C3): MPS + CPU for low-information / non-finite rows
    if entry.device == "auto":
        # CPU (or CUDA), never Apple MPS: Toto-2.0-313m returns non-finite quantiles on MPS for a constant context
        # (CPU is fine; Stage B, 2026-11-13), and the benchmarks ran every teacher on CPU: labels = the measured teacher.
        entry.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GuardedMPS(entry) if entry.device == "mps_guarded" else build_model(entry)
    meta = {
        "teacher": entry.name,
        "checkpoint": entry.checkpoint,
        "package": f"{pkg}=={md.version(pkg)}",
        "levels": list(QUANTILE_LEVELS),
        "native_levels": list(model.native_levels),
        "horizon": spec.horizon,
        "context_length": spec.context_length,
        "corpus_key": spec.key(),
        "device": entry.device,
        "git_sha": git_sha(),
    }
    out = write_cache(
        entry.name,
        spec.key(),
        corpus.ids,
        corpus.context,
        lambda c: model.forecast(c, spec.horizon),
        meta,
        log=log,
    )
    if isinstance(model, GuardedMPS):  # record how the labels were produced
        man_path = out / "manifest.json"
        man = json.loads(man_path.read_text())
        man["mps_guard"] = {
            "check_first_batch": model.check,
            "cpu_rows_this_run": model.fallback_rows,
        }
        man_path.write_text(json.dumps(man, indent=1))
        log(f"[{entry.name}] MPS guard: {model.fallback_rows} rows on CPU, check {model.check}")
    log(str(check_cache(entry.name, spec.key(), len(corpus.ids))))
    return out
