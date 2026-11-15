"""Recompute cached teacher windows whose labels are all-zero (t0 replaces NaN/Inf with 0.0 internally) on CPU.

    uv run python scripts/repair_cache.py --teacher t0-gamma --corpus-key 48dd0113d4
Rewrites only the affected shards; records the repair in the manifest. Raises if CPU output is still degenerate.
"""

import argparse
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tinycast.config import ModelEntry, load
from tinycast.data.cache import _decode, _encode, cache_dir
from tinycast.eval.benchmark import ROOT
from tinycast.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--corpus-key", required=True)
a = ap.parse_args()
d = cache_dir(a.teacher, a.corpus_key)
man = json.loads((d / "manifest.json").read_text())
K, H = len(man["levels"]), man["horizon"]
corpus = np.load(ROOT / "data" / "processed" / f"corpus_train_{a.corpus_key}.npz")
pos = {w: i for i, w in enumerate(corpus["ids"])}
entry = load(ROOT / "configs" / "teachers" / f"{a.teacher}.yaml", ModelEntry)
entry.device = "cpu"
model = None
repaired = []
for name in sorted(man["shards"]):
    t = pq.read_table(d / name)
    ids = t["window_id"].to_numpy(zero_copy_only=False)
    q = _decode(t["q"].to_pylist(), K, H).astype(np.float32)
    bad = np.where((q == 0).all(axis=(1, 2)))[0]
    if not len(bad):
        continue
    model = model or build_model(entry)
    ctx = corpus["context"][[pos[w] for w in ids[bad]]]
    q[bad] = model.forecast(ctx, H)  # raises if CPU also returns all-zero
    pq.write_table(
        pa.table({"window_id": pa.array(ids.tolist()), "q": pa.array(_encode(q), pa.binary())}),
        d / name,
    )
    repaired += [str(w) for w in ids[bad]]
    print(f"{name}: recomputed {len(bad)} window(s) on CPU")
man.setdefault("repairs", []).append(
    {"rule": "all-zero rows recomputed on CPU", "windows": repaired}
)
(d / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"repaired {len(repaired)} windows")
