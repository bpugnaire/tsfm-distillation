"""Is any benchmark series (or piece of one) inside our training regimen? -> results/contamination/<benchmark>.json

Training side (everything the student or its fine-tuning can see, ALL positions of every series, conservatively):
  the 11 multi-domain training datasets, the held-out pedestrian dataset (fine-tuning only), French electricity (RTE).
  Synthetic data is random by construction; teacher rollouts start from the multi-domain series (covered).
Benchmark side: GIFT-Eval (Salesforce/GiftEval) and fev (autogluon/fev_datasets), every dataset / frequency / column.

    uv run python scripts/contamination_check.py
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from tinycast.data.sources import hf_datasets as H
from tinycast.eval.contamination import Fingerprint, check_dataset

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "raw" / "benchmarks"


def training_series() -> dict[str, list[np.ndarray]]:
    out = {name: [s.values for s in H.load(name)] for name in H.DATASETS}
    try:
        from tinycast.data.sources.rte_eco2mix import series_arrays

        out["rte_eco2mix"] = [v for _, v in series_arrays().values()]
    except FileNotFoundError:
        pass
    return out


def _series_from_table(t: pa.Table) -> list[np.ndarray]:
    """Every numeric sequence column (univariate or multivariate rows) -> list of 1-D arrays."""
    out = []
    for col in t.column_names:
        if col in ("id", "item_id", "timestamp", "start", "freq"):
            continue
        typ = t.schema.field(col).type
        if not (pa.types.is_list(typ) or pa.types.is_large_list(typ)):
            continue
        inner = typ.value_type
        while pa.types.is_list(inner) or pa.types.is_large_list(inner):
            inner = inner.value_type
        if not (pa.types.is_floating(inner) or pa.types.is_integer(inner)):
            continue  # text / categorical sequences (e.g. holiday names in fev)
        for row in t.column(col).to_pylist():
            if row is None or len(row) == 0:
                continue
            if isinstance(row[0], list):  # multivariate: one series per variate
                out += [np.array(v, dtype=np.float64) for v in row if v is not None]
            elif isinstance(row[0], (int, float)) or row[0] is None:
                out.append(np.array([np.nan if v is None else v for v in row], dtype=np.float64))
    return out


def gift_eval() -> dict[str, list[np.ndarray]]:
    out = {}
    for f in sorted((BENCH / "GiftEval").rglob("data-*.arrow")):
        name = "/".join(f.relative_to(BENCH / "GiftEval").parts[:-1])
        with ipc.open_stream(f) as r:
            out[name] = _series_from_table(r.read_all())
    return out


def fev() -> dict[str, list[np.ndarray]]:
    out = {}
    for f in sorted((BENCH / "fev_datasets").rglob("*.parquet")):
        name = "/".join(f.relative_to(BENCH / "fev_datasets").parts[:-1])
        out.setdefault(name, []).extend(_series_from_table(pq.read_table(f)))
    return out


if __name__ == "__main__":
    train = training_series()
    fp = Fingerprint.build(train)
    print(
        f"training fingerprint: {len(fp.hashes):,} chunk hashes from {len(train)} datasets",
        flush=True,
    )
    outdir = ROOT / "results" / "contamination"
    outdir.mkdir(parents=True, exist_ok=True)
    for bench, loader in (("gift_eval", gift_eval), ("fev", fev)):
        rep = {}
        for name, series in loader().items():
            rep[name] = check_dataset(fp, series)
            r = rep[name]
            if r["seen_share"] > 0:
                print(
                    f"{bench:9s} {name:38s} seen {100 * r['seen_share']:6.2f}%  tail {100 * r['seen_share_tail']:6.2f}%  <- {r['matched_training_datasets']}",
                    flush=True,
                )
        clean = sorted(k for k, v in rep.items() if v["seen_share"] == 0)
        summary = {
            "benchmark": bench,
            "training_datasets": fp.names,
            "n_benchmark_datasets": len(rep),
            "clean": clean,
            "contaminated": {k: v["seen_share"] for k, v in rep.items() if v["seen_share"] > 0},
            "per_dataset": rep,
        }
        (outdir / f"{bench}.json").write_text(json.dumps(summary, indent=1))
        print(f"{bench}: {len(clean)}/{len(rep)} datasets with zero overlap", flush=True)
