"""Paired comparisons shown on the website that no study table contains (same windows, cluster bootstrap over series).

    uv run python scripts/site_pairs.py
-> results/site_pairs.json

Both models must have been run on the same suite (results/<suite>/<model>.json, e.g. by scripts/benchmark.py).
< 1 means a is better.
"""

from __future__ import annotations

import json

import numpy as np

from tinycast.eval.benchmark import ROOT
from tinycast.eval.metrics import paired_ratio_ci

SUITES = ("fresh_fr_power", "shapes")
PAIRS = [  # (a, b, question)
    ("student-toto-onnx-int8", "student-toto-onnx-fp32", "int8 vs fp32, same student"),
    ("student-toto-onnx-fp32", "student-preview-gt-m",
     "distilled general student (1M windows, Toto labels) vs a student trained on real futures of 12k "
     "pre-cutoff French electricity windows + 12k synthetic"),
]  # fmt: skip


def wql(suite: str, model: str) -> tuple[np.ndarray, list[str]]:
    r = json.loads((ROOT / "results" / suite / f"{model}.json").read_text())
    return np.array([np.nan if v is None else v for v in r["per_series"]["wql"]]), r["ids"]


def main() -> None:
    out = []
    for suite in SUITES:
        s = json.loads((ROOT / "results" / suite / "_suite.json").read_text())
        groups = [m.get("series_id", i) for m, i in zip(s["meta"], s["ids"], strict=True)]
        for a, b, q in PAIRS:
            wa, ia = wql(suite, a)
            wb, ib = wql(suite, b)
            if ia != ib or ia != s["ids"]:
                raise ValueError(f"{suite}: {a} and {b} were not scored on the same windows")
            p, lo, hi = paired_ratio_ci(wa, wb, groups=np.array(groups))
            out.append({"suite": suite, "a": a, "b": b, "question": q, "ratio": [p, lo, hi]})
            print(f"{suite:15s} {a} / {b}: {p:.3f} [{lo:.3f}, {hi:.3f}]")
    (ROOT / "results" / "site_pairs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
