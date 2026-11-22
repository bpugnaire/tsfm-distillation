"""Paired GIFT-Eval comparisons (same 55 short-term configs): per-config CRPS ratio a/b, geometric mean, bootstrap CI.

    uv run python scripts/gift_eval_pairs.py
-> results/gift_eval/pairs.json

Configs are the resampling unit (one score per config, the leaderboard's granularity). < 1 means a is better.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tinycast.eval.benchmark import ROOT
from tinycast.eval.metrics import paired_ratio_ci

GE = ROOT / "results" / "gift_eval"
LB = ROOT / "external" / "gift-eval" / "results"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
PAIRS = [  # (a, b, question); names are result folders under results/gift_eval/ or the leaderboard
    ("tinycast-kd5m_3m", "Toto-2.0-4m@ctx512-univariate", "site student (3M windows, Timestral-5M) vs Toto-2.0-4m, same 512 context"),
    ("tinycast-kd5m_3m", "Toto-2.0-4m", "site student vs Toto-2.0-4m as on the leaderboard (4096 context)"),
    ("tinycast-kd5m_3m", "Timestral-5M", "site student vs its teacher Timestral-5M as on the leaderboard"),
    ("tinycast-kd5m_3m", "tinycast-gt_3m", "teacher labels vs real future, 3M windows"),
    ("tinycast-kd5m_3m+flip", "tinycast-kd5m_3m", "site student: sign-flip averaging vs plain"),
    ("tinycast-c3-kd_toto_1m", "Toto-2.0-4m@ctx512-univariate", "Toto student vs Toto-2.0-4m, same 512 context, univariate"),
    ("tinycast-c3-kd_toto_1m", "Toto-2.0-4m", "Toto student vs Toto-2.0-4m as on the leaderboard (4096 context)"),
    ("tinycast-c3-kd_toto_1m", "tinycast-c2", "Toto student vs Timestral-taught student"),
    ("tinycast-c2", "Toto-2.0-4m@ctx512-univariate", "Timestral-taught student vs Toto-2.0-4m, same 512 context"),
    ("Toto-2.0-4m@ctx512-univariate", "Toto-2.0-4m", "Toto-2.0-4m: 512 vs 4096 context"),
]  # fmt: skip


def load(name: str) -> pd.Series:
    d = GE / name if (GE / name).exists() else LB / name
    df = pd.read_csv(d / "all_results.csv").drop_duplicates("dataset").set_index("dataset")
    return df[CRPS].astype(float)


def main() -> None:
    out = []
    for a, b, q in PAIRS:
        sa, sb = load(a), load(b)
        cfg = sorted(set(sa.index) & set(sb.index))
        p, lo, hi = paired_ratio_ci(sa.loc[cfg].to_numpy(), sb.loc[cfg].to_numpy())
        wins = int((sa.loc[cfg] < sb.loc[cfg]).sum())
        out.append(
            {
                "a": a,
                "b": b,
                "question": q,
                "n_configs": len(cfg),
                "ratio": [p, lo, hi],
                "a_wins": wins,
            }
        )
        print(f"{q}: {p:.3f} [{lo:.3f}, {hi:.3f}], a better on {wins}/{len(cfg)}")
    # leaderboard-style score (geometric mean of CRPS / Seasonal Naive) for local runs that are not in summary.json
    naive = load("Seasonal_Naive")
    scores = {}
    for name in ("Toto-2.0-4m@ctx512-univariate",):
        s = load(name)
        scores[name] = float(np.exp(np.log(s / naive.loc[s.index]).mean()))
        print(f"{name}: CRPS ratio to Seasonal Naive {scores[name]:.3f} on {len(s)} configs")
    (GE / "pairs.json").write_text(json.dumps({"pairs": out, "scores": scores}, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
