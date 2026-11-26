"""Core claim with 3 training seeds: teacher labels (Timestral-5M) vs real future, same windows, at 1M and 3M windows.

    uv run python scripts/seeds_summary.py  -> results/seeds_core_claim.json

GIFT-Eval short (55 configs): per seed, the paired ratio KD/GT over configs; pooled, the ratio of the per-config
geometric means over the 3 seeds, with a bootstrap that resamples configs AND, within each arm, seeds (so the interval
covers training randomness, not only the choice of test configs). Seed 0 runs: C14 kd5m_1m, C13 gt_1m, C15 kd5m_3m / gt_3m.
"""

from __future__ import annotations

import json

import numpy as np

from scripts.gift_eval_pairs import load
from tinycast.eval.benchmark import ROOT
from tinycast.eval.metrics import paired_ratio_ci

RUNS = {
    "1m": {"kd": ["tinycast-kd5m_1m", "tinycast-kd5m_1m_s1", "tinycast-kd5m_1m_s2"],
           "gt": ["tinycast-gt_1m", "tinycast-gt_1m_s1", "tinycast-gt_1m_s2"]},
    "3m": {"kd": ["tinycast-kd5m_3m", "tinycast-kd5m_3m_s1", "tinycast-kd5m_3m_s2"],
           "gt": ["tinycast-gt_3m", "tinycast-gt_3m_s1", "tinycast-gt_3m_s2"]},
}  # fmt: skip


def main() -> None:
    naive = load("Seasonal_Naive")
    out = {}
    rng = np.random.default_rng(0)
    for scale, arms in RUNS.items():
        S = {k: [load(m) for m in v] for k, v in arms.items()}
        cfg = sorted(
            set.intersection(*(set(s.index) for v in S.values() for s in v)) & set(naive.index)
        )
        L = {
            k: np.log(np.stack([s.loc[cfg].to_numpy() / naive.loc[cfg].to_numpy() for s in v]))
            for k, v in S.items()
        }
        scores = {k: [float(np.exp(x.mean())) for x in L[k]] for k in L}
        per_seed = [list(paired_ratio_ci(np.exp(L["kd"][i]), np.exp(L["gt"][i]))) for i in range(3)]
        pooled = float(np.exp(L["kd"].mean(0).mean() - L["gt"].mean(0).mean()))
        boot = []
        for _ in range(4000):
            c = rng.integers(0, len(cfg), len(cfg))
            sk, sg = rng.integers(0, 3, 3), rng.integers(0, 3, 3)
            boot.append(np.exp(L["kd"][sk][:, c].mean() - L["gt"][sg][:, c].mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[scale] = {"n_configs": len(cfg), "scores": scores, "kd_over_gt_per_seed": per_seed,
                      "kd_over_gt_pooled": [pooled, float(lo), float(hi)]}  # fmt: skip
        print(
            f"{scale}: KD {[f'{x:.3f}' for x in scores['kd']]}  GT {[f'{x:.3f}' for x in scores['gt']]}"
        )
        print(f"    per seed KD/GT {[f'{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]' for r in per_seed]}")
        print(f"    pooled KD/GT {pooled:.3f} [{lo:.3f}, {hi:.3f}] (configs and seeds resampled)")
    # does the gap widen from 1M to 3M? ratio of the two KD/GT ratios, configs and seeds resampled (same configs)
    LL = {}
    for scale, arms in RUNS.items():
        S = {k: [load(m) for m in v] for k, v in arms.items()}
        LL[scale] = {
            k: np.log(np.stack([s.loc[cfg].to_numpy() / naive.loc[cfg].to_numpy() for s in v]))
            for k, v in S.items()
        }

    def gap(d, c, idx):
        return d["kd"][idx[0]][:, c].mean() - d["gt"][idx[1]][:, c].mean()

    w = np.exp(
        gap(LL["3m"], slice(None), (slice(None), slice(None)))
        - gap(LL["1m"], slice(None), (slice(None), slice(None)))
    )
    bw = []
    for _ in range(4000):
        c = rng.integers(0, len(cfg), len(cfg))
        i3 = (rng.integers(0, 3, 3), rng.integers(0, 3, 3))
        i1 = (rng.integers(0, 3, 3), rng.integers(0, 3, 3))
        bw.append(np.exp(gap(LL["3m"], c, i3) - gap(LL["1m"], c, i1)))
    lo, hi = np.percentile(bw, [2.5, 97.5])
    out["widening_3m_over_1m"] = [float(w), float(lo), float(hi)]
    print(f"widening: (KD/GT at 3M) / (KD/GT at 1M) = {w:.3f} [{lo:.3f}, {hi:.3f}]")
    (ROOT / "results" / "seeds_core_claim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
