"""Robustness of fine-tuned students from existing runs (no training) -> results/study/diagnostics/robustness.json

1. Ensembling: average the quantiles of the 3 seeds of each Stage-D (real-data) fine-tune; paired vs the mean single seed.
2. Forgetting: electricity / pedestrian fine-tuned students (1 year) scored on the OTHER suites vs the zero-shot student.
"""

import json

import numpy as np

from tinycast.config import QUANTILE_LEVELS, BenchConfig, load
from tinycast.eval import metrics as M
from tinycast.eval.benchmark import ROOT, load_suite
from tinycast.models.student import StudentForecaster

runs = json.loads((ROOT / "results" / "study" / "stage_d_final" / "runs.json").read_text())
ZS = runs["distilled_zero_shot"]


def suite_and_naive(name):
    cfg = load(ROOT / "configs" / "eval" / f"{name}.yaml", BenchConfig)
    s = load_suite(cfg.suite, cfg)
    naive = np.array(
        json.loads((ROOT / "results" / name / "seasonal_naive.json").read_text())["per_series"][
            "wql"
        ],
        float,
    )
    return s, naive, [m.get("series_id", i) for m, i in zip(s.meta, s.ids, strict=True)]


def predict(run, s):
    return StudentForecaster.from_run(str(ROOT / run)).forecast(s.context, 64)


out = {"ensemble": {}, "forgetting": {}}
for d, suite in (("fr", "fresh_fr_power"), ("ped", "pedestrian_test")):
    s, naive, groups = suite_and_naive(suite)
    for h in ("1w", "1m", "1y"):
        qs = [predict(runs[f"ft_warm_{d}_{h}_distilled_s{k}"], s) for k in range(3)]
        w_single = np.mean([M.wql(s.future, q, QUANTILE_LEVELS) for q in qs], axis=0)
        w_ens = M.wql(s.future, np.mean(qs, axis=0), QUANTILE_LEVELS)
        rel = M.bootstrap_ci(M.relative_scores(w_ens, naive), "geomean", groups=groups)
        vs = M.paired_ratio_ci(w_ens, w_single, groups=groups)
        out["ensemble"][f"{d}_{h}"] = {"ensemble_rel_wql": rel, "ensemble_vs_single": vs}
        print(
            f"ensemble of 3 seeds {d} {h}: rel {rel[0]:.3f}  vs average single seed {vs[0]:.3f} [{vs[1]:.3f}, {vs[2]:.3f}]",
            flush=True,
        )
for src_d, src_suite in (("fr", "fresh_fr_power"), ("ped", "pedestrian_test")):
    for other in ("shapes", "multidomain_test", "fresh_fr_power", "pedestrian_test"):
        if other == src_suite:
            continue
        s, naive, groups = suite_and_naive(other)
        w0 = M.wql(s.future, predict(ZS, s), QUANTILE_LEVELS)
        w1 = np.mean(
            [
                M.wql(
                    s.future,
                    predict(runs[f"ft_warm_{src_d}_1y_distilled_s{k}"], s),
                    QUANTILE_LEVELS,
                )
                for k in range(3)
            ],
            axis=0,
        )
        vs = M.paired_ratio_ci(w1, w0, groups=groups)
        out["forgetting"][f"{src_d}_1y_on_{other}"] = vs
        print(
            f"fine-tuned on {src_d} (1y), scored on {other:17s}: vs zero-shot student {vs[0]:.3f} [{vs[1]:.3f}, {vs[2]:.3f}]",
            flush=True,
        )
d = ROOT / "results" / "study" / "diagnostics"
d.mkdir(parents=True, exist_ok=True)
(d / "robustness.json").write_text(json.dumps(out, indent=1))
