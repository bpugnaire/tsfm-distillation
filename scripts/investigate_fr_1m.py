"""Diagnostic for the Stage-D anomaly: fine-tuning on the LAST MONTH of electricity hurts (1.14x zero-shot).

H2 overfitting: fine-tune for 25 / 100 / 300 / 1000 steps (no early stopping) and score each on the fresh test.
H3 seasonal regime: fine-tune on the same calendar month ONE YEAR EARLIER (targets 2025-09-15 .. 2025-10-14) and on the
   month that matches the test season one year earlier (targets 2025-10-15 .. 2025-11-12), same steps.
1 seed, diagnostic only (-> results/study/diagnostics/fr_1m.json).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tinycast.config import QUANTILE_LEVELS, BenchConfig, TrainConfig, from_dict, load, to_dict
from tinycast.data.target_domain import history_corpora
from tinycast.distill.train import train
from tinycast.eval import metrics as M
from tinycast.eval.benchmark import ROOT, load_suite
from tinycast.models.student import StudentForecaster

INIT = ROOT / "runs" / "2026-11-13_stage_c_lite_kd_mix_m_300k"


def score(run_or_model, suite, naive):
    sf = (
        StudentForecaster.from_run(str(run_or_model))
        if isinstance(run_or_model, Path)
        else run_or_model
    )
    w = M.wql(suite.future, sf.forecast(suite.context, 64), QUANTILE_LEVELS)
    groups = [m["series_id"] for m in suite.meta]
    return w, M.bootstrap_ci(M.relative_scores(w, naive), "geomean", groups=groups)


if __name__ == "__main__":
    bcfg = load(ROOT / "configs" / "eval" / "fresh_fr_power.yaml", BenchConfig)
    suite = load_suite("fresh_fr_power", bcfg)
    naive = np.array(
        json.loads((ROOT / "results" / "fresh_fr_power" / "seasonal_naive.json").read_text())[
            "per_series"
        ]["wql"],
        float,
    )
    groups = [m["series_id"] for m in suite.meta]
    w0, zs = score(INIT, suite, naive)
    out = {"zero_shot": zs, "runs": []}
    print(f"zero-shot student: {zs[0]:.3f} [{zs[1]:.3f}, {zs[2]:.3f}]", flush=True)
    variants = [("last month (Sep 15 - Oct 14, 2026)", None)]
    variants += [
        ("same month 1 year earlier (Sep 15 - Oct 14, 2025)", pd.Timestamp("2025-10-15", tz="UTC"))
    ]
    variants += [
        (
            "test-season month 1 year earlier (Oct 15 - Nov 12, 2025)",
            pd.Timestamp("2025-11-13", tz="UTC"),
        )
    ]
    base = to_dict(load(ROOT / "configs" / "train" / "study_base_m.yaml", TrainConfig))
    base["student"] = json.loads((INIT / "student_config.json").read_text())
    base["loss"]["alpha"] = 0.0
    for label, end in variants:
        corpora = history_corpora("fr_power", 720, mode="warm", end_time=end)
        for steps in [25, 100, 300, 1000] if end is None else [100, 1000]:
            cfg_d = json.loads(json.dumps(base))
            cfg_d["name"] = f"diag_fr1m_{'last' if end is None else end.strftime('%Y%m%d')}_{steps}"
            cfg_d["optim"].update(
                max_steps=steps,
                lr=3e-4,
                eval_every=steps,
                warmup_steps=max(5, steps // 20),
                patience=99,
                batch_size=128,
            )
            run = train(
                from_dict(TrainConfig, cfg_d), corpora=corpora, init_from=INIT, log_every=10**6
            )
            w, r = score(run, suite, naive)
            vs = M.paired_ratio_ci(w, w0, groups=groups)
            out["runs"].append(
                {
                    "history": label,
                    "steps": steps,
                    "rel_wql": r,
                    "vs_zero_shot": vs,
                    "run": str(run.relative_to(ROOT)),
                }
            )
            print(
                f"{label:55s} steps {steps:5d}: rel {r[0]:.3f}  vs zero-shot {vs[0]:.3f} [{vs[1]:.3f}, {vs[2]:.3f}]",
                flush=True,
            )
    d = ROOT / "results" / "study" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fr_1m.json").write_text(json.dumps(out, indent=1))
