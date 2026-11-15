"""Long-horizon measurement: how far can the 64-step student go by rolling out, vs models with their own long decoding.

    uv run python scripts/long_horizon.py [--suites fresh_fr_power,...] [--horizons 64,128,256,512] [--models a,b]
-> results/long_horizon/<suite>@<H>/<model>.json + summary.json (same format as results/<suite>/)

Every model sees the same 512-step context (Timestral could read up to 32k steps; giving it more would measure context
length, not horizon). Each comparator uses its own long-horizon method through its official API:
Timestral decodes up to 1024 steps in one block; Toto and Chronos-Bolt use their package's decoding.
Fresh electricity has 28 days of targets (2026-10-15 -> 11-12), so H = 512 leaves ~8 daily origins per series.
"""

from __future__ import annotations

import argparse
import json

from tinycast.config import BenchConfig, ModelEntry, load
from tinycast.eval.benchmark import ROOT, load_suite, run_model, summarize, write_suite
from tinycast.models import build_model
from tinycast.models.rollout import RolloutForecaster

STUDENT_RUN = (
    "runs/2026-11-15_stage_c2_kd_v2_1m"  # the site's student (selected on validation, Stage C2)
)
COMPARATORS = [
    ModelEntry(kind="timestral", name="timestral-28m", checkpoint="mistralai/Timestral-28M"),
    ModelEntry(kind="timestral", name="timestral-5m", checkpoint="mistralai/Timestral-5M"),
    ModelEntry(kind="toto", name="toto-2.0-4m", checkpoint="Datadog/Toto-2.0-4m"),
    ModelEntry(
        kind="chronos_bolt", name="chronos-bolt-tiny", checkpoint="amazon/chronos-bolt-tiny"
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="fresh_fr_power,shapes,multidomain_test")
    ap.add_argument("--horizons", default="64,128,256,512")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    wanted = set(a.models.split(",")) if a.models else None
    for suite_name in a.suites.split(","):
        base_cfg = load(ROOT / "configs" / "eval" / f"{suite_name}.yaml", BenchConfig)
        for H in [int(h) for h in a.horizons.split(",")]:
            cfg = BenchConfig(suite=suite_name, horizon=H, context_length=512, n_seeds=base_cfg.n_seeds,
                              latency=False, season=base_cfg.season, models=[])  # fmt: skip
            if (
                suite_name == "fresh_fr_power"
            ):  # tolerate the one missing DST hour (see fresh_test_windows)
                from tinycast.data.sources.rte_eco2mix import fresh_suite

                suite = fresh_suite(cfg, min_finite=0.99)
            else:
                suite = load_suite(suite_name, cfg)
            out = ROOT / "results" / "long_horizon" / f"{suite_name}@{H}"
            write_suite(suite, out)
            sj = json.loads(
                (out / "_suite.json").read_text()
            )  # keep ids/meta, drop the arrays (rebuildable)
            sj = {k: v for k, v in sj.items() if k not in ("context", "future")}
            (out / "_suite.json").write_text(json.dumps(sj))
            print(f"== {suite_name} @ {H}: {len(suite.ids)} windows", flush=True)
            naive = ModelEntry(kind="seasonal_naive", name="seasonal_naive")
            if cfg.season:  # same rule as scripts/benchmark.py
                naive.season = list(suite.season) if cfg.season == "freq" else cfg.season
            entries: list[tuple[ModelEntry, object]] = [(naive, None)]
            from tinycast.models.student import StudentForecaster

            stu = StudentForecaster.from_run(STUDENT_RUN, name="student")
            for mode in ("median", "paths"):
                e = ModelEntry(
                    kind="student_rollout", name=f"student-rollout-{mode}", checkpoint=STUDENT_RUN
                )
                entries.append((e, RolloutForecaster(stu, e.name, mode=mode)))
            entries += [(e, None) for e in COMPARATORS]
            for e, model in entries:
                if wanted and e.name not in wanted and e.name != "seasonal_naive":
                    continue
                m = model or build_model(e)
                r = run_model(m, e, suite, cfg, out)
                slim = {
                    k: v for k, v in r.items() if k != "forecasts_f16"
                }  # ~50 MB per model at H=512
                (out / f"{e.name}.json").write_text(json.dumps(slim))
                print(f"  {e.name:24s} {r['wall_seconds_full_suite']:7.1f}s", flush=True)
            s = summarize(out)
            for row in s["rows"]:
                print(
                    f"  {row['model']:24s} relWQL {row['rel_wql']:.3f} [{row['rel_wql_ci'][0]:.3f},{row['rel_wql_ci'][1]:.3f}]  cov80 {row['mean_cov80']:.2f}"
                )
            (out / "_run.json").write_text(
                json.dumps({"student_run": STUDENT_RUN, "context_length": 512})
            )


if __name__ == "__main__":
    main()
