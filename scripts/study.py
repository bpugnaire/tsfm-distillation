"""Run one stage of the distillation study: corpus -> teacher cache -> train each arm -> evaluate -> table.

    uv run python scripts/study.py configs/study/stage_a.yaml

Resumable: finished arms (run dir with metrics.json listed in results/study/<name>/runs.json) are skipped.
Outputs: results/study/<name>/<suite>/summary.json, results/study/<name>/table.json (+ table.md).
Comparisons between arms are PAIRED (same windows), with a cluster bootstrap over series.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tinycast.config import BenchConfig, ModelEntry, TrainConfig, from_dict, load, to_dict
from tinycast.data.cache import cache_teacher_on_corpus
from tinycast.data.corpus import build_corpus
from tinycast.distill.train import specs, train
from tinycast.eval import metrics as M
from tinycast.eval.benchmark import ROOT, load_suite, run_model, summarize, write_suite
from tinycast.models import build_model
from tinycast.models.student import StudentForecaster


def arm_config(study: dict, arm: dict) -> TrainConfig:
    base = to_dict(load(ROOT / study["base"], TrainConfig))
    base["name"] = f"{study['name']}_{arm['name']}"
    base["data"]["mix"] = arm["mix"]
    base["loss"]["alpha"] = arm.get("alpha", base["loss"]["alpha"])
    base["data"]["teacher"] = (
        arm.get("teacher", study.get("teacher")) if base["loss"]["alpha"] > 0 else None
    )
    for section, values in arm.get("override", {}).items():  # e.g. {loss: {kd: quantile_huber}}
        base[section].update(values)
    base["student"]["quantile_levels"] = list(base["student"]["quantile_levels"])
    return from_dict(TrainConfig, base, where=f"arm {arm['name']}")


def per_series(out: Path, model: str, key: str = "wql") -> np.ndarray:
    r = json.loads((out / f"{model}.json").read_text())
    return np.array([np.nan if v is None else v for v in r["per_series"][key]], dtype=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--only-eval", action="store_true")
    a = ap.parse_args()
    study = yaml.safe_load(Path(a.study).read_text())
    sdir = ROOT / "results" / "study" / study["name"]
    sdir.mkdir(parents=True, exist_ok=True)
    runs_file = sdir / "runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.exists() else {}

    # ---------------------------------------------------------------- train every arm
    for arm in study["arms"]:
        if arm["name"] in runs and (ROOT / runs[arm["name"]] / "metrics.json").exists():
            print(f"[{arm['name']}] done already: {runs[arm['name']]}", flush=True)
            continue
        if a.only_eval:
            continue
        cfg = arm_config(study, arm)
        tr_spec, va_spec = specs(cfg)
        print(f"[{arm['name']}] corpus {dict(tr_spec.mix)} (key {tr_spec.key()})", flush=True)
        corpus = build_corpus(tr_spec)
        build_corpus(va_spec)
        if cfg.loss.alpha > 0:
            entry = load(ROOT / "configs" / "teachers" / f"{cfg.data.teacher}.yaml", ModelEntry)
            cache_teacher_on_corpus(entry, tr_spec, corpus)
        run = train(cfg, log_every=500)
        runs[arm["name"]] = str(run.relative_to(ROOT))
        runs_file.write_text(json.dumps(runs, indent=1))

    # ---------------------------------------------------------------- evaluate every arm on every suite
    table = {"study": study["name"], "reference_arm": study["reference_arm"], "suites": {}}
    for suite_name in study["eval_suites"]:
        bcfg = load(ROOT / "configs" / "eval" / f"{suite_name}.yaml", BenchConfig)
        bcfg.latency = False
        suite = load_suite(bcfg.suite, bcfg)
        out = sdir / suite_name
        write_suite(suite, out)
        sn = ModelEntry(kind="seasonal_naive", name="seasonal_naive")
        if bcfg.season:
            sn.season = list(suite.season) if bcfg.season == "freq" else bcfg.season
        run_model(build_model(sn), sn, suite, bcfg, out)
        for arm_name, run in runs.items():
            entry = ModelEntry(kind="student", name=arm_name, checkpoint=str(ROOT / run))
            run_model(
                StudentForecaster.from_run(str(ROOT / run), name=arm_name), entry, suite, bcfg, out
            )
        summ = summarize(out)
        groups = [m.get("series_id", i) for m, i in zip(suite.meta, suite.ids, strict=True)]
        ref = study["reference_arm"]
        rows = {}
        for r in summ["rows"]:
            row = {"rel_wql": r["rel_wql"], "rel_wql_ci": r["rel_wql_ci"], "cov80": r["mean_cov80"]}
            if r["model"] in runs and ref in runs and r["model"] != ref:
                p, lo, hi = M.paired_ratio_ci(
                    per_series(out, r["model"]), per_series(out, ref), groups=groups
                )
                row["vs_reference"] = [p, lo, hi]
            rows[r["model"]] = row
        teachers_summary = ROOT / "results" / suite_name / "summary.json"
        teach = {}
        if teachers_summary.exists():
            for r in json.loads(teachers_summary.read_text())["rows"]:
                if r["kind"] not in ("student", "student_onnx", "seasonal_naive"):
                    teach[r["model"]] = {
                        "rel_wql": r["rel_wql"],
                        "rel_wql_ci": r["rel_wql_ci"],
                        "n_params": r["n_params"],
                    }
        table["suites"][suite_name] = {
            "arms": rows,
            "zero_shot_models": teach,
            "n_windows": len(suite.ids),
            "flag": suite.flag,
        }

    (sdir / "table.json").write_text(json.dumps(table, indent=1))
    lines = [
        f"# {study['name']}: relative WQL vs seasonal naive (95% CI); paired ratio vs `{study['reference_arm']}` (<1 = better)",
        "",
    ]
    for s, d in table["suites"].items():
        lines += [
            f"## {s} ({d['n_windows']} windows){'  — ' + d['flag'] if d['flag'] else ''}",
            "",
            "| arm | rel. WQL | 95% CI | vs reference | 80% cov. |",
            "|---|---|---|---|---|",
        ]
        for m, r in sorted(d["arms"].items(), key=lambda kv: kv[1]["rel_wql"]):
            vs = r.get("vs_reference")
            vs_s = f"{vs[0]:.3f} [{vs[1]:.3f}, {vs[2]:.3f}]" if vs else "–"
            lines.append(
                f"| {m} | {r['rel_wql']:.3f} | [{r['rel_wql_ci'][0]:.3f}, {r['rel_wql_ci'][1]:.3f}] | {vs_s} | {r['cov80']:.2f} |"
            )
        if d["zero_shot_models"]:
            lines += [
                "",
                "zero-shot models (same suite, same baseline): "
                + ", ".join(
                    f"{k} {v['rel_wql']:.3f}"
                    for k, v in sorted(
                        d["zero_shot_models"].items(), key=lambda kv: kv[1]["rel_wql"]
                    )
                ),
            ]
        lines.append("")
    (sdir / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
