"""Run one stage of the distillation study: corpus -> teacher cache -> train each arm -> evaluate -> table.

    uv run python scripts/study.py configs/study/stage_a.yaml

Resumable: finished arms (run dir with metrics.json listed in results/study/<name>/runs.json) are skipped.
Outputs: results/study/<name>/<suite>/summary.json, results/study/<name>/table.json (+ table.md).
Comparisons between arms are PAIRED (same windows), with a cluster bootstrap over series.
"""

import argparse
import json
import sys
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
    base["seed"] = arm.get(
        "seed", base["seed"]
    )  # init + batch order; data.seed (which windows) stays fixed
    base["data"]["mix"] = arm["mix"]
    base["loss"]["alpha"] = arm.get("alpha", base["loss"]["alpha"])
    base["data"]["teacher"] = (
        arm.get("teacher", study.get("teacher")) if base["loss"]["alpha"] > 0 else None
    )
    for section, values in arm.get("override", {}).items():  # e.g. {loss: {kd: quantile_huber}}
        base[section].update(values)
    base["student"]["quantile_levels"] = list(base["student"]["quantile_levels"])
    return from_dict(TrainConfig, base, where=f"arm {arm['name']}")


def wait_for_data(tr_spec, va_spec, members: list[str], name: str, poll: float = 60.0) -> None:
    """Block until the training/validation corpora and every teacher cache are complete (written by --prepare-only)."""
    import time

    from tinycast.data.cache import cache_dir
    from tinycast.data.corpus import corpus_path

    n = sum(tr_spec.counts().values())

    def ready() -> bool:
        if not (corpus_path(tr_spec).exists() and corpus_path(va_spec).exists()):
            return False
        for m in members:
            man = cache_dir(m, tr_spec.key()) / "manifest.json"
            if not man.exists() or sum(json.loads(man.read_text())["shards"].values()) < n:
                return False
        return True

    import os

    t0 = time.time()
    prep = int(os.environ.get("PIPELINE_PREP_PID", "0"))
    while not ready():
        if (
            prep
        ):  # the preparer died without producing the data: fail now instead of waiting forever
            try:
                os.kill(prep, 0)
            except ProcessLookupError as err:
                if not ready():
                    raise RuntimeError(
                        f"[{name}] the --prepare-only process ({prep}) exited before the data was ready"
                    ) from err
        time.sleep(poll)
    print(f"[{name}] data ready after waiting {time.time() - t0:.0f}s", flush=True)


def run_of(ref: str) -> Path:
    """'<stage>:<arm>' or 'best_val:<stage>+<stage>' (best validation score; fine-tuned runs excluded)."""
    if ref.startswith(
        "best_val:"
    ):  # the run with the best VALIDATION score across the listed stages
        cands = []
        for stage in ref.removeprefix("best_val:").split("+"):
            for run in json.loads(
                (ROOT / "results" / "study" / stage / "runs.json").read_text()
            ).values():
                m = json.loads((ROOT / run / "metrics.json").read_text())
                if m.get("finetune") is None and m.get("best_val_wql_geomean") is not None:
                    cands.append((m["best_val_wql_geomean"], run))
        best = min(cands)
        print(
            f"  init selected on validation: {best[1]} (val WQL {best[0]:.4f}) among {len(cands)} runs",
            flush=True,
        )
        return ROOT / best[1]
    stage, arm_name = ref.split(":")
    return (
        ROOT / json.loads((ROOT / "results" / "study" / stage / "runs.json").read_text())[arm_name]
    )


def finetune_arm(study: dict, arm: dict) -> Path:
    """Stage D: light fine-tuning on a target domain's own history (ground truth), from a distilled student or scratch.

    arm: {name, type: finetune, init: "<stage>:<arm>" | "scratch", student_from: "<stage>:<arm>" (architecture when
    init = scratch), domain: fr_power | pedestrian, history: <steps>, steps: <int>, lr: <float>}
    Stage F options: trainable (all | head | last), l2sp (anchor weight), replay (fraction of training windows drawn
    from data/processed/replay_pool_*.npz, trained on their teacher labels; target windows keep their real future).
    """
    from tinycast.data.target_domain import history_corpora

    init = None if arm["init"] == "scratch" else run_of(arm["init"])
    arch_run = init or run_of(arm["student_from"])
    base = to_dict(load(ROOT / study["base"], TrainConfig))
    base["name"] = f"{study['name']}_{arm['name']}"
    base["seed"] = arm.get("seed", base["seed"])
    base["student"] = json.loads((arch_run / "student_config.json").read_text())
    # GT-only by default; with "teacher", the history is labelled by that teacher once (cached) and alpha mixes the two
    base["loss"]["alpha"] = arm.get("alpha", 0.0) if arm.get("teacher") else 0.0
    base["data"]["teacher"] = arm.get("teacher")
    base["optim"].update(
        max_steps=arm["steps"],
        lr=arm.get("lr", 3e-4),
        eval_every=max(25, arm["steps"] // 20),
        warmup_steps=max(10, arm["steps"] // 20),
        patience=6,
        batch_size=arm.get("batch_size", 128),
        trainable=arm.get("trainable", "all"),
        l2sp=arm.get("l2sp", 0.0),
    )
    cfg = from_dict(TrainConfig, base, where=f"finetune arm {arm['name']}")
    corpora = history_corpora(
        arm["domain"],
        arm["history"],
        cfg.student.context_length,
        cfg.student.horizon,
        seed=0,  # same fine-tuning windows for every seed; the seed only changes init noise / batch order
        mode=arm.get("mode", "cold"),
    )
    tq_override = None
    if arm.get("replay"):  # mix distillation windows (teacher labels) into the fine-tuning set
        from tinycast.data.corpus import Corpus

        tr_t, va_t = corpora
        ok = np.isfinite(tr_t.future).all(axis=1)  # the label trick below needs a complete future
        tr_t = Corpus(tr_t.ids[ok], tr_t.source[ok], tr_t.context[ok], tr_t.future[ok])
        pool = np.load(ROOT / "data" / "processed" / arm["replay_pool"], allow_pickle=False)
        r = arm["replay"]
        n_r = min(len(pool["ids"]), int(round(len(tr_t.ids) * r / (1 - r))))
        pick = np.sort(
            np.random.default_rng(base["seed"]).choice(len(pool["ids"]), n_r, replace=False)
        )
        K = len(cfg.student.quantile_levels)
        tr_c = Corpus(np.concatenate([tr_t.ids, pool["ids"][pick]]),
                      np.concatenate([tr_t.source, pool["source"][pick]]),
                      np.concatenate([tr_t.context, pool["context"][pick]]),
                      np.concatenate([tr_t.future, pool["future"][pick]]))  # fmt: skip
        # target rows: "teacher" = the real future on every level, so the teacher pinball equals the real-future loss
        tq_override = np.concatenate(
            [np.repeat(tr_t.future[:, None, :], K, axis=1), pool["q"][pick]]
        ).astype(np.float32)
        corpora = (tr_c, va_t)

        cfg.loss.alpha = 1.0  # every row is trained on its label (teacher or real future)
        print(
            f"[{arm['name']}] replay: {len(tr_t.ids)} target + {n_r} distillation windows",
            flush=True,
        )
    if (
        "epochs" in arm
    ):  # steps proportional to the data (capped): avoids 170 epochs on a 1-week history
        import math

        steps = max(
            100,
            min(
                arm["steps"], math.ceil(arm["epochs"] * len(corpora[0].ids) / cfg.optim.batch_size)
            ),
        )
        cfg.optim.max_steps = steps
        cfg.optim.eval_every = max(25, steps // 20)
        cfg.optim.warmup_steps = max(10, steps // 20)
    print(
        f"[{arm['name']}] fine-tune {arm['domain']} history={arm['history']} train={len(corpora[0].ids)} init={arm['init']}",
        flush=True,
    )
    teacher_key = None
    if arm.get("teacher"):
        from tinycast.distill.weighting import parse

        class _HistorySpec:  # cache key for "this domain's history", distinct per domain / mode / length
            horizon, context_length = cfg.student.horizon, cfg.student.context_length

            def key(self) -> str:
                return f"ft_{arm['domain']}_{arm.get('mode', 'cold')}{arm['history']}"

        teacher_key = _HistorySpec().key()
        for member in parse(arm["teacher"])[1]:
            entry = load(ROOT / "configs" / "teachers" / f"{member}.yaml", ModelEntry)
            cache_teacher_on_corpus(entry, _HistorySpec(), corpora[0])
    return train(
        cfg,
        log_every=100,
        corpora=corpora,
        init_from=init,
        teacher_corpus_key=teacher_key,
        teacher_q_override=tq_override,
        extra_metrics={
            "finetune": {
                **{k: arm[k] for k in ("domain", "history", "steps", "init")},
                "teacher": arm.get("teacher"),
                "alpha": cfg.loss.alpha,
                "mode": arm.get("mode", "cold"),
                "epochs": arm.get("epochs"),
                "init_run": str(init) if init else None,
                "seed": cfg.seed,
                "trainable": cfg.optim.trainable,
                "l2sp": cfg.optim.l2sp,
                "replay": arm.get("replay", 0.0),
            }
        },
    )


def per_series(out: Path, model: str, key: str = "wql") -> np.ndarray:
    """Per-window scores of `model`; for a multi-seed arm (files model_s0, model_s1...), the mean over seeds."""
    files = (
        [out / f"{model}.json"]
        if (out / f"{model}.json").exists()
        else sorted(out.glob(f"{model}_s[0-9]*.json"))
    )
    if not files:
        raise FileNotFoundError(f"{model} not evaluated in {out}")
    vals = [
        np.array(
            [np.nan if v is None else v for v in json.loads(f.read_text())["per_series"][key]],
            dtype=float,
        )
        for f in files
    ]
    return np.nanmean(np.stack(vals), axis=0)


def expand_seeds(study: dict) -> list[dict]:
    """Arms x study['seeds'] -> name_s<seed> (existing/finetune arms are not expanded)."""
    seeds = study.get("seeds")
    if not seeds:
        return study["arms"]
    out = []
    for arm in study["arms"]:
        if arm.get("type") and not (arm["type"] == "finetune" and arm.get("seeded", False)):
            out.append(arm)
            continue
        out += [
            {**arm, "name": f"{arm['name']}_s{sd}", "seed": sd, "base_arm": arm["name"]}
            for sd in seeds
        ]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--only-eval", action="store_true")
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="build corpora + teacher caches for every arm, no training",
    )
    ap.add_argument(
        "--wait-for-data",
        action="store_true",
        help="wait for a --prepare-only process instead of building data",
    )
    a = ap.parse_args()
    study = yaml.safe_load(Path(a.study).read_text())
    study["arms"] = expand_seeds(study)
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
        if (
            arm.get("type") == "existing"
        ):  # reuse a run from an earlier stage as an arm (e.g. zero-shot reference)
            runs[arm["name"]] = str(run_of(arm["run"]).relative_to(ROOT))
            runs_file.write_text(json.dumps(runs, indent=1))
            continue
        if arm.get("type") == "finetune":
            run = finetune_arm(study, arm)
            runs[arm["name"]] = str(run.relative_to(ROOT))
            runs_file.write_text(json.dumps(runs, indent=1))
            continue
        cfg = arm_config(study, arm)
        tr_spec, va_spec = specs(cfg)
        print(f"[{arm['name']}] corpus {dict(tr_spec.mix)} (key {tr_spec.key()})", flush=True)
        members = []
        if cfg.loss.alpha > 0:
            from tinycast.distill.weighting import parse

            members = parse(cfg.data.teacher)[1]  # composite teachers: cache every member
        if a.wait_for_data:  # a --prepare-only process builds the data; wait until it is complete
            wait_for_data(tr_spec, va_spec, members, arm["name"])
        corpus = build_corpus(tr_spec)
        build_corpus(va_spec)
        for member in members:
            entry = load(ROOT / "configs" / "teachers" / f"{member}.yaml", ModelEntry)
            cache_teacher_on_corpus(entry, tr_spec, corpus)
        if a.prepare_only:
            print(f"[{arm['name']}] data ready", flush=True)
            continue
        run = train(cfg, log_every=500)
        runs[arm["name"]] = str(run.relative_to(ROOT))
        runs_file.write_text(json.dumps(runs, indent=1))

    if a.prepare_only:
        sys.exit(0)
    # ---------------------------------------------------------------- evaluate every arm on every suite
    table = {"study": study["name"], "reference_arm": study["reference_arm"], "suites": {}}
    for suite_name in study["eval_suites"]:
        base_suite, _, cap = suite_name.partition(
            "@"
        )  # "fresh_fr_power@168": contexts cut to the last 168 steps
        bcfg = load(ROOT / "configs" / "eval" / f"{base_suite}.yaml", BenchConfig)
        bcfg.latency = False
        suite = load_suite(bcfg.suite, bcfg)
        if cap:
            suite.context[:, : -int(cap)] = (
                np.nan
            )  # cold start: only the last `cap` steps are known
            suite.name = suite_name
        out = sdir / suite_name
        write_suite(suite, out)
        sn = ModelEntry(kind="seasonal_naive", name="seasonal_naive")
        if bcfg.season:
            sn.season = list(suite.season) if bcfg.season == "freq" else bcfg.season
        run_model(build_model(sn), sn, suite, bcfg, out)
        zoo_dir = ROOT / "results" / suite_name
        zoo = study.get("zoo", [])
        zoo = (
            ["seasonal_naive"] if zoo and "seasonal_naive" not in zoo else []
        ) + zoo  # summarize() divides by it
        for zname in zoo:  # zero-shot references on this exact suite (computed once)
            if (zoo_dir / f"{zname}.json").exists():
                continue
            zentry = (
                ModelEntry(kind="seasonal_naive", name="seasonal_naive")
                if zname == "seasonal_naive"
                else load(ROOT / "configs" / "teachers" / f"{zname}.yaml", ModelEntry)
            )
            if zname == "seasonal_naive" and bcfg.season:
                zentry.season = sn.season
            if zentry.device == "auto":
                zentry.device = "cpu"
            print(f"  zoo {zname} on {suite_name}", flush=True)
            write_suite(suite, zoo_dir)
            run_model(build_model(zentry), zentry, suite, bcfg, zoo_dir)
        if study.get("zoo"):
            summarize(zoo_dir)
        arm_suites = {x["name"]: x.get("eval_suites") for x in study["arms"]}
        for arm_name, run in runs.items():
            if arm_suites.get(arm_name) and suite_name not in arm_suites[arm_name]:
                continue  # e.g. an electricity fine-tune is not evaluated on pedestrians
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
        contrasts = []
        for a_arm, b_arm, question in study.get("pairs", []):
            b_name = b_arm.removeprefix("zoo:")
            b_dir = ROOT / "results" / suite_name if b_arm.startswith("zoo:") else out
            files = [out / f"{a_arm}.json", *out.glob(f"{a_arm}_s[0-9]*.json")]
            files += [b_dir / f"{b_name}.json", *b_dir.glob(f"{b_name}_s[0-9]*.json")]
            files = [f for f in files if f.exists()]
            try:
                pa, pb = per_series(out, a_arm), per_series(b_dir, b_name)
            except FileNotFoundError:
                continue  # an arm not evaluated on this suite (e.g. a domain-specific fine-tune)
            for f in files:  # paired comparisons need the exact same windows
                if json.loads(f.read_text())["ids"] != suite.ids:
                    raise ValueError(f"{f}: windows differ from suite {suite_name}; re-run it")
            p, lo, hi = M.paired_ratio_ci(pa, pb, groups=groups)
            contrasts.append({"a": a_arm, "b": b_arm, "question": question, "ratio": [p, lo, hi]})
        table["suites"][suite_name] = {
            "arms": rows,
            "pairs": contrasts,
            "seed_spread": {
                base: sorted(
                    r["rel_wql"]
                    for m, r in rows.items()
                    if m.rsplit("_s", 1)[0] == base and m != base
                )
                for base in {x.get("base_arm") for x in study["arms"]} - {None}
            },
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
        if d["pairs"]:
            lines += [
                "",
                "| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |",
                "|---|---|---|---|",
            ]
            for c in d["pairs"]:
                p, lo, hi = c["ratio"]
                verdict = (
                    "a better" if hi < 1 else "b better" if lo > 1 else "no detectable difference"
                )
                lines.append(
                    f"| {c['question']}: `{c['a']}` vs `{c['b']}` | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {verdict} |"
                )
        lines.append("")
    (sdir / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
