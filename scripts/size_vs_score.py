"""Every model we scored, by size: our distilled students (all general-purpose runs) and the foundation models.

    uv run python scripts/size_vs_score.py   -> results/size_vs_score.json (read by scripts/build_web_data.py)

Our suites: scores from the study tables (students) and results/<suite>/summary.json (foundation models); all rows of a
suite are on the same windows. GIFT-Eval: leaderboard rule (geometric mean over the 55 short-term configs of CRPS / Seasonal
Naive). Student parameter counts are read from the checkpoints; foundation-model counts from our own loading of each model
(results/<suite>/summary.json), from the leaderboard, or from the cited note.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import yaml

from tinycast.eval.benchmark import ROOT

SUITES = ("fresh_fr_power", "shapes", "multidomain_test")
GE, LB = ROOT / "results" / "gift_eval", ROOT / "external" / "gift-eval" / "results"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
# general-purpose students (no fine-tuning runs, no seed replicas, no negative controls); GIFT-Eval result folder if any
STUDENTS = {
    "2026-11-13_stage_c_lite_gt_mix_m_300k": None,
    "2026-11-13_stage_c_lite_kd_mix_s_300k": None,
    "2026-11-13_stage_c_lite_kd_mix_m_300k": None,
    "2026-11-14_stage_c2_kd_v2_300k": "tinycast-c2-300k-ctx512",
    "2026-11-15_stage_c2_kd_v2_1m": "tinycast-c2",
    "2026-11-16_stage_c3_kd_toto_1m": "tinycast-c3-kd_toto_1m",
    "2026-11-16_stage_c3_kd_t0_1m": "tinycast-c3-kd_t0_1m",
    "2026-11-16_stage_c3_kd_ens3_1m_2": "tinycast-c3-kd_ens3_1m",
    "2026-11-19_stage_c5_209m_kd_209m_1m_full": "tinycast-c5-209m",
    "2026-11-20_stage_c6_ctx2048_kd_ctx2048_1m": "tinycast-c6-ctx2048-1m",
    "2026-11-20_stage_c7_size_l_28m": "tinycast-c7-l_28m",
    "2026-11-20_stage_c7_size_l_209m": "tinycast-c7-l_209m",
    "2026-11-20_stage_c8_mt_kd_mt_300k": "tinycast-c8-mt-300k",
    "2026-11-21_stage_c9_mt_scale_mt_1m": "tinycast-c9-mt_1m",
    "2026-11-21_stage_c9_mt_scale_mt_l_1m_209m": "tinycast-c9-mt_l_1m_209m",
    "2026-11-21_stage_c9_mt_ctx2048_mt_2048_1m": "tinycast-c9-mt_2048_1m",
    "2026-11-21_stage_c10_v3_mt_v3_300k": "tinycast-mt_v3_300k",
    "2026-11-21_stage_c11_ctx_mt_2048_p16": "tinycast-mt_2048_p16",
    "2026-11-21_stage_c11_ctx_mt_2048_p64": "tinycast-mt_2048_p64",
    "2026-11-21_stage_c12_varctx_mt_2048_p16_var": "tinycast-mt_2048_p16_var",
    "2026-11-21_stage_c13_gt_vs_kd_gt_300k": "tinycast-gt_300k",
    "2026-11-21_stage_c13_gt_vs_kd_mix_300k": "tinycast-mix_300k",
    "2026-11-21_stage_c13_gt_vs_kd_gt_1m": "tinycast-gt_1m",
    "2026-11-21_stage_c13_gt_vs_kd_mix_1m": "tinycast-mix_1m",
    "2026-11-21_stage_c14_teacher5m_kd5m_1m": "tinycast-kd5m_1m",
    "2026-11-22_stage_c15_3m_gt_3m": "tinycast-gt_3m",
    "2026-11-22_stage_c15_3m_kd5m_3m": "tinycast-kd5m_3m",
    "2026-11-22_stage_c16_hybrid_hyb5m_1m": "tinycast-hyb5m_1m",
}
PUBLISHED = "2026-11-22_stage_c15_3m_kd5m_3m"  # the site model since 2026-11-22 (scripts/build_web_data.py STUDENT)
TEACHER_LABEL = {
    "timestral-5m": "Timestral-5M",
    "timestral-28m": "Timestral-28M",
    "timestral-209m": "Timestral-209M",
    "toto-2.0-313m": "Toto-2.0-313m",
    "t0-gamma": "t0-gamma",
}
ZOO_LABEL = {
    "timestral-5m": "Timestral-5M", "timestral-28m": "Timestral-28M", "timestral-209m": "Timestral-209M",
    "toto-2.0-4m": "Toto-2.0-4m", "toto-2.0-22m": "Toto-2.0-22m", "toto-2.0-313m": "Toto-2.0-313m",
    "chronos-2": "Chronos-2", "chronos-2-small": "Chronos-2-small", "chronos-bolt-tiny": "Chronos-Bolt-tiny",
    "chronos-bolt-mini": "Chronos-Bolt-mini", "t0-gamma": "t0-gamma", "ttm-r2": "TTM-R2", "ttm-r2.1-l1": "TTM-R2.1",
}  # fmt: skip
TEACHERS = {
    "timestral-28m",
    "timestral-209m",
    "toto-2.0-313m",
    "t0-gamma",
}  # models we distilled from
# GIFT-Eval leaderboard rows -> (label, params source): zero-shot rows with a parameter count, no leakage flag
GIFT_ZOO = {
    "Timestral-5M": "timestral-5m", "Timestral-28M": "timestral-28m", "Timestral-209M": "timestral-209m",
    "Toto-2.0-4m": "toto-2.0-4m", "Toto-2.0-22m": "toto-2.0-22m", "Toto-2.0-313m": "toto-2.0-313m",
    "Chronos-2": "chronos-2", "TTM-R2-Pretrained": "ttm-r2",
}  # fmt: skip
SMALL = (
    ROOT / "results" / "gift_eval" / "small_models.json"
)  # scripts/small_models.py: parameter counts from Hugging Face
SMALL_ZOO = {
    "TimeH2O-v1-tiny": "TimeH2O-v1-tiny",
    "FlowState-9.1M": "FlowState",
    "Moirai_small": "Moirai-small",
}


def gift_series(name: str) -> pd.Series:
    d = GE / name if (GE / name).exists() else LB / name
    df = pd.read_csv(d / "all_results.csv").drop_duplicates("dataset").set_index("dataset")
    return df[CRPS].astype(float)


def gift_score(name: str, naive: pd.Series) -> tuple[float, int]:
    s = gift_series(name)
    return float(np.exp(np.log(s / naive.loc[s.index]).mean())), len(s)


def student_info(run: str) -> dict:
    d = ROOT / "runs" / run
    sc = json.loads((d / "student_config.json").read_text())
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    sd = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    n = int(sum(v.numel() for v in sd.values()))
    mix = cfg["data"].get("mix") or {}
    windows = sum(mix.values()) if mix else cfg["data"]["n_train_windows"]
    teacher = cfg["data"].get("teacher")
    alpha = cfg["loss"]["alpha"]
    size = {64: "s", 128: "m", 192: "l"}.get(sc["d_model"], f"d{sc['d_model']}")
    mt = sc["arch"].endswith("_mt")
    gt_src = cfg["data"].get("gt_sources") or []
    if teacher is None or alpha == 0:
        t_lab = "ground truth only"
    elif gt_src:
        t_lab = (
            f"real future on {'+'.join(gt_src)}, {TEACHER_LABEL.get(teacher, teacher)} on the rest"
        )
    elif 0 < alpha < 1:
        t_lab = f"{alpha:g} × {TEACHER_LABEL.get(teacher, teacher)} + real future"
    elif teacher.startswith("ensemble"):
        t_lab = "3-teacher ensemble"
    else:
        t_lab = TEACHER_LABEL.get(teacher, teacher)
    data = "data v3" if any("v3" in k for k in mix) else None
    n_w = (
        f"{windows // 1_000_000}M"
        if windows >= 1_000_000 and windows % 1_000_000 == 0
        else f"{windows // 1000}k"
    )
    bits = [f"{size}{' multi-token' if mt else ''}", f"{n_w} windows", t_lab]
    if sc["context_length"] != 512:
        bits.append(f"{sc['context_length']} steps")
    if data:
        bits.append(data)
    if sc["context_length"] != 512 and sc.get("patch_size") != 64:
        bits.append(f"patch {sc['patch_size']}")
    if cfg["data"].get("var_context"):
        bits.append("variable histories")
    return {
        "run": run,
        "label": " · ".join(bits),
        "n_params": n,
        "size": size,
        "multi_token": mt,
        "context": sc["context_length"],
        "windows": windows,
        "teacher": t_lab,
        "distilled": t_lab != "ground truth only",
        "published": run == PUBLISHED,
    }


def study_scores() -> dict[str, dict[str, dict]]:
    """run dir -> suite -> {score, ci, cov80, source} from every study table that scored it on our standard suites."""
    out: dict[str, dict[str, dict]] = {}
    for runs_f in sorted((ROOT / "results" / "study").glob("*/runs.json")):
        tab_f = runs_f.with_name("table.json")
        if not tab_f.exists():
            continue
        runs, tab = json.loads(runs_f.read_text()), json.loads(tab_f.read_text())
        for suite, s in tab["suites"].items():
            if suite not in SUITES:
                continue
            for arm, a in s["arms"].items():
                run = runs.get(arm, "").removeprefix("runs/")
                if run in STUDENTS:
                    out.setdefault(run, {})[suite] = {
                        "score": a["rel_wql"], "ci": a["rel_wql_ci"], "cov80": a["cov80"],
                        "source": str(tab_f.relative_to(ROOT)),
                    }  # fmt: skip
    return out


def main() -> None:
    scores = study_scores()
    naive = gift_series("Seasonal_Naive")
    students = []
    for run, gift_dir in STUDENTS.items():
        e = student_info(run)
        e["scores"] = scores.get(run, {})
        if gift_dir:
            g, n = gift_score(gift_dir, naive)
            e["scores"]["gift"] = {
                "score": g,
                "ci": None,
                "n_configs": n,
                "source": f"results/gift_eval/{gift_dir}",
            }
        students.append(e)
    zoo: dict[str, dict] = {}
    for suite in SUITES:
        summ = json.loads((ROOT / "results" / suite / "summary.json").read_text())
        for r in summ["rows"]:
            m = r["model"]
            if m not in ZOO_LABEL:
                continue
            z = zoo.setdefault(
                m,
                {
                    "model": m,
                    "label": ZOO_LABEL[m],
                    "n_params": r["n_params"],
                    "teacher": m in TEACHERS,
                    "scores": {},
                },
            )
            z["scores"][suite] = {
                "score": r["rel_wql"],
                "ci": r["rel_wql_ci"],
                "cov80": r["mean_cov80"],
                "source": f"results/{suite}/summary.json",
            }
    lb = {r["model"]: r for r in json.loads((GE / "summary.json").read_text())["rows"]}
    for lb_name, m in GIFT_ZOO.items():
        z = zoo[m]
        z["scores"]["gift"] = {
            "score": lb[lb_name]["CRPS"],
            "ci": None,
            "source": "results/gift_eval/summary.json",
        }
        if lb[lb_name]["leakage"] == "Yes":
            z["gift_leakage"] = True
    n_small = {r["model"]: r["n_params"] for r in json.loads(SMALL.read_text())["rows"]}
    for lb_name, label in SMALL_ZOO.items():
        zoo[lb_name] = {
            "model": lb_name,
            "label": label,
            "n_params": n_small[lb_name],
            "teacher": False,
            "scores": {
                "gift": {
                    "score": lb[lb_name]["CRPS"],
                    "ci": None,
                    "source": "results/gift_eval/summary.json · params: results/gift_eval/small_models.json",
                }
            },
        }
    raws = {
        "model": "raws-tinycast",
        "label": "raws-labs TinyCast",
        "n_params": n_small["TinyCast"],
        "teacher": False,
        "scores": {},
    }
    raws["scores"]["gift"] = {
        "score": lb["TinyCast"]["CRPS"],
        "ci": None,
        "source": "results/gift_eval/summary.json (full history)",
    }
    g512, _ = gift_score("raws-tinycast@hist512", naive)
    raws["scores"]["gift_512"] = {
        "score": g512,
        "ci": None,
        "source": "results/gift_eval/raws-tinycast@hist512",
    }
    t512, _ = gift_score("Toto-2.0-4m@ctx512-univariate", naive)
    zoo["toto-2.0-4m"]["scores"]["gift_512"] = {
        "score": t512,
        "ci": None,
        "source": "results/gift_eval/Toto-2.0-4m@ctx512-univariate",
    }
    out = {"source": "scripts/size_vs_score.py", "students": students, "zoo": [*zoo.values(), raws]}
    p = ROOT / "results" / "size_vs_score.json"
    p.write_text(json.dumps(out, indent=1))
    for e in students:
        print(
            f"{e['n_params']:>9,}  {e['label']:<60} "
            + "  ".join(f"{k}={v['score']:.3f}" for k, v in e["scores"].items())
        )
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
