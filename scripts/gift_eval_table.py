"""GIFT-Eval comparison on the configs we ran: our student vs every leaderboard model, same configs, same normalisation.

    uv run python scripts/gift_eval_table.py [--ours tinycast-c2]
-> results/gift_eval/summary.json + table.md

Leaderboard rule (external/gift-eval/notebooks/timestral.ipynb, last cell): for each config divide MASE[0.5] and the
mean weighted quantile loss (CRPS) by Seasonal Naive's, then take the geometric mean over configs. We restrict every
model to the configs we evaluated (short term: 55), so the numbers differ from the public all-97 leaderboard.
Leaderboard numbers come from the models' own runs (their own context lengths), with their `testdata_leakage` flag.
Parameter counts are filled only for models we measured ourselves.
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from tinycast.eval.benchmark import ROOT

LB = ROOT / "external" / "gift-eval" / "results"
MEASURED_PARAMS = {  # from our own runs (results/*/summary.json, results/external/ttm)
    "Toto-2.0-4m": 4.1e6, "Toto-2.0-22m": 21.9e6, "Toto-2.0-313m": 313e6, "Timestral-5M": 5.2e6,
    "Timestral-28M": 27.8e6, "Timestral-209M": 209e6, "TTM-R2-Pretrained": 0.81e6, "Chronos-2": 119e6,
}  # fmt: skip
COLS = {"MASE": "eval_metrics/MASE[0.5]", "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss"}


def gmean(x: pd.Series) -> float:
    return float(math.exp(x.map(math.log).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="tinycast-c2")
    a = ap.parse_args()
    ours = pd.read_csv(ROOT / "results" / "gift_eval" / a.ours / "all_results.csv").set_index(
        "dataset"
    )
    configs = list(ours.index)
    naive = pd.read_csv(LB / "Seasonal_Naive" / "all_results.csv").set_index("dataset").loc[configs]
    rows = []
    entries = [(a.ours, ours, {"testdata_leakage": "see note", "org": "this project"})]
    for d in sorted(
        (ROOT / "results" / "gift_eval").glob("tinycast-*")
    ):  # our other students, same configs
        if d.name != a.ours and (d / "all_results.csv").exists():
            df = pd.read_csv(d / "all_results.csv").set_index("dataset")
            if set(configs) <= set(df.index):
                entries.append(
                    (
                        d.name,
                        df.loc[configs],
                        {"testdata_leakage": "see note", "org": "this project"},
                    )
                )
    for d in sorted(LB.iterdir()):
        f = d / "all_results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).drop_duplicates("dataset").set_index("dataset")
        if not set(configs) <= set(df.index):
            continue  # model did not report every config we ran
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        entries.append((d.name, df.loc[configs], cfg))
    for name, df, cfg in entries:
        r = {"model": name, "org": cfg.get("org"), "leakage": cfg.get("testdata_leakage"),
             "type": cfg.get("model_type"), "params": MEASURED_PARAMS.get(name)}  # fmt: skip
        ok = True
        for k, col in COLS.items():
            ratio = df[col].astype(float) / naive[col].astype(float)
            if not (ratio > 0).all() or ratio.isna().any():
                ok = False
                break
            r[k] = gmean(ratio)
        if ok:
            rows.append(r)
    t = pd.DataFrame(rows).sort_values("CRPS").reset_index(drop=True)
    t["CRPS_rank"] = t.index + 1
    (ROOT / "results" / "gift_eval" / "summary.json").write_text(
        json.dumps(
            {
                "configs": configs,
                "n_configs": len(configs),
                "ours": a.ours,
                "rows": t.to_dict("records"),
            },
            indent=1,
        )
    )
    md = [f"# GIFT-Eval, {len(configs)} short-term configs (geometric mean of ratio to Seasonal Naive, lower is better)", "",
          "| rank | model | org | params | leakage | CRPS | MASE |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for _, r in t.iterrows():
        p = (
            f"{r['params'] / 1e6:.1f}M"
            if isinstance(r["params"], float) and r["params"] == r["params"]
            else ""
        )
        md.append(
            f"| {r['CRPS_rank']} | {'**' + r['model'] + '**' if r['model'].startswith('tinycast') else r['model']} | {r['org'] or ''} | {p} | {r['leakage'] or ''} | {r['CRPS']:.3f} | {r['MASE']:.3f} |"
        )
    (ROOT / "results" / "gift_eval" / "table.md").write_text("\n".join(md))
    print(
        "\n".join(
            md[:4]
            + [
                m
                for m in md[4:]
                if any(
                    k in m
                    for k in (
                        a.ours,
                        "Toto-2.0-4m",
                        "Timestral-5M",
                        "Timestral-28M",
                        "TTM-R2-Pre",
                        "chronos_bolt",
                        "Seasonal_Naive",
                        "| 1 |",
                        "| 2 |",
                        "| 3 |",
                    )
                )
            ]
        )
    )
    print(f"{len(t)} models compared")


if __name__ == "__main__":
    main()
