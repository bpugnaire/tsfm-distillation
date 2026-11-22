"""Our student against the other small forecasting models on GIFT-Eval short-term (same 55 configs).

    uv run python scripts/small_models.py [--refresh-sizes]
-> results/gift_eval/small_models.json

Scores: geometric mean over configs of CRPS / Seasonal Naive (the leaderboard's aggregation), plus a paired ratio
ours / theirs with a 95% bootstrap interval over configs. Parameter counts are the `safetensors.total` field of each
model's Hugging Face page, fetched once and stored in the output (with the fetch date) so the file is reproducible
offline. Leaderboard rows are the authors' own submissions; history length differs by model (ours reads 512 steps).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request

import numpy as np

from scripts.gift_eval_pairs import GE, load
from tinycast.eval.metrics import paired_ratio_ci

OURS = "tinycast-kd5m_3m"
OUT = GE / "small_models.json"
# (result folder or leaderboard name, Hugging Face id for the size, label, note)
MODELS = [
    (OURS, None, "Our student", "512-step history"),
    ("TinyCast", "raws-labs/tinycast", "raws-labs/tinycast", "2048-step history"),
    ("raws-tinycast@hist512", "raws-labs/tinycast", "raws-labs/tinycast, 512 history", "cut to 512 steps like ours (our run)"),
    ("TTM-R2-Pretrained", "ibm-granite/granite-timeseries-ttm-r2", "TTM-R2", "point forecasts only"),
    ("Toto-2.0-4m", "Datadog/Toto-2.0-4m", "Toto-2.0-4m", "4096-step history"),
    ("Toto-2.0-4m@ctx512-univariate", "Datadog/Toto-2.0-4m", "Toto-2.0-4m, 512 history", "512 steps like ours (our run)"),
    ("Timestral-5M", "mistralai/Timestral-5M", "Timestral-5M (our teacher)", ""),
    ("TimeH2O-v1-tiny", "h2oai/timeh2o-v1-tiny", "TimeH2O-v1-tiny", ""),
    ("FlowState-9.1M", "ibm-research/flowstate", "FlowState", ""),
    ("Moirai_small", "Salesforce/moirai-1.1-R-small", "Moirai-small", ""),
]  # fmt: skip
OUR_PARAMS_FILE = "results/export/c1x_kd5m_3m.json"


def hf_params(repo: str) -> int:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        n = (json.load(r).get("safetensors") or {}).get("total")
    if not n:
        raise ValueError(f"{repo}: no safetensors parameter count on its Hugging Face page")
    return int(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-sizes", action="store_true")
    a = ap.parse_args()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    sizes = {} if a.refresh_sizes else old.get("sizes", {})
    from tinycast.eval.benchmark import ROOT

    sizes[OURS] = {
        "n_params": json.loads((ROOT / OUR_PARAMS_FILE).read_text())["n_params"],
        "source": OUR_PARAMS_FILE,
    }
    for _, repo, *_ in MODELS:
        if repo and repo not in sizes:
            sizes[repo] = {"n_params": hf_params(repo), "source": f"https://huggingface.co/api/models/{repo}",
                           "fetched": dt.date.today().isoformat()}  # fmt: skip
    naive, ours = load("Seasonal_Naive"), load(OURS)
    rows = []
    for name, repo, label, note in MODELS:
        s = load(name)
        cfg = sorted(set(s.index) & set(ours.index) & set(naive.index))
        if len(cfg) != 55:
            raise ValueError(f"{name}: {len(cfg)} configs in common, expected 55")
        score = float(np.exp(np.log(s.loc[cfg] / naive.loc[cfg]).mean()))
        r = (
            None
            if name == OURS
            else list(paired_ratio_ci(ours.loc[cfg].to_numpy(), s.loc[cfg].to_numpy()))
        )
        wins = None if name == OURS else int((ours.loc[cfg] < s.loc[cfg]).sum())
        n = sizes[OURS if repo is None else repo]["n_params"]
        rows.append({"model": name, "label": label, "note": note, "n_params": n, "crps_rel": score,
                     "ours_over_theirs": r, "ours_better_on": wins})  # fmt: skip
        print(
            f"{label:34s} {n / 1e6:6.2f}M  {score:.3f}  "
            + (
                f"ours/theirs {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}], ours better on {wins}/55"
                if r
                else ""
            )
        )
    OUT.write_text(json.dumps({"n_configs": 55, "rows": rows, "sizes": sizes}, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
