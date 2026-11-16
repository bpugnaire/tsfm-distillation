"""Toto-2.0 on GIFT-Eval with a chosen context length: the leaderboard notebook's code, context as a parameter.

    .venv-gift-toto/bin/python scripts/gift_eval_toto.py --size 4m --context 512 [--datasets a,b]
-> results/gift_eval/Toto-2.0-<size>@ctx<context>/all_results.csv

Copied from external/gift-eval/notebooks/toto_2_0.ipynb (commit 0b57dd6): `to_univariate=False` (Toto reads all variates),
`Toto2GluonTSModelConfig(prediction_length, context_length, target_dim, past_feat_dynamic_real_dim)`, batch 512, same
metrics. Only `context_length` changes (leaderboard: 4096). Env `.venv-gift-toto`: toto-2 2.0.0 needs Python 3.12 and
gluonts >= 0.16, so gift-eval is installed with --no-deps on gluonts 0.17; `--context 4096` on a few configs checks that
this reproduces the leaderboard numbers. CPU only (Toto gives non-finite output on Apple MPS for constant inputs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "raw" / "gift_eval"))
sys.path.insert(0, str(ROOT / "scripts"))
from gift_eval_student import COLUMNS, PRETTY, SHORT  # noqa: E402  (same config list and names as our student)


class AxisFix:
    """toto2's GluonTS wrapper returns multivariate QuantileForecasts as (quantile, variate, time), the layout of
    gluonts 0.15 (the leaderboard's); gluonts 0.17 (needed by toto-2) expects (quantile, time, variate). Swap the last
    two axes of multivariate forecasts only; checked by reproducing the leaderboard's ett1/H numbers."""

    def __init__(self, predictor):
        self.p = predictor

    def predict(self, dataset, **kw):
        from gluonts.model.forecast import QuantileForecast

        for f in self.p.predict(dataset, **kw):
            if f.forecast_array.ndim == 3:
                f = QuantileForecast(forecast_arrays=f.forecast_array.transpose(0, 2, 1), start_date=f.start_date,
                                     forecast_keys=f.forecast_keys, item_id=f.item_id)  # fmt: skip
            yield f


def main() -> None:
    import torch
    from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model

    from gift_eval.data import Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4m")
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--univariate", action="store_true", help="split multivariate series like our student does")
    a = ap.parse_args()
    torch.set_num_threads(8)
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
               RMSE(forecast_type="mean"), NRMSE(forecast_type="mean"), ND(),
               MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]  # fmt: skip
    props = json.loads((ROOT / "external" / "gift-eval" / "notebooks" / "dataset_properties.json").read_text())
    name = f"Toto-2.0-{a.size}"
    model = Toto2Model.from_pretrained(f"Datadog/{name}", map_location="cpu").eval()
    tag = f"{name}@ctx{a.context}" + ("-univariate" if a.univariate else "")
    out_dir = ROOT / "results" / "gift_eval" / (a.out or tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t_all = [], time.time()
    for ds_name in a.datasets.split(",") if a.datasets else SHORT:
        uni = a.univariate and Dataset(name=ds_name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL").target_dim != 1
        ds = Dataset(name=ds_name, term="short", to_univariate=uni, storage_env_var="GIFT_EVAL")
        cfg = Toto2GluonTSModelConfig(prediction_length=ds.prediction_length, context_length=a.context,
                                      target_dim=1 if uni else ds.target_dim, past_feat_dynamic_real_dim=ds.past_feat_dynamic_real_dim)  # fmt: skip
        predictor = AxisFix(Toto2GluonTSModel(model, cfg).eval().create_predictor(batch_size=512, device="cpu"))
        if "/" in ds_name:
            key, freq = ds_name.split("/")
            key = PRETTY.get(key.lower(), key.lower())
        else:
            key = PRETTY.get(ds_name.lower(), ds_name.lower())
            freq = props[key]["frequency"]
        t0 = time.time()
        res = evaluate_model(predictor, test_data=ds.test_data, metrics=metrics, axis=None, mask_invalid_label=True,
                             allow_nan_forecast=False, seasonality=get_seasonality(ds.freq))  # fmt: skip
        row = {"dataset": f"{key}/{freq}/short", "model": tag}
        row.update({f"eval_metrics/{c}": float(res[c].iloc[0]) for c in COLUMNS})
        row.update({"domain": props[key]["domain"], "num_variates": props[key]["num_variates"]})
        rows.append(row)
        print(f"{row['dataset']:40s} MASE {row['eval_metrics/MASE[0.5]']:.4f} "
              f"CRPS {row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} ({time.time() - t0:.1f}s)", flush=True)  # fmt: skip
        with open(out_dir / "all_results.csv", "w", newline="") as f:  # rewritten after each config: resumable output
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "config.json").write_text(json.dumps({"model": name, "context_length": a.context, "device": "cpu",
        "gift_eval_commit": "0b57dd6", "gluonts": __import__("gluonts").__version__, "toto": "toto-2==2.0.0",
        "univariate": a.univariate, "wall_seconds": round(time.time() - t_all, 1)}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
