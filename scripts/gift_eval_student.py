"""GIFT-Eval with the official harness: our ONNX student as a GluonTS predictor, run in `.venv-gift`.

    .venv-gift/bin/python scripts/gift_eval_student.py --onnx web/public/models/c2_kd_v2_1m.fp32.onnx --name tinycast-c2 \
        [--terms short] [--datasets m4_hourly,...]
-> results/gift_eval/<name>/all_results.csv  (the leaderboard's format, one row per dataset/freq/term config)

Conventions copied from external/gift-eval/notebooks/chronos.ipynb (commit 0b57dd6, 2026-11-13): dataset list and pretty
names, `to_univariate` when target_dim > 1, `evaluate_model(..., axis=None, mask_invalid_label=True,
allow_nan_forecast=False, seasonality=get_seasonality(freq))`, the 11 metrics, quantile levels 0.1..0.9.
The student reads the last 512 steps (shorter series are left-padded with NaN, which it treats as missing) and forecasts
64 steps in one pass; every short-term horizon is <= 64. Medium/long terms (up to 900 steps) are not run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "raw" / "gift_eval"))

SHORT = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]
PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
OURS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
EMPTY: list[int] = []  # rows per predict() call with no observed history at all
GAP: list[int] = []  # rows whose last 512 steps were all missing (older data used, gap removed)
GIFT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COLUMNS = ["MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS", "RMSE[mean]",
           "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss"]  # fmt: skip


class OnnxStudentPredictor:
    """GluonTS-style predictor: predict(test_data.input) -> list[QuantileForecast]."""

    def __init__(
        self, onnx_path: str, prediction_length: int, context: int = 512, batch: int = 1024
    ):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.H, self.T, self.batch = prediction_length, context, batch
        if prediction_length > 64:
            raise ValueError(f"student horizon is 64, asked {prediction_length}")
        self.idx = [OURS.index(q) for q in GIFT_LEVELS]
        self.keys = [str(q) for q in GIFT_LEVELS] + ["mean"]

    def predict(self, test_data_input, **kwargs):
        from gluonts.model.forecast import QuantileForecast

        entries = list(test_data_input)
        x = np.full((len(entries), self.T), np.nan, dtype=np.float32)
        for i, e in enumerate(entries):
            full = np.asarray(e["target"], dtype=np.float32)
            t = full[-self.T :]
            if not np.isfinite(t).any() and np.isfinite(full).any():
                # the last 512 steps are all missing but older data exists (kdd_cup_2018, bitbrains gaps): read the
                # last 512 *observed* values instead, gap removed. Counted in config.json (gap_skipped_rows).
                obs = full[np.isfinite(full)]
                t = obs[-self.T :]
                GAP.append(1)
            x[i, self.T - len(t) :] = t
        q = np.concatenate(
            [
                self.sess.run(None, {"context": x[i : i + self.batch]})[0]
                for i in range(0, len(x), self.batch)
            ]
        )[:, :, : self.H]  # (N, 13, H)
        # A series with no observed history at all (e.g. electricity/W MT_178: 184 NaN) has nothing to scale by:
        # forecast 0 for it, counted in config.json (empty_context_rows), never silently.
        empty = ~np.isfinite(x).any(axis=1)
        q[empty] = 0.0
        EMPTY.append(int(empty.sum()))
        if not np.isfinite(q).all():
            raise ValueError("non-finite student forecast")
        out = []
        for i, e in enumerate(entries):
            arr = q[i, self.idx]  # (9, H)
            arr = np.concatenate(
                [arr, arr[GIFT_LEVELS.index(0.5)][None]], axis=0
            )  # "mean" = the median
            out.append(QuantileForecast(forecast_arrays=arr, forecast_keys=self.keys,
                                        start_date=e["start"] + len(e["target"]), item_id=e.get("item_id")))  # fmt: skip
        return out


def main() -> None:
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--terms", default="short")
    ap.add_argument("--datasets", default=None)
    ap.add_argument(
        "--context",
        type=int,
        default=512,
        help="history the ONNX student reads (its trained context)",
    )
    a = ap.parse_args()
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(),
               NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=GIFT_LEVELS)]  # fmt: skip
    props = json.loads(
        (ROOT / "external" / "gift-eval" / "notebooks" / "dataset_properties.json").read_text()
    )
    out_dir = ROOT / "results" / "gift_eval" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    names = a.datasets.split(",") if a.datasets else SHORT
    t_all = time.time()
    for ds_name in names:
        for term in a.terms.split(","):
            if "/" in ds_name:
                key, freq = ds_name.split("/")
                key = PRETTY.get(key.lower(), key.lower())
            else:
                key = PRETTY.get(ds_name.lower(), ds_name.lower())
                freq = props[key]["frequency"]
            uni = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
            ds = Dataset(name=ds_name, term=term, to_univariate=uni)
            t0 = time.time()
            res = evaluate_model(
                OnnxStudentPredictor(a.onnx, ds.prediction_length, context=a.context), test_data=ds.test_data, metrics=metrics,
                batch_size=1024, axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                seasonality=get_seasonality(ds.freq),
            )  # fmt: skip
            row = {"dataset": f"{key}/{freq}/{term}", "model": a.name}
            row.update({f"eval_metrics/{c}": float(res[c].iloc[0]) for c in COLUMNS})
            row.update({"domain": props[key]["domain"], "num_variates": props[key]["num_variates"]})
            rows.append(row)
            print(f"{row['dataset']:40s} H={ds.prediction_length:3d} n={len(ds.test_data):6d} "
                  f"MASE {row['eval_metrics/MASE[0.5]']:.3f} CRPS {row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)  # fmt: skip
    with open(out_dir / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "config.json").write_text(json.dumps({
        "model": a.name, "onnx": a.onnx, "context_length": a.context, "horizon": 64, "terms": a.terms,
        "gift_eval_commit": "0b57dd6937eb9a69a3ae771f73ab36188bb33e47", "wall_seconds": round(time.time() - t_all, 1),
        "empty_context_rows": int(sum(EMPTY)), "empty_context_rule": "no observed history at all: forecast 0",
        "gap_skipped_rows": int(sum(GAP)), "gap_rule": "last 512 steps all missing: last 512 observed values",
    }, indent=1))  # fmt: skip
    print(f"{len(rows)} configs -> {out_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
