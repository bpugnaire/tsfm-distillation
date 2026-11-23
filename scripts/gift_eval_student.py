"""GIFT-Eval with the official harness: our ONNX student as a GluonTS predictor, run in `.venv-gift`.

    .venv-gift/bin/python scripts/gift_eval_student.py --onnx web/public/models/c2_kd_v2_1m.fp32.onnx --name tinycast-c2 \
        [--terms short] [--datasets m4_hourly,...]
-> results/gift_eval/<name>/all_results.csv  (the leaderboard's format, one row per dataset/freq/term config)

Conventions copied from external/gift-eval/notebooks/chronos.ipynb (commit 0b57dd6, 2026-11-13): dataset list and pretty
names, `to_univariate` when target_dim > 1, `evaluate_model(..., axis=None, mask_invalid_label=True,
allow_nan_forecast=False, seasonality=get_seasonality(freq))`, the 11 metrics, quantile levels 0.1..0.9.
The student reads the last 512 steps (shorter series are left-padded with NaN, which it treats as missing) and forecasts
64 steps in one pass; every short-term horizon is <= 64. Medium/long terms (480-900 steps) need --rollout paths: the
student is rolled out in 64-step blocks with sampled paths (tinycast/models/rollout.py, the mode validated in
results/long_horizon/), quantiles taken across --paths paths.

    .venv-gift/bin/python scripts/gift_eval_student.py --onnx ... --name ... --terms medium,long --datasets med_long --rollout paths
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


DS_COUNTS: dict[int, int] = {}  # downsampling factor -> series (--downsample)


def dominant_period(x: np.ndarray, tail: int = 16384, sig: float = 4.0) -> int | None:
    """Dominant period by FFT over the last `tail` observed steps (gaps interpolated); None if the peak is not at least
    `sig` x the median spectral power (Reverso's spectral test, beta = 4)."""
    x = np.asarray(x, dtype=np.float64)[-tail:]
    ok = np.isfinite(x)
    if ok.sum() < 64:
        return None
    idx = np.arange(len(x))
    x = np.interp(idx, idx[ok], x[ok]) - np.nanmean(x[ok])
    p = np.abs(np.fft.rfft(x)) ** 2
    p[0] = 0.0
    if len(p) < 4 or np.median(p[1:]) <= 0:
        return None
    j = int(np.argmax(p[1:]) + 1)
    if p[j] < sig * np.median(p[1:]):
        return None
    period = len(x) / j
    return int(round(period)) if period >= 2 and period <= len(x) / 2 else None


def block_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Average non-overlapping blocks of k steps, aligned to the end of the series (NaN-aware)."""
    n = (len(x) // k) * k
    with np.errstate(all="ignore"):
        return np.nanmean(x[len(x) - n :].reshape(-1, k), axis=1)


def upsample(q: np.ndarray, k: int, H: int) -> np.ndarray:
    """(K, Hc) quantile paths at k-step resolution -> (K, H) at the original step, linear between block centres."""
    centres = np.arange(q.shape[1]) * k + (k - 1) / 2
    t = np.arange(H)
    return np.stack([np.interp(t, centres, row) for row in q])


MED_LONG = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]  # gift-eval notebooks


class _Base:
    """Adapter for RolloutForecaster: forecast(ctx, h) -> (n, 13, h) from the ONNX student (64-step blocks)."""

    def __init__(self, run):
        self.run = run

    def forecast(self, ctx, h):
        return self.run(np.ascontiguousarray(ctx, dtype=np.float32))[:, :, :h]


class OnnxStudentPredictor:
    """GluonTS-style predictor: predict(test_data.input) -> list[QuantileForecast].

    Inference-time options (journal 2026-11-21, docs/notes/gap-analysis.md), fixed rules from the literature, not tuned:
    * flip: average f(x) and -f(-x) with mirrored quantile levels (Reverso, raws-labs/tinycast);
    * downsample: if the dominant period P leaves fewer than M = 8 cycles in the context T, average blocks of
      k = floor(M * P / T) steps, forecast ceil(H / k) coarse steps and interpolate back (Reverso's rule).
    """

    def __init__(self, onnx_path: str, prediction_length: int, context: int = 512, batch: int = 1024,
                 flip: bool = False, downsample: bool = False, rollout: str = "none", paths: int = 32):  # fmt: skip
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("GIFT_ORT_THREADS", "8"))
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.H, self.T, self.batch = prediction_length, context, batch
        self.flip, self.downsample = flip, downsample
        if prediction_length > 64 and rollout == "none":
            raise ValueError(
                f"student horizon is 64, asked {prediction_length}: pass --rollout paths"
            )
        self.roll = None
        if prediction_length > 64:
            import sys

            sys.path.insert(0, str(ROOT))
            from tinycast.models.rollout import RolloutForecaster

            self.roll = RolloutForecaster(_Base(self._run), "student-rollout", mode=rollout, n_paths=paths,
                                          context_length=context, batch=batch)  # fmt: skip
        self.idx = [OURS.index(q) for q in GIFT_LEVELS]
        self.keys = [str(q) for q in GIFT_LEVELS] + ["mean"]

    def _run(self, x: np.ndarray) -> np.ndarray:
        q = np.concatenate(
            [
                self.sess.run(None, {"context": x[i : i + self.batch]})[0]
                for i in range(0, len(x), self.batch)
            ]
        )
        if self.flip:
            qn = np.concatenate(
                [
                    self.sess.run(None, {"context": -x[i : i + self.batch]})[0]
                    for i in range(0, len(x), self.batch)
                ]
            )
            q = 0.5 * (q - qn[:, ::-1])  # level tau of -f(-x) is minus level 1 - tau
        return q

    def predict(self, test_data_input, **kwargs):
        from gluonts.model.forecast import QuantileForecast

        entries = list(test_data_input)
        x = np.full((len(entries), self.T), np.nan, dtype=np.float32)
        ks = np.ones(len(entries), dtype=int)
        for i, e in enumerate(entries):
            full = np.asarray(e["target"], dtype=np.float32)
            t = full[-self.T :]
            if not np.isfinite(t).any() and np.isfinite(full).any():
                # the last 512 steps are all missing but older data exists (kdd_cup_2018, bitbrains gaps): read the
                # last 512 *observed* values instead, gap removed. Counted in config.json (gap_skipped_rows).
                full = full[np.isfinite(full)]
                t = full[-self.T :]
                GAP.append(1)
            if self.downsample:
                per = dominant_period(full)
                k = int(8 * per // self.T) if per else 1
                if k >= 2 and len(full) // k >= 32:
                    t = block_mean(full, k)[-self.T :].astype(np.float32)
                    ks[i] = k
            DS_COUNTS[int(ks[i])] = DS_COUNTS.get(int(ks[i]), 0) + 1
            x[i, self.T - len(t) :] = t
        if (
            self.roll is not None
        ):  # chunks of series: N x paths contexts at once would not fit in memory
            # rows with no observed history are left out of the rollout (the student cannot scale them) and get the
            # same documented 0 forecast as in the short-term path below
            keep = np.isfinite(x).any(axis=1)
            q = np.zeros((len(x), 13, self.H), dtype=np.float32)
            xk = x[keep]
            if len(xk):
                q[keep] = np.concatenate(
                    [self.roll.forecast(xk[i : i + 256], self.H) for i in range(0, len(xk), 256)]
                )
        else:
            q = self._run(x)  # (N, 13, 64)
        # A series with no observed history at all (e.g. electricity/W MT_178: 184 NaN) has nothing to scale by:
        # forecast 0 for it, counted in config.json (empty_context_rows), never silently.
        empty = ~np.isfinite(x).any(axis=1)
        q[empty] = 0.0
        EMPTY.append(int(empty.sum()))
        if not np.isfinite(q).all():
            raise ValueError("non-finite student forecast")
        out = []
        for i, e in enumerate(entries):
            k = int(ks[i])
            qi = q[i, :, : self.H] if k == 1 else upsample(q[i, :, : -(-self.H // k)], k, self.H)
            arr = qi[self.idx]  # (9, H)
            arr = np.concatenate(
                [arr, arr[GIFT_LEVELS.index(0.5)][None]], axis=0
            )  # "mean" = the median
            out.append(QuantileForecast(forecast_arrays=arr, forecast_keys=self.keys,
                                        start_date=e["start"] + len(e["target"]), item_id=e.get("item_id")))  # fmt: skip
        return out


def _eval_config(job):
    """One GIFT-Eval config (runs in a worker process when --workers > 1); returns (row, (empty, gap, ds_counts))."""
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

    ds_name, term, onnx, context, flip, downsample, name, rollout, paths = job
    part = (
        ROOT / "results" / "gift_eval" / name / "parts" / f"{ds_name.replace('/', '_')}_{term}.json"
    )
    if part.exists():  # finished in an earlier (crashed) run: reuse it
        d = json.loads(part.read_text())
        return d["row"], tuple(d["counts"])
    EMPTY.clear()
    GAP.clear()
    DS_COUNTS.clear()
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(),
               NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=GIFT_LEVELS)]  # fmt: skip
    props = json.loads(
        (ROOT / "external" / "gift-eval" / "notebooks" / "dataset_properties.json").read_text()
    )
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
        OnnxStudentPredictor(onnx, ds.prediction_length, context=context, flip=flip, downsample=downsample,
                             rollout=rollout, paths=paths),
        test_data=ds.test_data, metrics=metrics, batch_size=1024, axis=None, mask_invalid_label=True,
        allow_nan_forecast=False, seasonality=get_seasonality(ds.freq),
    )  # fmt: skip
    row = {"dataset": f"{key}/{freq}/{term}", "model": name}
    row.update({f"eval_metrics/{c}": float(res[c].iloc[0]) for c in COLUMNS})
    row.update({"domain": props[key]["domain"], "num_variates": props[key]["num_variates"]})
    print(f"{row['dataset']:40s} H={ds.prediction_length:3d} n={len(ds.test_data):6d} "
          f"MASE {row['eval_metrics/MASE[0.5]']:.3f} CRPS {row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)  # fmt: skip
    counts = (int(sum(EMPTY)), int(sum(GAP)), dict(DS_COUNTS))
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(
        json.dumps({"row": row, "counts": counts})
    )  # saved per config: a crash keeps finished work
    return row, counts


def main() -> None:

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
    ap.add_argument("--flip", action="store_true", help="average f(x) and -f(-x)")
    ap.add_argument(
        "--workers", type=int, default=4, help="configs evaluated in parallel processes"
    )
    ap.add_argument("--downsample", action="store_true", help="Reverso-style period downsampling")
    ap.add_argument(
        "--rollout", default="none", choices=["none", "paths", "median"], help="horizons > 64"
    )
    ap.add_argument("--paths", type=int, default=32)
    a = ap.parse_args()
    out_dir = ROOT / "results" / "gift_eval" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    names = MED_LONG if a.datasets == "med_long" else a.datasets.split(",") if a.datasets else SHORT
    jobs = [
        (ds_name, term, a.onnx, a.context, a.flip, a.downsample, a.name, a.rollout, a.paths)
        for ds_name in names
        for term in a.terms.split(",")
    ]
    t_all = time.time()
    if (
        a.workers > 1
    ):  # configs in parallel processes: GluonTS bookkeeping is single-threaded Python
        from concurrent.futures import ProcessPoolExecutor

        os.environ["GIFT_ORT_THREADS"] = str(max(1, 8 // a.workers))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_eval_config, jobs))
    else:
        results = [_eval_config(j) for j in jobs]
    rows = [r for r, _ in results]
    for _, (empty, gap, ds_counts) in results:
        EMPTY.append(empty)
        GAP.append(gap)
        for k, v in ds_counts.items():
            DS_COUNTS[int(k)] = DS_COUNTS.get(int(k), 0) + v
    with open(out_dir / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "config.json").write_text(json.dumps({
        "model": a.name, "onnx": a.onnx, "context_length": a.context, "horizon": 64, "terms": a.terms,
        "gift_eval_commit": "0b57dd6937eb9a69a3ae771f73ab36188bb33e47", "wall_seconds": round(time.time() - t_all, 1),
        "empty_context_rows": int(sum(EMPTY)), "empty_context_rule": "no observed history at all: forecast 0",
        "gap_skipped_rows": int(sum(GAP)), "gap_rule": "last 512 steps all missing: last 512 observed values",
        "flip": a.flip, "downsample": a.downsample, "rollout": a.rollout, "paths": a.paths, "downsample_factors": {str(k): v for k, v in sorted(DS_COUNTS.items())},
    }, indent=1))  # fmt: skip
    print(f"{len(rows)} configs -> {out_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
