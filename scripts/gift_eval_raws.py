"""raws-labs/tinycast (146k params, Apache-2.0) on GIFT-Eval short-term with a chosen history length.

    .venv-raws/bin/python -P scripts/gift_eval_raws.py --history 512 [--configs m4_weekly,...]
-> results/gift_eval/raws-tinycast@hist<N>/all_results.csv  (leaderboard format)

Their package is also called `tinycast`: run it in `.venv-raws` with `python -P` so our repo is not on the import path.
The loop copies `tinycast.eval.evaluate` (external/raws-tinycast, commit 628c0be): same metrics, `to_univariate` rule,
the `no_daily` flag on bizitobs_l2c, period-alignment downsampling and flip invariance (their published defaults).
The only change: every series is cut to its last `--history` observed steps before it reaches their predictor, and the
period-alignment factor is computed on the cut series. Their model reads 2048 steps; shorter inputs are left-padded
with NaN by their predictor, as for any short series. CPU fp32 here; the leaderboard run used bf16 on CUDA.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "raw" / "gift_eval"))


class History:
    """Wraps their predictor: each entry's target is cut to its last `n` steps."""

    def __init__(self, predictor, n: int):
        self.p, self.n = predictor, n
        self.prediction_length = predictor.prediction_length

    def predict(self, dataset, **kw):
        def cut(entries):
            for e in entries:
                e = dict(e)
                t = np.asarray(e["target"])
                if t.shape[-1] > self.n:
                    e["start"] = e["start"] + (t.shape[-1] - self.n)
                    e["target"] = t[..., -self.n :]
                yield e

        return self.p.predict(cut(dataset), **kw)


def main() -> None:
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from tinycast.downsample import freq_to_seconds, period_alignment_factor
    from tinycast.predictor import TinyCastPredictor

    from tinycast import eval as E  # theirs (raws-labs), thanks to python -P

    assert "raws-tinycast" in E.__file__, (
        f"imported the wrong tinycast: {E.__file__} (run with python -P)"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--configs", default=None)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    ckpt = str(
        Path(
            snapshot_download(
                "raws-labs/tinycast", revision="9635a27b66a73fedd6e88120c72c7f91427d216e"
            )
        )
        / "model.safetensors"
    )
    metrics, props = E._build_metrics(), E._load_dataset_properties()
    wanted = set(a.configs.split(",")) if a.configs else None
    out = ROOT / "results" / "gift_eval" / f"raws-tinycast@hist{a.history}"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "all_results.csv"
    with open(f, "w", newline="") as fh:
        csv.writer(fh).writerow(E.LEADERBOARD_COLUMNS)
    t_all = time.time()
    for ds_name, term, ds_config, ds_key, ds_freq in E.iter_configs(None, "short"):
        if wanted and ds_name not in wanted:
            continue
        t0 = time.time()
        to_uni = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
        ds = Dataset(name=ds_name, term=term, to_univariate=to_uni)
        kw = dict(
            device=os.environ.get("RAWS_DEVICE", "cpu"), force_flip_invariance=True, use_amp=1
        )
        if "l2c" in ds_name.lower():
            kw["no_daily"] = True
        fs = freq_to_seconds(ds_freq)
        if fs is not None:
            k = period_alignment_factor((np.asarray(e["target"], dtype=np.float64)[..., -a.history :]
                                         for e in ds.test_data.input), fs, horizon=ds.prediction_length)  # fmt: skip
            if k > 1:
                kw["downsample_factor"] = k
        pred = TinyCastPredictor(prediction_length=ds.prediction_length, checkpoint_path=ckpt, freq=ds_freq,
                                 domain=props[ds_key]["domain"], **kw)  # fmt: skip
        res = evaluate_model(History(pred, a.history), test_data=ds.test_data, metrics=metrics, batch_size=64, axis=None,
                             mask_invalid_label=True, allow_nan_forecast=False, seasonality=get_seasonality(ds.freq))  # fmt: skip
        cols = ["MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS", "RMSE[mean]",
                "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss"]  # fmt: skip
        with open(f, "a", newline="") as fh:
            csv.writer(fh).writerow([ds_config, f"raws-tinycast@hist{a.history}", *[res[c].iloc[0] for c in cols],
                                     props[ds_key]["domain"], props[ds_key]["num_variates"]])  # fmt: skip
        print(f"{ds_config:40s} MASE {res['MASE[0.5]'].iloc[0]:.4f} CRPS {res['mean_weighted_sum_quantile_loss'].iloc[0]:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)  # fmt: skip
    print(f"done in {time.time() - t_all:.0f}s -> {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
