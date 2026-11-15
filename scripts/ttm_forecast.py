"""IBM Granite TinyTimeMixer (TTM-r2) forecasts on our suites, run in its OWN env.

granite-tsfm pins torch/pandas versions that clash with ours, so this script runs in `.venv-ttm` and only writes files:
    .venv-ttm/bin/python scripts/ttm_forecast.py --suite shapes
reads   results/<suite>/_suite.json                  (the exact contexts every other model saw)
writes  results/external/ttm/<name>/<suite>.npz      (context as seen, point forecast (N, 64), latency json)
Scoring happens in the main env (kind: precomputed, tinycast/models/precomputed.py).

TTM facts (checked on granite-tsfm 0.3.12, 2026-11-16; see docs/notes/teacher-apis.md):
* `get_model(..., context_length=512, prediction_length=64)` picks the 512-96 checkpoint and cuts the output to 64
  (`prediction_filter_length`). `prefer_l1_loss=True` picks the L1-trained variant (a median forecast).
* Point forecasts only (loss mse / l1). Its internal std scaler uses `past_observed_mask`, so missing values are
  passed as masked; we fill them first (linear inside, nearest value at the edges) because NaN inputs propagate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"ttm-r2": False, "ttm-r2.1-l1": True}  # name -> prefer_l1_loss
# The L1 checkpoint (512-96-ft-l1-r2.1) is frequency-prefix tuned: it needs a frequency token per window.
# Frequencies copied from tinycast/data/sources/hf_datasets.DATASETS (this env cannot import tinycast); monthly is
# not in TTM's map, so it is "oov" (unknown), like the synthetic shapes.
TOKEN = {"oov": 0, "30min": 6, "h": 7, "d": 8}
SUITE_FREQ = {"shapes": "oov", "fresh_fr_power": "h"}
DATASET_FREQ = {
    "monash_traffic": "h", "monash_kdd_cup_2018": "h", "monash_rideshare": "h", "taxi_30min": "30min",
    "monash_weather": "d", "monash_temperature_rain": "d", "monash_saugeenday": "d", "monash_covid_deaths": "d",
    "exchange_rate": "d", "monash_tourism_monthly": "oov", "monash_fred_md": "oov", "monash_pedestrian_counts": "h",
}  # fmt: skip


def freq_tokens(suite: str, ids: list[str]) -> np.ndarray:
    if suite in SUITE_FREQ:
        return np.full(len(ids), TOKEN[SUITE_FREQ[suite]], dtype=np.int64)
    return np.array([TOKEN[DATASET_FREQ[i.split("/")[0]]] for i in ids], dtype=np.int64)


def fill(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.isfinite(x)
    out = x.copy()
    for i in range(len(x)):
        o = obs[i]
        if o.all():
            continue
        if not o.any():
            out[i] = 0.0
            continue
        idx = np.arange(x.shape[1])
        out[i] = np.interp(idx, idx[o], x[i, o])
    return out.astype(np.float32), obs


def main() -> None:
    from tsfm_public.toolkit.get_model import get_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--horizon", type=int, default=64)
    a = ap.parse_args()
    s = json.loads((ROOT / "results" / a.suite / "_suite.json").read_text())
    ctx = np.array(
        [[np.nan if v is None else v for v in c] for c in s["context"]], dtype=np.float64
    )
    T = ctx.shape[1]
    tokens = freq_tokens(a.suite, s["ids"])
    spec = importlib.util.spec_from_file_location("lat", ROOT / "tinycast" / "eval" / "latency.py")
    lat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lat)  # type: ignore[union-attr]

    for name, l1 in VARIANTS.items():
        key = get_model(
            "ibm-granite/granite-timeseries-ttm-r2", context_length=T, prediction_length=a.horizon,
            prefer_l1_loss=l1, return_model_key=True,
        )  # fmt: skip
        model = get_model(
            "ibm-granite/granite-timeseries-ttm-r2", context_length=T, prediction_length=a.horizon,
            prefer_l1_loss=l1,
        ).eval()  # fmt: skip
        n_params = sum(p.numel() for p in model.parameters())

        class F:  # Forecaster-like wrapper for the shared latency harness
            tok: np.ndarray | None = (
                None  # per-window frequency tokens; None -> hourly (latency runs)
            )

            def __init__(self, net):
                self.net = net

            @torch.no_grad()
            def forecast(self, x, horizon):
                xf, obs = fill(np.asarray(x, dtype=np.float64))
                tk = self.tok if self.tok is not None else np.full(len(xf), TOKEN["h"])
                out = self.net(
                    past_values=torch.from_numpy(xf)[..., None],
                    past_observed_mask=torch.from_numpy(obs)[..., None],
                    freq_token=torch.from_numpy(np.asarray(tk, dtype=np.int64)),
                )
                return out.prediction_outputs[..., 0].numpy()[:, :horizon]

        f = F(model)
        t0 = time.perf_counter()
        chunks = []
        for i in range(0, len(ctx), 64):
            f.tok = tokens[i : i + 64]
            chunks.append(f.forecast(ctx[i : i + 64], a.horizon))
        f.tok = None
        point = np.concatenate(chunks)
        wall = time.perf_counter() - t0
        if not np.isfinite(point).all():
            raise ValueError(f"{name}: non-finite forecasts on {a.suite}")
        latency = lat.measure(f, T, a.horizon)
        out = ROOT / "results" / "external" / "ttm" / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"{a.suite}.npz", context=ctx.astype(np.float32), point=point.astype(np.float32),
            ids=np.array(s["ids"]),
        )  # fmt: skip
        (out / f"{a.suite}.json").write_text(
            json.dumps(
                {
                    "model": name, "hf_repo": "ibm-granite/granite-timeseries-ttm-r2", "model_key": key,
                    "n_params": n_params, "package": "granite-tsfm==0.3.12", "torch": torch.__version__,
                    "latency": latency, "wall_seconds_full_suite": round(wall, 3),
                    "kind": "point forecast (scored as a degenerate distribution: all quantiles = the point)",
                    "freq_tokens": "per window from the dataset frequency (shapes and monthly: oov)",
                },
                indent=1,
            )
        )  # fmt: skip
        print(f"{name} [{key}] {n_params:,} params · {a.suite}: {len(point)} windows · {wall:.1f}s · "
              f"t1/b1 p50 {latency['threads1_batch1']['p50_ms']:.2f} ms")  # fmt: skip


if __name__ == "__main__":
    main()
