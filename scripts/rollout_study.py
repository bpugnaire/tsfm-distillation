"""Teacher rollouts: start on real context, feed the teacher's own forecasts back in -> long series.
-> results/rollouts.json   (owner idea 2026-11-13; prototype, small)

A forecast is a distribution; to continue the series one path must be chosen from it. Strategies:
  median       feed back the median (what Toto/Timestral do internally beyond one shot)
  independent  per step, u ~ U(0.01, 0.99) iid, value = quantile function at u
  copula       u_t = Phi(z_t), z_t = rho z_{t-1} + sqrt(1 - rho^2) eps  (temporally correlated path)
The quantile function is linear interpolation between the 13 levels (u clipped to [0.01, 0.99]).

Per generated block b we measure, relative to the real context (last 512 steps before generation):
  amplitude  = std of the block's step-to-step changes / same on the real context   (1 = same roughness)
  spread     = std of the block's values / std of 64-step real windows               (1 = same variability)
  drift      = |mean(block) - mean(real context)| / std(real context)
"""

import argparse
import json

import numpy as np
from scipy.stats import norm

from tinycast.config import QUANTILE_LEVELS, ModelEntry
from tinycast.eval.benchmark import ROOT, git_sha
from tinycast.models import build_model

LV = np.array(QUANTILE_LEVELS)


def sample_path(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """q: (B, K, H) quantiles; u: (B, H) in (0, 1) -> (B, H) values on the interpolated quantile function."""
    u = np.clip(u, LV[0], LV[-1])
    idx = np.clip(np.searchsorted(LV, u) - 1, 0, len(LV) - 2)
    w = (u - LV[idx]) / (LV[idx + 1] - LV[idx])
    lo = np.take_along_axis(q, idx[:, None, :], 1)[:, 0]
    hi = np.take_along_axis(q, (idx + 1)[:, None, :], 1)[:, 0]
    return lo * (1 - w) + hi * w


def rollout(
    model, ctx: np.ndarray, blocks: int, strategy: str, rng, rho: float = 0.9
) -> np.ndarray:
    series = ctx.copy()
    z = rng.standard_normal(len(ctx))
    for _ in range(blocks):
        q = model.forecast(series[:, -512:], 64)
        if strategy == "median":
            nxt = q[:, list(QUANTILE_LEVELS).index(0.5)]
        elif strategy == "independent":
            nxt = sample_path(q, rng.uniform(size=(len(ctx), 64)))
        else:
            zs = np.empty((len(ctx), 64))
            for t in range(64):
                z = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(len(ctx))
                zs[:, t] = z
            nxt = sample_path(q, norm.cdf(zs))
        series = np.concatenate([series, nxt.astype(np.float32)], axis=1)
    return series


def block_stats(series: np.ndarray, n_ctx: int, blocks: int) -> dict:
    real = series[:, :n_ctx]
    real_rough = np.nanstd(np.diff(real, axis=1), axis=1)
    real_win = np.nanmean(
        [np.nanstd(real[:, i : i + 64], axis=1) for i in range(0, n_ctx - 63, 64)], axis=0
    )
    mu, sd = np.nanmean(real, axis=1), np.nanstd(real, axis=1)
    out = {"amplitude": [], "spread": [], "drift": []}
    for b in range(blocks):
        blk = series[:, n_ctx + 64 * b : n_ctx + 64 * (b + 1)]
        out["amplitude"].append(float(np.median(np.std(np.diff(blk, axis=1), axis=1) / real_rough)))
        out["spread"].append(float(np.median(np.std(blk, axis=1) / real_win)))
        out["drift"].append(float(np.median(np.abs(blk.mean(1) - mu) / sd)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--blocks", type=int, default=16)
    a = ap.parse_args()
    from tinycast.data.corpus import CorpusSpec, build_corpus
    from tinycast.data.shapes import SHAPE_NAMES, make_shape

    seeds = {
        "real_electricity": build_corpus(CorpusSpec("val", a.n, 0, 0, 512, 64)).context,
        "shapes": np.stack(
            [
                make_shape(
                    SHAPE_NAMES[i % len(SHAPE_NAMES)], 900_100 + i, 512, 64, randomize=True
                ).context
                for i in range(a.n)
            ]
        ),
    }
    seeds = {
        k: np.where(np.isnan(v), np.nanmedian(v, axis=1, keepdims=True), v).astype(np.float32)
        for k, v in seeds.items()
    }
    teachers = [
        ("timestral", "timestral-28m", "mistralai/Timestral-28M"),
        ("toto", "toto-2.0-22m", "Datadog/Toto-2.0-22m"),
    ]
    res = {"n": a.n, "blocks": a.blocks, "git_sha": git_sha(), "stats": {}, "examples": {}}
    for kind, name, ck in teachers:
        m = build_model(ModelEntry(kind=kind, name=name, checkpoint=ck, batch_size=64))
        for src, ctx in seeds.items():
            for strat in ("median", "independent", "copula"):
                s = rollout(m, ctx, a.blocks, strat, np.random.default_rng(0))
                st = block_stats(s, 512, a.blocks)
                res["stats"][f"{name}|{src}|{strat}"] = st
                print(
                    f"{name:14s} {src:17s} {strat:12s} spread b1 {st['spread'][0]:.2f} b8 {st['spread'][7]:.2f} b16 {st['spread'][-1]:.2f}"
                    f" | rough b1 {st['amplitude'][0]:.2f} b16 {st['amplitude'][-1]:.2f} | drift b16 {st['drift'][-1]:.2f}",
                    flush=True,
                )
                if name == "timestral-28m":
                    res["examples"][f"{src}|{strat}"] = [
                        [round(float(v), 3) for v in s[i, 256::2]] for i in range(3)
                    ]
    (ROOT / "results" / "rollouts.json").write_text(json.dumps(res))
