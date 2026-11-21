"""Data for the two explainer posts (training data; inside the student) -> web/public/data/explain.json.

Everything is recomputed from the code that built the published student (Stage C3, Toto-2.0-313m labels):
  - the training corpus is rebuilt with the same functions and checked window-id by window-id against the stored corpus;
  - the teacher labels are read from the cache shards the training read;
  - the inference trace runs the trained checkpoint layer by layer and is checked against Student.forward.

    uv run python scripts/build_explainers.py
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch

from tinycast.config import TrainConfig, load
from tinycast.data import shapes as S
from tinycast.data.cache import _decode, cache_dir
from tinycast.data.corpus import ROOT, _synth_seeds
from tinycast.data.sources import data_v2 as V
from tinycast.data.sources import hf_datasets as H
from tinycast.data.sources.rte_eco2mix import series_arrays
from tinycast.data.splits import load_splits
from tinycast.data.synthetic import TRAIN_SEED_MAX, synthetic_series
from tinycast.data.windows import fresh_test_windows, window_id
from tinycast.distill.train import specs
from tinycast.models.student import StudentForecaster

RUN = ROOT / "runs" / "2026-11-16_stage_c3_kd_toto_1m"
TEACHER = "toto-2.0-313m"
OUT = ROOT / "web" / "public" / "data" / "explain.json"
DOMAIN_CHRONOS = {  # chronos_datasets entries of the data v2 pool (domains of the GiftEvalPretrain ones are in the yaml)
    "monash_traffic": "traffic",
    "monash_rideshare": "mobility",
    "taxi_30min": "mobility",
    "monash_weather": "climate",
    "exchange_rate": "finance",
    "monash_tourism_monthly": "tourism",
    "monash_fred_md": "economics",
}


def r(a, nd: int = 4):
    """JSON-friendly rounded list, NaN -> None."""
    a = np.asarray(a, dtype=np.float64)
    return (
        [None if not np.isfinite(v) else round(float(v), nd) for v in a.ravel()]
        if a.ndim <= 1
        else [r(row, nd) for row in a]
    )


def downsample(v: np.ndarray, n: int = 900) -> list:
    """Bin means (ignoring NaN) so a long series fits a chart."""
    k = int(np.ceil(len(v) / n))
    pad = (-len(v)) % k
    b = np.concatenate([v, np.full(pad, np.nan)]).reshape(-1, k)
    with np.errstate(all="ignore"):
        return r(np.nanmean(b, axis=1), 3), k


def read_shard_rows(
    corpus_key: str, rows: list[int], shard_size: int = 4096
) -> dict[int, tuple[str, np.ndarray]]:
    """Teacher quantiles for corpus rows, straight from the shard files (written in corpus order)."""
    d = cache_dir(TEACHER, corpus_key)
    man = json.loads((d / "manifest.json").read_text())
    out = {}
    for row in rows:
        t = pq.read_table(d / f"shard_{row // shard_size:05d}.parquet")
        j = row % shard_size
        out[row] = (
            t["window_id"][j].as_py(),
            _decode([t["q"][j].as_py()], len(man["levels"]), man["horizon"])[0],
        )
    return out


# ---------------------------------------------------------------------------------------------------------- data
def data_section(log=print) -> dict:
    cfg = load(RUN / "config.yaml", TrainConfig)
    tr_spec, va_spec = specs(cfg)
    key = tr_spec.key()
    T, Hh = tr_spec.context_length, tr_spec.horizon
    stored = np.load(ROOT / "data" / "processed" / f"corpus_train_{key}.npz")
    ids, src = stored["ids"], stored["source"]
    counts = {k: int(v) for k, v in Counter(src.tolist()).items()}
    log(f"corpus {key}: {len(ids)} windows {counts}")

    # real part: rebuild with the training code and check the ids match the stored corpus
    t0 = time.time()
    pool = V.load_v2()
    ctx, fut, wids, seasons, quota = V.windows_v2(
        pool, "train", counts["multidomain_v2"], tr_spec.seed, T, Hh
    )
    rebuilt = np.array([window_id("md2", i, 0) for i in wids])
    real_rows = np.flatnonzero(src == "multidomain_v2")
    if not np.array_equal(rebuilt, ids[real_rows]):
        raise ValueError("rebuilt data v2 windows do not match the stored corpus")
    log(f"data v2 rebuilt and matched in {time.time() - t0:.0f}s ({len(pool)} series)")

    import yaml

    gp = yaml.safe_load((ROOT / "configs" / "data" / "gift_pretrain_v2.yaml").read_text())[
        "subsets"
    ]
    domain = {**DOMAIN_CHRONOS, **gp}
    n_series = Counter(s.dataset for s in pool)
    ds_rows = [
        {
            "dataset": nm,
            "domain": domain.get(nm, "other"),
            "windows": int(q),
            "series": int(n_series[nm]),
        }
        for nm, q in sorted(quota.items(), key=lambda kv: -kv[1])
    ]
    idx = json.loads(V.INDEX.read_text())

    # one real series: the whole history, its position splits, and the windows the corpus took from it
    by_series: dict[str, list[int]] = {}
    for j, w in enumerate(wids):
        by_series.setdefault(w.split("@")[0], []).append(j)
    pick_ds = "monash_traffic"
    sk = max((k for k in by_series if k.startswith(pick_ds + "/")), key=lambda k: len(by_series[k]))
    ser = next(s for s in pool if f"{s.dataset}/{s.series_id}" == sk)
    L = len(ser.values)
    origins = sorted(int(wids[j].split("@")[1]) for j in by_series[sk])
    vals, bin_k = downsample(ser.values)
    # the candidate grid windows_v2 drew from: stride = horizon, one random offset per series
    first, last = max(64, int(H.SPLITS["train"][0] * L)), int(H.SPLITS["train"][1] * L) - Hh
    grid = [o for o in range(origins[0] % Hh, last + 1, Hh) if o >= first]
    if not set(origins) <= set(grid):
        raise ValueError("sampled origins are not on the stride grid")
    o_mid = next(
        o for o in origins[len(origins) // 2 :] if o + Hh in origins and o + 2 * Hh in grid
    )
    z0 = o_mid - T
    series_ex = {
        "dataset": pick_ds,
        "series_id": ser.series_id,
        "freq": "hourly",
        "length": L,
        "bin": bin_k,
        "values": vals,
        "splits": {k: [int(a * L), int(b * L)] for k, (a, b) in H.SPLITS.items()},
        "origins": origins,
        "grid": grid,
        "zoom": {  # consecutive grid slots: contexts overlap, targets tile the series without overlap
            "start": z0,
            "values": r(ser.values[z0 : o_mid + 3 * Hh], 4),
            "slots": [o_mid, o_mid + Hh, o_mid + 2 * Hh],
            "sampled": [o in origins for o in (o_mid, o_mid + Hh, o_mid + 2 * Hh)],
        },
    }

    # synthetic part: what each seed generates (same rng draws as synthetic_series)
    seeds = _synth_seeds("train", counts["synth"], tr_spec.seed)
    kinds: Counter = Counter()
    shape_names: Counter = Counter()
    for s in seeds:
        rng = np.random.default_rng([int(s), 7919])
        u = rng.uniform()
        if u < 0.4:
            kinds["kernel"] += 1
        elif u < 0.8:
            kinds["shape"] += 1
            shape_names[S.SHAPE_NAMES[rng.integers(len(S.SHAPE_NAMES))]] += 1
        else:
            kinds["mix"] += 1
    synth_rows = np.flatnonzero(src == "synth")
    if not np.array_equal(
        np.array([window_id("synth", str(int(s)), 0) for s in seeds]), ids[synth_rows]
    ):
        raise ValueError("synthetic seeds do not match the stored corpus")

    def kind_of(seed: int) -> str:
        u = np.random.default_rng([seed, 7919]).uniform()
        return "kernel" if u < 0.4 else "shape" if u < 0.8 else "mix"

    # example windows with their teacher labels: 3 real (different domains) + 3 synthetic (one per kind)
    ex_real = []
    for want in ("PEMS03", "london_smart_meters_with_missing", "beijing_air_quality"):
        j = next(j for j, w in enumerate(wids) if w.startswith(want + "/"))
        ex_real.append(("real", int(real_rows[j]), want, ctx[j], fut[j]))
    ex_syn = []
    for kind in ("kernel", "shape", "mix"):
        j = next(j for j, s in enumerate(seeds) if kind_of(int(s)) == kind)
        v = synthetic_series(int(seeds[j]), T + Hh, T)
        ex_syn.append(("synth", int(synth_rows[j]), kind, v[:T], v[T:]))
    labels = read_shard_rows(key, [e[1] for e in ex_real + ex_syn])
    examples = []
    for kind, row, name, c, f in ex_real + ex_syn:
        wid, q = labels[row]
        if wid != ids[row]:
            raise ValueError(f"cache row {row} is {wid}, corpus says {ids[row]}")
        examples.append(
            {
                "source": kind,
                "name": name,
                "row": row,
                "window_id": wid,
                "context": r(c, 4),
                "future": r(f, 4),
                "teacher": r(q, 4),
                "n_missing": int(np.isnan(c).sum()),
            }
        )

    # the cache on disk
    d = cache_dir(TEACHER, key)
    man = json.loads((d / "manifest.json").read_text())
    shard_bytes = [p.stat().st_size for p in sorted(d.glob("shard_*.parquet"))]
    cache = {
        "path": f"data/cache/{TEACHER}/{key}/",
        "n_shards": len(man["shards"]),
        "shard_size": 4096,
        "n_windows": int(sum(man["shards"].values())),
        "bytes": int(sum(shard_bytes)),
        "manifest": {k: v for k, v in man.items() if k != "shards"},
        "first_shards": dict(list(man["shards"].items())[:2]),
    }

    # the fresh electricity timeline (test only: electricity is never in the distillation data)
    sp = load_splits()
    rte = series_arrays()
    ts, v = rte["France"]
    daily = v[: len(v) // 24 * 24].reshape(-1, 24).mean(1)
    fresh = [w for w in fresh_test_windows(rte, sp) if w.series_id == "France"]
    return {
        "corpus_key": key,
        "val_corpus_key": va_spec.key(),
        "counts": counts,
        "context_length": T,
        "horizon": Hh,
        "datasets": ds_rows,
        "n_pool_series": len(pool),
        "contamination": {
            "candidates": idx["n_candidates"],
            "kept": len(idx["kept"]),
            "dropped_by_dataset": idx["dropped_by_dataset"],
        },
        "position_splits": {k: list(v) for k, v in H.SPLITS.items()},
        "series_example": series_ex,
        "synthetic": {
            "kinds": dict(kinds),
            "shapes": dict(shape_names.most_common()),
            "seed_ranges": {
                "train": [0, TRAIN_SEED_MAX // 2],
                "val": [TRAIN_SEED_MAX // 2, TRAIN_SEED_MAX],
                "gallery_and_tests": [TRAIN_SEED_MAX, None],
            },
        },
        "examples": examples,
        "cache": cache,
        "validation": {k: int(n) for k, n in va_spec.counts().items()},
        "rte": {
            "series": sorted(rte),
            "daily_start": str(ts[0].date()),
            "daily": r(daily, 0),
            "train_end": str(sp.train_end.date()),
            "val_start": str(sp.val_start.date()),
            "val_end": str(sp.val_end.date()),
            "fresh_start": str(sp.fresh_start.date()),
            "last": str(ts[-1].date()),
            "n_fresh_windows_france": len(fresh),
        },
    }


# ---------------------------------------------------------------------------------------------------------- model
GROUPS = [  # (prefix, group, role)
    ("net.embed", "patch embedding", "knowledge"),
    ("net.pos", "position table", "range"),
    ("net.query", "query token", "knowledge"),
    ("self_attn", "attention", "knowledge"),
    ("linear", "feed-forward", "knowledge"),
    ("norm", "layer norms", "plumbing"),
    ("net.head", "quantile head", "range"),
]


def param_groups(model: torch.nn.Module) -> list[dict]:
    out: dict[str, dict] = {}
    for name, p in model.named_parameters():
        g = next((grp, role) for pre, grp, role in GROUPS if pre in name)
        layer = name.split("layers.")[1].split(".")[0] if "layers." in name else None
        k = f"{g[0]}|{layer}"
        e = out.setdefault(
            k,
            {
                "group": g[0],
                "role": g[1],
                "layer": None if layer is None else int(layer),
                "params": 0,
                "shapes": [],
            },
        )
        e["params"] += p.numel()
        e["shapes"].append([name, list(p.shape)])
    return list(out.values())


@torch.no_grad()
def model_section(log=print) -> dict:
    fc = StudentForecaster.from_run(str(RUN))
    m = fc.model.eval()
    c = m.c
    net = m.net
    sp = load_splits()
    rte = series_arrays()
    wins = [w for w in fresh_test_windows(rte, sp) if w.series_id == "France"]
    w = wins[len(wins) // 2]
    x = torch.tensor(w.context)[None]

    z, mask, loc, scale = m.prepare(x)
    p = c.patch_size
    zp = z.view(1, net.n_patches, p)
    mp = mask.view(1, net.n_patches, p).to(z.dtype)
    emb_in = torch.cat([zp * mp, mp], dim=-1)
    emb = net.embed(emb_in)
    tok = torch.cat([emb, net.query.expand(1, -1, -1)], dim=1) + net.pos
    h = tok
    layers = []
    for lyr in net.encoder.layers:
        hn = lyr.norm1(h)
        _, att = lyr.self_attn(hn, hn, hn, need_weights=True, average_attn_weights=False)
        h_attn = h + lyr.self_attn(hn, hn, hn, need_weights=False)[0]
        h_new = lyr(h)
        layers.append(
            {
                "query_attention": r(
                    att[0, :, -1, :], 4
                ),  # (heads, 33): where the query token looks
                "attention_mean": r(att[0].mean(0), 3),  # (33, 33), averaged over heads
                "norm_in": r(h[0].norm(dim=-1), 3),
                "attn_update": r((h_attn - h)[0].norm(dim=-1), 3),
                "ffn_update": r((h_new - h_attn)[0].norm(dim=-1), 3),
            }
        )
        h = h_new
    final = net.norm(h[:, -1])
    raw = net.head.proj(final).view(len(c.quantile_levels), c.horizon)
    zq = net.head(final)
    q = m.scaler.unscale(zq, loc[..., None], scale[..., None])[0]
    q_ref = m(x)[0]
    err = float((q - q_ref).abs().max() / scale.item())
    if err > 1e-5:
        raise ValueError(f"trace does not reproduce Student.forward (max rel err {err})")

    # scale invariance, measured: multiply the input by 1000 and shift it, compare forecasts in units of the scale
    a, b = 1000.0, 5.0e4
    q2 = m(x * a + b)[0]
    inv_err = float(((q2 - b) / a - q_ref).abs().max() / scale.item())

    n_params = m.n_params()
    groups = param_groups(m)
    i_med = c.quantile_levels.index(0.5)
    return {
        "config": {
            k: getattr(c, k)
            for k in (
                "context_length",
                "horizon",
                "patch_size",
                "d_model",
                "n_layers",
                "n_heads",
                "ff_mult",
                "use_asinh",
            )
        }
        | {"quantile_levels": list(c.quantile_levels), "n_patches": net.n_patches},
        "n_params": n_params,
        "groups": groups,
        "trace": {
            "series_id": w.series_id,
            "target_start": str(w.target_start),
            "context": r(w.context, 1),
            "future": r(w.future, 1),
            "loc": round(loc.item(), 2),
            "scale": round(scale.item(), 2),
            "z": r(z[0], 4),
            "patch_inputs": r(emb_in[0], 4),  # (32, 2p)
            "embeddings": r(emb[0], 3),  # (32, d)
            "query": r(net.query[0, 0], 3),
            "pos_norm": r(net.pos[0].norm(dim=-1), 3),
            "emb_norm": r(emb[0].norm(dim=-1), 3),
            "layers": layers,
            "final": r(final[0], 3),
            "head_raw": r(raw, 4),  # (K, H) before softplus
            "z_quantiles": r(zq[0], 4),
            "quantiles": r(q, 1),
            "i_med": i_med,
            "forward_check_rel_err": err,
            "scale_invariance_rel_err": inv_err,
            "invariance_transform": [a, b],
        },
    }


def main() -> None:
    t0 = time.time()
    out = {
        "source": "scripts/build_explainers.py",
        "run": str(RUN.relative_to(ROOT)),
        "teacher": TEACHER,
    }
    out["model"] = model_section()
    print(f"model trace done ({time.time() - t0:.0f}s)")
    out["data"] = data_section()
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(
        f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
