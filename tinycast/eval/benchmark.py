"""Run any Forecaster on a suite -> results/<suite>/<model>.json (+ _suite.json, summary.json)."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinycast.config import QUANTILE_LEVELS, BenchConfig, to_dict
from tinycast.eval import latency as L
from tinycast.eval import metrics as M

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Suite:
    name: str
    ids: list[str]
    context: np.ndarray  # (N, T)
    future: np.ndarray  # (N, H)
    season: list[int | None]  # used by the MASE scale only (evaluator side)
    meta: list[dict]
    flag: str | None = None  # e.g. "teachers may have seen this data"


def load_suite(name: str, cfg: BenchConfig) -> Suite:
    if name == "shapes":
        from tinycast.data.shapes import gallery

        g = [
            s for k in range(cfg.n_seeds) for s in gallery(cfg.context_length, cfg.horizon, seed=k)
        ]
        return Suite(
            name,
            [f"{s.name}#{s.meta['seed'] - 1_000_000}" for s in g],
            np.stack([s.context for s in g]),
            np.stack([s.future for s in g]),
            [s.meta["period"] for s in g],
            [dict(s.meta, future_noiseless=s.future_noiseless.tolist()) for s in g],
        )
    if name in ("multidomain_test", "pedestrian_test"):
        from tinycast.data.sources import hf_datasets as H

        held = name == "pedestrian_test"
        series = [s for s in H.load_all(include_held_out=True) if (s.dataset in H.HELD_OUT) == held]
        ctx, fut, ids, seasons = H.windows(
            series, "test", 600, seed=12345, context_length=cfg.context_length, horizon=cfg.horizon
        )
        return Suite(
            name, ids, ctx, fut, list(seasons),
            [dict(series_id=i.split("@")[0], dataset=i.split("/")[0]) for i in ids],
            flag="public datasets: teachers may have seen them in pretraining",
        )  # fmt: skip
    if name == "fresh_fr_power":
        from tinycast.data.sources.rte_eco2mix import fresh_suite

        return fresh_suite(cfg)
    raise ValueError(f"unknown suite {name}")


def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "tinycast", "configs", "scripts"],
            cwd=ROOT,
            text=True,
        )
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "uncommitted"


def checkpoint_size_mb(checkpoint: str | None) -> float | None:
    if not checkpoint:
        return None
    p = Path(checkpoint)
    if p.exists():
        files = [p] if p.is_file() else list(p.rglob("*"))
    else:
        try:
            from huggingface_hub import snapshot_download

            d = Path(
                snapshot_download(
                    checkpoint, allow_patterns=["*.safetensors"], local_files_only=True
                )
            )
            files = list(d.rglob("*.safetensors"))
        except Exception:
            return None
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 2)


def per_series_metrics(suite: Suite, q: np.ndarray) -> dict[str, np.ndarray]:
    y, lv = suite.future, QUANTILE_LEVELS
    scale = np.array(
        [M.mase_scale(c[None], s)[0] for c, s in zip(suite.context, suite.season, strict=True)]
    )
    return {
        "wql": M.wql(y, q, lv),
        "wql13": M.wql(y, q, lv, use_levels=None),
        "mase": np.array(
            [
                M.mase(y[i : i + 1], q[i : i + 1], lv, suite.context[i : i + 1], suite.season[i])[0]
                for i in range(len(y))
            ]
        ),
        "cov80": M.coverage(y, q, lv, 0.1, 0.9),
        "cov98": M.coverage(y, q, lv, 0.01, 0.99),
        "width80": M.interval_width(q, lv, 0.1, 0.9, scale),
    }


def _clean(x):
    return [None if not np.isfinite(v) else round(float(v), 5) for v in x]


def run_model(model, entry, suite: Suite, cfg: BenchConfig, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    q = model.forecast(suite.context, cfg.horizon)
    wall = time.perf_counter() - t0
    mets = per_series_metrics(suite, q)
    res = {
        "model": entry.name,
        "kind": entry.kind,
        "checkpoint": entry.checkpoint,
        "suite": suite.name,
        "suite_flag": suite.flag,
        "quantile_levels": list(QUANTILE_LEVELS),
        "native_levels": list(getattr(model, "native_levels", QUANTILE_LEVELS)),
        "n_params": getattr(model, "n_params", 0),
        "size_mb": checkpoint_size_mb(entry.checkpoint),
        "fallbacks": sorted(set(getattr(model, "fallbacks", []))),
        "wall_seconds_full_suite": round(wall, 3),
        "hardware": L.hardware(),
        "git_sha": git_sha(),
        "config": to_dict(cfg),
        "per_series": {k: _clean(v) for k, v in mets.items()},
        "ids": suite.ids,
        "groups": [m.get("series_id", i) for m, i in zip(suite.meta, suite.ids, strict=True)],
        "forecasts_f16": np.asarray(q, dtype=np.float16).tolist(),
    }
    if cfg.latency:
        res["latency"] = L.measure(model, cfg.context_length, cfg.horizon)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.name}.json").write_text(json.dumps(res))
    return res


def write_suite(suite: Suite, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_suite.json").write_text(
        json.dumps(
            {
                "suite": suite.name,
                "flag": suite.flag,
                "ids": suite.ids,
                "context": [_clean(c) for c in suite.context],
                "future": [_clean(f) for f in suite.future],
                "season": suite.season,
                "meta": suite.meta,
            }
        )
    )


def summarize(out_dir: Path, baseline: str = "seasonal_naive") -> dict:
    files = sorted(p for p in out_dir.glob("*.json") if not p.name.startswith(("_", "summary")))
    runs = {json.loads(p.read_text())["model"]: json.loads(p.read_text()) for p in files}
    if baseline not in runs:
        raise ValueError(f"baseline {baseline} missing from {out_dir}")
    suite_meta = json.loads((out_dir / "_suite.json").read_text())
    suite_groups = [
        m.get("series_id", i) for m, i in zip(suite_meta["meta"], suite_meta["ids"], strict=True)
    ]
    base = np.array(runs[baseline]["per_series"]["wql"], dtype=float)
    base_mase = np.array(runs[baseline]["per_series"]["mase"], dtype=float)
    rows = []
    for name, r in runs.items():
        ps = {
            k: np.array([np.nan if v is None else v for v in vals], dtype=float)
            for k, vals in r["per_series"].items()
        }
        rel = M.relative_scores(ps["wql"], base)
        rel_mase = M.relative_scores(ps["mase"], base_mase)
        groups = r.get("groups") or suite_groups  # cluster bootstrap over series
        g, lo, hi = M.bootstrap_ci(rel, "geomean", groups=groups)
        gm, lom, him = M.bootstrap_ci(rel_mase, "geomean", groups=groups)
        lat = r.get("latency", {})
        rows.append(
            {
                "model": name,
                "kind": r["kind"],
                "n_params": r["n_params"],
                "size_mb": r["size_mb"],
                "native_levels": r["native_levels"],
                "rel_wql": g,
                "rel_wql_ci": [lo, hi],
                "rel_mase": gm,
                "rel_mase_ci": [lom, him],
                "mean_wql": float(np.nanmean(ps["wql"])),
                "mean_cov80": float(np.nanmean(ps["cov80"])),
                "mean_cov98": float(np.nanmean(ps["cov98"])),
                "tails_extrapolated": len(r["native_levels"]) < len(QUANTILE_LEVELS),
                "latency_ms_t1_b1": lat.get("threads1_batch1", {}).get("p50_ms"),
                "latency_ms_t8_b32": lat.get("threads8_batch32", {}).get("p50_ms"),
                "n_windows": int(np.isfinite(rel).sum()),
                "n_clusters": len(set(groups)),
            }
        )
    rows.sort(key=lambda r: r["rel_wql"])
    summary = {"suite": out_dir.name, "baseline": baseline, "rows": rows, "git_sha": git_sha()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
