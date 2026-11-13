"""make bench S=<suite> [M=model1,model2] -> results/<suite>/*.json + summary.json"""

import argparse

from tinycast.config import BenchConfig, load
from tinycast.eval.benchmark import ROOT, load_suite, run_model, summarize, write_suite
from tinycast.models import build_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--models", default=None, help="comma-separated subset of model names")
    ap.add_argument("--no-latency", action="store_true")
    a = ap.parse_args()
    cfg = load(ROOT / "configs" / "eval" / f"{a.suite}.yaml", BenchConfig)
    if a.no_latency:
        cfg.latency = False
    suite = load_suite(cfg.suite, cfg)
    out = ROOT / "results" / cfg.suite
    write_suite(suite, out)
    wanted = set(a.models.split(",")) if a.models else None
    for entry in cfg.models:
        if wanted and entry.name not in wanted:
            continue
        if cfg.season and entry.kind in ("seasonal_naive", "ets"):
            entry.season = list(suite.season) if cfg.season == "freq" else cfg.season
        print(f"[{entry.name}] building", flush=True)
        model = build_model(entry)
        r = run_model(model, entry, suite, cfg, out)
        lat = r.get("latency", {}).get("threads1_batch1", {}).get("p50_ms")
        print(
            f"[{entry.name}] mean WQL {sum(v for v in r['per_series']['wql'] if v is not None) / len(r['ids']):.4f}  latency t1/b1 p50 {lat} ms",
            flush=True,
        )
        del model
    s = summarize(out)
    for row in s["rows"]:
        print(
            f"{row['model']:18s} relWQL {row['rel_wql']:.3f} [{row['rel_wql_ci'][0]:.3f},{row['rel_wql_ci'][1]:.3f}]  cov80 {row['mean_cov80']:.2f}"
        )
