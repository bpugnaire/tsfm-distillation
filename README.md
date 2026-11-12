# tinycast

Distilling large time-series foundation models (Datadog **Toto 2.0**, The Forecasting Company **t0**)
into a tiny (~0.2–3M parameter) probabilistic forecaster that runs live in your browser.

**Status: work in progress.** Every number in this repo comes from a file under `results/`
produced by a committed script + config; anything not computed yet is `TBD`.

## Planned articles
1. Evaluating forecasting foundation models properly (leakage, fresh data, baselines, CIs)
2. Distilling a time-series foundation model (loss design for probabilistic outputs)
3. Compressing a forecaster to run in the browser (ONNX with the scaler inside, int8, parity)
4. Building the demo (draw mode, shapes gallery, Pareto page)

## Reproduce
```bash
make setup          # uv sync (Python 3.12) + pre-commit
make test           # fast tests
make data           # RTE eCO2mix -> data/processed/ (~5 min download)
make bench S=shapes # 20 shapes x 10 seeds x model zoo -> results/shapes/
make bench S=fresh_fr_power
make web-data       # results/ -> web/public/data/
cd web && npm install && npm run dev   # Draw / Gallery / Results / Lab notes
```

## Status (2026-11-12)
Done: evaluation harness, teacher zoo, shapes gallery, fresh-data leaderboard, student + KD losses (tested),
ONNX/int8 export, demo site. **Not done: any distillation experiment.** The only student so far is a 3-minute
ground-truth-only *preview* used to build the pipeline. See `docs/journal.md`.
