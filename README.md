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
make bench S=shapes # 20 canonical shapes x model zoo -> results/shapes/
make web-dev        # demo site (web/)
```
