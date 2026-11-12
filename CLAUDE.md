# CLAUDE.md — tinycast

> Working name: **tinycast** (rename freely: update `pyproject.toml`, the package folder and this file).
> Read this file fully, then read `PLAN.md`, before doing anything in this repo.

## 1. What this project is

A portfolio + learning project: **distill large time-series forecasting foundation models (TSFMs) into a tiny probabilistic forecaster that runs live in the browser**, and publish what was learned as a series of articles.

Three deliverables, in order of importance:

1. **A rigorous, reproducible distillation study.** Teachers: Datadog **Toto 2.0** and The Forecasting Company **t0** (both Apache-2.0). Student: a ~0.3–3M parameter quantile forecaster. Key claim to test: *"a student trained on teacher outputs beats the same student trained on ground truth alone, at 50–100× fewer parameters"*, and how close it gets to the teachers.
2. **An interactive demo website** with:
   - **Draw mode** (inspired by the Timestral "draw any time series and see a live forecast" demo): the user draws a series on a canvas, the student forecasts it live in the browser (onnxruntime-web), with a quantile fan chart and an in-browser latency readout.
   - **Shapes gallery** (inspired by Meta FAIR's e-means project page, which compares methods side by side on canonical toy datasets): a grid of canonical shapes × models, each cell showing the forecast, the error and the latency.
   - **Results page**: accuracy-vs-latency / accuracy-vs-size Pareto charts and ablations.
3. **Four articles** (drafts in `docs/articles/`) on topics rarely covered well online: evaluation of TSFMs, distillation techniques, compression for the browser, and the demo. The owner writes them, using the journal and the figures this repo produces.

### Who you are working with
The owner is an ML engineer who wants to **learn distillation and showcase model-tuning skills in interviews**. So:
- **Explain non-trivial choices.** When you pick a loss, a normalisation, an architecture or a metric, state the reason in 2–4 sentences in your reply *and* in `docs/journal.md`. Prefer clarity to cleverness.
- **Propose before large changes.** For anything that changes architecture, data splits, metrics or consumes > 1 GPU-hour, write the proposal (what, why, cost, alternatives) and wait for approval.
- **Do not hide the learning.** Keep the core pieces written by hand in plain PyTorch: losses, training loop, normalisation, metrics. Avoid high-level frameworks (no Lightning, no AutoGluon training loops) for those parts.

## 2. Ground rules (non-negotiable)

1. **Never fabricate results.** Every number in a README, article draft, chart or journal entry must come from a file under `results/` produced by a committed script + config. If a number isn't computed yet, write `TBD`.
2. **No evaluation leakage.**
   - The **fresh evaluation set** (French electricity data published *after* the teachers' release dates) is the headline benchmark. Its time cutoff lives in `configs/data/splits.yaml` and is enforced by a test (`tests/test_leakage.py`).
   - Never train, tune, or select checkpoints on the fresh test split. Use the validation split.
   - Public benchmarks (e.g. GIFT-Eval) are *secondary* and must be flagged as "teachers may have seen this data".
3. **Honest baselines are mandatory** in every results table: seasonal naive, ETS (statsforecast), the **official small Toto 2.0 checkpoints** (e.g. 4m / 22m — the thing our student must justify itself against), and the same student architecture trained on ground truth only.
4. **Licences.**
   - Teachers for distillation: Toto 2.0 and t0 only (Apache-2.0). t0 weights are gated on Hugging Face: the owner must accept terms and provide `HF_TOKEN`.
   - Mistral **Timestral** (modified MIT with a revenue clause) and Amazon **Chronos-2** may be used **as evaluation comparators only**. Do not distill from Timestral unless the owner explicitly approves after reading its licence.
   - Record every dataset's licence and source URL in `data/LICENSES.md` before using it.
5. **Verify third-party APIs; don't guess them.** Model package APIs (Toto 2.0, tfc-t0, Chronos-2, Timestral) change fast. Before writing an adapter, read the installed package's README/source (`python -c "import x, inspect; ..."`, or the repo on GitHub) and pin the version in `pyproject.toml`. Write down what you learned in `docs/notes/teacher-apis.md`.
6. **Reproducibility.** Every run has: a config file, a seed, a git SHA, and an output dir `runs/<date>_<name>/` with `config.yaml`, `metrics.json`, and the checkpoint (or a pointer to it on HF Hub). Results that go into articles are copied/aggregated into `results/` by a script.
7. **Respect the compute budget** (section 6). Measure before you scale.

## 3. Tech stack

**Python (research + training)**
- Python **3.11** (t0's package supports 3.11–3.13), managed with **uv**. `uv sync`, `uv run ...`.
- PyTorch (CUDA when available, CPU fallback for everything except big caching/training runs).
- numpy, pandas, **polars** + **pyarrow** (parquet caches), **statsforecast** (ETS / seasonal naive baselines), scipy.
- Teachers: `toto` (Datadog Toto 2.0, version per their README), `tfc-t0`; comparators: `chronos-forecasting` (Chronos-2), Timestral (per Mistral's docs).
- Configs: YAML files under `configs/` loaded into typed dataclasses (`tinycast/config.py`). No Hydra — keep it inspectable.
- Experiment tracking: **Weights & Biases** (free tier), with `WANDB_MODE=offline` as the default fallback. Always also write `metrics.json` locally.
- Quality: **ruff** (lint + format), **pytest**, **pre-commit**. Type hints everywhere in `tinycast/`; `mypy` optional.
- Export: `torch.onnx.export` (opset ≥ 17), `onnxruntime` (parity tests), `onnxruntime.quantization` (int8 dynamic/static).

**Web demo**
- **Vite + TypeScript**, no heavy framework (vanilla TS + small components) unless the owner asks for Svelte.
- **onnxruntime-web** (WASM backend; WebGPU optional) for the student.
- Charts: **D3** (fan charts, gallery small multiples, Pareto).
- Static hosting: **GitHub Pages** (built by GitHub Actions from `web/`).
- Optional live teacher backend: a **Hugging Face Space** (FastAPI, CPU) in `space/`, exposing `POST /forecast`. Precomputed teacher outputs are the default, so the site works with no backend.

## 4. Repository layout

```
tinycast/
├── CLAUDE.md                 # this file
├── PLAN.md                   # phased implementation plan (source of truth for "what next")
├── README.md                 # project front page: headline chart, demo link, article links
├── pyproject.toml            # uv-managed; pinned teacher package versions
├── Makefile                  # common commands (see section 5)
├── configs/
│   ├── data/                 # datasets, splits.yaml (time cutoffs), window sampling
│   ├── teachers/             # one yaml per teacher/comparator (checkpoint id, context len, batch size)
│   ├── student/              # architecture sizes (s/m/l)
│   ├── train/                # distillation recipes (loss mix, alpha, horizon weights, lr…)
│   └── eval/                 # benchmark suites (shapes, fresh_fr_power, gift_subset)
├── tinycast/                 # python package
│   ├── config.py
│   ├── data/
│   │   ├── shapes.py         # canonical shape generators (context + true future, seeded)
│   │   ├── synthetic.py      # KernelSynth-style GP kernel compositions + augmentations
│   │   ├── sources/          # rte_eco2mix.py, open_meteo.py, hf_datasets.py
│   │   ├── windows.py        # window sampler (context/horizon, strides, masks)
│   │   └── cache.py          # teacher-output cache (sharded parquet, read/write, integrity)
│   ├── models/
│   │   ├── base.py           # Forecaster protocol (see section 7)
│   │   ├── baselines.py      # seasonal naive, ETS
│   │   ├── teachers/         # toto.py, t0.py, chronos2.py, timestral.py  (adapters only)
│   │   └── student/          # patch-transformer + MLP variants, quantile head, scaler
│   ├── distill/
│   │   ├── losses.py         # pinball, teacher-quantile matching, sample-pinball, horizon weights
│   │   ├── weighting.py      # multi-teacher per-sample weighting / gating
│   │   └── train.py          # plain PyTorch training loop
│   ├── eval/
│   │   ├── metrics.py        # WQL/CRPS-approx, MASE, coverage, interval width, fidelity-to-teacher
│   │   ├── latency.py        # CPU p50/p95 latency, batch 1 and batch 32, threads=1 and 8
│   │   ├── benchmark.py      # runs any Forecaster on a suite → results/*.json
│   │   └── plots.py          # all figures for README/articles (matplotlib, consistent style)
│   └── export/
│       ├── onnx.py           # export + parity check
│       └── quantize.py
├── scripts/                  # thin CLIs: cache_teachers.py, train.py, benchmark.py, export.py, build_web_data.py
├── notebooks/
│   ├── exploration/          # local scratch (not used for results)
│   └── kaggle/               # small notebooks that `pip install` this repo and call scripts/
├── tests/                    # pytest: metrics, shapes determinism, leakage, cache, onnx parity
├── runs/                     # git-ignored run outputs
├── results/                  # small, committed, aggregated JSON/CSV + figures used in articles
├── data/                     # git-ignored raw/processed data; data/LICENSES.md is committed
├── web/                      # Vite + TS demo site
│   ├── public/models/        # student .onnx files (fp32 + int8), copied by build script
│   ├── public/data/          # gallery.json, results.json, precomputed teacher forecasts
│   └── src/                  # draw/, gallery/, results/, lib/ (ort session, scaler, charts)
├── space/                    # optional HF Space (FastAPI) serving teacher forecasts
└── docs/
    ├── journal.md            # dated lab notebook (see section 8)
    ├── decisions/            # short ADRs: NNNN-title.md
    ├── notes/                # teacher-apis.md, compute.md, gotchas.md
    └── articles/             # A1–A4 kits (A<N>-slug.md) + figures/ per article
```

## 5. Commands

Keep the `Makefile` in sync with this list.

```bash
make setup          # uv sync, pre-commit install
make lint           # ruff check + ruff format --check
make test           # pytest -q (fast tests only; mark slow ones with @pytest.mark.slow)
make test-all       # including slow tests (teacher loading, onnx parity)
make data           # download + process raw data per configs/data/*
make cache T=toto   # run a teacher over the training windows → data/cache/<teacher>/
make train C=configs/train/<recipe>.yaml
make bench S=shapes # run benchmark suite S for all models in configs/eval/<S>.yaml
make export R=runs/<run>   # onnx + int8 + parity check → web/public/models/
make web-data       # build gallery.json / results.json for the site from results/
make web-dev        # cd web && npm run dev
make web-build      # cd web && npm run build
make figures        # regenerate every figure in results/figures from results/*.json
```

Run `make lint test` before telling the owner a task is done.

## 6. Compute budget and environment

- **Local machine**: CPU-only development, unit tests, small debug runs (≤ 5 minutes), latency benchmarks, web dev.
- **Kaggle free GPU** (T4/P100, ~30 h/week) or **Colab**: teacher caching and student training. Kaggle notebooks live in `notebooks/kaggle/`, `pip install` the repo from GitHub at a pinned commit, call `scripts/*.py`, and push artifacts to the HF Hub (`<owner>/tinycast-*`) or W&B.
- **Budget targets** (revise in `docs/notes/compute.md` after measuring in Phase 1):
  - Teacher caching: ≤ 20 GPU-hours total for all teachers.
  - One student training run: ≤ 2 GPU-hours (small runs ≤ 20 min for sweeps).
  - The whole project: ≤ 100 GPU-hours.
- Before any job estimated at > 1 GPU-hour: post the estimate (throughput measured on a small sample × size) and wait for approval.
- Store teacher outputs as **float16** quantiles; regenerate synthetic contexts from seeds instead of storing them.

## 7. Core abstractions

```python
# tinycast/models/base.py
class Forecaster(Protocol):
    name: str
    quantile_levels: tuple[float, ...]      # the project-wide grid (see below)
    def forecast(
        self,
        context: np.ndarray,                # (batch, T) float32, NaN = missing
        horizon: int,
        past_covariates: np.ndarray | None = None,     # (batch, C, T)    — optional
        future_covariates: np.ndarray | None = None,   # (batch, C, T+H)  — optional
    ) -> np.ndarray: ...                    # (batch, len(quantile_levels), horizon)
```

- **Project quantile grid**: `(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)`. Adapters convert each model's native output to this grid (e.g. Toto's distribution → quantiles from ≥ 256 samples or analytically if supported; t0 / Chronos-2 / Timestral → select or interpolate from native quantiles). Document each conversion in `docs/notes/teacher-apis.md`.
- **Scaling**: every model receives raw values. The student owns its scaler (instance normalisation on the context: robust location/scale + optional `asinh`). Its forward pass is `raw → scaled → net → unscaled`, and the **exported ONNX includes the scaler** so the browser receives raw-space quantiles.
- **Student defaults** (change via proposal): context 512, horizon 64, patch 16, 4 transformer layers, d_model 128, 4 heads, quantile head with monotonicity enforced (cumulative softplus deltas around the median). Size variants s/m/l ≈ 0.3M / 1M / 3M params.
- **Univariate first.** Covariates (temperature from Open-Meteo) are a later, optional phase.

## 8. Journal, decisions and articles

- `docs/journal.md`: add a dated entry whenever you finish a task that produced results or a decision. Format:
  ```
  ## YYYY-MM-DD — <short title>
  **Did:** … **Result:** (numbers + path to results file) … **Surprise / learned:** … **Next:** …
  ```
  Surprises and failed attempts are the most valuable article material. Always log them.
- `docs/decisions/NNNN-title.md`: one short ADR (context, options, decision, consequences) per significant choice (metric definition, split dates, student architecture, loss).
- `docs/articles/A<N>-slug.md`: there are **four articles only** (A1 evaluation, A2 distillation, A3 compression, A4 demo; see the table in `PLAN.md`). Other phases only feed the journal and `results/figures/`. At each article milestone, create an **article kit**: a suggested title, a 5–8 bullet outline, the key figures (regenerated via `make figures` into `docs/articles/NN-slug/figures/`), a table of the headline numbers with their source files, and 3 "things that surprised us" pulled from the journal. **Do not write the article prose** unless asked; the owner writes it.

## 9. Coding conventions

- Small, reviewable commits with conventional messages (`feat:`, `fix:`, `exp:`, `docs:`, `refactor:`, `test:`). Don't commit data, checkpoints, `runs/` or secrets.
- Teacher adapters are thin and isolated: no teacher-specific code outside `tinycast/models/teachers/`.
- Pure functions for metrics and losses, each with unit tests against hand-computed examples (and cross-checked against a reference implementation where one exists).
- Shape generators and window samplers are **deterministic given a seed**, which a test enforces.
- Every figure uses `tinycast/eval/plots.py` styles (consistent colours per model across all figures and the website; define the palette once in `configs/style.yaml` and export it to `web/src/lib/palette.ts`).
- Secrets via environment variables only (`HF_TOKEN`, `WANDB_API_KEY`); provide `.env.example`.
- Frontend: TypeScript strict mode; no network calls except to the static assets and the optional Space URL in `web/.env`.

## 10. Definition of done (any task)

- Code + tests pass (`make lint test`).
- If it produced numbers: they are in `results/` (or `runs/`), reproducible from a committed config, and summarised in the journal.
- If it changed behaviour or a decision: an ADR or journal entry exists.
- `PLAN.md` checkboxes updated.
- Your reply to the owner: what was done, the key numbers, what you learned or found surprising, and the proposed next step.

## 11. Things to avoid

- Claiming the student "beats" a teacher without the fresh-split evidence and confidence intervals (bootstrap over series; ≥ 3 seeds for headline runs).
- Tuning on the test split, or picking the best of many runs on test.
- Silent fallbacks (e.g. an adapter quietly returning point forecasts as all quantiles). Raise instead.
- Large dependencies in the web bundle (> 1 MB JS excluding ort-wasm), or student models > 10 MB.
- Rewriting teacher architectures, or fine-tuning teachers. That's out of scope unless the owner adds it.
