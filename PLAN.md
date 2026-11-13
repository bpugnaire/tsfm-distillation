# PLAN.md — tinycast implementation plan

Source of truth for *what to do next*. Work top to bottom. Tick boxes as you go. Every phase ends with a journal entry and a short "phase review" message to the owner. **Only four milestones produce an article kit** (see CLAUDE.md §8), on topics that are rarely covered well online.

Effort estimates assume evenings/weekends. GPU estimates are on a free Kaggle T4 and must be re-measured.

---

## Overview

| Phase | Outcome | GPU |
|---|---|---|
| 0 | Repo, tooling, CI skeleton | 0 |
| 1 | Teacher zoo + **shapes gallery benchmark** (no training) | ~1 h |
| 2 | Real data, fresh test split, **teacher cache** | ~10–20 h |
| 3 | Student architecture, trained on ground truth only (the floor) | ~3 h |
| 4 | **Single-teacher distillation** + ablations | ~10 h |
| 5 | **Multi-teacher** distillation + generalisation to shapes | ~10 h |
| 6 | Compression: ONNX, int8, browser parity + latency | ~0 |
| 7 | **Demo website**: Draw / Gallery / Results | 0 |
| 8 (opt.) | Covariates (temperature), live teacher Space, WebGPU | ~5 h |

### Articles (4)

| # | Article | Covers phases | Kit due after |
|---|---|---|---|
| A1 | **Evaluating forecasting foundation models properly**: teacher zoo, toy-shapes gallery, leakage, fresh data, baselines, CIs | 1–2 | Phase 2 |
| A2 | **Distilling a time-series foundation model**: loss design for probabilistic outputs, α / horizon weighting, data efficiency, multi-teacher weighting, honest comparison with official tiny checkpoints | 3–5 | Phase 5 |
| A3 | **Compressing a forecaster to run in the browser**: ONNX with the scaler inside, int8, parity testing, real browser latency | 6 | Phase 6 |
| A4 | **Building the demo**: Draw mode, e-means-style gallery, Pareto page; communicating ML results interactively (+ short retrospective) | 7 | Phase 7 |

Ordering rationale: Phase 1 produces the gallery data **before any training**, which de-risks the teacher adapters and the metrics. Its findings feed A1 together with Phase 2.

---

## Phase 0 — Scaffolding (½–1 day)

- [x] `uv init` with Python 3.11 → **3.12** (toto-2 requires it; ADR 0001); `pyproject.toml` with the package `tinycast`, dev deps (ruff, pytest, pre-commit), and core deps (torch, numpy, pandas, polars, pyarrow, scipy, statsforecast, pyyaml, wandb, matplotlib).
- [x] Create the folder layout from CLAUDE.md §4 (empty modules with docstrings stating their responsibility).
- [x] `Makefile` with the targets in CLAUDE.md §5 (stubs allowed).
- [x] `.gitignore` (data/, runs/, *.onnx outside web/public/models, .env, wandb/), `.env.example`.
- [x] `pre-commit` (ruff, ruff-format, end-of-file, trailing-whitespace, large-file check 5 MB).
- [x] GitHub Actions: `ci.yml` (lint + fast tests on CPU). Pages deploy workflow comes in Phase 7.
- [x] `tinycast/config.py`: typed dataclasses + YAML loader with validation (fail on unknown keys).
- [x] `docs/journal.md` (first entry), `docs/decisions/0001-project-scope.md`, `data/LICENSES.md` (empty table), `docs/notes/compute.md`.
- [x] README skeleton: one-paragraph pitch, "status: work in progress", planned article list.

**Done when:** `make setup lint test` passes locally and in CI.

---

## Phase 1 — Teacher zoo & shapes gallery (no training)

Goal: a uniform `Forecaster` interface for all models, a canonical **shapes** suite, a correct metrics + latency harness, and a gallery like the e-means page's toy-dataset comparison, but for forecasting.

### 1.1 Metrics (`tinycast/eval/metrics.py`) — write tests first
- [x] `pinball_loss(y, q_pred, levels)`; **WQL** (weighted quantile loss, normalised by Σ|y|), our CRPS approximation; document the approximation in ADR `0002-metrics.md`.
- [x] **MASE** (seasonal period from config; fall back to lag-1 when no seasonality).
- [x] Coverage of the 80% (0.1–0.9) and 98% (0.01–0.99) intervals; mean interval width (scaled).
- [x] **Fidelity** metrics for later phases: student-vs-teacher quantile loss, median MAE vs teacher median.
- [x] Aggregation: per series → per suite with **geometric mean of relative scores vs seasonal naive** (as in GIFT-Eval) + arithmetic mean; bootstrap 95% CIs over series.
- [x] Tests: hand-computed examples; a check that WQL ≈ CRPS for a Gaussian with many quantiles (compare with the closed-form Gaussian CRPS).

### 1.2 Shapes library (`tinycast/data/shapes.py`)
Each generator returns `(context, future_true, future_noiseless, meta)` from `(seed, length, horizon, params)`. Fixed set for the gallery (seeded), randomised variants for training later.
- [x] Sine · two seasonalities (daily+weekly-like) · trend + seasonality · **multiplicative** seasonality (growing amplitude) · damped trend · exponential growth · logistic saturation
- [x] Level shift (step) · trend break · spikes/outliers in context · heteroscedastic noise · missing-value gaps
- [x] Random walk · mean-reverting AR(1) · intermittent demand (mostly zeros) · sawtooth · square wave · chirp (rising frequency) · impulse + exponential decay · constant + noise
- [x] Tests: determinism per seed, shapes and dtypes, no NaN in future, meta records period and generating params.
- [x] A quick visual check: `scripts/plot_shapes.py` → `results/figures/shapes_grid.png`.

### 1.3 Model adapters (`tinycast/models/`)
For each: read the official README/source first, pin the version, write findings in `docs/notes/teacher-apis.md` (install, checkpoint ids, max context, native output format, how you convert to the project quantile grid, CPU feasibility).
- [x] Baselines: seasonal naive (with empirical residual quantiles), ETS via statsforecast (with its intervals → quantiles).
- [x] **Toto 2.0**: the official small checkpoints (4m, 22m) + one larger checkpoint as teacher (pick the largest that keeps caching within budget; measure in 1.5).
- [x] **t0**: t0-gamma (the current release) if accessible, otherwise t0-alpha/beta; HF gated access via `HF_TOKEN`.
- [x] **Chronos-2** (comparator).
- [x] **Timestral** 5M/28M/209M (comparator; owner confirmed licence eligibility 2026-11-13) (comparator only; see licence rule).
- [ ] Contract test (`tests/test_forecaster_contract.py`, slow): every adapter returns `(B, 13, H)`, monotone in the quantile axis, finite, and handles NaNs in the context.

### 1.4 Benchmark harness (`tinycast/eval/benchmark.py`, `latency.py`)
- [x] `make bench S=shapes` runs all models configured in `configs/eval/shapes.yaml` and writes `results/shapes/<model>.json` (per-series metrics + forecasts as float16 for the gallery).
- [x] Latency: CPU, `torch.set_num_threads(1)` and `8`, batch 1 and 32, warm-up then 50 timed runs → p50/p95. Record the hardware in the JSON. Also record param count and checkpoint size on disk.
- [x] `scripts/build_web_data.py --suite shapes` → `web/public/data/gallery.json` (shapes × models: context, quantiles, truth, WQL, MASE, latency).

### 1.5 Measure for later phases
- [x] Teacher throughput (CPU + Apple MPS; Kaggle T4 not measured) (windows/second) on CPU and on a Kaggle T4, at context 512 / horizon 64 → `docs/notes/compute.md`, plus a caching budget estimate for Phase 2.

### 1.6 Material for A1 (no article kit yet)
- [x] Save the gallery figure (shapes × models small multiples), a summary table (relative WQL, latency, params), and the surprises (e.g. failure modes on multiplicative seasonality / intermittent series / level shifts) to the journal and `results/figures/`.

**Done when:** gallery.json exists for ≥ 18 shapes × ≥ 7 models, the contract test passes, and the numbers are reproducible with one command.

---

## Phase 2 — Real data, fresh test split, teacher cache

### 2.1 Data sources (`tinycast/data/sources/`)
- [x] **RTE éCO2mix** (French national + regional electricity consumption, ~15/30 min): download script, resample to hourly (configurable), handle DST and gaps. Record the licence.
- [ ] **Open-Meteo** historical weather (temperature for the same regions) — store now for optional Phase 8 covariates.
- [ ] Optional extra real training data from open HF time-series corpora (e.g. Chronos / GIFT-Eval pretraining datasets). Document each licence; keep the total small.
- [x] Synthetic training corpus (`synthetic.py`): KernelSynth-style random GP kernel compositions + randomised versions of the shape generators + TSMixup-style mixing. Deterministic from seeds.

### 2.2 Splits (`configs/data/splits.yaml`) and leakage tests
- [x] Define `train_end`, `val_end`, and the **fresh test window** (data strictly after the latest teacher release date used; record the dates and their sources in ADR `0003-splits.md`).
- [x] `tests/test_leakage.py`: no training/validation window has a future timestamp ≥ `val_end`, no test window overlaps train, and gallery shapes use seeds disjoint from training seeds.
- [ ] Secondary suite: a small GIFT-Eval subset, flagged "possibly seen by teachers".

### 2.3 Windowing (`windows.py`)
- [x] Sample (context=512, horizon=64) windows with configurable stride, a minimum observed fraction, and NaN masks. Window ids are stable hashes (source, series_id, start index) so caches can be joined.

### 2.4 Teacher cache (`cache.py`, `scripts/cache_teachers.py`)
- [x] Sharded parquet: `window_id`, teacher, quantiles float16 `(13, 64)`; optional `samples` for Toto (e.g. 64 samples float16) on a subset, used by the sample-pinball loss in Phase 4.
- [x] Resumable (skip completed shards), integrity check (counts, NaN check, monotonicity), and a manifest with teacher version + config hash.
- [ ] Size the corpus from the 1.5 measurements: start with ~300k windows per teacher, then scale if the budget allows. **Post the estimate, wait for approval, then run on Kaggle.**
- [x] Also benchmark all models (teachers, Toto small checkpoints, baselines) on the fresh test split → `results/fresh_fr_power/`.

### 2.5 Article kit A1 — "Evaluating forecasting foundation models properly"
- [ ] `docs/articles/A1-evaluation.md` kit combining Phases 1–2.
- [ ] Figures: the shapes gallery; a timeline of splits vs teacher release dates; teacher rankings on the fresh split vs a public benchmark (do they change?); the effect of CIs/seeds on the conclusions; the metric definitions (WQL vs CRPS check).
- [ ] Angle: what most TSFM comparisons get wrong (leakage, missing baselines, single numbers without CIs, ignoring latency) and how this harness avoids it.

**Done when:** leakage tests pass, teacher caches are complete and validated, and the fresh-split leaderboard exists.

---

## Phase 3 — Student architecture & ground-truth floor

- [x] `tinycast/models/student/`: scaler (robust instance norm + optional asinh), patch embedding with NaN mask channel, a small transformer encoder, a quantile head with monotone construction. Also an **MLP-only variant** (N-HiTS-like) as an architecture ablation.
- [x] Size presets s/m/l (≈0.3M/1M/3M); print the param count at startup.
- [x] `tinycast/distill/train.py`: plain PyTorch loop: AdamW, cosine schedule with warm-up, grad clipping, AMP on GPU, gradient accumulation, early stopping on **validation WQL**, best-checkpoint saving, W&B + `metrics.json` logging.
- [x] Loss: pinball vs ground truth (`L_gt`).
- [x] Overfit test: the student overfits 64 windows to ~0 loss (catches bugs cheaply; make it a slow test).
- [ ] Train s/m/l on ground truth only (same windows as the cache) → **"GT-only floor"**, 3 seeds for m.
- [x] Short ADR `0004-student-architecture.md`.

**Done when:** GT-only students are evaluated on the fresh split + shapes, with results in `results/`.

---

## Phase 4 — Single-teacher distillation (the core study)

### 4.1 Losses (`tinycast/distill/losses.py`) — each with unit tests
- [x] `L_kd_quantile`: per-level regression of student quantiles onto teacher quantiles, in **scaled space** (Huber), to test the "match the teacher's quantiles" view.
- [x] `L_kd_pinball_teacher` (also replaces the sample-pinball loss: Toto 2.0 has no samples): pinball of the student against the teacher's **quantile values treated as pseudo-observations**, weighted per level.
- [~] `L_kd_sample_pinball` — superseded: Toto 2.0 emits no samples; equivalent = mass-weighted `L_kd_pinball_teacher` (Toto samples): expected pinball under teacher samples. Its minimiser is exactly the teacher's quantile, a principled distribution-matching loss. **Explain why in the journal; good article material.**
- [ ] Horizon weighting `w(h)` (uniform / linear increasing / learned), after DistilTS's observation that short horizons dominate the loss.
- [x] Mixed objective `L = (1-α)·L_gt + α·L_kd`.

### 4.2 Experiments (each = a config in `configs/train/`; small sweeps first)
- [ ] E1: teacher ∈ {Toto, t0} × α ∈ {0, 0.5, 0.9, 1.0} for student m.
- [ ] E2: KD loss variant (4.1) at the best α.
- [ ] E3: horizon weighting on/off.
- [ ] E4: **data efficiency**: training windows ∈ {30k, 100k, 300k}: does KD help more when data is scarce?
- [ ] E5: student size s/m/l at the best recipe → points for the Pareto chart.
- [ ] Headline runs: 3 seeds, bootstrap CIs, on the fresh split + shapes + GIFT subset (flagged).
- [ ] Simple hyperparameter tuning (lr, weight decay, warm-up, batch size) on **validation only**, with a small random search budget (≤ 20 short runs); log everything.

### 4.3 Material for A2 (kit comes after Phase 5)
- [ ] Save to the journal and `results/figures/`: α sweep curves, loss-variant comparison, data-efficiency curves, fidelity vs accuracy scatter, first Pareto chart (WQL vs latency / params incl. Toto 4m/22m, Timestral 5M/28M).
- [ ] Honest verdict: does KD beat the GT-only floor? Where doesn't it?

**Done when:** a recommended single-teacher recipe exists with seeds + CIs and is compared fairly to the official small checkpoints.

---

## Phase 5 — Multi-teacher distillation & generalisation

- [ ] `weighting.py`: per-window teacher weights from each teacher's error on that window's ground truth (softmax(−τ·WQL)), with τ tuned on validation; compare to a uniform average and to the best single teacher. Optional: a tiny learned gating network on context features.
- [ ] Oracle analysis: how often is each teacher best, by shape/domain? (cheap, and explains the result)
- [ ] Generalisation study: train with/without the synthetic corpus share → effect on **shapes** (needed for Draw mode) vs **fresh electricity**. Pick the mix for the demo model.
- [ ] Optional, only if the budget allows: hidden-state alignment to a teacher's patch embeddings on a subset (linear projector + cosine loss). Report cost vs benefit.
- [ ] Final model selection (on validation), then a final evaluation on the fresh split once. Freeze the "demo student" (fp32) and record its card in `results/model_card.md`.

### Article kit A2 — "Distilling a time-series foundation model"
- [ ] `docs/articles/A2-distillation.md` kit combining Phases 3–5: the GT-only floor, the loss variants (with the sample-pinball derivation), α / horizon weighting, data efficiency, multi-teacher weighting + oracle analysis, and the final Pareto chart against the official tiny checkpoints.
- [ ] Angle: why classification-style KD doesn't transfer to probabilistic forecasting, and what does work, with negative results included.

**Done when:** a frozen demo student with a model card, a final leaderboard, and an oracle/weighting analysis.

---

## Phase 6 — Compression & browser parity

- [x] `export/onnx.py`: export student **including the scaler** (input: raw context + NaN mask, length 512, left-padded; output: raw-space quantiles). Dynamic batch axis.
- [x] Parity test (slow): torch vs onnxruntime on 1k stress windows; tolerance 1e-4 *relative to max(scale, |q−loc|)* (abs 1e-4 is impossible in float32 at 50,000 MW). Run by `scripts/export.py`.
- [x] `export/quantize.py`: int8 dynamic quantisation (and static with calibration if needed); report the accuracy delta on the fresh split and shapes.
- [ ] Browser latency: a tiny page in `web/` that times onnxruntime-web WASM (1 and 4 threads) and optionally WebGPU, on ≥ 2 machines; record results in `results/latency_browser.json`.
- [x] Model files ≤ 10 MB (target: ~1–4 MB fp32, < 1 MB int8 for size s/m).
- [ ] Copy artifacts to `web/public/models/` and push to the HF Hub with the model card.

### Article kit A3 — "Compressing a forecaster to run in the browser"
- [ ] `docs/articles/A3-compression.md` kit: export pitfalls met (and fixed), fp32 vs int8 accuracy deltas by shape, parity-test design, browser latency across machines/threads/backends, file sizes.

---

## Phase 7 — Demo website (`web/`)

Design: one consistent visual language (palette from `configs/style.yaml`), fast first load, works offline after load, mobile-friendly.

### 7.1 Draw mode (home page)
- [x] Canvas: the user draws freehand left→right; strokes are resampled to a uniform grid (up to 512 points; shorter drawings are left-padded with the NaN mask). Buttons: clear, undo, "add noise", "repeat my pattern" (tiles the drawing so seasonality emerges).
- [x] Live forecast (changed with owner 2026-11-13: the forecast starts at the pen tip and runs every frame, ~2 ms; hover = backtest from any point); draw the median line plus 50% / 80% / 98% fan bands.
- [x] HUD: in-browser latency (ms, rolling median), model size, int8/fp32 toggle, horizon slider (≤ 64).
- [x] Presets: load any gallery shape into the canvas.
- [ ] Optional "compare with teachers" toggle: calls the Space if configured. Otherwise show it only for presets, using precomputed forecasts.

### 7.2 Shapes gallery (e-means-style)
- [x] Grid: rows = shapes, columns = models (seasonal naive, ETS, Toto large, t0, Chronos-2, Timestral, Toto 4m/22m, **student fp32/int8**). Each cell is a small multiple with context, truth and fan, plus WQL and CPU latency under the plot.
- [ ] Column headers show params and median latency; best-in-row is highlighted; you can sort columns by accuracy or speed; clicking a cell opens it large in Draw mode.
- [x] Data from `web/public/data/gallery.json` (built by `make web-data`).

### 7.3 Results page
- [ ] Interactive Pareto charts (WQL vs latency, log x; WQL vs params), with model toggles and CI whiskers.
- [ ] Ablation charts from Phases 4–5; a "how it works" diagram (teacher → cache → student → ONNX → browser); links to articles, repo, HF model.

### 7.4 Ship
- [ ] GitHub Actions: build `web/` and deploy to Pages on push to `main`.
- [ ] Lighthouse check (performance ≥ 90), and a manual test on mobile.
- [ ] README: headline Pareto figure, GIF of Draw mode, demo link, article links, a reproducibility section (`make` commands).

### Article kit A4 — "Building the demo"
- [ ] `docs/articles/A4-demo.md` kit: design choices for Draw mode (resampling, throttling, fan rendering) and the gallery, what makes an ML demo convincing, a GIF of Draw mode, plus a short project retrospective (what worked, what didn't, what I'd do next).

---

## Phase 8 — Optional extensions
- [ ] Known-future covariates (temperature) for the electricity student; t0, Toto and Timestral all support covariates. Does distillation transfer covariate use?
- [ ] HF Space (`space/`): FastAPI `POST /forecast {series, horizon, models}` → quantiles for Toto/t0/Chronos-2 on CPU; CORS for the Pages domain; basic rate limiting; a cold-start banner in the UI.
- [ ] WebGPU backend; a "race" view (student vs teacher timing side by side).
- No article for Phase 8; fold any findings into A2–A4 if they're published later, or into the README.
- [ ] Tabular side-quest: distill TabPFN/TabICL into LightGBM on 2D toy datasets with a decision-boundary gallery (non-commercial licence: note it).

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Teacher APIs differ from expectations / gated access delays | Phase 1 contract tests; read sources first; start the t0 access request on day 1 |
| Caching exceeds the free GPU budget | Measure in 1.5; float16; fewer windows; smaller teacher checkpoint; regenerate synthetic contexts from seeds |
| KD doesn't beat the GT-only floor | Still a valid, publishable result; analyse when/why (data size, shape type); emphasise the fidelity + compression story |
| The official tiny checkpoints (Toto 4m, Timestral 5M) beat our student | Report it honestly; position the student by browser size/latency, domain specialisation, or multi-teacher gains |
| Drawn series are out of distribution (arbitrary scale, no time unit) | Robust scaler; synthetic + shapes share in training; "repeat my pattern" button; evaluate on shapes |
| Leakage via benchmarks the teachers saw | The fresh split is the headline; public benchmarks are flagged secondary |

## Interview showcase checklist (keep updated in README "What this demonstrates")
- Probabilistic loss design (pinball / WQL / sample-based distribution matching) and why classification-style KD doesn't transfer directly
- Rigorous evaluation: leakage control, fresh data, baselines, seeds + bootstrap CIs, ablations
- Budget-aware training on free GPUs: caching teachers, throughput measurement, small-run sweeps
- Model compression & deployment: ONNX, int8, onnxruntime-web, latency measurement
- Product thinking: an interactive demo that communicates a technical result
