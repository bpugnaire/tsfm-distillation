# Lab journal

## 2026-11-12 — Phases 0–2 (+ preview pipeline for 3, 6, 7)

**Did:** scaffolding (uv, Python 3.12, ruff, pytest, CI); metrics with hand-computed tests; 20 seeded shapes;
baselines; adapters for Toto 2.0 (4m/22m/313m), t0-gamma, Chronos-2 (+small) after reading their sources;
benchmark + latency harness; RTE éCO2mix download (13 series, 2012 → today); splits + leakage tests;
synthetic corpus; teacher cache; student + KD losses + training loop; ONNX/int8 export; demo site with lab notes.

**Result:**
- Shapes (200 series = 20 shapes × 10 seeds), relative WQL vs seasonal naive, cluster-bootstrap 95% CI
  (`results/shapes/summary.json`): Toto-313m 0.409 [0.376, 0.447], Toto-22m 0.443, t0-gamma 0.452, Chronos-2-small 0.460,
  Chronos-2 0.469, Toto-4m 0.484 [0.441, 0.534], ETS 1.161.
- Fresh FR power (312 windows / 13 series, targets ≥ 2026-10-15) (`results/fresh_fr_power/summary.json`):
  Toto-313m 0.556 [0.519, 0.592], Chronos-2 0.560, Toto-22m 0.573, Chronos-2-small 0.577, t0-gamma 0.599, Toto-4m 0.636, ETS 1.249.
- CPU latency, 1 thread, batch 1, p50 (M1 Pro): Toto-4m 10.7 ms, Toto-22m 31 ms, Chronos-2-small 20 ms, Chronos-2 61 ms,
  t0-gamma 102 ms, Toto-313m 275 ms.
- Teacher throughput, batch 64 (`results/throughput.json`): Toto-313m 54 w/s CPU, 179 w/s MPS; t0-gamma 83 / 206 w/s.
- PREVIEW student (m, 0.95M params, GT-only, 3000 steps, 170 s on MPS, 1 seed): fresh 0.589 [0.538, 0.641],
  shapes 0.526 [0.463, 0.599]. int8 ONNX: same WQL to 3 decimals, 3.9 MB → 1.27 MB (`results/export/preview_gt_m.json`).

**Surprise / learned:**
1. *Equal-weight quantile averages are biased CRPS estimators, with a grid-dependent sign*: 9 levels +9%, our 13 levels −17%
   (calibrated Gaussian). Training uses probability-mass level weights. (ADR 0002, `tests/test_metrics.py`)
2. *Toto 2.0 is `toto-2`, not `toto-ts`*; it emits 9 fixed quantiles and no samples → the planned sample-pinball loss is
   replaced by pinball against mass-weighted teacher quantiles (same minimiser; unit-tested).
3. *Chronos-2 silently clamps out-of-range levels and its raw quantiles cross* (0.4–1.6% of cells). Sorted + recorded.
4. *Rankings change between suites*: Chronos-2 is 5th on shapes, 2nd on fresh data; t0-gamma 3rd → 5th.
5. *Shapes where TSFMs lose to seasonal naive*: level shift (t0-gamma 2.35×, Chronos-2-small 1.17×), sawtooth and square wave
   (Toto-4m 1.30× / 1.46×). Impulse-decay has inverted scaling (Toto-4m best). No model wins everywhere.
6. *All TSFMs under-cover on fresh data* (80% intervals contain 66–71%).
7. *Window-level bootstrap CIs were too narrow on the fresh split* (overlapping targets, 13 series) → cluster bootstrap.
8. *ONNX*: no `asinh` symbolic in the TorchScript exporter (written from basic ops). Abs-diff parity is impossible in float32
   at 50,000 MW (relative tolerance). int8: keeping the patch embedding fp32 matters (p99 5.8% → 2.8%,
   `results/export/int8_sensitivity_preview_gt_m.json`); "keep the head fp32" did nothing because the dynamic quantiser never touches `Gemm`.
   Parity error ≠ accuracy loss: WQL unchanged.
9. The preview student "beating" Toto-4m on fresh data is **not** a valid claim: 1 seed, overlapping CIs, and it is in-domain
   (trained on these series' past) while the teachers are zero-shot.
10. Gotchas: lightning needs `setuptools<81`; t0 needs `jaxtyping>=0.3`; npm skipped rolldown's native binding;
    `ruff format` also formats Python blocks inside Markdown (excluded `*.md`).

**Next (owner decisions, nothing launched):** experiment design for Phase 3–4: fair comparison protocol (in-domain vs zero-shot),
corpus size, teacher choice per level/window, seeds/power. Teacher caching ≈ 25–28 min per teacher per 300k windows on MPS (needs approval).

## 2026-11-13 — Correction: seasonal-naive baseline bug; Timestral; data for distillation

**Did:** enabled Timestral as a comparator (owner: personal project, licence clause does not apply; vendor post-processing off, same
policy as t0). Measured what teachers output on different input families (owner question: "can we distil on pure random noise?").
Rebuilt Draw mode (forecast origin = pen tip, hover = backtest), added Status and Learn pages.

**CORRECTION (affects every relative score of 2026-11-12):** `SeasonalNaive` stepped back one season too many for horizon steps
beyond the period (step 25 with m=24 copied y[T-48] instead of y[T-24]). It was a valid but *weaker* baseline, so every model looked
better relative to it. Found when the Learn page extracted the function's source. Fixed + hand-computed test
(`tests/test_metrics.py::test_seasonal_naive_uses_most_recent_season`), baseline re-run, summaries recomputed. Rankings unchanged.

**Result (corrected, relative WQL, cluster-bootstrap 95% CI):**
- Fresh FR power (`results/fresh_fr_power/summary.json`): Timestral-209M 0.616 [0.574, 0.660], Timestral-28M 0.617, Toto-313m 0.621,
  Chronos-2 0.626, Toto-22m 0.641, Timestral-5M 0.643, Chronos-2-small 0.645, preview student 0.658, t0-gamma 0.670, Toto-4m 0.711, ETS 1.396.
- Shapes (`results/shapes/summary.json`): Timestral-209M 0.402 [0.366, 0.443], Toto-313m 0.439, Timestral-28M 0.447, Toto-22m 0.475,
  t0-gamma 0.485, Timestral-5M 0.492, Chronos-2-small 0.494, Chronos-2 0.504, Toto-4m 0.519, preview student 0.564, ETS 1.246.
- Input families (`results/input_families.json`, 200 contexts each; Toto-22m / t0-gamma): median movement (std of the median forecast
  / 80% band width) white noise 0.001 / 0.004, random walk 0.03 / 0.06, KernelSynth 0.27 / 0.35, random shapes 0.63 / 0.64,
  real electricity 0.66 / 0.78. Diversity across contexts: 0.006 / 0.024 (white noise) vs 1.04 / 1.07 (real).

**Surprise / learned:**
1. *Showing code to a learner is a code review*: extracting `SeasonalNaive._one` for the Learn page exposed the bug. Also: a
   `str.replace` patch that matches nothing fails silently; patch scripts must assert.
2. *On white noise the teachers are correct and useless for distillation*: flat median, constant calibrated band (3.28 MAD
   ≈ 2.56σ/0.8σ), the same answer for every context. Structured random noise (KernelSynth) carries ~70% of the real-data signal.
3. *On fresh data the top four models are tied* (0.616–0.626, overlapping CIs): Timestral-28M = Timestral-209M.
4. Timestral-5M fails on sawtooth (2.17× seasonal naive) and level shift (1.22×) while the 209M is the best model overall.

**Next:** owner decisions in `docs/status.yaml`; the input-family result suggests an experiment: same student, teacher labels on
{real, KernelSynth, shapes, white noise} contexts, evaluated on fresh + shapes.

## 2026-11-13 — Teacher rollouts, data sources, study design

**Did:** prototype of owner's idea "start the teacher on real data and loop back to generate long series"
(`scripts/rollout_study.py`, 24 series × 2 sources × 2 teachers × 3 path strategies × 16 blocks of 64, 24 s on CPU).
Surveyed open multi-domain sources. Wrote ADR 0005 (proposed study design).

**Result (`results/rollouts.json`, median over series; 1 = like the real context):**
- median feedback: spread 0.90–0.96 at block 1 → 0.36–0.83 at block 16; roughness down to 0.17 (shapes): series get smooth and flat.
- independent per-step sampling: roughness ×1.9–2.0 from block 1, ×2.5–7 at block 16: jagged, self-amplifying.
- correlated (Gaussian copula, ρ = 0.9): block 1 spread 1.00–1.01, roughness 0.92–1.09 (realistic); block 16 spread ×3.6–9.5, drift 0.9–3.4 std.
- Data sources (HF, checked 2026-11-13): Monash (CC BY 4.0), LOTSA (174 configs, Apache-2.0 declared, per-dataset licences
  underneath), GiftEvalPretrain (Apache-2.0), BOOM (Apache-2.0), chronos_datasets (67 subsets, licence "other": per subset).

**Surprise / learned:** the "path" choice decides whether rollouts collapse or explode; neither median feedback (what TSFMs do
internally for long horizons) nor independent sampling produces realistic long series. Short correlated rollouts are a plausible generator.
Public corpora are likely in the teachers' pretraining: fine for *training* the student, not for headline *evaluation*.

**Next:** owner approval of ADR 0005 Stage A.
