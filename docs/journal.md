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
