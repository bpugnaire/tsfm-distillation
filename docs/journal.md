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

## 2026-11-13 — Stage A (which inputs should the teacher label?) + cache overflow bug

**Did:** Stage A of the study (`configs/study/stage_a.yaml`): teacher Timestral-209M, student s (0.22M), 30k windows/arm, 3000 steps,
1 seed. Arms: KD on real multi-domain / synthetic / teacher rollouts / white noise; ground truth (GT) on multi-domain / synthetic.
Zero-shot zoo benchmarked on the held-out multi-domain test.

**BUG found and fixed:** teacher caches stored raw float16 → overflow above 65504: 2.9% (multi-domain) and 4.9% (rollout)
of labels were inf (integrity check only tested NaN; pinball gradients stay finite so training did not crash).
Fix: per-window normalised float16 + float32 loc/scale (`cache.py`, ENCODING scaled_f16_v2), finite check, regression test.
The two affected arms were re-run; invalid runs kept as `runs/INVALID_fp16overflow_*`.

**Result (`results/study/stage_a/table.md`; paired ratio a/b, cluster bootstrap 95% CI; <1 = a better):**
- Teacher labels vs ground truth, **real** inputs: 0.83 [0.80, 0.87] shapes, 0.86 [0.83, 0.88] fresh electricity, 0.97 [0.93, 1.01] multi-domain.
- Teacher labels vs ground truth, **synthetic** inputs: 1.12 [1.10, 1.15] shapes (GT better), 1.00 elsewhere (no difference).
- Synthetic vs real inputs (KD): 0.43 shapes, 1.23 multi-domain test, 1.00 fresh electricity → each wins on its own distribution.
- Rollouts vs real inputs (KD): 1.06 shapes, 1.09 multi-domain, 0.98 [0.95, 1.01] fresh → no gain at this scale.
- White noise vs synthetic (KD): 5.7× shapes, 1.46× multi-domain, 1.48× fresh → worst everywhere (as predicted).
- Absolute level: best student on fresh electricity ≈ seasonal naive (1.02–1.04) vs teachers zero-shot 0.61–0.71.
  On the multi-domain test: best student 0.734 vs Toto-4m 0.721, Timestral-5M 0.660, t0-gamma 0.511 (best there).

**Surprise / learned:**
1. Distillation helps most when inputs are **real**: the teacher's distribution is a cleaner target than one noisy observed future.
   On synthetic inputs the generator's own future is already a good target, so teacher labels add nothing (or hurt on shapes).
2. No single input source wins: each is best on its own distribution → mixing sources is the obvious next step.
3. t0-gamma is the best zero-shot model on the multi-domain test (0.511) but among the worst on fresh electricity (0.668).
4. A 0.22M student at 30k windows does not transfer zero-shot to electricity (≈ naive). Scale (data/size) and fine-tuning are open.

**Next:** propose Stage D pilot (fine-tuning, headline) and Stage C (scale) before B; owner approval needed.

## 2026-11-13 — Stage D pilot (light fine-tuning on "your data") + a 2 h stall

**Did:** fine-tuned the Stage-A student selected on validation (kd_multidomain, 0.22M) on 1 week / 1 month / 1 year of a target
domain's own history (ground truth), vs the same architecture from scratch; domains: French electricity (fresh test) and pedestrian
counts (held out of distillation, public). 1000 steps, lr 3e-4 (fine-tune) / 1e-3 (scratch), 1 seed (`results/study/stage_d_pilot/table.md`).

**Result (relative WQL; paired ratios a/b on the same windows):**
- Fine-tuning works **with enough history**: 1 year → electricity 0.748 (zero-shot 1.035; ratio 0.72 [0.68, 0.76]),
  pedestrians 0.740 (zero-shot 0.948; 0.78 [0.76, 0.81]).
- Distilled init vs scratch: **1 week 0.60 [0.54, 0.65] (electricity), 0.63 [0.60, 0.66] (pedestrians)**: pretraining matters most
  when data is scarce. 1 month: tie. 1 year: scratch slightly better on electricity (1.05 [1.03, 1.07]), distilled on pedestrians (0.98).
- **Short-history fine-tuning hurts** vs not fine-tuning: 1 week 1.27 / 1.28, 1 month (electricity) 1.07, with collapsed
  calibration (80% intervals cover 32–36%).
- vs big models zero-shot (headline): **not reached yet**. Electricity, 1 year: 1.06× Toto-2.0-4m, 1.17× Timestral-5M, 1.22× Timestral-209M.
  Pedestrians, 1 year: 1.14× Toto-4m, 1.61× Timestral-209M. A 0.22M model trained *from scratch* on 1 year of electricity (0.713)
  ≈ Toto-4m zero-shot (0.709).

**Surprise / learned:**
1. The short-history failures look like a **design flaw of the pilot**, not of fine-tuning: (a) contexts were cut to the history
   window (≈ 60 observed steps for 1 week) while the test uses 512-step contexts (train/test mismatch); (b) 1000 steps on ~750
   windows ≈ 170 epochs, and the early-stopping set came from the same week (too similar to stop overfitting). To fix before the final D.
2. With a year of data, a tiny local model is already strong: the bar for "distillation helps" is the scratch model, not only the teachers.
3. Stall: the Mac slept ~2h20 during C-lite caching (shards 20→30, 10:41→13:03). Queue now runs under `caffeinate -i -s`.

**Next:** final Stage D after C-lite/B: start from the C-lite winner (m), fix contexts (full past allowed, supervision limited to
the history period) and scale steps with data; 3 seeds.

## 2026-11-13 — Stage C-lite (scale): a 0.95M student beats Toto-2.0-4m on two suites

**Did:** teacher Timestral-209M; 300k windows (150k synthetic + 150k real multi-domain); student m (0.95M) 15k steps; contrasts:
KD vs GT, m vs s (0.22M), 300k vs 30k windows (same number of passes), vs Stage-A best. 1 seed (`results/study/stage_c_lite/table.md`).

**Result (relative WQL, zero-shot; paired ratios, 95% cluster-bootstrap CI):**
| | shapes | multi-domain (flagged) | fresh electricity |
|---|---|---|---|
| kd_mix_m_300k | 0.460 | 0.703 | 0.795 |
| vs Toto-2.0-4m zero-shot | **0.89 [0.85, 0.93]** | 0.98 [0.95, 1.01] (tie) | 1.12 [1.08, 1.17] |
| vs Timestral-5M | **0.94 [0.89, 0.98]** | 1.07 | 1.24 |
| vs its teacher Timestral-209M | 1.14 | 1.12 | 1.29 |
| teacher labels vs ground truth (same data) | **0.85** | **0.96** | **0.88** |
| 300k vs 30k windows | **0.74** | **0.91** | **0.87** |
| 0.95M vs 0.22M student | **0.92** | **0.97** | **0.93** |

**Surprise / learned:**
1. Scale fixed most of Stage A's gap: on electricity the student went from 1.035 (≈ naive) to 0.795, zero-shot, never having seen
   electricity. Data (×10) was the biggest lever, then teacher labels, then size.
2. A 0.95M student (4.3× smaller than Toto-2.0-4m, 5.5× smaller than Timestral-5M) beats both on shapes and ties Toto-4m on the
   multi-domain test. It still trails them on electricity (12–24%) and its teacher everywhere (12–29%).
3. Distillation's advantage over ground truth *persists at scale* (0.85–0.96, all significant): not a small-data artefact.

**Next:** Stage B (teachers × α, s-size, running) and final Stage D (fine-tuning; its init is chosen on validation and will very
likely be this run: val WQL 0.1377, the best so far).

## 2026-11-13 — Contamination check against GIFT-Eval and fev

**Did:** content-level check (`tinycast/eval/contamination.py`, `scripts/contamination_check.py`): 18.2M chunk hashes of every
training / fine-tuning series vs every GIFT-Eval and fev series. ADR 0006.

**Result (`results/contamination/`):** overlap in GIFT-Eval covid_deaths / kdd_cup_2018 / saugeenday (100%), temperature_rain (3.4%),
m4_hourly (9.2%, from our pedestrian counts), m4_monthly (0.08%); fev fred_md_2025 (53%). Zero overlap: 50/56 GIFT-Eval, 99/100 fev datasets.

**Surprise / learned:** (1) first version flagged car_parts at 23% and solar at 3%: false positives from low-information chunks
(a spike on a flat line looks identical in any series once z-normalised) → require ≥ 8 distinct values (+ regression test).
(2) M4-hourly contains pedestrian-count series: a dataset we treated as a clean "held-out domain" is partly inside a famous benchmark.
(3) 4 of our 11 training datasets are GIFT-Eval *test* sets: name-level checks would have caught those, but not (2).

## 2026-11-13 — Stage B interrupted: Toto on Apple MPS + constant context

Stage B stopped after 9/33 runs (gt ×3, kd_timestral ×3, kd_timestral_a05 ×3 done) while caching Toto-2.0-313m: the output check
refused non-finite quantiles. Cause: window 564 of the multi-domain corpus is a perfectly constant series; **Toto-313m on Apple
MPS returns non-finite values for it, on CPU it is fine.** I had moved teacher caching to MPS for speed: reverted (teachers always
run on CPU, as in the benchmarks), partial MPS caches deleted, regression test added. The final Stage D started meanwhile and selects
its init among C-lite + the 9 finished B runs (s-size, val ≥ 0.144 vs C-lite 0.1377): to re-check once B is complete.
Lesson: a speed-up that changes *where* a teacher runs changes the teacher; the "never silently fix outputs" check caught it.

## 2026-11-13 — Final Stage D (fine-tuning on "your data") and Stage B (teachers × α, 3 seeds)

**Stage D final** (`results/study/stage_d_final/table.md`): init = C-lite student (0.95M, selected on validation, val WQL 0.1377);
warm = full past as context, fine-tuning targets limited to the last week/month/year (3 seeds); cold = only that period exists
(contexts cut at train AND test; zero-shot models scored on the same cut contexts). Steps = 20 epochs, cap 1500.
| warm, relative WQL (seed range) | electricity | vs Toto-4m | vs Timestral-5M | vs Timestral-209M | pedestrians | vs Toto-4m | vs Timestral-5M |
|---|---|---|---|---|---|---|---|
| zero-shot student | 0.795 | | | | 0.759 | | |
| 1 week | 0.788 (0.78–0.81) | 1.12 | 1.24 | 1.29 | 0.659 | 1.02 (tie) | 1.22 |
| 1 month | 0.907 (0.88–0.93) | 1.28 | 1.42 | 1.48 | 0.619 | 0.96 | 1.15 |
| 1 year | **0.686 (0.66–0.70)** | **0.98 (tie)** | 1.09 | 1.13 | **0.591** | **0.91** | 1.10 |
- Distilled init vs scratch: decisive with little data (electricity 1 week 0.54, cold 1 week 0.68; pedestrians cold 1 week 0.78),
  ~equal with a year (electricity 1.04, pedestrians 0.95).
- Cold start: electricity 1 week 0.966 (1.18× Toto-4m on the same 1-week contexts), pedestrians 1 week 0.816 (ties Toto-4m).
  The large teachers barely suffer from 1-week contexts (Timestral-209M 0.615 vs 0.614); Toto-4m does (0.816 vs 0.709).
- **Anomaly:** electricity, 1 *month* of fine-tuning is worse than none (1.14× zero-shot) while 1 week and 1 year help or are
  neutral; scratch is worse too. Hypothesis (untested): the last month before the test (mid-Sep → mid-Oct) is a seasonal regime
  that differs from the test (mid-Oct → Nov), and real-time RTE data are provisional. To investigate before any claim.

**Stage B** (`results/study/stage_b/table.md`; student s, 30k real multi-domain windows, 3 seeds; paired on seed means):
- Teacher labels vs ground truth, confirmed with 3 seeds: 0.91 shapes, 0.97 multi-domain, 0.88 electricity.
- Teacher choice matters little: Toto-313m vs Timestral-209M 0.95 / 1.02 / 0.98; t0-gamma vs Timestral 1.03 / 1.00 / **1.07**
  (t0 is the best zero-shot model on the multi-domain test but the worst teacher for electricity).
- Ensemble of 3 vs Timestral alone: 0.98 / **0.98** / **0.98** (small, significant on two suites). Per-window error weighting vs plain
  ensemble: no gain (1.00 / 0.995 / 1.017), although t0 was the per-window best on 55% of training windows.
- α = 0.5 vs 1: never better on shapes or electricity (1.02–1.10); pure distillation (α = 1) wins, even for Toto whose tails are
  not native (only +1.3% for α = 0.5 on the multi-domain test).
- Seed-to-seed spread: ±0.002–0.013 on the multi-domain test, ±0.02–0.10 on shapes / electricity: 1-seed differences < 5% there are noise.

**What moved the needle (all stages):** data ×10 (9–26%) > teacher labels vs truth (3–15%) > student ×4 (3–8%) ≈ fine-tuning with a
year of your data (12–22%) > ensemble of teachers (~2%) ≈ which teacher (2–7%) > α (1 is best) > per-window weighting (none).

**Headline verdict so far:** with a year of its own history, the 0.95M student **matches Toto-2.0-4m on electricity and beats it by 9% on
pedestrians**, i.e. it matches the out-of-the-box accuracy of the *smallest* official checkpoint at 4.3× fewer parameters. It does **not**
yet reach Timestral-5M (9–10% behind) or the 200–300M models (13–29% behind).

## 2026-11-13 — Diagnostic: why does 1 month of electricity fine-tuning hurt?

`scripts/investigate_fr_1m.py` → `results/study/diagnostics/fr_1m.json` (1 seed; ratio vs the zero-shot student, <1 = helps).
- H1 data problem (consolidated → real-time switch on 2026-09-01): **rejected**. No level discontinuity, no gaps, and 24-h profiles
  are nearly identical across datasets/years (the "peak hour" flip was two near-equal peaks, midday 1.10 vs evening 1.11).
- H2 overfitting: **partly**. Last month: 1.05 (25 steps), 1.09 (100), 1.08 (300), 1.17 (1000): harm grows with steps but is there at 25.
- H3 seasonal regime: **rejected**. Same calendar month one year earlier (Sep–Oct 2025): 0.91 (100 steps) / 0.99 (1000), it *helps*;
  the test-season month one year earlier (Oct–Nov 2025): 1.05 / 1.20, it hurts.
- Conclusion: one month of supervision is a high-variance, idiosyncratic sample; whether it helps depends on the month, and more
  steps amplify it. A year averages it out. Prediction for the next experiment: teacher labels on the same month should remove most
  of the harm (they replace the month's single noisy realisations).

## 2026-11-13 — Stage E: fine-tune on your history with TEACHER labels

**Did:** same as final Stage D warm (C-lite init, 1 week / 1 month / 1 year, 3 seeds), but the fine-tuning target is Timestral-209M's
forecast of each history window (kd, alpha = 1) or a 50/50 mix with the real future (mix, alpha = 0.5). The teacher labels the history
once (73k windows, ~10 min CPU). References: Stage-D fine-tunes on the real future (gt), same seeds (`results/study/stage_e/table.md`).

**Result (paired, seed-averaged):**
- Prediction confirmed: electricity, 1 month: kd vs gt **0.79 [0.76, 0.81]**; vs no fine-tune 0.90 (gt was 1.14: it hurt).
- **Data efficiency: teacher labels on 1 month ≈ real future on 1 year** (electricity 1.03 [1.00, 1.06], pedestrians 0.995 [0.98, 1.01]).
- kd vs gt: pedestrians 0.91 (1 w) / 0.95 (1 m) / 0.99 (1 y); electricity 0.99 / 0.79 / 0.97: largest gain where supervision is short or noisy.
- Target chosen on the fine-tuning VALIDATION set (never test): electricity mix (1 w), kd (1 m), kd (1 y); pedestrians mix (all).
- **Headline (validation-selected, electricity, 1 year, kd, 0.677): vs Toto-2.0-4m 0.96 [0.93, 0.98] (better), vs Timestral-5M
  1.06 [1.02, 1.10], vs Timestral-209M (the teacher) 1.10 [1.07, 1.14].** Pedestrians (mix, 1 y, 0.582; public data, flagged):
  vs Toto-4m 0.90, vs Timestral-5M 1.08, vs Timestral-209M 1.27.
- Seed spread: ≤ 0.027 (electricity), ≤ 0.002 (pedestrians).

**Surprise / learned:**
1. The best-looking number (mix, 1 year: ties Timestral-5M, 1.02 [1.00, 1.05]) is the best of 9 combinations *read on test*; validation
   picks kd (1.06). Choosing on test would have inflated the headline by ~4%. Recorded as a cautionary example; not claimed.
2. Teacher labels act as a regulariser on your own data: the 1-month idiosyncrasy that made raw fine-tuning hurt disappears.
3. The contrasts mix-vs-zoo and "kd 1 month vs gt 1 year" were added after a first read of the results (descriptive; no selection).

## 2026-11-13 — Owner reframing: student+FT vs teacher+FT; robustness from existing runs

Owner: fine-tune on real data (not teacher outputs); target claim "distil then fine-tune is more efficient and robust than fine-tuning
the SOTA teacher" (ADR 0007). Checked: only Chronos-2 ships fine-tuning (`fit`, full / LoRA); Toto 2.0, t0, Timestral are inference-only.
Robustness from existing Stage-D runs (`scripts/robustness_check.py`): 3-seed ensemble = free 5% gain on electricity 1 y (0.686 → 0.659),
≤ 0.7% elsewhere; forgetting is strong (3.5× worse on shapes, 1.2–1.5× on other domains after a 1-year fine-tune).

## 2026-11-13 — Chronos-2 fine-tuning cost on a laptop
Vendor `fit()` defaults (batch 256, 1000 steps): Chronos-2-small ~18 min/run on MPS; Chronos-2 (120M) does not fit in MPS memory
for full fine-tuning and needs ~8–12 h per run on CPU (LoRA is not faster here). Our student fine-tunes in ~1 min. (`docs/notes/compute.md`)

## 2026-11-15 — Website refresh: distilled student on Draw, new Play page, findings-first Study
**Did:** Draw now runs the Stage C-lite distilled student (0.95M, `kd_mix_m_300k`) instead of the ground-truth preview, with a toggle to
overlay the same network trained without the teacher (`gt_mix_m_300k`) and the teacher's precomputed forecast. New Play page: the
visitor draws the next 64 steps of fresh electricity windows or shapes, then sees the truth, the student (live) and Timestral-209M.
Results: suite tabs (fresh electricity / multi-domain / shapes), students in every leaderboard, decluttered Pareto, a fine-tuning
curve (Stage D, 3 seeds). Study: 7 key-finding cards (numbers read from the study tables), stages collapsed. Stale "no distillation
results yet" banners removed. Study page height 38.8k px -> 2.2k px; no horizontal scroll at 390 px.
**Result:** all numbers via `scripts/build_web_data.py` from `results/` (HIGHLIGHTS list, `finetune_curves`, `play`).
**Surprise / learned:** the "one week of data halves the error" framing was wrong: the 0.54 ratio is distilled vs *scratch*; the fine-tuned
student only matches its zero-shot score. Card reworded. The distilled ONNX files are git-ignored (`web/public/models/clite_*`) until
the owner decides whether a Timestral-distilled student may be published (data/LICENSES.md note).
**Next:** swap in the Stage C2 winner if it beats C-lite on validation; owner decision on publishing the student.

## 2026-11-16 — Stage C2 results, Chronos-Bolt comparators, lighter site
**Did:** Stage C2 finished (student m, Timestral-28M labels). Added Chronos-Bolt tiny (8.7M) and mini (21M) as zero-shot comparators
(`ChronosBoltAdapter`: native 0.1–0.9 levels only, IQF tails, crossing rate recorded). Site: Draw now runs the C2 1M-window student,
Results opens on a "models of its size" chart (everything ≤ 30M params), Study shows findings only, licence and model-detail text removed.
**Result:** `results/study/stage_c2/table.md`. `kd_v2_1m` (1M windows of data v2): shapes 0.458, multi-domain 0.697, fresh electricity
0.675 (C-lite: 0.460 / 0.703 / 0.795). Validation (same 2000 windows for every run): 0.1369 vs C-lite 0.1377, so it replaces C-lite on
the site by validation, not by test. vs same-size zero-shot models (`results/{shapes,fresh_fr_power,multidomain_test}/summary.json`):
best on shapes (Toto-4m 0.519, Timestral-5M 0.492, Bolt-tiny 0.686, Bolt-mini 0.651); electricity behind Timestral-5M 0.641 and
Bolt-mini 0.651, tied with Bolt-tiny 0.674, ahead of Toto-4m 0.709; multi-domain behind Bolt (0.651 / 0.637, likely trained on these
series) and Timestral-5M 0.660, ahead of Toto-4m 0.721.
**Surprise / learned:** data v2 at 300k was *worse* than the old data on multi-domain (1.05) and only helped on electricity (0.94): the
gain comes from volume (1M vs 300k: 0.94 / 0.94 / 0.89). Timestral-28M labels cost 7% on shapes vs 209M and nothing elsewhere.
Electricity is no longer domain-zero-shot for C2 (data v2 has other energy series; the RTE series stay out).
**Next:** TTM-r2 (~1M params, the closest-size comparator) needs `granite-tsfm`, which pins torch 2.11 / pandas 3: run it in a separate
env and score its saved forecasts here. Fine-tuning curve on the site is still the C-lite student: rerun Stage D with C2.

## 2026-11-16 — Long horizons: rolling the 64-step student out vs native long decoding
**Did:** `tinycast/models/rollout.py` (median feedback, or 100 sampled paths with an AR(1) Gaussian copula, rho 0.9) and
`scripts/long_horizon.py`: C2 student vs Timestral-28M / 5M (one block up to 1024 steps), Toto-2.0-4m and Chronos-Bolt-tiny
(their own decoding), horizons 64/128/256/512, same 512-step context for all. Read Timestral's source to see how it does long
horizons (patch 32, 32 forecast-query tokens = 1024 steps per block, context up to 32,768, median feedback past 1024).
**Result:** `results/long_horizon/table.md` (paired ratios on the same windows). Sampled paths beat median feedback by 3–8% and
fix calibration (cov80 at 512: 0.72 vs 0.30 on shapes, 0.78 vs 0.36 multi-domain). On real data the gap to the teacher stays
flat: electricity 1.10 -> 0.99, multi-domain 1.09 -> 1.12 from 64 to 512 steps. On synthetic shapes it opens at once: 1.02 at 64,
1.35 at 128, 1.47 at 512, while the teacher's one-shot score stays at 0.45. Electricity at 512 h: no model beats daily seasonal
naive (8 origins, 104 windows, every model under-covers: cov80 0.37–0.50).
**Surprise / learned:** rollout is good enough on seasonal real data; it fails on shapes where the next block depends on a
long-range trend or phase the student must re-infer from its own sampled continuation. That is the case for multi-token decoding.
One hour (2026-10-25 00:00 UTC, the DST night) is missing in every RTE series: long fresh windows now accept <= 1% missing targets
(`min_finite`, headline suite unchanged at 312 windows). Data processing should be fixed at the source.
**Next:** owner decision on multi-token decoding (ADR) and on GIFT-Eval / other teachers.

## 2026-11-16 — GIFT-Eval short-term (official harness) and a t0 labelling bug
**Did:** official GIFT-Eval code (commit 0b57dd6) in `.venv-gift`, our C2 student as an ONNX GluonTS predictor
(`scripts/gift_eval_student.py`), 55 short-term configs; comparison with every leaderboard model on the same 55 configs
(`scripts/gift_eval_table.py`, geometric mean of MASE / CRPS ratios to Seasonal Naive, the leaderboard rule).
**Result:** `results/gift_eval/table.md`. Student CRPS 0.616, MASE 0.822, rank 141/171. Same configs: Toto-2.0-4m 0.527 / 0.714,
Timestral-5M 0.494 / 0.673, Timestral-28M (teacher) 0.478 / 0.652, TTM-R2 0.939 / 1.005, Chronos-Bolt-small 0.543.
Largest losses vs Toto-4m: bizitobs 10 s (x2.7–3.3), covid_deaths (x2.1), jena_weather 10 min (x2.0), us_births daily (x1.9).
**Surprise / learned:** context length dominates: leaderboard runs gave Toto-2.0 4096 steps and Timestral up to 32,768; our
student reads 512 (hours to days at 10 s / 10 min). 489 series have no history at all (forecast 0, counted) and 1,387 have
their last 512 steps missing (we read the last 512 observed values). Caveats: the C2 student's training data is filtered per
series against GIFT-Eval, but its teacher (Timestral-28M) may have seen test data; checkpoint selection used the old
multi-domain validation split, which includes earlier windows of 4 GIFT datasets (selection only, not training).
t0 bug: `t0.quantile.sanitize_predictions` replaces NaN/Inf with 0.0 and logs only; 2 of 360k Stage C3 labels were all-zero
(bitcoin series at ~1e18 overflow float32 statistics). Adapter now raises on all-zero outputs and rescales rows above 1e12 by
a power of ten (exact: 2e-7 at 1e15); the 2 windows recomputed (`scripts/repair_cache.py`). Stage B caches and all
benchmark results checked: no all-zero t0 rows.
**Next:** same-context comparison (Toto-4m / Timestral-5M at 512 on GIFT) to separate context from distillation quality.

## 2026-11-16 — Stage C3: other teachers on the C2 recipe (+ a memory crash)
**Did:** same 1M windows and student as C2, labels from Toto-2.0-313m, t0-gamma, and the mean of three teachers (+ Timestral-28M).
Scored on our 3 suites (`results/study/stage_c3/table.md`) and GIFT-Eval short-term, 55 configs (`results/gift_eval/table.md`).
**Result:** vs the C2 student (Timestral-28M labels), paired ratios: Toto 0.99 shapes (ns) / 1.03 multi-domain / 0.97 electricity;
t0 1.12 / 1.00 (ns) / 1.05; ensemble 1.00 (ns) / 0.99 (ns) / 1.03. GIFT-Eval CRPS (ratio to Seasonal Naive): C2 0.616, Toto
0.627, ensemble 0.627, t0 0.640; per-config paired vs C2: Toto 1.019 [1.002, 1.039], ensemble 1.018 [1.004, 1.035], t0 1.040
[1.009, 1.074]. Validation ranked the ensemble first (0.1342 vs 0.1369 for C2): validation did not predict test here.
**Surprise / learned:** at 1M windows the teacher barely matters (all within 4%), and the Stage B ensemble gain (+2% at 30k
windows) disappears. The Apache-only student (Toto labels) is within 2% of the Timestral-taught one on GIFT-Eval and better on
electricity: a publishable option. The first ensemble attempt loaded 3 full caches and stacked them (> 20 GB): the Mac froze and
rebooted on a watchdog panic. Ensemble caches are now averaged shard by shard on disk (530 MB peak).
**Next:** owner decision: publish the Toto student (Apache teachers) or keep C2 on the site; same-context GIFT comparison.

## 2026-11-16 — GIFT-Eval gap to Toto-2.0-4m, decomposed
**Did:** ran Toto-2.0-4m on the same 55 short-term configs with the leaderboard notebook's code (`scripts/gift_eval_toto.py`,
`.venv-gift-toto`; gluonts 0.17 needed an axis swap for multivariate forecasts). Context 4096 on 4 configs reproduces the
leaderboard to 4 decimals (`results/gift_eval/repro_toto4m_ctx4096`), then context 512 multivariate and 512 univariate.
**Result:** CRPS (ratio to Seasonal Naive): Toto 4096 multivariate 0.527, 512 multivariate 0.563, 512 univariate 0.563; our C2
student (512, univariate) 0.616. Paired student / Toto@512-univariate: 1.093 [1.047, 1.146], student better on 16/55 configs.
Teacher labelling speed at 2048 vs 512 context: Timestral-28M CPU 116 vs 372 w/s, Toto-313m guarded MPS 57 vs 138 w/s.
**Surprise / learned:** reading variates jointly gives Toto nothing here; context length is worth ~7% for Toto; 9% remains at equal
context and equal univariate setup (size, data coverage: worst on covid_deaths, us_births/D, 10 s and 10 min data).
**Next:** pilot of a 2048-context student (patch 64, same 32 tokens) on the C2 300k windows, proposed to the owner.

## 2026-11-20 — Explainer posts: training data, and the student traced layer by layer
**Did:** two new site posts (`web/post-data.html`, `web/post-student.html`) fed by `scripts/build_explainers.py` →
`web/public/data/explain.json`. The script rebuilds the published student's training corpus with the training code and checks it
window id by window id against `data/processed/corpus_train_48dd0113d4.npz`. It reads example labels from the Toto cache
shards, and runs the C3 Toto checkpoint layer by layer on one fresh France window, checked against `Student.forward`.
**Result:** corpus = 500,000 real windows (18,114 series, 41 datasets, 302 of 18,416 candidate series dropped by the
contamination filter) + 500,000 synthetic (200,140 Gaussian-process, 200,201 shapes, 99,659 mixes). Labels: 245 shards,
1.70 GB. Parameters: 946,368, of which feed-forward 56%, attention 28%, head 11% (the head and the 33-row position table
fix the output and input range). Scale check: input × 1000 + 50,000 moves the forecast identically to within 0.002% of the
scale. The trace reproduces `Student.forward` exactly.
**Surprise / learned:** the data v2 sampler only uses part of each series' stride-64 grid (80 of 190 slots on the example
traffic sensor), because of the per-dataset quota. Energy + traffic make up 64% of the real windows: the quota is per dataset,
and those two domains have the most datasets.
**Next:** owner review of the two pages in a browser; C7 (2.9M) result to fill the "bigger still" row (TBD).

## 2026-11-20 — Stage C6 (2048-step student, 1M windows) and C7 (2.9M student, 28M vs 209M labels)
**Did:** C6: same 1M windows as C2, Timestral-28M labels at 2048 steps, patch 64 x 32 tokens, truncation p 0.5, full 50k steps.
C7: student l (d_model 192, 6 layers, 2.92M params) on the C2 windows with Timestral-28M and 209M labels. All on our suites and
GIFT-Eval short-term (55 configs). (The Mac slept 12.5 h with the lid closed on battery; synthetic generation at 2112 points cost
~5 h because GP sampling grows ~cubically with length.)
**Result:** GIFT-Eval CRPS (ratio to Seasonal Naive) and paired vs C2 (m, 512, 28M labels, 0.616): C5 m+209M 0.612 (0.995),
C6 2048 0.603 (0.979 [0.951, 1.009]), C7 l+28M 0.608 (0.988), l+209M 0.607 (0.986 [0.974, 0.997]); Toto-2.0-4m at 512 0.563
(0.915), at 4096 0.527, Timestral-5M 0.494. Our suites: C6 vs C2 = shapes 1.00, multi-domain 0.97 (ns), electricity 1.06 (worse);
C7 209M vs 28M labels with student l = 0.97 / 0.99 / 0.98 (ns), with student m = 0.99 / 1.00 / 1.00; l vs m = 0.98 on multi-domain.
**Surprise / learned:** no single lever moves the student more than ~2%: 3x the parameters gives ~1.4%, a better teacher ~0.5%,
4x the context ~2%. The 9% gap to Toto-4m at equal context is not size, teacher or history. Candidate: the architecture's output
bottleneck (one summary vector decodes all 13 x 64 outputs; Toto and Timestral decode per-patch tokens), the same change the
long-horizon note proposes. The 209M advantage grows a little with the bigger student (0.5% -> 1.5-3%), consistent with a weak
capacity gap, not significant.
**Next:** ADR for a multi-token decoder student (also needed for long horizons); test at 300k windows first.

## 2026-11-21 — Stage C9 (night): the multi-token decoder at scale
**Did:** multi-token students (ADR 0008) on cached labels, each paired with its single-token twin: m 512 steps 28M labels (vs C2),
l 2.9M 512 steps 209M labels (vs C7 l_209m), m 2048 steps + truncation 28M labels (vs C6). Our suites + GIFT-Eval (55 configs).
**Result:** GIFT-Eval CRPS (ratio to Seasonal Naive) and paired vs the twin: m 512 0.603 (0.979 [0.959, 0.997], 33/55 better),
l 209M 0.598 (0.986 [0.973, 0.999]), m 2048 0.603 (0.999, ns). Our suites vs twin: m 512 = shapes 1.024 (worse), multi-domain
0.985, electricity 1.045 (worse); l 209M = shapes 0.893, multi-domain 0.983, electricity 1.022 (ns); m 2048 = 1.029 / 0.993 /
1.005. Best student so far on GIFT-Eval: multi-token l + 209M labels (0.598); at 512 steps Toto-2.0-4m is 0.563 (gap 6%, was 9%).
The l multi-token student beats Toto-2.0-4m on all three of our suites (0.78 / 0.93 / 0.94).
**Surprise / learned:** the pilot's consistent 2-3% gain (300k windows) shrinks at 1M windows and turns mixed on our small suites
(worse on electricity and shapes for size m); on GIFT-Eval it holds at ~2% for 512 steps and vanishes at 2048 steps. Gains
do not add up: context, decoder, size and teacher each give 1-2% and overlap. The largest single combination effect is the
bigger student with the multi-token decoder and 209M labels (shapes 0.865 vs the m multi-token student).
**Next:** decide the default student (m multi-token, 512) and the release candidate; data mix / volume is the remaining lever.

## 2026-11-21 — Inference tricks, raws-labs/tinycast at equal history, data v3
**Did:** (1) GIFT-Eval of the C9 multi-token student (m, 512) with flip symmetrisation, Reverso-style downsampling (M = 8
cycles, beta = 4), and both (`scripts/gift_eval_student.py --flip --downsample`). (2) raws-labs/tinycast through their own
eval loop with histories cut to 2048 and 512 raw steps (`scripts/gift_eval_raws.py`, MPS; full history reproduces the
leaderboard within 1%). (3) Data v3: v2 + 18 GiftEvalPretrain subsets (bull, cockatoo, hog, pdb excluded as fev tasks).
**Result:** flip: CRPS 0.603 -> 0.596 (0.989 [0.973, 1.004]), MASE unchanged. Downsampling: worse, CRPS 1.031 [1.010, 1.058],
MASE 1.040; it fires on ~27% of series (k = 4-5 on 5-min data), big wins on bizitobs 10 s (-17%) but losses on 5-min sets
(bizitobs_l2c/5T +54%). raws-labs/tinycast: full 0.570, 2048 raw steps 0.571, 512 raw steps 0.612 (MASE 0.819); ours at 512:
0.603 (MASE 0.801), at 2048: 0.603. Data v3: 49,200 series (v2 18,114), 55 datasets, same 302 dropped, but only 0.30B points
(v2 0.26B) because of the 15M-points cap per subset.
**Surprise / learned:** at equal 512-step history our 0.87M student beats the 146k model; its whole lead is long history
(+7% from 512 to 2048), which our students cannot use (0.603 at both). Reverso's downsampling rule is calibrated for a
2048-step context: scaled to 512 it fires 4x more often and hurts. Flip is a small, cheap gain (~1%, borderline).
**Next:** pilots on 300k windows: data v3 vs v2; long history at fine resolution (2048 steps, patch 16 = 128 tokens, or a
multi-scale input).

## 2026-11-21 — Pilots C10 (data v3) and C11 (long history at fine resolution)
**Did:** 300k-window multi-token students, Timestral-28M labels, 15k steps. C10: data v3 vs v2 (512 steps). C11: 2048 steps at
patch 16 (128 tokens) and patch 64 (32 tokens), no truncation, vs the 512-step C8 student on 2048-step suites. GIFT-Eval for all.
**Result:** GIFT-Eval CRPS vs C8 (0.624): v3 0.635 (1.017 [1.005, 1.030], worse); 2048/p64 0.624 (1.000); 2048/p16 0.621
(0.996 [0.978, 1.014], MASE 0.824 vs 0.837). Our suites: v3 = shapes 1.016, multi-domain 1.000, electricity 1.030 (worse);
2048/p16 = shapes 1.132 (worse), multi-domain 0.961, electricity 0.942 (better); 2048/p64 = 0.974 / 0.954 / 1.011.
**Surprise / learned:** more diverse data (55 datasets, 2.7x series, same window count) did not help at 300k windows: the
per-dataset quota shrinks, so it may need more windows to pay. Fine-resolution long history helps where long history exists
(electricity -6%, 5-min / 15-min GIFT sets -10 to -15%) but loses on short series (weekly sets, covid_deaths, us_births +13-21%):
trained on mostly full 2048-step contexts, the student sees short GIFT series as 80-90% padding. Truncation would fix the
mismatch but breaks label consistency; the clean fix is variable-length contexts in the corpus, labelled by the teacher as cut.
**Next:** owner decision: variable-length-context corpus (consistent labels), pipeline overlap + batch-size benchmark.

## 2026-11-22 — Night: C12 (variable histories), C13 (real future vs teacher vs mix), C14 (Timestral-5M teacher)
**Did:** C12: 2048-step multi-token student (patch 16) on a corpus where half the windows were cut to a random history length
before Timestral-28M labelled them. C13: multi-token student m, 512 steps, trained on the real future (alpha 0), a 50/50 mix
(0.5) and teacher labels (1, existing) at 300k and 1M windows. C14: Timestral-5M as teacher on the 1M windows. All on our
suites and GIFT-Eval (4 parallel workers). Pipelined runs (labels on CPU during training). The batch-size benchmark ran out
of MPS memory at its largest setting and its printed lines were lost (piped through tail): to redo.
**Result (GIFT-Eval CRPS, ratio to Seasonal Naive):** real future 0.679 (300k) / 0.664 (1M); mix 0.644 / 0.621; teacher labels
0.624 / 0.603. Teacher vs real future, paired: 0.919 [0.894, 0.945] at 300k, 0.908 [0.878, 0.934] at 1M, better on 50/55
configs both times; mix vs teacher at 1M 1.031 (teacher better). Timestral-5M teacher: 0.593, vs 28M 0.984 [0.965, 1.003]
(33/55): the best student under 1M so far. Variable histories: 0.622 vs 0.621 (no change). Our suites disagree on shapes:
real future beats teacher labels there (KD/GT 1.10 at 300k, 1.19 at 1M) while teacher labels win on electricity (0.95, 0.96)
and tie on the public datasets.
**Surprise / learned:** at 300k -> 1M windows the distillation advantage on GIFT-Eval does not shrink (8.1% -> 9.2%): we are
well below the threshold where supervised training catches up. The shapes suite favours ground truth because half the
training data is synthetic with exact generator continuations, i.e. shapes are in-distribution for ground-truth training:
GIFT-Eval is the honest guide. A smaller, faster teacher (5M) is at least as good for a 1M student (capacity gap).
Consistent variable-length labels did not fix long-history use.
**Next:** Timestral-5M labels for data scaling (faster); redo the batch-size benchmark without the OOM setting.

## 2026-11-22 — Stage C15: 3M windows, real future vs Timestral-5M labels
**Did:** 3M windows (1.5M synthetic incl. 1M seeds from the new high range + 1.5M data v2), multi-token student m, 512 steps,
150k steps (~13 passes), memory-mapped corpus and labels, pipelined (5M-teacher labels during the real-future training).
(First launch lost the night: the synthetic seed space held only 500k training seeds, the preparer crashed and the trainer
waited forever; both fixed, relaunched at 10:30, done 19:57.)
**Result:** GIFT-Eval CRPS, data-scaling curve: real future 0.679 (300k), 0.664 (1M), 0.671 (3M); teacher labels 0.624 (300k,
28M), 0.603 (1M, 28M), 0.593 (1M, 5M), **0.582 (3M, 5M)**. Teacher vs real future at 3M: 0.868 [0.819, 0.916] (46/55).
3M vs 1M: teacher labels 0.983 [0.963, 1.000]; real future 1.011 (no gain). Our suites (3M, teacher labels): beats Toto-2.0-4m
on all three (0.93 shapes, 0.94 public, 0.92 electricity) and its own teacher Timestral-5M on shapes (0.979); electricity
-7% vs 1M. Real future still wins on shapes (KD/GT 1.23).
**Surprise / learned:** the opposite of the scaling-law expectation for our regime: real-future training stops improving
after 1M windows, while distillation keeps improving, so the gap widens (8% at 300k, 9% at 1M, 13% at 3M). The student
beats its teacher on synthetic shapes (in-distribution data).
**Next:** best student under 1M is now kd5m_3m (0.582): GIFT-Eval with flip, site update; more data with the 5M teacher.

## 2026-11-22 — Site: 3M student by default, long forecasts in Draw, small-model comparison
**Did:** Site model switched to the C15 3M-window student (`c1x_kd5m_3m`, Timestral-5M labels; benchmarks
`results/{shapes,fresh_fr_power}/student-site-onnx-*.json`). Draw now forecasts to the edge of the pad by rolling the
student out in 64-step blocks with sampled paths (browser port of `tinycast/models/rollout.py`, paths mode; 128 paths, 12
while the pen moves). New `scripts/small_models.py` → `results/gift_eval/small_models.json`: our student against every
model under 15M on the 55 GIFT-Eval short-term configs, parameter counts from Hugging Face. **Result:** ours 0.582; ties
raws-labs/tinycast at its 2048-step history (ours/theirs 1.022 [0.977, 1.068]) and beats it at 512 steps (0.952
[0.918, 0.987]); 3% behind Toto-2.0-4m at 512 steps, 18–19% behind Timestral-5M and TimeH2O-v1-tiny; beats Moirai-small
(0.875) and TTM-R2 (point forecasts, 0.620). Flip averaging on the 3M student: 0.998 [0.988, 1.009], nothing (it gave
~1% on the 1M student). **Surprise / learned:** the free inference trick vanishes once the student has more data, as if
the student had learned the sign symmetry itself. **Next:** C16 hybrid result, then C17 (10M windows) overnight.

## 2026-11-23 — C16: per-source hybrid target (real future on synthetic, teacher on real)
**Did:** `data.gt_sources: [synth]` (teacher quantiles of synthetic windows replaced by their true future, which makes the
teacher pinball equal the ground-truth pinball; checked to 7 digits), same 1M windows / student / 50k steps as C14 kd5m_1m.
**Result:** vs pure teacher labels: shapes 0.779 [0.724, 0.832], multidomain_test 1.022 [1.008, 1.035], fresh_fr_power
1.004 [0.986, 1.023], GIFT-Eval 1.025 [1.011, 1.040] (`results/study/stage_c16_hybrid/table.md`, `runs/c17_decision.txt`).
**Surprise / learned:** the shapes win of real-future training transfers completely, but it costs 2–2.5% on held-out real
data: the teacher's view of synthetic series is useful for real data, the generator's own draw is not. The pre-registered
rule (hybrid only if its GIFT interval is below 1) kept C17 on pure distillation. **Also:** C17 first launch failed in 1
minute (data v4 missing from `VERSIONS`; an edit script without an existence check). Fixed and relaunched (2026-11-22 23:56:53).
**Next:** C17 (10M windows) results in the morning; GIFT-Eval medium/long with the rolled-out 3M student.

## 2026-11-23 — C17: 10M windows (data v4), same compute as C15
**Did:** 5M data-v4 + 5M synthetic windows, Timestral-5M labels (1,150 windows/s on CPU, ~2.4 h for 10M; integrity check
clean), multi-token m, 150k steps (`configs/study/stage_c17_10m.yaml`). Pipeline 23:57 → 09:15.
**Result:** vs the 3M student (C15): shapes 1.056 [1.034, 1.081], multidomain_test 1.008 [0.998, 1.020], fresh_fr_power
0.994 [0.980, 1.008]; GIFT-Eval short 0.585 vs 0.582, paired 1.004 [0.980, 1.031], better on 21/55
(`results/study/stage_c17_10m/table.md`, `results/gift_eval/tinycast-kd5m_10m`). vs Toto-2.0-4m: 0.913 fresh, 0.947
public, tie shapes; vs Timestral-5M: tie on fresh, 3% behind elsewhere. **Surprise / learned:** the data-scaling curve is
flat from 3M to 10M at fixed compute; the gain from 1M to 3M came from more distinct series, and that need is met.
**Also (my error):** the queued medium/long evaluation never started: its wait loop `pgrep -f "runs/night_c17.sh"` matched
its own command line. Lost 09:15 → 20:25; launched by hand. Use a PID file or `pgrep -f '^bash runs/...'` next time.
**Next:** GIFT-Eval medium/long (running); then compute/capacity rather than data (longer training, or size l on 3M).
