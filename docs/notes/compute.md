# Compute notes (measured 2026-11-12, Apple M1 Pro, 16 GB)

Source: `results/throughput.json`, timing runs in the journal.

| job | CPU | MPS (M1 GPU) |
|---|---|---|
| Toto-313m teacher, windows/s (b=64) | 54 | 179 |
| t0-gamma teacher, windows/s (b=64) | 83 | 206 |
| Toto-22m teacher, windows/s | 377 | 1377 |
| student s / m / l, train steps/s (b=256) | 6.0 / 2.6 / 1.2 | 39 / 17 / 8 |
| synthetic series generation | 140 series/s (GP Cholesky) | – |

Estimates: cache 300k windows ≈ 28 min (Toto-313m) / 24 min (t0-gamma) on MPS. One m-student run of 10k steps ≈ 10 min MPS.
Kaggle T4 not measured yet. **Owner rule (2026-11-12): ask before any long local run on the Mac.**

## Fine-tuning cost on this Mac (measured 2026-11-13, `results/compute/chronos2_finetune_timing.txt`)
Chronos-2 official `fit()` with vendor defaults (batch 256, 1000 steps), electricity 1-year history, context 512:
| model | Apple GPU (MPS) | CPU |
|---|---|---|
| Chronos-2-small (28M), full / LoRA | 1.06 / 1.03 s/step → ~18 min | 6.3 / 6.1 s/step → ~1h45 |
| Chronos-2 (120M), full | **out of memory** | 29.7 s/step → ~8 h |
| Chronos-2 (120M), LoRA | 39 s/step → ~11 h | 44 s/step → ~12 h |
| our student (0.95M), full fine-tune (Stage D, 1 year) | ~1 min (≤ 1500 steps, batch 128) | – |
