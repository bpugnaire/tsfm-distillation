# 0002 — Metric definitions

**Status:** accepted 2026-11-12

## Context
Models emit different quantile sets: Toto 2.0 only 0.1…0.9; t0-gamma and Chronos-2 emit 0.01…0.99.
Our project grid has 13 levels (0.01, 0.05, 0.1…0.9, 0.95, 0.99).

## Decision
- **Headline: WQL on the 9 core levels 0.1…0.9** = `mean_τ 2Σ_t ρ_τ / Σ_t|y|` (GluonTS
  `mean_weighted_sum_quantile_loss`, what GIFT-Eval calls "CRPS"). Every model is scored on levels it
  produced itself, so Toto is not judged on tails we extrapolated for it.
- Aggregate: per-series WQL / seasonal-naive WQL → **geometric mean** (GIFT-Eval style) with a
  **percentile bootstrap 95% CI over series**. Arithmetic means are reported too.
- Secondary: MASE (median; seasonal scale, lag-1 fallback), 80%/98% coverage, 80% width / MASE scale,
  `wql13` (all 13 levels). Coverage98 is flagged for models whose tails are extrapolated.

## Finding that shaped this (tests/test_metrics.py)
Equal-weight quantile-loss averages are *not* unbiased CRPS estimates. For a calibrated Gaussian,
E[ρ_τ] = σφ(z_τ), largest at the median, so:

| estimator | value | vs exact CRPS (1/√π = 0.564) |
|---|---|---|
| 9 levels 0.1…0.9 (GIFT-Eval) | 0.615 | **+9%** |
| our 13 levels, equal weights | 0.466 | **−17%** |
| 99 uniform levels | 0.568 | ≈ exact |

Ranking models with the 9-level metric is fine (same bias for all); calling it CRPS is loose.
**Training** losses weight each level by its probability mass (midpoint rule), which removes the bias.

## Consequences
Numbers are comparable with GIFT-Eval conventions; the absolute WQL is not a true CRPS.
