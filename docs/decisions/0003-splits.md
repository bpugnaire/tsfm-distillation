# 0003 — Data splits and the fresh test window

**Status:** proposed 2026-11-12 (owner to review)

## Context
Headline claims must rest on data no model could have seen. Model releases (HF `createdAt`): Chronos-2 2025-10-30,
Toto 2.0 2026-04-14, t0-gamma 2026-10-12, Timestral 2026-10-14. RTE éCO2mix: consolidated to 2026-08-31, real-time after.

## Decision (`configs/data/splits.yaml`, enforced by `tests/test_leakage.py`)
Splits are on **target** timestamps (UTC): train ≤ 2025-12-31; validation 2026-01-01 → 2026-08-31; fresh test ≥ 2026-10-15.
2026-09-01 → 10-14 is never a target. Fresh windows: daily origins at 00:00 UTC, horizon 64 h.
Shapes gallery seeds ≥ 1e6; training seeds < 1e6 (train/val use disjoint halves).

## Consequences
- Only ~4 weeks of fresh targets (312 windows, 13 series) → CIs ≈ ±0.04 relative WQL; cluster bootstrap over series is required.
  The window grows by ~13 windows/day: re-running later tightens it.
- Release date is a proxy for training cutoff (the true cutoff is earlier, so this is conservative).
- Real-time data are provisional (later revised by RTE); identical for all models.

## Alternatives
Drop Timestral (never enabled) → fresh_start could be 2026-10-13: +2 days only. Drop t0-gamma for t0-beta (2026-09-16) → +4 weeks.
