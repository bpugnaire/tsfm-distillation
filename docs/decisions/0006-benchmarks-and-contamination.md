# 0006 — External benchmarks and contamination control

**Status:** PROPOSED 2026-11-13 (owner approval needed for the retraining it implies)

## Context
Once a student is good enough we want standard, comparable numbers: **GIFT-Eval** (Salesforce, Apache-2.0, 28 datasets /
97 configurations, the reference leaderboard) and **fev** (AutoGluon, 100 tasks). The owner asked whether we can make sure
these benchmarks are not inside our training regimen.

## Evidence: content-level check (`scripts/contamination_check.py`, `results/contamination/*.json`)
Every 32-step z-normalised chunk (≥ 8 distinct values) of every series the student or its fine-tuning can see is hashed and
looked up in the benchmarks. Names are not trusted: the same series circulate under different names.
- Overlap found: GIFT-Eval covid_deaths, kdd_cup_2018, saugeenday (100%), temperature_rain (3.4%), m4_monthly (0.08%:
  tourism/FRED series), m4_hourly (9.2%: **our pedestrian counts**, used only for fine-tuning); fev fred_md_2025 (53%).
- **50/56 GIFT-Eval and 99/100 fev datasets have zero overlap.**
- GIFT-Eval's own non-leaking pretraining corpus (GiftEvalPretrain, 975 GB) contains 7 of our 11 training datasets and
  pedestrian counts, and none of the 4 GIFT test datasets above: consistent with the check.

## Decision (proposed)
1. **Drop from distillation training** the 4 datasets that ARE GIFT-Eval test sets: monash_covid_deaths, monash_kdd_cup_2018,
   monash_saugeenday, monash_temperature_rain (small: ~940 series together). Retrain the final student (Stage C winner recipe).
2. Re-run the content check; the release gate is **zero overlap** on every benchmark task we report.
3. Tasks still flagged after (1) are **excluded and listed** in the paper (expected: m4_monthly via tourism/FRED, fev
   fred_md_2025), not silently kept. A student fine-tuned on pedestrian counts is never evaluated on m4_hourly.
4. Suite: **GIFT-Eval short-term** configurations (horizons ≤ 64 match the student's one-shot horizon; medium/long-term need
   autoregressive rollout: later), plus the matching fev tasks, plus our fresh electricity (the only suite clean for the
   *teachers* too).
5. Teacher caveat: the teachers' own pretraining may include GIFT-Eval/fev test data (the GIFT leaderboard flags such models).
   Student-vs-teacher comparisons on public benchmarks are flagged; the headline stays on fresh electricity.

## Consequences
One retraining of the final student (~1.5 h on the Mac) and a GIFT-Eval/fev evaluation harness. Stage A–C conclusions stand
(they compare arms on the same suites), but the released model is the clean retrain.
