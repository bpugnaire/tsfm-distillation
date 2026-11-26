# How far are we from a publishable result? (2026-11-23)

"Publishable" here: a workshop paper / arXiv preprint, or an article whose claims survive a hostile reviewer. Numbers below
come from the files cited; nothing new was computed for this note.

## 1. Candidate claims and their current evidence

| # | Claim | Evidence now | Strength |
|---|---|---|---|
| C1 | At equal windows, teacher labels beat the real future, and the gap **grows** with data (0.919 → 0.908 → 0.868 at 300k / 1M / 3M) | C13, C15; GIFT-Eval paired over 55 configs, e.g. 0.868 [0.817, 0.916] at 3M | strong effect, **1 seed per arm** |
| C2 | A smaller teacher teaches better (5M > 28M > 209M labels for a 0.87M student) | C14 vs C9 vs C5/C7, GIFT 0.593 vs 0.603 | small effect (~2%), 1 seed: **needs seeds** |
| C3 | Using the real future only on synthetic windows wins the in-house shapes test but loses 2.5% on GIFT-Eval | C16 | clean, 1 seed; a good negative result |
| C4 | Data stops helping between 3M and 10M windows at fixed compute | C17 (tie, 1.004) | 1 seed; C19 (2x steps) running tonight decides the reading |
| C5 | A 0.87M distilled student ties raws-labs/tinycast (146k) at full history and beats it by 5% at equal history; beats Toto-2.0-4m on fresh electricity (0.913 for the 10M student) | `results/gift_eval/small_models.json`, `stage_c17_10m` | solid comparison, but see 2.3 |
| C6 | Free inference tricks fade with data (flip: ~1% at 1M windows, 0.998 at 3M) | `results/gift_eval/tinycast-c9-mt_1m+flip`, `tinycast-kd5m_3m+flip` | 2 points only |

C1 is the paper. It contradicts the usual reading of distillation scaling laws (Busbridge et al. 2025: supervised training
catches up with enough data) in the regime we can reach, and the time-series literature has little on it.

## 2. What a reviewer would reject today

1. **Single seeds.** Almost every Stage C number is one training seed; the intervals cover test windows / configs, not training
   noise. CLAUDE.md requires >= 3 seeds for headline runs. Needed: 3 seeds of KD and GT at 1M and 3M (C1), 3 seeds of the 5M vs
   28M teacher at 1M (C2). Cost: 1M runs ~70 min each, 3M runs ~6 h each on this Mac: 8 x 70 min + 4 x 6 h ≈ 33 h GPU.
2. **Model selected on a test set.** The site/headline student (C15 kd5m_3m) was chosen for its GIFT-Eval score. Fix: state a
   pre-registered rule (largest data scale, chosen before seeing GIFT; or validation), report every arm, never only the best.
3. **Partial benchmark.** We report GIFT-Eval short (55 of 97 configs). The leaderboard ranks all 97; medium/long with our
   rollout is running now. Without it, "beats X on GIFT-Eval" must say "short-term".
4. **Unequal histories in comparisons.** Leaderboard models read 2048–4096 steps; ours 512. We already run 512-step versions of
   Toto-2.0-4m and raws-labs/tinycast; every comparison table must show both.
5. **Contamination statement.** Training data = GiftEvalPretrain subsets (built to exclude GIFT-Eval test) plus our per-series
   filter (`results/contamination/`); teachers' pretraining overlap is unknown (Timestral, Toto). Needs a short, explicit
   section; the fresh French electricity split (after all release dates) is our only leakage-free real test.
6. **Teacher licence.** The best students are Timestral-taught (modified MIT with a revenue clause). Fine for a study; any
   released checkpoint needs a licence check, or the Toto-taught student (Apache-2.0) as the released one.
7. **Related work.** DistilTS (arXiv 2601.12785), raws-labs/tinycast, Reverso, TTM, Chronos-Bolt, distillation scaling laws,
   capacity-gap papers. Our claim must be positioned against DistilTS in particular (time-series distillation, horizon weighting).

## 3. Already in good shape
- Paired bootstrap intervals everywhere; a leakage-safe fresh split enforced by a test; strong baselines (seasonal naive, ETS,
  official small Toto, same-architecture GT student, raws-labs/tinycast at equal history).
- Every number reproducible from committed configs; journal with dated failures (good material for a "pitfalls" section).
- Controlled ablations: teacher size, data size, target (GT / mix / KD / hybrid), decoder, context, inference tricks.

## 4. Shortest path (proposal)
1. Finish GIFT medium/long (tonight) → full 97-config score.
2. Seeds for C1 and C2 (~33 h GPU, 3–4 nights).
3. Write the protocol section first (selection rule, seeds, contamination), then the C1 figure: KD/GT ratio vs windows with
   seed bands.
4. Optional: one point above 3M for C1 (3M → 10M GT run, ~6 h GPU at 150k steps) to see whether the gap keeps growing.

## Update 2026-11-27: 3 seeds for C1
`results/seeds_core_claim.json` (`scripts/seeds_summary.py`): teacher labels / real future on GIFT-Eval short, configs and
seeds resampled: **1M 0.903 [0.862, 0.941], 3M 0.880 [0.831, 0.925]**, every seed below 1. Widening from 1M to 3M:
0.975 [0.942, 1.006], not significant. C1 should therefore be stated as "teacher labels beat the real future by ~10-12% at
1M and 3M windows, and only distillation improves with more data", not "the gap grows". Item 2.1 (seeds) is done for C1;
C2 (teacher size) still has one seed.
