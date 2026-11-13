# 0005 — Distillation study design (generator- and teacher-driven data)

**Status:** PROPOSED 2026-11-13, waiting for owner approval. Nothing below has been run.

## Context
- Owner direction (2026-11-13): electricity is the final demonstrator, not the focus. Use several data sources and/or
  generators, and let the teachers (Toto 2.0, t0, Timestral; all allowed) produce the training targets.
- Evidence so far: teachers on white noise give one flat answer for every context (`results/input_families.json`);
  KernelSynth ≈ 70–75% of real-data "richness". Teacher rollouts are realistic for 1–3 blocks with correlated sampling,
  then collapse (median feedback) or drift (sampling) (`results/rollouts.json`).
- Fixing a fairness problem: the preview student was trained on electricity (in-domain) while teachers are zero-shot.
  **In this study the student never sees electricity in training**, so the fresh electricity test is zero-shot for everyone.

## Question
Which **inputs** (real multi-domain, structured synthetic, teacher rollouts, noise) and which **targets** (ground truth,
one teacher, several teachers) give the best small student, per unit of compute?

## Design (staged; each stage is decided after reading the previous one)
**Stage A, input source (pilot, ≈ 30 min on the Mac GPU).** One teacher (Timestral-209M: best on shapes, native tails), α = 1,
student size s, 30k windows per arm, 1 seed. Arms:
1. real multi-domain contexts (Monash / LOTSA subsets, train portion only; licences recorded first)
2. KernelSynth + randomised shapes
3. teacher rollouts from real multi-domain starts (1–3 blocks, correlated sampling, ρ = 0.9)
4. white noise (negative control: expected to fail)
5. ground truth only on arms 1 and 2 (the floors)
Cost: caching 4 × 30k windows ≈ 10–15 min (measured teacher throughput ≈ 180–210 windows/s on MPS for 200–300M teachers;
Timestral-209M on MPS still to measure) + 6 trainings × ~2 min.

**Stage B, teachers (≈ 2.5 h).** Best input mix from A; teachers ∈ {Timestral-209M, Toto-313m, t0-gamma, mean ensemble,
per-window error weighting}; α ∈ {0.5, 1}; student m; 100k windows; 3 seeds.

**Stage C, scale (≈ 2–3 h).** Windows ∈ {30k, 100k, 300k}; rollout length ∈ {1, 3, 8} blocks; student s/m/l → Pareto chart.

## Evaluation (unchanged harness)
- **Shapes** (200 series, seeds disjoint from training).
- **Fresh electricity** (zero-shot for all, published after every model): the demonstrator.
- **Held-out multi-domain** subset, flagged "teachers may have seen this".
- Metrics: relative WQL (cluster bootstrap), coverage, fidelity to the teacher. Model selection on validation only.

## Alternatives considered
- Full factorial (inputs × teachers × α × size × seeds): > 20 GPU-hours; staged design gets the main answers in < 6 h.
- Electricity-only training: rejected by the owner (no fixation on one domain).

## Open sub-decisions
- Baseline season on real data: use the dataset's sampling frequency (hourly → 24, daily → 7…), the GIFT-Eval convention,
  instead of auto-detection (which made ETS non-seasonal on electricity). Recommended.
