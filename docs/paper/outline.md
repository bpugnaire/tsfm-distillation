# Paper plan: "Distilling time-series foundation models is easy: a 1M-parameter student, lightly fine-tuned, matches the giants on your data"

Working title; the claim is a **hypothesis to test**, not a result. Written 2026-11-13, before any study result,
so that the analysis plan is fixed in advance (what we compare, how, with which statistics).

## 1. Claims we want to be able to make (or refute)
- **C1 (simplicity).** A standard recipe (teacher cache + pinball-based distillation loss + a small patch transformer)
  trains a useful student in < 1 GPU-hour, without per-teacher tricks.
- **C2 (what matters).** A ranked list of the ingredients that move accuracy the most, with paired effect sizes and CIs.
- **C3' (headline, revised 2026-11-13, ADR 0007).** Distil a small student, then fine-tune it on your own real history: this is
  more **efficient** (fine-tuning and inference cost) and more **robust** (seed stability, cheap ensembling) than fine-tuning the SOTA
  teacher on the same data, at comparable accuracy. Tested against teachers fine-tuned on the same history (new Stage F).
- ~~C3 (original)~~ After a *light* fine-tuning on a target domain's own history (minutes of compute, ≤ 1 year of data),
  the distilled student matches the **zero-shot** accuracy of the large teachers on that domain's future, at ~100–1000× fewer
  parameters and a fraction of the latency.
- **C4 (honesty).** Where the student does *not* match (domains, horizons, tails), and why.

## 2. Research questions → stages
| RQ | question | stage | factors |
|---|---|---|---|
| RQ1 | Which **inputs** should the teacher label? | A | real multi-domain · synthetic (KernelSynth/shapes) · teacher rollouts · white noise (control) |
| RQ2 | Teacher **labels vs ground truth**? | A, B | α ∈ {0, 0.5, 1} |
| RQ3 | Which **teacher(s)**? | B | Timestral-209M · Toto-313m · t0-gamma · mean ensemble · per-window error weighting |
| RQ4 | Which **distillation loss**? | B | pinball vs teacher (mass-weighted) · per-level Huber · level weighting on/off · horizon weighting |
| RQ5 | How does it **scale**? | C | windows {30k, 100k, 300k} · student {0.2M, 1M, 3M} · rollout length {1, 3, 8} · input mixes |
| RQ6 | How much **fine-tuning** closes the gap? | D | target history {1 week, 1 month, 1 year} · steps {100, 1000} · from distilled vs from scratch |

Stage D target domains (never seen in distillation): French electricity (fresh test after every model release; clean)
and Melbourne pedestrian counts (public: flagged "teachers may have seen it").

## 3. Fixed protocol
- **Evaluation suites:** shapes (200 synthetic series), held-out multi-domain test (last 15% of 9 public datasets, flagged),
  fresh French electricity (zero-shot for every model), pedestrian test (Stage D).
- **Metric:** relative WQL vs seasonal naive (9 core levels; baseline season from the sampling frequency), geometric mean;
  plus 80% coverage, MASE, fidelity to the teacher.
- **Comparisons between arms are paired** (same windows): geometric mean of per-window ratios, cluster bootstrap over series
  (`metrics.paired_ratio_ci`). An effect "moves the needle" only if its 95% CI excludes 1.
- **Selection** on the shared validation set only (multi-domain val, 70–85% of each series). Test suites are read once per arm.
- **Seeds:** 1 for exploratory stages (A, C sweeps), 3 for every arm that enters a headline table (B winners, D).
- **Reference recipe** for "what moved the needle": the best Stage-A arm; every ablation changes one factor from it.

## 4. Figures and tables planned
1. Ablation "needle chart": one bar per factor = paired ratio vs the reference recipe (log scale, 95% CI), sorted by effect.
2. Input-source table (RQ1) on three suites.
3. Teacher × α grid (RQ2–3).
4. Scaling curves (RQ5): accuracy vs windows, vs student size; Pareto accuracy vs CPU latency with the teachers.
5. Fine-tuning curve (RQ6): accuracy vs amount of target history, horizontal lines = teachers zero-shot.
6. Failure analysis: per-shape and per-dataset heat maps (where the student loses).

## 5. Threats to validity (to address in the text)
- Teachers may have pretrained on the public multi-domain data (training is fine; evaluation flagged; headline on fresh data).
- One fresh domain (13 correlated series) → wide CIs; pedestrian counts as a second, flagged, domain.
- Single seed in exploratory stages → effects smaller than the seed-to-seed spread must be re-run with 3 seeds.
- Student sees no electricity before Stage D: C3 compares *student + fine-tuning* with *teachers zero-shot*; fine-tuning the
  teachers is out of scope (stated explicitly, not hidden).

## 6. Compute budget (measured throughputs, Apple M1 Pro; see docs/notes/compute.md)
Stage A ≈ 35 min (running 2026-11-13) · B ≈ 2.5 h · C ≈ 2–3 h · D ≈ 1 h. Each stage starts only after owner approval.
