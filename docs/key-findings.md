# Key findings that are not obvious

What this project measured that surprised us, or that most people would guess wrong. Each finding names its evidence
(results files, journal entries). Scores are GIFT-Eval short-term CRPS relative to Seasonal Naive (lower is better) unless
stated otherwise. Last updated 2026-11-22.

## 1. A smaller teacher can teach better than a bigger, stronger one

Same 1M training windows, same 0.87M multi-token student, only the teacher changes:

| teacher | its own GIFT-Eval score | student's GIFT-Eval score |
|---|---|---|
| Timestral-209M | 0.465 | (single-token runs: ties the 28M, C5) |
| Timestral-28M | 0.478 | 0.603 |
| **Timestral-5M** | 0.494 (the weakest) | **0.593** (the best) |

Evidence: `results/study/stage_c14_teacher5m`, `results/gift_eval/tinycast-kd5m_1m`; 209M vs 28M in C5 and C7.

**Intuition.** A student can only copy the part of the teacher's behaviour that it can represent. A 209M model's extra skill
lives in patterns a 1M network cannot express, so its labels carry that skill as noise the student cannot fit: the student
ends up matching the teacher about as well either way (distillation loss 0.289 vs 0.291 for 209M vs 28M labels, C5), and
gains nothing. A 5M teacher's function is much closer to what 1M parameters can express, so more of its knowledge actually
transfers. Distillation research calls this the capacity gap (Cho & Hariharan 2019; Mirzadeh et al. 2020, "teacher
assistant"). The practical bonus: the 5M teacher labels several times faster, which is what limits data scaling on a laptop.

## 2. Copying the teacher beats learning from what actually happened, except on synthetic data

Same windows, same student, only the training target changes (C13):

| | real future | 50/50 mix | teacher labels |
|---|---|---|---|
| 300k windows | 0.679 | 0.644 | **0.624** (better on 50/55 GIFT configs) |
| 1M windows | 0.664 | 0.621 | **0.603** (better on 50/55 GIFT configs) |

On fresh electricity, teacher labels also win (4–5%). On our synthetic *shapes* test, the opposite: the real future wins
by 10% (300k) and 19% (1M). Evidence: `results/study/stage_c13_gt_vs_kd`, `results/gift_eval/tinycast-{gt,mix}_{300k,1m}`.

**Intuition: why teacher labels win on real data.** The real future of a window is one draw out of everything that could
have happened: tomorrow's demand is high partly because it happened to be cold. Training on it, the student sees a noisy
target and has to average many windows to work out the distribution. The teacher's 13 quantiles describe that distribution
directly, for every window, informed by its pretraining on far more series than we have. Each training example therefore
carries much more information and much less noise: the student learns the spread, not one sample of it.

**Intuition: why the real future wins on synthetic shapes.** Half of our training data is synthetic, and there the "real
future" is not a noisy draw from an unknown process: it is the generator's own continuation, from the very generator
families the shapes test is built from. Training on it teaches the student the generator itself, exactly. The teacher has
never seen our generator; its forecast of a synthetic sawtooth is an approximation (Timestral-28M scores 0.45 on shapes,
not 0). Copying it caps the student at the teacher's error on data where the truth is clean and learnable. In short:
teacher labels help when the truth is noisy and the teacher knows more than our data; the truth helps when it is clean
and the test comes from the same source. This is also why our shapes suite flatters ground-truth training and why
GIFT-Eval is the guide for decisions.

**Tested (C16): use the real future only on synthetic windows, teacher labels on real ones.** Same 1M windows and student.
Against pure teacher labels: shapes −22% [0.724, 0.832] (the real-future gain, fully kept), fresh electricity a tie
(1.004 [0.986, 1.023]), held-out public data +2.2% [1.008, 1.035], GIFT-Eval **+2.5% [1.011, 1.040] worse**. So the hybrid
buys our in-house shapes score and pays for it on everything else. The teacher's forecasts of synthetic series carry
something useful for real data that the generator's own continuation does not: the generator's future is itself one random
draw of a Gaussian process (not "the truth" except on the hand-built shape families), while the teacher answers with a
calibrated spread and a real-data prior. Evidence: `results/study/stage_c16_hybrid`, `results/gift_eval/tinycast-hyb5m_1m`.

## 3. More data makes distillation pull further ahead, not less

Theory (Busbridge et al. 2025) predicts supervised training catches up once the student has seen enough data. Here the
opposite happened (C13, C15):

| windows | real future | teacher labels | teacher / real future |
|---|---|---|---|
| 300k | 0.679 | 0.624 (28M) | 0.919 |
| 1M | 0.664 | 0.603 (28M), 0.593 (5M) | 0.908 |
| 3M | 0.671 | **0.582** (5M) | **0.868** [0.819, 0.916] |

Real-future training stops improving after 1M windows; distillation keeps improving. **Intuition.** With the real future,
more windows mostly add more noisy single draws; a 1M-parameter model saturates on what it can extract from them. Teacher
labels are already denoised distributions, so each extra window adds a clean, informative example, and the student keeps
moving towards the teacher (at 3M it even beats Timestral-5M on our shapes test). We are far below the data scale where the
theory's crossover would happen (raws-labs and Reverso train on ~150M windows of real futures).

**Then it stopped (C17): 10M windows at the same compute as 3M did not help.** Same student, teacher and 150k steps
(38M samples seen: 3.8 passes over 10M windows instead of 12.8 over 3M). GIFT-Eval 1.004 [0.980, 1.031] vs the 3M student,
fresh electricity 0.994 [0.980, 1.008], held-out public data 1.008 [0.998, 1.020], shapes 5.6% worse [1.034, 1.081].
**Intuition.** Between 1M and 3M windows the student was still seeing too few distinct series; by 3M it is not any more,
and at a fixed step budget the extra windows only replace repeats. The shapes loss fits this: each synthetic window is
seen 3.4x less often, and shapes reward knowing our own generator well. The next lever is compute or capacity, not windows.
Evidence: `results/study/stage_c17_10m`, `results/gift_eval/tinycast-kd5m_10m`.

## 4. At equal history, our 0.87M student beats a 146k model that beats it on the leaderboard

raws-labs/tinycast (146k params) scores 0.570 on the leaderboard and our best under 1M 0.603. Cut both to 512 steps of
history: theirs 0.612, ours 0.603. Its whole lead is long history (+7% from 512 to 2048 steps), which our students do not yet
use: 2048-step versions score the same as 512-step ones (C6, C9, C11, C12). Evidence: `results/gift_eval/raws-tinycast@hist*`.

## 5. Things that did not matter as much as expected
- **Which teacher, at scale:** Toto-313m, t0-gamma, Timestral-28M and an ensemble of three all within 4% at 1M windows (C3).
  The ensemble gain seen at 30k windows (2%) disappeared.
- **3x the parameters:** 0.95M -> 2.9M gave about 1.4% (C7).
- **Periodicity detection** (raws-labs' signature idea): ~20% at a short training budget, ~0.5% at full budget (their own
  retrain ablation). It saves data rather than adding skill.
- **Bigger batches on an Apple GPU are slower:** 4,733 windows/s at batch 256, 2,934 at 1,024, 2,626 at 2,048; bf16 does not
  help (benchmark 2026-11-22).

## 6. Things that silently cost us, and would cost anyone
- **Early stopping with a cosine schedule:** stopping when validation plateaus skips the low-learning-rate phase; it made the
  209M-taught student look 3–10% worse than it was (C5a vs its full-schedule rerun).
- **Validation can pick the wrong model:** the 3-teacher ensemble had the best validation score and lost on every test (C3).
- **Packages that hide failures:** t0 replaces NaN/Inf with 0 and only logs it (2 of 360k labels were all-zero forecasts);
  Chronos and Chronos-Bolt clamp unsupported quantile levels with a log warning. Both pass naive integrity checks.
- **Inference tricks do not transfer blindly:** sign-flip averaging gives ~1%, but Reverso's downsampling rule, tuned for
  2048-step histories, fires 4x too often at 512 and costs 3%.
- **A 16 GB laptop panics, it does not slow down:** two memory-heavy jobs together (training with two teachers' labels in RAM
  + a multi-worker rollout evaluation) exhausted the memory compressor and the kernel watchdog rebooted the Mac twice in
  this project. Queue them, never overlap them; save evaluation results per item so a crash keeps finished work.
- **Random noise as distillation input teaches nothing** (5.7x worse than structured synthetic, Stage A): the teacher
  answers every noise window the same way.
