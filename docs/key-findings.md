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

## 3. Distillation keeps improving with data where real-future training stalls

Theory (Busbridge et al. 2025) predicts supervised training catches up once the student has seen enough data. Here the
opposite happened (C13, C15):

| windows | real future | teacher labels | teacher / real future |
|---|---|---|---|
| 300k | 0.679 | 0.624 (28M) | 0.919 |
| 1M | 0.664 | 0.603 (28M), 0.593 (5M) | 0.908 |
| 3M | 0.671 | **0.582** (5M) | **0.868** [0.819, 0.916] |

**Checked with 3 training seeds (2026-11-27, same teacher Timestral-5M at both scales, `results/seeds_core_claim.json`):**

| windows | real future (3 seeds) | teacher labels (3 seeds) | teacher / real future, configs + seeds resampled |
|---|---|---|---|
| 1M | 0.664 / 0.646 / 0.655 | 0.593 / 0.593 / 0.587 | **0.903** [0.862, 0.941] |
| 3M | 0.671 / 0.653 / 0.658 | 0.582 / 0.584 / 0.577 | **0.880** [0.831, 0.925] |

Teacher labels win at both scales on every seed (10-12%). **Whether the gap widens is not established:** 3M vs 1M ratio
0.975 [0.942, 1.006]. The earlier 0.908 -> 0.868 mixed teachers (28M at 1M, 5M at 3M). What holds: real-future training
does not improve from 1M to 3M (0.655 -> 0.661 on average), distillation does (0.591 -> 0.581). Training noise is ~3%
for real-future students and ~1% for distilled ones. **Intuition.** With the real future,
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

## 5. Fine-tuning: worth it only where the zero-shot model is weak, and only with replay (Stage F)

Site student, 2 clean domains, 1 week / 1 month / 1 year of history, 9 techniques, 3 seeds (`results/study/stage_f_finetune`).
- **Where zero-shot is already good (fresh electricity, level with its teacher), fine-tuning does not pay:** worse with
  1 week or 1 month of history (full fine-tuning 1.22x / 1.31x the error), a tie with a year.
- **Where it is weaker (pedestrian counts), gains grow with history:** -2.5% / -9% / -16% for 1w / 1m / 1y.
- **It forgets a lot**, even with 27k trainable weights: head-only fine-tuning doubles the error on the shapes.
  **Replaying distillation windows (50% of batches, teacher labels) removes the forgetting** (shapes 1.02x, public data
  0.99x) at almost no cost on the target. Intuition: in a 0.87M model the output head is shared by every domain, so
  "freeze the backbone" is not a protection; showing the old data again is.
- Labels chosen by the realised future help the in-house shapes test and hurt real-world benchmarks, twice: the
  per-source hybrid (C16) and "209M teacher where it beat 5M by 10% on the window" (C18, GIFT 1.027 worse, shapes 0.87).

**Confirmed on fresh data (Stage G, pedestrian counts published after every model's release, arm fixed in advance):**
the 0.87M student fine-tuned with replay on one year of history beats its teacher Timestral-5M by 12% [11%, 13%],
every model up to 28M parameters, and ties Toto-2.0-313m (360x its size); Chronos-2, t0-gamma and Timestral-209M stay
3-8% ahead. Without forgetting (shapes 1.02x, public data 1.00x). `results/study/stage_g_fresh_ped/headline.txt`.

**On five fresh datasets (Stage H, same pre-registered arm):** the fine-tuned student beats its zero-shot teacher on
3 of 5: pedestrians -12%, bike counters -8%, solar -6% (solar: also beats Chronos-2, ties 200-300M models). It loses on
wind (+2.4%) and electricity (+6.7%). **Intuition.** Fine-tuning teaches each series' own repeating profile (where a
counter is, how much solar a region has installed). Wind follows weather that no history reveals, and national-scale
electricity is a smooth aggregate the general model already handles. `results/fresh_headline.json`.

## 6. Things that did not matter as much as expected
- **Which teacher, at scale:** Toto-313m, t0-gamma, Timestral-28M and an ensemble of three all within 4% at 1M windows (C3).
  The ensemble gain seen at 30k windows (2%) disappeared.
- **3x the parameters:** 0.95M -> 2.9M gave about 1.4% (C7).
- **Periodicity detection** (raws-labs' signature idea): ~20% at a short training budget, ~0.5% at full budget (their own
  retrain ablation). It saves data rather than adding skill.
- **Bigger batches on an Apple GPU are slower:** 4,733 windows/s at batch 256, 2,934 at 1,024, 2,626 at 2,048; bf16 does not
  help (benchmark 2026-11-22).

## 7. Things that silently cost us, and would cost anyone
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
