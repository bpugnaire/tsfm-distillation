# 0007 — Headline claim: distilled student + fine-tuning vs teacher + fine-tuning

**Status:** PROPOSED 2026-11-13 (owner reframing; design and compute need approval)

## Context (owner, 2026-11-13)
"Fine-tune on the real data, not on the teacher output. Show that we can distil foundation teachers into very efficient models with
great out-of-the-box performance, which after a not-so-costly fine-tuning on a dedicated series beat the teachers. Distil a small
student then fine-tune on your use case is more efficient and robust (easier ensembling, e.g.) than taking the SOTA teacher and
fine-tuning it on your data."
This adds **fine-tuning the teachers** to the scope (CLAUDE.md §11 listed it as out of scope unless the owner adds it).

## Decisions
1. Fine-tuning target = **real data** (Stage D protocol). Stage E (teacher labels on your history) becomes a secondary result.
2. The claim is tested against **teachers fine-tuned on the same history, same budget rules** (not only teachers zero-shot).
3. The claim is measured on three axes, each reported separately (no single "winner"):
   - **accuracy** after fine-tuning (paired, 3 seeds where affordable);
   - **cost**: fine-tuning wall time / memory on the same machine, inference latency (CPU, browser), parameters;
   - **robustness**: seed spread, forgetting on other suites, gain from ensembling k fine-tuned models at equal inference cost.
4. **Update (owner, 2026-11-13): teachers that ship no fine-tuning (Toto 2.0, t0, Timestral) are compared zero-shot**: that is what a
   user can actually do with them, and being fine-tunable is precisely the student's advantage. Only Chronos-2, which has an official
   fine-tuning API, is compared fine-tuned. The custom-loop plan below is dropped.
   ~~Teacher fine-tuning recipes: **Chronos-2 via its official `fit()` (full and LoRA)**; Toto 2.0 (22m, 313m) and Timestral (28M, 209M)
   via our own loop around their modules (pinball loss on native quantiles, AdamW, early stopping on the same fine-tuning validation
   set). Stated in the paper as "our recipe" (no vendor fine-tuning code exists for them; checked 2026-11-13).~~

## Evidence already available (`results/study/diagnostics/robustness.json`, no new training)
- Ensemble of the 3 seeds of the electricity 1-year fine-tune: 0.659 (single seed 0.686; paired 0.947 [0.938, 0.955]).
- Forgetting: a student fine-tuned on electricity is 3.5× worse on shapes, 1.4× on the multi-domain test, 1.2× on pedestrians.

## Risks
Accuracy alone will likely favour the fine-tuned 200M+ teachers; the claim may hold only as an **efficiency** claim
(accuracy per unit of fine-tuning compute / inference cost). We report it either way.
