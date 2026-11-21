# Why a 146k model beats our 0.9M students: gap analysis (2026-11-21)

Sources: raws-labs/tinycast repo and evidence files (external/raws-tinycast, commit 628c0be; numbers recomputed here from
their per-configuration CSVs, normalised by their seasonal-naive file), Reverso (Fu et al., ICML 2026, arXiv 2602.17634),
Distillation Scaling Laws (Busbridge et al., ICML 2025, arXiv 2502.08606), Towards Neural Scaling Laws for TSFMs (Yao et al.,
ICLR 2025, arXiv 2410.12360), DistilTS (arXiv 2601.12785), TiRex (arXiv 2505.23719), and our own stages A–C9.

## 1. Capacity is not the bottleneck
raws-labs/tinycast: 146,505 params, GIFT-Eval CRPS 0.570 on our 55 short configs (0.545 on all 97). Reverso-Small 0.55M:
MASE 0.721 (all 97). Ours, best under 1M: 0.603. Our own C7: 3x the parameters gave ~1.4%.

## 2. Training data: ~500x less, and less diverse (largest gap)
| | series | time points | windows seen in training |
|---|---|---|---|
| ours (data v2) | ~18k used (68k downloaded, 1,500 cap per subset) | ~0.4B | 12.8M (50k steps x 256) |
| raws-labs/tinycast | full GiftEvalPretrain (~4.5M) + KernelSynth + 250k synthetic | ~230B | ~150M (36,621 x 4,096) |
| Reverso | full GiftEvalPretrain, rebalanced (<=100k samples/dataset/epoch, <=48/series) + 1M synthetic | ~230B | ~1M points per step |
GiftEvalPretrain has 152 subsets (975 GB); we use 46 of the smaller ones. Evidence that volume and diversity matter:
our 300k -> 1M windows = 3.5% on GIFT-Eval; raws budget 50M -> 150M samples = 3.5% (0.748 -> 0.722, their optimisation
family); scaling laws for TSFMs hold out of distribution only with diverse data (Yao et al.).

## 3. Synthetic data: narrow, and shaped like our own test suite
Reverso: removing synthetic data costs 6.5% MASE (0.738 -> 0.786), the largest single ablation. Its three families (also
raws-labs'): GP over a 38-kernel bank with 19 periods normalised by length, spike/pulse trains, trend-seasonal-impulse,
length 4096. Ours: KernelSynth with 5 fixed absolute periods (12...168), plus randomised versions of the 20 gallery shapes.
The second part makes our *shapes* suite partly in-distribution: our strong shapes scores (beating Toto-4m) overstate
generalisation; GIFT-Eval is the honest guide.

## 4. Free inference-time gains we do not use
raws-labs deployed profiles: sign symmetrisation (average f(x) and -f(-x)) 1.1%, period-alignment downsampling 2.0%,
both 3.1% (0.563 -> 0.545). Reverso: downsampling 1.8%. Both apply to an already-trained student: evaluation only.

## 5. Augmentation
Reverso: all six augmentations together 2.3% (downsampling, amplitude modulation, vertical and horizontal flip, censor,
mixup; each alone ~0). We have random truncation only (+2.5% on GIFT-Eval in the C4 pilot).

## 6. Architecture choices that matter at small size
- Sequence mixer: plain attention is the weakest at Reverso scale (MASE 0.750–0.762 vs 0.725 for conv + DeltaNet, ~4%).
  Ours is plain attention.
- Decoder: attention decoder head vs linear 2% (Reverso); our multi-token decoder gave ~2% (C9).
- Multi-scale input channels (effective context 16k at 2048 tokens): 0.5–2% (Reverso).
- Normalisation: both use per-window min-max; we use median / mean-abs-deviation + asinh. Untested here.
- Periodicity prior (raws-labs): ~20% at a short budget, ~0.5% at the full budget (their detector-retrain at 36,621 steps:
  0.548 without vs 0.546 with). A data-efficiency device, not the source of their lead.

## 7. Distillation vs ground truth at scale
Both strong small models are trained on ground truth only. Distillation scaling laws (Busbridge et al.): distillation beats
supervised training only below a student-size-dependent compute/data threshold. Our "teacher labels beat ground truth"
evidence (Stage A, C-lite: 12–15%) is from the 30k–300k-window regime. With 100x more real data the advantage may vanish,
and teacher labelling (CPU-bound, 100–440 windows/s) would block scaling. To test: same large corpus, GT vs KD vs mix.

## 8. Recipe details (smaller)
raws-labs: 4-block autoregressive rollout with scheduled sampling, WSD schedule, LR 3e-3, batch 4096, uniform soup of the
last 8 checkpoints. Reverso: AdamW 5e-4, WSD, weight decay 0.1, MAE loss. DistilTS: horizon-weighted KD (later steps
weighted up), small gains.

## Proposed order (laptop-feasible)
1. Inference tricks on existing students (eval only, ~30 min per GIFT run). Expected ~3%.
2. Data: more GiftEvalPretrain subsets (disk: 369 GB free), no 1,500-series cap, Reverso-style rebalancing, contamination
   filter rerun; Reverso/raws synthetic families at length 4096. Then GT vs KD vs mix on the same windows.
3. Augmentations (the Reverso set).
4. Architecture: a conv / linear-recurrent mixer variant, min-max normalisation, multi-scale input.
