# 0004 — Student architecture

**Status:** accepted for the preview 2026-11-12 (defaults from CLAUDE.md §7)

Patch transformer: context 512 → 32 patches of 16 (+ observed-mask channel), 2-layer MLP embedding, learned positions and a
forecast token, pre-norm encoder (s: 64×3, m: 128×4, l: 192×6), monotone head (median + cumulative softplus gaps) → 13×64.
Sizes 0.22M / 0.95M / 2.92M. Scaler inside the module: median / mean-abs-deviation + asinh (basic ops, ONNX-exportable).
MLP ablation (`arch: mlp`) exists but is parameter-heavier at equal width (flat 1024-d input); match params before comparing.
Loss weights levels by probability mass (ADR 0002).
