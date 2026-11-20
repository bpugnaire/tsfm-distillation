# 0008 — Multi-token decoder for the student

**Status:** ACCEPTED for a pilot 2026-11-20 (owner). Full-scale adoption depends on the pilot.

## Context
Stages C2–C7 moved the student by at most ~2% on GIFT-Eval: 3x the parameters (~1.4%), Timestral-209M vs 28M labels (~0.5%),
4x the context (~2%). At equal 512-step univariate input, Toto-2.0-4m is still ~9% ahead (journal 2026-11-16 / 11-20).
The student compresses everything into one d_model vector (a single query token) and one linear layer emits all 13 x 64
outputs. Toto 2.0 and Timestral decode the forecast with one token per future patch (Timestral: 32 forecast-query tokens,
two-stream attention; `timestral/model.py`). That is the last structural difference we have not tested.

## Options
1. Keep the single query token, widen it (C7: little gain).
2. **Multi-token decoder**: `horizon / out_patch_size` learned query tokens (default 4 x 16 steps), shared monotone head per
   token, context tokens masked from the query tokens. Horizon becomes "number of query tokens" (long horizons later).
3. Autoregressive patch decoding (Toto-style): more passes, harder to export; not now.

## Decision
Option 2, `arch: patch_transformer_mt` (`tinycast/models/student/__init__.py: MultiTokenNet`). Student m: 866,640 params
(single-query: 946,368). ONNX parity 7e-7. Pilot: same 300k windows and Timestral-28M labels as C2's `kd_v2_300k`, same
15k steps; scored on our three suites and GIFT-Eval short-term.

## Consequences
If it helps, it becomes the default architecture and the long-horizon path (more query tokens). If not, the remaining
gap is about data/training rather than architecture, and the next lever is the data mix.
