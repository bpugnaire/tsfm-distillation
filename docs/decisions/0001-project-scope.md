# 0001 — Project scope, stack and licences

**Status:** accepted 2026-11-12 (stack deviations pending owner review)

## Context
Distil TSFMs (Toto 2.0, t0) into a ≤3M-param quantile forecaster that runs in the browser; publish a
rigorous study + demo. Constraints: free GPUs only, no fabricated numbers, fresh-data evaluation.

## Decisions
- **Python 3.12** (not 3.11 as first planned): `toto-2` (the Toto 2.0 package) requires ≥ 3.12;
  `tfc-t0` supports 3.11–3.13 and `chronos-forecasting` ≥ 3.10, so 3.12 is the only common choice.
- **Toto 2.0 is `toto-2` (`from toto2 import Toto2Model`), not `toto-ts`** (that is Toto 1.0, which pins
  numpy==1.26.4 and huggingface-hub<1 and is incompatible with t0). All teachers share one env.
- **t0 is not gated** (t0-alpha/beta/gamma: `gated=False`, Apache-2.0, checked 2026-11-12). No HF_TOKEN needed.
- **Timestral disabled.** Its Modified MIT licence withdraws *all* rights (evaluation included) when the
  user's company or employer has > $20M monthly revenue. The owner must confirm eligibility before we
  even run it as a comparator. ~~The adapter raises `PermissionError` until then.~~
  **Update 2026-11-13:** owner confirmed this is a personal project, so the revenue clause does not apply. Timestral is
  enabled as a **comparator** (5M/28M/209M). Distilling *from* Timestral still needs an explicit owner decision (CLAUDE.md §2.4).
- Chronos-2: comparator only (Apache-2.0).

## Consequences
- The plan's Toto "sample-pinball" loss has no samples to use (Toto 2.0 outputs 9 fixed quantiles).
  Replaced by the equivalent "pinball against teacher quantiles as mass-weighted pseudo-observations"
  (see `tinycast/distill/losses.py` and the journal).
