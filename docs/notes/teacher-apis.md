# Teacher / comparator APIs (verified against installed sources, 2026-11-12)

All installed together in one Python 3.12 env (`uv sync --extra teachers`).

## Toto 2.0 — `toto-2==2.0.0`
- Install: `toto-2` (the HF card says `pip install toto-models`, a meta-package that requires `toto-2`).
  **Not** `toto-ts` — that is Toto 1.0 (pins numpy 1.26.4 / huggingface-hub<1).
- Import: `from toto2 import Toto2Model`; `Toto2Model.from_pretrained("Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}")`.
- Call: `model.forecast({"target": (B,V,T), "target_mask": bool (B,V,T), "series_ids": long (B,V)}, horizon, has_missing_values=...)`.
- Output: `(9, B, V, H)` quantiles at fixed knots **0.1…0.9**. `QuantileKnotsOutputHead.forward(x, q=None)` ignores `q`:
  no arbitrary levels, **no samples**. Quantiles are sorted internally (`.sort(dim=0)`).
- Internal scaler: causal patched std scaler + `asinh`. Decoding: next-patch, block decoding with median feedback for long horizons.
- **Our conversion:** 0.1…0.9 native; 0.01/0.05/0.95/0.99 by IQF exponential tails (Park et al. 2021), the same rule t0 uses
  (`tinycast/models/base.py::extend_tails_exponential`). Flagged via `native_levels`; KD can mask these levels.
- Deps gotchas: imports gluonts → lightning → `pkg_resources` ⇒ pin `setuptools<81`.

## t0 — `tfc-t0[torch]==0.6.1`
- Import name is **`t0`**: `from t0 import T0Forecaster`; `T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma")`.
- Checkpoints: t0-alpha (102M, 2026-06-09), t0-beta (256M, 2026-09-16), **t0-gamma** (256M, 2026-10-12). Not gated.
- Call: `model.predict(np.ndarray (B,T), horizon, quantile_levels=...)` → `Forecast` with `.quantiles` `(B, H, Q)`; NaN = missing.
- Native: 21 levels 0.01, 0.05, 0.1, 0.15 … 0.95, 0.99 → **our 13 levels are all native**. Off-grid levels are interpolated;
  beyond 0.01/0.99 IQF exponential tails.
- Post-processing heuristics (`symmetric_averaging`, `clip_nonnegative`, `snap_integers`, `trim_leading_missing`) are OFF by default;
  the authors recommend enabling them. **We keep them off** (raw model, comparable with the others). Worth an ablation.
- Deps gotcha: annotations like `Float[ArrayT, ...]` with a TypeVar need **jaxtyping ≥ 0.3** (0.2.29 raises at import).

## Chronos-2 — `chronos-forecasting==2.3.2` (comparator only)
- `from chronos import Chronos2Pipeline`; `Chronos2Pipeline.from_pretrained("amazon/chronos-2" | "autogluon/chronos-2-small")`.
- `predict_quantiles(tensor (B,1,T), prediction_length, quantile_levels)` → list of `(V, H, Q)`.
- Trained levels: small = our 13 levels exactly; base = 21 levels (0.01…0.99). **Requests outside the trained range are
  clamped to the edge with only a log warning** — we only request trained levels.
- **Raw quantiles cross** on 0.4% (small) / 1.6% (base) of gallery cells (worst: 3.3% of the quantile span). We sort them
  (monotone rearrangement; never increases pinball loss) and record the rate in each result's `fallbacks`.

## Timestral (comparator) — `timestral` from GitHub, pinned commit a534bdb
- Install: `git+https://github.com/mistralai/timestral.git@<sha>` (not on PyPI). Code Apache-2.0; weights Modified MIT (revenue clause;
  owner confirmed eligibility 2026-11-13).
- `from timestral import TimestralForecaster`; `TimestralForecaster.from_pretrained("mistralai/Timestral-{5M,28M,123M,209M}", device=...)`.
- `forecast(context (B,T) with NaN, prediction_length)` → `Forecast.quantiles` `(B, V, H, Q)`; 21 native levels → our 13 all native.
- **Defaults differ from t0:** `symmetric_averaging=True` (also forecasts −context and averages; 2× cost) and `clip_nonnegative=True`.
  We pass both `False` (raw-model policy, like t0). Ablation candidate.
- Block-autoregressive decoding with median feedback beyond 32 patches (1024 steps); horizon 64 is one shot.
- **Does not run on Apple MPS** (uses float64 internally; MPS has no float64). CPU throughput (M1 Pro, batch 128):
  209M ≈ 124 windows/s, 28M ≈ 524 windows/s; faster than Toto-313m (54) and t0-gamma (83) on CPU.
