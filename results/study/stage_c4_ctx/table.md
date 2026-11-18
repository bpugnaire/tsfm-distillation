# stage_c4_ctx: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_ctx512` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ctx2048 | 0.462 | [0.417, 0.515] | 0.981 [0.941, 1.026] | 0.87 |
| c2_ctx512 | 0.471 | [0.424, 0.521] | – | 0.85 |
| kd_ctx2048_trunc | 0.486 | [0.441, 0.536] | 1.032 [0.982, 1.086] | 0.88 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 vs 512 (same windows, same teacher): `kd_ctx2048` vs `c2_ctx512` | 0.981 | [0.941, 1.026] | no detectable difference |
| random history truncation (p = 0.5) vs none: `kd_ctx2048_trunc` vs `kd_ctx2048` | 1.052 | [1.023, 1.083] | b better |
| context 2048 + truncation vs 512: `kd_ctx2048_trunc` vs `c2_ctx512` | 1.032 | [0.982, 1.086] | no detectable difference |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048` vs `zoo:toto-2.0-4m` | 0.960 | [0.920, 1.001] | no detectable difference |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048` vs `zoo:timestral-5m` | 1.071 | [1.037, 1.109] | b better |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ctx2048 | 0.677 | [0.644, 0.715] | 0.937 [0.912, 0.967] | 0.82 |
| kd_ctx2048_trunc | 0.693 | [0.658, 0.732] | 0.959 [0.936, 0.985] | 0.79 |
| c2_ctx512 | 0.723 | [0.683, 0.764] | – | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 vs 512 (same windows, same teacher): `kd_ctx2048` vs `c2_ctx512` | 0.937 | [0.912, 0.967] | a better |
| random history truncation (p = 0.5) vs none: `kd_ctx2048_trunc` vs `kd_ctx2048` | 1.023 | [1.005, 1.041] | b better |
| context 2048 + truncation vs 512: `kd_ctx2048_trunc` vs `c2_ctx512` | 0.959 | [0.936, 0.985] | a better |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048` vs `zoo:toto-2.0-4m` | 1.050 | [1.030, 1.073] | b better |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048` vs `zoo:timestral-5m` | 1.118 | [1.095, 1.143] | b better |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ctx2048 | 0.756 | [0.716, 0.796] | 0.992 [0.964, 1.020] | 0.66 |
| c2_ctx512 | 0.762 | [0.718, 0.805] | – | 0.63 |
| kd_ctx2048_trunc | 0.763 | [0.724, 0.804] | 1.002 [0.973, 1.033] | 0.60 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 vs 512 (same windows, same teacher): `kd_ctx2048` vs `c2_ctx512` | 0.992 | [0.964, 1.020] | no detectable difference |
| random history truncation (p = 0.5) vs none: `kd_ctx2048_trunc` vs `kd_ctx2048` | 1.010 | [0.996, 1.024] | no detectable difference |
| context 2048 + truncation vs 512: `kd_ctx2048_trunc` vs `c2_ctx512` | 1.002 | [0.973, 1.033] | no detectable difference |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048` vs `zoo:toto-2.0-4m` | 1.072 | [1.039, 1.105] | b better |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048` vs `zoo:timestral-5m` | 1.172 | [1.132, 1.215] | b better |
