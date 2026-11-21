# stage_c9_mt_ctx2048: relative WQL vs seasonal naive (95% CI); paired ratio vs `c6_single_2048` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c6_single_2048 | 0.452 | [0.411, 0.499] | – | 0.87 |
| mt_2048_1m | 0.465 | [0.422, 0.515] | 1.029 [1.015, 1.047] | 0.86 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 2048 steps, 1M windows: `mt_2048_1m` vs `c6_single_2048` | 1.029 | [1.015, 1.047] | b better |
| multi-token 2048 student vs Toto-2.0-4m (2048 context): `mt_2048_1m` vs `zoo:toto-2.0-4m` | 0.966 | [0.924, 1.013] | no detectable difference |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_2048_1m | 0.656 | [0.622, 0.693] | 0.993 [0.979, 1.005] | 0.79 |
| c6_single_2048 | 0.660 | [0.626, 0.700] | – | 0.81 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 2048 steps, 1M windows: `mt_2048_1m` vs `c6_single_2048` | 0.993 | [0.979, 1.005] | no detectable difference |
| multi-token 2048 student vs Toto-2.0-4m (2048 context): `mt_2048_1m` vs `zoo:toto-2.0-4m` | 1.016 | [0.994, 1.039] | no detectable difference |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c6_single_2048 | 0.724 | [0.681, 0.770] | – | 0.66 |
| mt_2048_1m | 0.728 | [0.683, 0.773] | 1.005 [0.998, 1.012] | 0.61 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 2048 steps, 1M windows: `mt_2048_1m` vs `c6_single_2048` | 1.005 | [0.998, 1.012] | no detectable difference |
| multi-token 2048 student vs Toto-2.0-4m (2048 context): `mt_2048_1m` vs `zoo:toto-2.0-4m` | 1.033 | [1.005, 1.062] | b better |
