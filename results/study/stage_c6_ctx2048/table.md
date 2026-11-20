# stage_c6_ctx2048: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_ctx512` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ctx2048_1m | 0.452 | [0.411, 0.499] | 0.997 [0.952, 1.045] | 0.87 |
| c2_ctx512 | 0.453 | [0.412, 0.497] | – | 0.82 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 (+ truncation) vs 512, 1M windows, Timestral-28M: `kd_ctx2048_1m` vs `c2_ctx512` | 0.997 | [0.952, 1.045] | no detectable difference |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048_1m` vs `zoo:toto-2.0-4m` | 0.939 | [0.900, 0.980] | a better |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048_1m` vs `zoo:timestral-5m` | 1.048 | [1.010, 1.089] | b better |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ctx2048_1m | 0.660 | [0.626, 0.700] | 0.973 [0.938, 1.006] | 0.81 |
| c2_ctx512 | 0.679 | [0.644, 0.714] | – | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 (+ truncation) vs 512, 1M windows, Timestral-28M: `kd_ctx2048_1m` vs `c2_ctx512` | 0.973 | [0.938, 1.006] | no detectable difference |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048_1m` vs `zoo:toto-2.0-4m` | 1.024 | [1.003, 1.047] | b better |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048_1m` vs `zoo:timestral-5m` | 1.089 | [1.069, 1.112] | b better |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c2_ctx512 | 0.681 | [0.628, 0.735] | – | 0.69 |
| kd_ctx2048_1m | 0.724 | [0.681, 0.770] | 1.064 [1.035, 1.096] | 0.66 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| context 2048 (+ truncation) vs 512, 1M windows, Timestral-28M: `kd_ctx2048_1m` vs `c2_ctx512` | 1.064 | [1.035, 1.096] | b better |
| 2048 student vs Toto-2.0-4m (2048 context): `kd_ctx2048_1m` vs `zoo:toto-2.0-4m` | 1.028 | [1.002, 1.053] | b better |
| 2048 student vs Timestral-5M (2048 context): `kd_ctx2048_1m` vs `zoo:timestral-5m` | 1.123 | [1.095, 1.157] | b better |
