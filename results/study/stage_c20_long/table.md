# stage_c20_long: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd5m_3m` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_3m | 0.467 | [0.421, 0.518] | – | 0.82 |
| long_1m | 0.555 | [0.500, 0.620] | 1.190 [1.126, 1.260] | 0.83 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| long-horizon student (2048 -> 1024, 1M windows) vs site student (512 -> 64, 3M windows), first 64 steps: `long_1m` vs `kd5m_3m` | 1.190 | [1.126, 1.260] | b better |
| long-horizon student vs Toto-2.0-4m: `long_1m` vs `zoo:toto-2.0-4m` | 1.153 | [1.101, 1.212] | b better |
| long-horizon student vs its teacher Timestral-5M: `long_1m` vs `zoo:timestral-5m` | 1.287 | [1.224, 1.355] | b better |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_3m | 0.659 | [0.624, 0.694] | – | 0.80 |
| long_1m | 0.695 | [0.661, 0.734] | 1.055 [1.029, 1.086] | 0.80 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| long-horizon student (2048 -> 1024, 1M windows) vs site student (512 -> 64, 3M windows), first 64 steps: `long_1m` vs `kd5m_3m` | 1.055 | [1.029, 1.086] | b better |
| long-horizon student vs Toto-2.0-4m: `long_1m` vs `zoo:toto-2.0-4m` | 1.077 | [1.054, 1.103] | b better |
| long-horizon student vs its teacher Timestral-5M: `long_1m` vs `zoo:timestral-5m` | 1.147 | [1.120, 1.175] | b better |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_3m | 0.657 | [0.610, 0.703] | – | 0.67 |
| long_1m | 0.790 | [0.734, 0.846] | 1.204 [1.163, 1.252] | 0.65 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| long-horizon student (2048 -> 1024, 1M windows) vs site student (512 -> 64, 3M windows), first 64 steps: `long_1m` vs `kd5m_3m` | 1.204 | [1.163, 1.252] | b better |
| long-horizon student vs Toto-2.0-4m: `long_1m` vs `zoo:toto-2.0-4m` | 1.121 | [1.088, 1.156] | b better |
| long-horizon student vs its teacher Timestral-5M: `long_1m` vs `zoo:timestral-5m` | 1.225 | [1.177, 1.276] | b better |
