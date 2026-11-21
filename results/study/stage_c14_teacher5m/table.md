# stage_c14_teacher5m: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd28m_1m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd28m_1m | 0.469 | [0.429, 0.515] | – | 0.84 |
| kd5m_1m | 0.500 | [0.453, 0.556] | 1.067 [1.040, 1.098] | 0.84 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher Timestral-5M vs 28M (same 1M windows, multi-token student): `kd5m_1m` vs `kd28m_1m` | 1.067 | [1.040, 1.098] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_1m | 0.686 | [0.654, 0.723] | 0.999 [0.984, 1.016] | 0.80 |
| kd28m_1m | 0.687 | [0.653, 0.723] | – | 0.79 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher Timestral-5M vs 28M (same 1M windows, multi-token student): `kd5m_1m` vs `kd28m_1m` | 0.999 | [0.984, 1.016] | no detectable difference |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_1m | 0.702 | [0.662, 0.744] | 0.996 [0.986, 1.008] | 0.66 |
| kd28m_1m | 0.705 | [0.662, 0.747] | – | 0.65 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher Timestral-5M vs 28M (same 1M windows, multi-token student): `kd5m_1m` vs `kd28m_1m` | 0.996 | [0.986, 1.008] | no detectable difference |
