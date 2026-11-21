# stage_c10_v3: relative WQL vs seasonal naive (95% CI); paired ratio vs `c8_mt_v2_300k` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c8_mt_v2_300k | 0.475 | [0.429, 0.526] | – | 0.83 |
| mt_v3_300k | 0.483 | [0.435, 0.540] | 1.016 [0.997, 1.036] | 0.82 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| data v3 vs v2 (multi-token, 300k windows, same labels teacher): `mt_v3_300k` vs `c8_mt_v2_300k` | 1.016 | [0.997, 1.036] | no detectable difference |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_v3_300k | 0.719 | [0.684, 0.754] | 1.000 [0.987, 1.014] | 0.78 |
| c8_mt_v2_300k | 0.719 | [0.683, 0.755] | – | 0.79 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| data v3 vs v2 (multi-token, 300k windows, same labels teacher): `mt_v3_300k` vs `c8_mt_v2_300k` | 1.000 | [0.987, 1.014] | no detectable difference |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c8_mt_v2_300k | 0.733 | [0.685, 0.780] | – | 0.70 |
| mt_v3_300k | 0.754 | [0.711, 0.798] | 1.030 [1.016, 1.043] | 0.73 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| data v3 vs v2 (multi-token, 300k windows, same labels teacher): `mt_v3_300k` vs `c8_mt_v2_300k` | 1.030 | [1.016, 1.043] | b better |
