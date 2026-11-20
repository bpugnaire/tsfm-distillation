# stage_c8_mt: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_single_300k` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mt_300k | 0.475 | [0.429, 0.526] | 0.977 [0.957, 0.999] | 0.83 |
| c2_single_300k | 0.487 | [0.441, 0.539] | – | 0.84 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token decoder vs single query token (same 300k windows, same labels): `kd_mt_300k` vs `c2_single_300k` | 0.977 | [0.957, 0.999] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `kd_mt_300k` vs `zoo:toto-2.0-4m` | 0.915 | [0.876, 0.958] | a better |
| multi-token student vs Timestral-5M zero-shot: `kd_mt_300k` vs `zoo:timestral-5m` | 0.967 | [0.929, 1.005] | no detectable difference |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mt_300k | 0.719 | [0.683, 0.755] | 0.968 [0.949, 0.989] | 0.79 |
| c2_single_300k | 0.742 | [0.705, 0.784] | – | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token decoder vs single query token (same 300k windows, same labels): `kd_mt_300k` vs `c2_single_300k` | 0.968 | [0.949, 0.989] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `kd_mt_300k` vs `zoo:toto-2.0-4m` | 0.997 | [0.963, 1.030] | no detectable difference |
| multi-token student vs Timestral-5M zero-shot: `kd_mt_300k` vs `zoo:timestral-5m` | 1.089 | [1.059, 1.117] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mt_300k | 0.733 | [0.685, 0.780] | 0.970 [0.950, 0.990] | 0.70 |
| c2_single_300k | 0.755 | [0.710, 0.799] | – | 0.63 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token decoder vs single query token (same 300k windows, same labels): `kd_mt_300k` vs `c2_single_300k` | 0.970 | [0.950, 0.990] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `kd_mt_300k` vs `zoo:toto-2.0-4m` | 1.034 | [1.002, 1.064] | b better |
| multi-token student vs Timestral-5M zero-shot: `kd_mt_300k` vs `zoo:timestral-5m` | 1.144 | [1.106, 1.181] | b better |
