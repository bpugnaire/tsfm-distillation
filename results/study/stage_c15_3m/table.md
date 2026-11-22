# stage_c15_3m: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd5m_1m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| gt_3m | 0.392 | [0.351, 0.439] | 0.783 [0.730, 0.837] | 0.83 |
| gt_1m | 0.396 | [0.350, 0.448] | 0.791 [0.733, 0.847] | 0.80 |
| kd5m_3m | 0.481 | [0.436, 0.533] | 0.962 [0.945, 0.978] | 0.83 |
| kd5m_1m | 0.500 | [0.453, 0.556] | – | 0.84 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 3M windows: Timestral-5M labels vs real future: `kd5m_3m` vs `gt_3m` | 1.228 | [1.149, 1.318] | b better |
| teacher labels: 3M vs 1M windows: `kd5m_3m` vs `kd5m_1m` | 0.962 | [0.945, 0.978] | a better |
| real future: 3M vs 1M windows: `gt_3m` vs `gt_1m` | 0.991 | [0.967, 1.014] | no detectable difference |
| 3M-window student vs Toto-2.0-4m zero-shot: `kd5m_3m` vs `zoo:toto-2.0-4m` | 0.927 | [0.890, 0.966] | a better |
| 3M-window student vs its teacher Timestral-5M: `kd5m_3m` vs `zoo:timestral-5m` | 0.979 | [0.960, 0.997] | a better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_3m | 0.676 | [0.644, 0.712] | 0.986 [0.972, 1.000] | 0.80 |
| kd5m_1m | 0.686 | [0.654, 0.723] | – | 0.80 |
| gt_3m | 0.691 | [0.653, 0.736] | 1.007 [0.984, 1.034] | 0.76 |
| gt_1m | 0.698 | [0.660, 0.743] | 1.017 [0.994, 1.044] | 0.76 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 3M windows: Timestral-5M labels vs real future: `kd5m_3m` vs `gt_3m` | 0.978 | [0.955, 1.000] | a better |
| teacher labels: 3M vs 1M windows: `kd5m_3m` vs `kd5m_1m` | 0.986 | [0.972, 1.000] | no detectable difference |
| real future: 3M vs 1M windows: `gt_3m` vs `gt_1m` | 0.990 | [0.973, 1.007] | no detectable difference |
| 3M-window student vs Toto-2.0-4m zero-shot: `kd5m_3m` vs `zoo:toto-2.0-4m` | 0.939 | [0.915, 0.963] | a better |
| 3M-window student vs its teacher Timestral-5M: `kd5m_3m` vs `zoo:timestral-5m` | 1.025 | [1.008, 1.045] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_3m | 0.651 | [0.603, 0.697] | 0.927 [0.905, 0.946] | 0.67 |
| kd5m_1m | 0.702 | [0.662, 0.744] | – | 0.66 |
| gt_1m | 0.736 | [0.690, 0.781] | 1.047 [1.031, 1.065] | 0.66 |
| gt_3m | 0.737 | [0.694, 0.778] | 1.050 [1.024, 1.073] | 0.72 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 3M windows: Timestral-5M labels vs real future: `kd5m_3m` vs `gt_3m` | 0.883 | [0.853, 0.912] | a better |
| teacher labels: 3M vs 1M windows: `kd5m_3m` vs `kd5m_1m` | 0.927 | [0.905, 0.946] | a better |
| real future: 3M vs 1M windows: `gt_3m` vs `gt_1m` | 1.002 | [0.982, 1.022] | no detectable difference |
| 3M-window student vs Toto-2.0-4m zero-shot: `kd5m_3m` vs `zoo:toto-2.0-4m` | 0.918 | [0.893, 0.940] | a better |
| 3M-window student vs its teacher Timestral-5M: `kd5m_3m` vs `zoo:timestral-5m` | 1.016 | [0.997, 1.033] | no detectable difference |
