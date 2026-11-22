# stage_c16_hybrid: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd5m_1m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| hyb5m_1m | 0.390 | [0.347, 0.438] | 0.779 [0.724, 0.832] | 0.81 |
| gt_1m | 0.396 | [0.350, 0.448] | 0.791 [0.733, 0.847] | 0.80 |
| kd5m_1m | 0.500 | [0.453, 0.556] | – | 0.84 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| hybrid (real future on synthetic, teacher on real) vs pure teacher labels: `hyb5m_1m` vs `kd5m_1m` | 0.779 | [0.724, 0.832] | a better |
| hybrid vs pure real future: `hyb5m_1m` vs `gt_1m` | 0.985 | [0.965, 1.005] | no detectable difference |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_1m | 0.686 | [0.654, 0.723] | – | 0.80 |
| gt_1m | 0.698 | [0.660, 0.743] | 1.017 [0.994, 1.044] | 0.76 |
| hyb5m_1m | 0.701 | [0.669, 0.738] | 1.022 [1.008, 1.035] | 0.80 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| hybrid (real future on synthetic, teacher on real) vs pure teacher labels: `hyb5m_1m` vs `kd5m_1m` | 1.022 | [1.008, 1.035] | b better |
| hybrid vs pure real future: `hyb5m_1m` vs `gt_1m` | 1.005 | [0.978, 1.030] | no detectable difference |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd5m_1m | 0.702 | [0.662, 0.744] | – | 0.66 |
| hyb5m_1m | 0.705 | [0.660, 0.749] | 1.004 [0.986, 1.023] | 0.68 |
| gt_1m | 0.736 | [0.690, 0.781] | 1.047 [1.031, 1.065] | 0.66 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| hybrid (real future on synthetic, teacher on real) vs pure teacher labels: `hyb5m_1m` vs `kd5m_1m` | 1.004 | [0.986, 1.023] | no detectable difference |
| hybrid vs pure real future: `hyb5m_1m` vs `gt_1m` | 0.959 | [0.942, 0.978] | a better |
