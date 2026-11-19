# stage_c5_209m: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_28m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_209m_1m_full | 0.453 | [0.408, 0.503] | 0.990 [0.959, 1.021] | 0.83 |
| c2_28m | 0.458 | [0.417, 0.503] | – | 0.83 |
| kd_209m_1m | 0.506 | [0.461, 0.555] | 1.106 [1.066, 1.149] | 0.85 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-209M vs 28M, full 50k steps for both: `kd_209m_1m_full` vs `c2_28m` | 0.990 | [0.959, 1.021] | no detectable difference |
| full schedule vs early-stopped (209M labels): `kd_209m_1m_full` vs `kd_209m_1m` | 0.895 | [0.865, 0.925] | a better |
| teacher: Timestral-209M vs 28M (same 1M windows, 512 steps): `kd_209m_1m` vs `c2_28m` | 1.106 | [1.066, 1.149] | b better |
| 209M-taught student vs Toto-2.0-4m zero-shot: `kd_209m_1m` vs `zoo:toto-2.0-4m` | 0.975 | [0.935, 1.018] | no detectable difference |
| 209M-taught student vs Timestral-5M zero-shot: `kd_209m_1m` vs `zoo:timestral-5m` | 1.030 | [0.974, 1.083] | no detectable difference |
| student vs its teacher Timestral-209M: `kd_209m_1m` vs `zoo:timestral-209m` | 1.259 | [1.210, 1.313] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_209m_1m_full | 0.693 | [0.662, 0.729] | 0.995 [0.982, 1.008] | 0.78 |
| c2_28m | 0.697 | [0.665, 0.731] | – | 0.77 |
| kd_209m_1m | 0.713 | [0.682, 0.747] | 1.023 [1.005, 1.045] | 0.80 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-209M vs 28M, full 50k steps for both: `kd_209m_1m_full` vs `c2_28m` | 0.995 | [0.982, 1.008] | no detectable difference |
| full schedule vs early-stopped (209M labels): `kd_209m_1m_full` vs `kd_209m_1m` | 0.972 | [0.956, 0.989] | a better |
| teacher: Timestral-209M vs 28M (same 1M windows, 512 steps): `kd_209m_1m` vs `c2_28m` | 1.023 | [1.005, 1.045] | b better |
| 209M-taught student vs Toto-2.0-4m zero-shot: `kd_209m_1m` vs `zoo:toto-2.0-4m` | 0.990 | [0.963, 1.017] | no detectable difference |
| 209M-taught student vs Timestral-5M zero-shot: `kd_209m_1m` vs `zoo:timestral-5m` | 1.081 | [1.056, 1.107] | b better |
| student vs its teacher Timestral-209M: `kd_209m_1m` vs `zoo:timestral-209m` | 1.136 | [1.106, 1.166] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_209m_1m_full | 0.674 | [0.627, 0.721] | 0.998 [0.983, 1.016] | 0.70 |
| c2_28m | 0.675 | [0.621, 0.730] | – | 0.69 |
| kd_209m_1m | 0.707 | [0.660, 0.755] | 1.048 [1.028, 1.067] | 0.76 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-209M vs 28M, full 50k steps for both: `kd_209m_1m_full` vs `c2_28m` | 0.998 | [0.983, 1.016] | no detectable difference |
| full schedule vs early-stopped (209M labels): `kd_209m_1m_full` vs `kd_209m_1m` | 0.953 | [0.943, 0.963] | a better |
| teacher: Timestral-209M vs 28M (same 1M windows, 512 steps): `kd_209m_1m` vs `c2_28m` | 1.048 | [1.028, 1.067] | b better |
| 209M-taught student vs Toto-2.0-4m zero-shot: `kd_209m_1m` vs `zoo:toto-2.0-4m` | 0.998 | [0.976, 1.020] | no detectable difference |
| 209M-taught student vs Timestral-5M zero-shot: `kd_209m_1m` vs `zoo:timestral-5m` | 1.104 | [1.067, 1.140] | b better |
| student vs its teacher Timestral-209M: `kd_209m_1m` vs `zoo:timestral-209m` | 1.152 | [1.118, 1.186] | b better |
