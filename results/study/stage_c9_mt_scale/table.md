# stage_c9_mt_scale: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_single_1m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_l_1m_209m | 0.406 | [0.368, 0.447] | 0.886 [0.861, 0.911] | 0.85 |
| c7_single_l_209m | 0.455 | [0.411, 0.504] | 0.993 [0.960, 1.028] | 0.76 |
| c2_single_1m | 0.458 | [0.417, 0.503] | – | 0.83 |
| mt_1m | 0.469 | [0.429, 0.515] | 1.024 [1.005, 1.045] | 0.84 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 1M windows, 512 steps, 28M labels: `mt_1m` vs `c2_single_1m` | 1.024 | [1.005, 1.045] | b better |
| multi-token vs single token, student l, 209M labels: `mt_l_1m_209m` vs `c7_single_l_209m` | 0.893 | [0.865, 0.919] | a better |
| multi-token: student l + 209M labels vs m + 28M labels: `mt_l_1m_209m` vs `mt_1m` | 0.865 | [0.839, 0.890] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `mt_1m` vs `zoo:toto-2.0-4m` | 0.903 | [0.875, 0.931] | a better |
| multi-token l student vs Toto-2.0-4m zero-shot: `mt_l_1m_209m` vs `zoo:toto-2.0-4m` | 0.781 | [0.753, 0.812] | a better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_l_1m_209m | 0.669 | [0.633, 0.707] | 0.959 [0.942, 0.977] | 0.79 |
| c7_single_l_209m | 0.680 | [0.647, 0.715] | 0.975 [0.959, 0.992] | 0.78 |
| mt_1m | 0.687 | [0.653, 0.723] | 0.985 [0.971, 0.998] | 0.79 |
| c2_single_1m | 0.697 | [0.665, 0.731] | – | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 1M windows, 512 steps, 28M labels: `mt_1m` vs `c2_single_1m` | 0.985 | [0.971, 0.998] | a better |
| multi-token vs single token, student l, 209M labels: `mt_l_1m_209m` vs `c7_single_l_209m` | 0.983 | [0.971, 0.995] | a better |
| multi-token: student l + 209M labels vs m + 28M labels: `mt_l_1m_209m` vs `mt_1m` | 0.973 | [0.957, 0.990] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `mt_1m` vs `zoo:toto-2.0-4m` | 0.953 | [0.920, 0.985] | a better |
| multi-token l student vs Toto-2.0-4m zero-shot: `mt_l_1m_209m` vs `zoo:toto-2.0-4m` | 0.928 | [0.899, 0.955] | a better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c7_single_l_209m | 0.651 | [0.604, 0.697] | 0.964 [0.940, 0.989] | 0.70 |
| mt_l_1m_209m | 0.665 | [0.619, 0.709] | 0.985 [0.963, 1.010] | 0.63 |
| c2_single_1m | 0.675 | [0.621, 0.730] | – | 0.69 |
| mt_1m | 0.705 | [0.662, 0.747] | 1.045 [1.022, 1.071] | 0.65 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| multi-token vs single token, 1M windows, 512 steps, 28M labels: `mt_1m` vs `c2_single_1m` | 1.045 | [1.022, 1.071] | b better |
| multi-token vs single token, student l, 209M labels: `mt_l_1m_209m` vs `c7_single_l_209m` | 1.022 | [0.996, 1.045] | no detectable difference |
| multi-token: student l + 209M labels vs m + 28M labels: `mt_l_1m_209m` vs `mt_1m` | 0.943 | [0.924, 0.961] | a better |
| multi-token student vs Toto-2.0-4m zero-shot: `mt_1m` vs `zoo:toto-2.0-4m` | 0.995 | [0.969, 1.020] | no detectable difference |
| multi-token l student vs Toto-2.0-4m zero-shot: `mt_l_1m_209m` vs `zoo:toto-2.0-4m` | 0.938 | [0.910, 0.965] | a better |
