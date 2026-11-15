# stage_c2: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd_old_28m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_v2_1m | 0.458 | [0.417, 0.503] | 0.927 [0.900, 0.953] | 0.83 |
| clite_ref | 0.460 | [0.415, 0.509] | 0.931 [0.899, 0.962] | 0.83 |
| kd_v2_300k | 0.487 | [0.441, 0.539] | 0.985 [0.959, 1.011] | 0.84 |
| kd_old_28m | 0.494 | [0.447, 0.550] | – | 0.85 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, ets 1.246

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-28M vs 209M labels (same old data): `kd_old_28m` vs `clite_ref` | 1.074 | [1.040, 1.112] | b better |
| data v2 vs old data (300k, same teacher): `kd_v2_300k` vs `kd_old_28m` | 0.985 | [0.959, 1.011] | no detectable difference |
| 1M vs 300k windows (data v2): `kd_v2_1m` vs `kd_v2_300k` | 0.941 | [0.917, 0.964] | a better |
| C2 best vs C-lite (overall): `kd_v2_1m` vs `clite_ref` | 0.995 | [0.960, 1.033] | no detectable difference |
| student (v2, 1M) vs Toto 2.0 4m zero-shot: `kd_v2_1m` vs `zoo:toto-2.0-4m` | 0.881 | [0.851, 0.913] | a better |
| student (v2, 1M) vs Timestral-5M zero-shot: `kd_v2_1m` vs `zoo:timestral-5m` | 0.931 | [0.901, 0.960] | a better |
| student (v2, 1M) vs its teacher Timestral-28M: `kd_v2_1m` vs `zoo:timestral-28m` | 1.025 | [1.003, 1.047] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_v2_1m | 0.697 | [0.665, 0.731] | 0.988 [0.952, 1.030] | 0.77 |
| clite_ref | 0.703 | [0.661, 0.749] | 0.997 [0.984, 1.012] | 0.79 |
| kd_old_28m | 0.705 | [0.666, 0.749] | – | 0.79 |
| kd_v2_300k | 0.742 | [0.705, 0.784] | 1.052 [1.024, 1.079] | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, timestral-28m 0.638, chronos-2-small 0.642, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-28M vs 209M labels (same old data): `kd_old_28m` vs `clite_ref` | 1.003 | [0.988, 1.016] | no detectable difference |
| data v2 vs old data (300k, same teacher): `kd_v2_300k` vs `kd_old_28m` | 1.052 | [1.024, 1.079] | b better |
| 1M vs 300k windows (data v2): `kd_v2_1m` vs `kd_v2_300k` | 0.939 | [0.917, 0.961] | a better |
| C2 best vs C-lite (overall): `kd_v2_1m` vs `clite_ref` | 0.992 | [0.955, 1.036] | no detectable difference |
| student (v2, 1M) vs Toto 2.0 4m zero-shot: `kd_v2_1m` vs `zoo:toto-2.0-4m` | 0.967 | [0.936, 0.999] | a better |
| student (v2, 1M) vs Timestral-5M zero-shot: `kd_v2_1m` vs `zoo:timestral-5m` | 1.057 | [1.030, 1.084] | b better |
| student (v2, 1M) vs its teacher Timestral-28M: `kd_v2_1m` vs `zoo:timestral-28m` | 1.093 | [1.064, 1.121] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_v2_1m | 0.675 | [0.621, 0.730] | 0.836 [0.798, 0.874] | 0.69 |
| kd_v2_300k | 0.755 | [0.710, 0.799] | 0.936 [0.912, 0.959] | 0.63 |
| clite_ref | 0.795 | [0.750, 0.840] | 0.985 [0.960, 1.009] | 0.76 |
| kd_old_28m | 0.807 | [0.775, 0.837] | – | 0.70 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: Timestral-28M vs 209M labels (same old data): `kd_old_28m` vs `clite_ref` | 1.016 | [0.991, 1.042] | no detectable difference |
| data v2 vs old data (300k, same teacher): `kd_v2_300k` vs `kd_old_28m` | 0.936 | [0.912, 0.959] | a better |
| 1M vs 300k windows (data v2): `kd_v2_1m` vs `kd_v2_300k` | 0.894 | [0.870, 0.916] | a better |
| C2 best vs C-lite (overall): `kd_v2_1m` vs `clite_ref` | 0.849 | [0.814, 0.887] | a better |
| student (v2, 1M) vs Toto 2.0 4m zero-shot: `kd_v2_1m` vs `zoo:toto-2.0-4m` | 0.952 | [0.924, 0.978] | a better |
| student (v2, 1M) vs Timestral-5M zero-shot: `kd_v2_1m` vs `zoo:timestral-5m` | 1.054 | [1.018, 1.090] | b better |
| student (v2, 1M) vs its teacher Timestral-28M: `kd_v2_1m` vs `zoo:timestral-28m` | 1.098 | [1.065, 1.134] | b better |
