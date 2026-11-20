# stage_c7_size: relative WQL vs seasonal naive (95% CI); paired ratio vs `m_28m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| m_209m | 0.453 | [0.408, 0.503] | 0.990 [0.959, 1.021] | 0.83 |
| l_209m | 0.455 | [0.411, 0.504] | 0.993 [0.960, 1.028] | 0.76 |
| m_28m | 0.458 | [0.417, 0.503] | – | 0.83 |
| l_28m | 0.467 | [0.428, 0.512] | 1.021 [1.001, 1.041] | 0.76 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher 209M vs 28M, student l (2.9M): `l_209m` vs `l_28m` | 0.973 | [0.944, 1.004] | no detectable difference |
| teacher 209M vs 28M, student m (0.95M): `m_209m` vs `m_28m` | 0.990 | [0.959, 1.021] | no detectable difference |
| student l vs m, Timestral-28M labels: `l_28m` vs `m_28m` | 1.021 | [1.001, 1.041] | b better |
| student l vs m, Timestral-209M labels: `l_209m` vs `m_209m` | 1.003 | [0.977, 1.031] | no detectable difference |
| l student (209M labels) vs Toto-2.0-4m zero-shot: `l_209m` vs `zoo:toto-2.0-4m` | 0.875 | [0.835, 0.919] | a better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| l_209m | 0.680 | [0.647, 0.715] | 0.975 [0.959, 0.992] | 0.78 |
| l_28m | 0.684 | [0.652, 0.719] | 0.981 [0.970, 0.993] | 0.78 |
| m_209m | 0.693 | [0.662, 0.729] | 0.995 [0.982, 1.008] | 0.78 |
| m_28m | 0.697 | [0.665, 0.731] | – | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher 209M vs 28M, student l (2.9M): `l_209m` vs `l_28m` | 0.994 | [0.982, 1.007] | no detectable difference |
| teacher 209M vs 28M, student m (0.95M): `m_209m` vs `m_28m` | 0.995 | [0.982, 1.008] | no detectable difference |
| student l vs m, Timestral-28M labels: `l_28m` vs `m_28m` | 0.981 | [0.970, 0.993] | a better |
| student l vs m, Timestral-209M labels: `l_209m` vs `m_209m` | 0.981 | [0.966, 0.995] | a better |
| l student (209M labels) vs Toto-2.0-4m zero-shot: `l_209m` vs `zoo:toto-2.0-4m` | 0.943 | [0.917, 0.969] | a better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| l_209m | 0.651 | [0.604, 0.697] | 0.964 [0.940, 0.989] | 0.70 |
| l_28m | 0.661 | [0.613, 0.709] | 0.979 [0.957, 1.000] | 0.63 |
| m_209m | 0.674 | [0.627, 0.721] | 0.998 [0.983, 1.016] | 0.70 |
| m_28m | 0.675 | [0.621, 0.730] | – | 0.69 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher 209M vs 28M, student l (2.9M): `l_209m` vs `l_28m` | 0.984 | [0.968, 1.004] | no detectable difference |
| teacher 209M vs 28M, student m (0.95M): `m_209m` vs `m_28m` | 0.998 | [0.983, 1.016] | no detectable difference |
| student l vs m, Timestral-28M labels: `l_28m` vs `m_28m` | 0.979 | [0.957, 1.000] | no detectable difference |
| student l vs m, Timestral-209M labels: `l_209m` vs `m_209m` | 0.966 | [0.952, 0.978] | a better |
| l student (209M labels) vs Toto-2.0-4m zero-shot: `l_209m` vs `zoo:toto-2.0-4m` | 0.918 | [0.892, 0.941] | a better |
