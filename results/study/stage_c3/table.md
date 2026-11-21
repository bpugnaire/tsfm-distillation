# stage_c3: relative WQL vs seasonal naive (95% CI); paired ratio vs `c2_ref` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_toto_1m | 0.451 | [0.414, 0.493] | 0.985 [0.963, 1.008] | 0.75 |
| kd_ens3_1m | 0.457 | [0.416, 0.504] | 0.999 [0.977, 1.022] | 0.76 |
| c2_ref | 0.458 | [0.417, 0.503] | – | 0.83 |
| kd_t0_1m | 0.512 | [0.466, 0.566] | 1.118 [1.080, 1.161] | 0.86 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: t0-gamma vs Timestral-28M (same 1M windows): `kd_t0_1m` vs `c2_ref` | 1.118 | [1.080, 1.161] | b better |
| teacher: Toto-2.0-313m vs Timestral-28M (same 1M windows): `kd_toto_1m` vs `c2_ref` | 0.985 | [0.963, 1.008] | no detectable difference |
| ensemble of 3 teachers vs Timestral-28M: `kd_ens3_1m` vs `c2_ref` | 0.999 | [0.977, 1.022] | no detectable difference |
| ensemble student vs Toto-2.0-4m zero-shot: `kd_ens3_1m` vs `zoo:toto-2.0-4m` | 0.881 | [0.844, 0.921] | a better |
| ensemble student vs Timestral-5M zero-shot: `kd_ens3_1m` vs `zoo:timestral-5m` | 0.930 | [0.897, 0.963] | a better |
| Toto-taught student vs Toto-2.0-4m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-4m` | 0.868 | [0.840, 0.899] | a better |
| Toto-taught student vs Toto-2.0-22m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-22m` | 0.949 | [0.916, 0.980] | a better |
| Toto-taught student vs its teacher, Toto-2.0-313m: `kd_toto_1m` vs `zoo:toto-2.0-313m` | 1.027 | [1.004, 1.050] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ens3_1m | 0.688 | [0.654, 0.726] | 0.986 [0.965, 1.007] | 0.67 |
| c2_ref | 0.697 | [0.665, 0.731] | – | 0.77 |
| kd_t0_1m | 0.697 | [0.662, 0.738] | 1.001 [0.974, 1.027] | 0.78 |
| kd_toto_1m | 0.717 | [0.683, 0.755] | 1.028 [1.009, 1.049] | 0.68 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: t0-gamma vs Timestral-28M (same 1M windows): `kd_t0_1m` vs `c2_ref` | 1.001 | [0.974, 1.027] | no detectable difference |
| teacher: Toto-2.0-313m vs Timestral-28M (same 1M windows): `kd_toto_1m` vs `c2_ref` | 1.028 | [1.009, 1.049] | b better |
| ensemble of 3 teachers vs Timestral-28M: `kd_ens3_1m` vs `c2_ref` | 0.986 | [0.965, 1.007] | no detectable difference |
| ensemble student vs Toto-2.0-4m zero-shot: `kd_ens3_1m` vs `zoo:toto-2.0-4m` | 0.954 | [0.930, 0.982] | a better |
| ensemble student vs Timestral-5M zero-shot: `kd_ens3_1m` vs `zoo:timestral-5m` | 1.042 | [1.020, 1.067] | b better |
| Toto-taught student vs Toto-2.0-4m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-4m` | 0.994 | [0.969, 1.020] | no detectable difference |
| Toto-taught student vs Toto-2.0-22m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-22m` | 1.052 | [1.030, 1.077] | b better |
| Toto-taught student vs its teacher, Toto-2.0-313m: `kd_toto_1m` vs `zoo:toto-2.0-313m` | 1.075 | [1.049, 1.102] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_toto_1m | 0.655 | [0.608, 0.702] | 0.970 [0.952, 0.989] | 0.62 |
| c2_ref | 0.675 | [0.621, 0.730] | – | 0.69 |
| kd_ens3_1m | 0.694 | [0.649, 0.739] | 1.029 [1.009, 1.048] | 0.61 |
| kd_t0_1m | 0.711 | [0.665, 0.756] | 1.053 [1.034, 1.074] | 0.73 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher: t0-gamma vs Timestral-28M (same 1M windows): `kd_t0_1m` vs `c2_ref` | 1.053 | [1.034, 1.074] | b better |
| teacher: Toto-2.0-313m vs Timestral-28M (same 1M windows): `kd_toto_1m` vs `c2_ref` | 0.970 | [0.952, 0.989] | a better |
| ensemble of 3 teachers vs Timestral-28M: `kd_ens3_1m` vs `c2_ref` | 1.029 | [1.009, 1.048] | b better |
| ensemble student vs Toto-2.0-4m zero-shot: `kd_ens3_1m` vs `zoo:toto-2.0-4m` | 0.980 | [0.953, 1.005] | no detectable difference |
| ensemble student vs Timestral-5M zero-shot: `kd_ens3_1m` vs `zoo:timestral-5m` | 1.084 | [1.052, 1.115] | b better |
| Toto-taught student vs Toto-2.0-4m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-4m` | 0.924 | [0.894, 0.951] | a better |
| Toto-taught student vs Toto-2.0-22m zero-shot: `kd_toto_1m` vs `zoo:toto-2.0-22m` | 1.025 | [1.000, 1.051] | b better |
| Toto-taught student vs its teacher, Toto-2.0-313m: `kd_toto_1m` vs `zoo:toto-2.0-313m` | 1.057 | [1.033, 1.083] | b better |
