# stage_c_lite: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd_mix_m_300k` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mix_m_300k | 0.460 | [0.415, 0.509] | – | 0.83 |
| kd_mix_s_300k | 0.499 | [0.450, 0.555] | 1.085 [1.059, 1.113] | 0.88 |
| gt_mix_m_300k | 0.540 | [0.485, 0.605] | 1.174 [1.131, 1.221] | 0.90 |
| kd_mix_m_30k | 0.620 | [0.550, 0.701] | 1.348 [1.283, 1.423] | 0.93 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| stage_a_best | 1.467 | [1.228, 1.788] | 3.188 [2.740, 3.726] | 0.84 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, ets 1.246

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth at scale (m, 300k): `kd_mix_m_300k` vs `gt_mix_m_300k` | 0.852 | [0.819, 0.884] | a better |
| student size: 0.95M vs 0.22M (300k): `kd_mix_m_300k` vs `kd_mix_s_300k` | 0.922 | [0.898, 0.944] | a better |
| data: 300k vs 30k windows (m): `kd_mix_m_300k` vs `kd_mix_m_30k` | 0.742 | [0.703, 0.780] | a better |
| C-lite recipe vs best Stage-A student: `kd_mix_m_300k` vs `stage_a_best` | 0.314 | [0.268, 0.365] | a better |
| student m vs Toto-2.0-4m zero-shot: `kd_mix_m_300k` vs `zoo:toto-2.0-4m` | 0.886 | [0.845, 0.930] | a better |
| student m vs Timestral-5M zero-shot: `kd_mix_m_300k` vs `zoo:timestral-5m` | 0.936 | [0.886, 0.984] | a better |
| student m vs its teacher Timestral-209M: `kd_mix_m_300k` vs `zoo:timestral-209m` | 1.144 | [1.107, 1.181] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mix_m_300k | 0.703 | [0.661, 0.749] | – | 0.79 |
| kd_mix_s_300k | 0.726 | [0.680, 0.775] | 1.033 [1.017, 1.049] | 0.80 |
| gt_mix_m_300k | 0.729 | [0.671, 0.790] | 1.037 [1.001, 1.072] | 0.80 |
| stage_a_best | 0.734 | [0.693, 0.777] | 1.044 [1.021, 1.071] | 0.83 |
| kd_mix_m_30k | 0.775 | [0.731, 0.825] | 1.103 [1.077, 1.131] | 0.82 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, timestral-28m 0.638, chronos-2-small 0.642, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth at scale (m, 300k): `kd_mix_m_300k` vs `gt_mix_m_300k` | 0.964 | [0.933, 0.999] | a better |
| student size: 0.95M vs 0.22M (300k): `kd_mix_m_300k` vs `kd_mix_s_300k` | 0.968 | [0.953, 0.983] | a better |
| data: 300k vs 30k windows (m): `kd_mix_m_300k` vs `kd_mix_m_30k` | 0.907 | [0.884, 0.928] | a better |
| C-lite recipe vs best Stage-A student: `kd_mix_m_300k` vs `stage_a_best` | 0.958 | [0.934, 0.979] | a better |
| student m vs Toto-2.0-4m zero-shot: `kd_mix_m_300k` vs `zoo:toto-2.0-4m` | 0.975 | [0.945, 1.013] | no detectable difference |
| student m vs Timestral-5M zero-shot: `kd_mix_m_300k` vs `zoo:timestral-5m` | 1.065 | [1.031, 1.103] | b better |
| student m vs its teacher Timestral-209M: `kd_mix_m_300k` vs `zoo:timestral-209m` | 1.120 | [1.083, 1.163] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_mix_m_300k | 0.795 | [0.750, 0.840] | – | 0.76 |
| kd_mix_s_300k | 0.853 | [0.818, 0.887] | 1.074 [1.043, 1.107] | 0.75 |
| gt_mix_m_300k | 0.904 | [0.865, 0.941] | 1.138 [1.114, 1.164] | 0.78 |
| kd_mix_m_30k | 0.910 | [0.870, 0.951] | 1.145 [1.097, 1.193] | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |
| stage_a_best | 1.035 | [0.971, 1.109] | 1.302 [1.247, 1.369] | 0.71 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth at scale (m, 300k): `kd_mix_m_300k` vs `gt_mix_m_300k` | 0.879 | [0.859, 0.898] | a better |
| student size: 0.95M vs 0.22M (300k): `kd_mix_m_300k` vs `kd_mix_s_300k` | 0.931 | [0.903, 0.959] | a better |
| data: 300k vs 30k windows (m): `kd_mix_m_300k` vs `kd_mix_m_30k` | 0.873 | [0.838, 0.911] | a better |
| C-lite recipe vs best Stage-A student: `kd_mix_m_300k` vs `stage_a_best` | 0.768 | [0.731, 0.802] | a better |
| student m vs Toto-2.0-4m zero-shot: `kd_mix_m_300k` vs `zoo:toto-2.0-4m` | 1.121 | [1.077, 1.168] | b better |
| student m vs Timestral-5M zero-shot: `kd_mix_m_300k` vs `zoo:timestral-5m` | 1.240 | [1.176, 1.299] | b better |
| student m vs its teacher Timestral-209M: `kd_mix_m_300k` vs `zoo:timestral-209m` | 1.294 | [1.238, 1.348] | b better |
