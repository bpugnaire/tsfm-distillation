# stage_c13_gt_vs_kd: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd_1m` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| gt_1m | 0.396 | [0.350, 0.448] | 0.844 [0.791, 0.894] | 0.80 |
| mix_1m | 0.404 | [0.362, 0.452] | 0.862 [0.824, 0.899] | 0.84 |
| gt_300k | 0.434 | [0.386, 0.489] | 0.925 [0.877, 0.972] | 0.81 |
| mix_300k | 0.444 | [0.396, 0.499] | 0.946 [0.906, 0.989] | 0.83 |
| kd_1m | 0.469 | [0.429, 0.515] | – | 0.84 |
| kd_300k | 0.475 | [0.429, 0.526] | 1.014 [0.984, 1.044] | 0.83 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 300k windows: teacher labels vs real future: `kd_300k` vs `gt_300k` | 1.096 | [1.050, 1.149] | b better |
| 300k windows: 50/50 mix vs real future: `mix_300k` vs `gt_300k` | 1.023 | [0.999, 1.049] | no detectable difference |
| 1M windows: teacher labels vs real future: `kd_1m` vs `gt_1m` | 1.185 | [1.119, 1.265] | b better |
| 1M windows: 50/50 mix vs real future: `mix_1m` vs `gt_1m` | 1.022 | [0.996, 1.051] | no detectable difference |
| real future: 1M vs 300k windows: `gt_1m` vs `gt_300k` | 0.912 | [0.880, 0.941] | a better |
| teacher labels: 1M vs 300k windows: `kd_1m` vs `kd_300k` | 0.986 | [0.958, 1.016] | no detectable difference |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mix_1m | 0.678 | [0.644, 0.717] | 0.987 [0.965, 1.009] | 0.77 |
| kd_1m | 0.687 | [0.653, 0.723] | – | 0.79 |
| gt_1m | 0.698 | [0.660, 0.743] | 1.016 [0.982, 1.051] | 0.76 |
| mix_300k | 0.718 | [0.683, 0.756] | 1.045 [1.021, 1.073] | 0.77 |
| kd_300k | 0.719 | [0.683, 0.755] | 1.046 [1.031, 1.062] | 0.79 |
| gt_300k | 0.724 | [0.677, 0.779] | 1.054 [1.001, 1.103] | 0.76 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 300k windows: teacher labels vs real future: `kd_300k` vs `gt_300k` | 0.992 | [0.952, 1.042] | no detectable difference |
| 300k windows: 50/50 mix vs real future: `mix_300k` vs `gt_300k` | 0.991 | [0.961, 1.024] | no detectable difference |
| 1M windows: teacher labels vs real future: `kd_1m` vs `gt_1m` | 0.984 | [0.952, 1.018] | no detectable difference |
| 1M windows: 50/50 mix vs real future: `mix_1m` vs `gt_1m` | 0.971 | [0.954, 0.986] | a better |
| real future: 1M vs 300k windows: `gt_1m` vs `gt_300k` | 0.964 | [0.941, 0.987] | a better |
| teacher labels: 1M vs 300k windows: `kd_1m` vs `kd_300k` | 0.956 | [0.942, 0.969] | a better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mix_1m | 0.701 | [0.653, 0.750] | 0.994 [0.980, 1.010] | 0.66 |
| kd_1m | 0.705 | [0.662, 0.747] | – | 0.65 |
| kd_300k | 0.733 | [0.685, 0.780] | 1.039 [1.025, 1.052] | 0.70 |
| gt_1m | 0.736 | [0.690, 0.781] | 1.043 [1.026, 1.062] | 0.66 |
| mix_300k | 0.764 | [0.720, 0.806] | 1.083 [1.070, 1.098] | 0.70 |
| gt_300k | 0.772 | [0.727, 0.817] | 1.096 [1.073, 1.119] | 0.75 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, chronos-bolt-mini 0.651, t0-gamma 0.668, chronos-bolt-tiny 0.674, toto-2.0-4m 0.709, ttm-r2.1-l1 1.159, ttm-r2 1.170, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 300k windows: teacher labels vs real future: `kd_300k` vs `gt_300k` | 0.949 | [0.930, 0.968] | a better |
| 300k windows: 50/50 mix vs real future: `mix_300k` vs `gt_300k` | 0.989 | [0.971, 1.006] | no detectable difference |
| 1M windows: teacher labels vs real future: `kd_1m` vs `gt_1m` | 0.959 | [0.942, 0.975] | a better |
| 1M windows: 50/50 mix vs real future: `mix_1m` vs `gt_1m` | 0.953 | [0.937, 0.967] | a better |
| real future: 1M vs 300k windows: `gt_1m` vs `gt_300k` | 0.952 | [0.930, 0.975] | a better |
| teacher labels: 1M vs 300k windows: `kd_1m` vs `kd_300k` | 0.962 | [0.951, 0.976] | a better |
