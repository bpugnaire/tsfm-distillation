# stage_a: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd_multidomain` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| gt_synth | 0.561 | [0.497, 0.636] | 0.383 [0.339, 0.434] | 0.89 |
| kd_synth | 0.630 | [0.561, 0.711] | 0.430 [0.382, 0.483] | 0.90 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| kd_multidomain | 1.467 | [1.228, 1.788] | – | 0.84 |
| kd_rollout | 1.555 | [1.321, 1.860] | 1.060 [1.018, 1.103] | 0.83 |
| gt_multidomain | 1.765 | [1.461, 2.170] | 1.203 [1.144, 1.259] | 0.73 |
| kd_noise | 3.592 | [2.947, 4.403] | 2.449 [2.186, 2.739] | 0.69 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, ets 1.246

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth, real inputs: `kd_multidomain` vs `gt_multidomain` | 0.831 | [0.795, 0.874] | a better |
| teacher labels vs ground truth, synthetic inputs: `kd_synth` vs `gt_synth` | 1.123 | [1.096, 1.152] | b better |
| synthetic vs real inputs (teacher labels): `kd_synth` vs `kd_multidomain` | 0.430 | [0.382, 0.483] | a better |
| synthetic vs real inputs (ground truth): `gt_synth` vs `gt_multidomain` | 0.318 | [0.277, 0.365] | a better |
| teacher rollouts vs real inputs: `kd_rollout` vs `kd_multidomain` | 1.060 | [1.018, 1.103] | b better |
| white noise vs structured synthetic inputs: `kd_noise` vs `kd_synth` | 5.698 | [4.862, 6.666] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_multidomain | 0.734 | [0.693, 0.777] | – | 0.83 |
| gt_multidomain | 0.759 | [0.691, 0.823] | 1.034 [0.994, 1.072] | 0.80 |
| kd_rollout | 0.800 | [0.752, 0.855] | 1.090 [1.065, 1.112] | 0.86 |
| gt_synth | 0.897 | [0.853, 0.950] | 1.223 [1.187, 1.257] | 0.73 |
| kd_synth | 0.899 | [0.859, 0.947] | 1.225 [1.189, 1.262] | 0.77 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |
| kd_noise | 1.316 | [1.186, 1.474] | 1.793 [1.682, 1.926] | 0.76 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, timestral-28m 0.638, chronos-2-small 0.642, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth, real inputs: `kd_multidomain` vs `gt_multidomain` | 0.967 | [0.933, 1.006] | no detectable difference |
| teacher labels vs ground truth, synthetic inputs: `kd_synth` vs `gt_synth` | 1.002 | [0.989, 1.014] | no detectable difference |
| synthetic vs real inputs (teacher labels): `kd_synth` vs `kd_multidomain` | 1.225 | [1.189, 1.262] | b better |
| synthetic vs real inputs (ground truth): `gt_synth` vs `gt_multidomain` | 1.182 | [1.122, 1.250] | b better |
| teacher rollouts vs real inputs: `kd_rollout` vs `kd_multidomain` | 1.090 | [1.065, 1.112] | b better |
| white noise vs structured synthetic inputs: `kd_noise` vs `kd_synth` | 1.464 | [1.366, 1.582] | b better |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |
| kd_rollout | 1.016 | [0.974, 1.061] | 0.982 [0.953, 1.008] | 0.85 |
| gt_synth | 1.028 | [1.000, 1.056] | 0.994 [0.944, 1.037] | 0.80 |
| kd_synth | 1.030 | [0.995, 1.066] | 0.996 [0.950, 1.038] | 0.76 |
| kd_multidomain | 1.035 | [0.971, 1.109] | – | 0.71 |
| gt_multidomain | 1.208 | [1.127, 1.298] | 1.168 [1.137, 1.201] | 0.71 |
| kd_noise | 1.528 | [1.412, 1.669] | 1.477 [1.426, 1.531] | 0.70 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth, real inputs: `kd_multidomain` vs `gt_multidomain` | 0.856 | [0.832, 0.879] | a better |
| teacher labels vs ground truth, synthetic inputs: `kd_synth` vs `gt_synth` | 1.002 | [0.988, 1.016] | no detectable difference |
| synthetic vs real inputs (teacher labels): `kd_synth` vs `kd_multidomain` | 0.996 | [0.950, 1.038] | no detectable difference |
| synthetic vs real inputs (ground truth): `gt_synth` vs `gt_multidomain` | 0.851 | [0.806, 0.894] | a better |
| teacher rollouts vs real inputs: `kd_rollout` vs `kd_multidomain` | 0.982 | [0.953, 1.008] | no detectable difference |
| white noise vs structured synthetic inputs: `kd_noise` vs `kd_synth` | 1.483 | [1.410, 1.570] | b better |
