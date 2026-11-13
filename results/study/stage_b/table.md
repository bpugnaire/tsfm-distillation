# stage_b: relative WQL vs seasonal naive (95% CI); paired ratio vs `kd_timestral` (<1 = better)

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| kd_toto_s1 | 1.384 | [1.159, 1.687] | – | 0.77 |
| kd_toto_s0 | 1.433 | [1.205, 1.736] | – | 0.77 |
| kd_toto_a05_s1 | 1.446 | [1.193, 1.780] | – | 0.75 |
| kd_weighted_s0 | 1.459 | [1.197, 1.799] | – | 0.75 |
| kd_timestral_s0 | 1.467 | [1.228, 1.788] | – | 0.84 |
| kd_ensemble_a05_s0 | 1.483 | [1.225, 1.827] | – | 0.73 |
| kd_ensemble_s1 | 1.491 | [1.248, 1.817] | – | 0.70 |
| kd_weighted_a05_s1 | 1.498 | [1.240, 1.835] | – | 0.75 |
| kd_t0_a05_s0 | 1.504 | [1.239, 1.851] | – | 0.77 |
| kd_toto_a05_s0 | 1.508 | [1.256, 1.848] | – | 0.77 |
| kd_ensemble_s2 | 1.512 | [1.238, 1.875] | – | 0.70 |
| kd_t0_s2 | 1.514 | [1.248, 1.874] | – | 0.80 |
| kd_weighted_s1 | 1.519 | [1.256, 1.874] | – | 0.70 |
| kd_ensemble_a05_s1 | 1.531 | [1.263, 1.883] | – | 0.75 |
| kd_ensemble_s0 | 1.531 | [1.269, 1.881] | – | 0.73 |
| kd_t0_a05_s2 | 1.543 | [1.262, 1.912] | – | 0.77 |
| kd_weighted_a05_s2 | 1.545 | [1.285, 1.895] | – | 0.72 |
| kd_timestral_a05_s1 | 1.550 | [1.295, 1.881] | – | 0.82 |
| kd_timestral_s1 | 1.558 | [1.313, 1.894] | – | 0.82 |
| kd_weighted_s2 | 1.562 | [1.269, 1.949] | – | 0.68 |
| kd_toto_s2 | 1.565 | [1.280, 1.944] | – | 0.75 |
| kd_weighted_a05_s0 | 1.576 | [1.297, 1.940] | – | 0.77 |
| kd_t0_a05_s1 | 1.595 | [1.314, 1.970] | – | 0.76 |
| gt_s1 | 1.596 | [1.309, 1.976] | – | 0.75 |
| kd_timestral_s2 | 1.603 | [1.327, 1.968] | – | 0.83 |
| kd_t0_s0 | 1.604 | [1.306, 2.019] | – | 0.77 |
| kd_timestral_a05_s0 | 1.605 | [1.332, 1.960] | – | 0.78 |
| kd_ensemble_a05_s2 | 1.622 | [1.325, 2.020] | – | 0.72 |
| kd_t0_s1 | 1.624 | [1.328, 2.018] | – | 0.78 |
| kd_toto_a05_s2 | 1.657 | [1.344, 2.063] | – | 0.73 |
| gt_s2 | 1.665 | [1.357, 2.071] | – | 0.76 |
| kd_timestral_a05_s2 | 1.678 | [1.381, 2.070] | – | 0.78 |
| gt_s0 | 1.765 | [1.461, 2.170] | – | 0.73 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, ets 1.246

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth (Timestral, 3 seeds): `kd_timestral` vs `gt` | 0.913 | [0.873, 0.959] | a better |
| teacher: Toto-313m vs Timestral-209M: `kd_toto` vs `kd_timestral` | 0.951 | [0.929, 0.972] | a better |
| teacher: t0-gamma vs Timestral-209M: `kd_t0` vs `kd_timestral` | 1.026 | [0.990, 1.065] | no detectable difference |
| ensemble of 3 vs Timestral alone: `kd_ensemble` vs `kd_timestral` | 0.984 | [0.965, 1.002] | no detectable difference |
| per-window weighting vs plain ensemble: `kd_weighted` vs `kd_ensemble` | 0.998 | [0.983, 1.012] | no detectable difference |
| alpha 0.5 vs 1 (Timestral): `kd_timestral_a05` vs `kd_timestral` | 1.046 | [1.021, 1.070] | b better |
| alpha 0.5 vs 1 (Toto: no native tails): `kd_toto_a05` vs `kd_toto` | 1.053 | [1.028, 1.079] | b better |
| alpha 0.5 vs 1 (ensemble): `kd_ensemble_a05` vs `kd_ensemble` | 1.020 | [0.997, 1.043] | no detectable difference |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_weighted_s2 | 0.714 | [0.664, 0.762] | – | 0.73 |
| kd_weighted_s1 | 0.716 | [0.672, 0.761] | – | 0.72 |
| kd_weighted_a05_s2 | 0.717 | [0.666, 0.766] | – | 0.78 |
| kd_ensemble_s2 | 0.718 | [0.673, 0.763] | – | 0.73 |
| kd_weighted_s0 | 0.718 | [0.672, 0.765] | – | 0.73 |
| kd_ensemble_s1 | 0.720 | [0.680, 0.761] | – | 0.72 |
| kd_ensemble_a05_s2 | 0.720 | [0.670, 0.769] | – | 0.78 |
| kd_ensemble_s0 | 0.722 | [0.681, 0.765] | – | 0.72 |
| kd_ensemble_a05_s0 | 0.726 | [0.676, 0.776] | – | 0.77 |
| kd_t0_a05_s2 | 0.728 | [0.677, 0.778] | – | 0.81 |
| kd_t0_s1 | 0.728 | [0.686, 0.775] | – | 0.80 |
| kd_weighted_a05_s0 | 0.730 | [0.674, 0.786] | – | 0.76 |
| kd_timestral_a05_s0 | 0.731 | [0.681, 0.781] | – | 0.82 |
| kd_t0_s0 | 0.731 | [0.688, 0.778] | – | 0.82 |
| kd_timestral_a05_s2 | 0.732 | [0.684, 0.780] | – | 0.84 |
| kd_toto_a05_s2 | 0.733 | [0.684, 0.781] | – | 0.80 |
| kd_ensemble_a05_s1 | 0.734 | [0.681, 0.785] | – | 0.77 |
| kd_timestral_s1 | 0.734 | [0.694, 0.776] | – | 0.82 |
| kd_timestral_s0 | 0.734 | [0.693, 0.777] | – | 0.83 |
| kd_t0_s2 | 0.734 | [0.691, 0.781] | – | 0.81 |
| kd_t0_a05_s0 | 0.735 | [0.683, 0.788] | – | 0.80 |
| kd_toto_a05_s1 | 0.737 | [0.682, 0.789] | – | 0.78 |
| kd_timestral_s2 | 0.737 | [0.695, 0.781] | – | 0.84 |
| kd_toto_a05_s0 | 0.739 | [0.688, 0.789] | – | 0.78 |
| kd_t0_a05_s1 | 0.742 | [0.692, 0.792] | – | 0.81 |
| kd_weighted_a05_s1 | 0.743 | [0.689, 0.796] | – | 0.76 |
| kd_toto_s1 | 0.743 | [0.696, 0.790] | – | 0.74 |
| kd_toto_s2 | 0.746 | [0.700, 0.793] | – | 0.76 |
| kd_timestral_a05_s1 | 0.746 | [0.697, 0.796] | – | 0.83 |
| kd_toto_s0 | 0.749 | [0.705, 0.794] | – | 0.73 |
| gt_s1 | 0.755 | [0.693, 0.816] | – | 0.81 |
| gt_s0 | 0.759 | [0.691, 0.823] | – | 0.80 |
| gt_s2 | 0.760 | [0.698, 0.819] | – | 0.80 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, timestral-28m 0.638, chronos-2-small 0.642, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth (Timestral, 3 seeds): `kd_timestral` vs `gt` | 0.966 | [0.937, 0.999] | a better |
| teacher: Toto-313m vs Timestral-209M: `kd_toto` vs `kd_timestral` | 1.016 | [1.005, 1.028] | b better |
| teacher: t0-gamma vs Timestral-209M: `kd_t0` vs `kd_timestral` | 0.998 | [0.987, 1.010] | no detectable difference |
| ensemble of 3 vs Timestral alone: `kd_ensemble` vs `kd_timestral` | 0.980 | [0.971, 0.989] | a better |
| per-window weighting vs plain ensemble: `kd_weighted` vs `kd_ensemble` | 0.995 | [0.985, 1.004] | no detectable difference |
| alpha 0.5 vs 1 (Timestral): `kd_timestral_a05` vs `kd_timestral` | 1.003 | [0.989, 1.017] | no detectable difference |
| alpha 0.5 vs 1 (Toto: no native tails): `kd_toto_a05` vs `kd_toto` | 0.987 | [0.974, 1.000] | a better |
| alpha 0.5 vs 1 (ensemble): `kd_ensemble_a05` vs `kd_ensemble` | 1.010 | [0.994, 1.026] | no detectable difference |

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| kd_ensemble_s0 | 0.918 | [0.864, 0.980] | – | 0.70 |
| kd_weighted_s0 | 0.963 | [0.897, 1.040] | – | 0.61 |
| kd_toto_s1 | 0.976 | [0.922, 1.036] | – | 0.64 |
| kd_toto_s0 | 0.990 | [0.940, 1.044] | – | 0.67 |
| kd_toto_a05_s0 | 0.992 | [0.933, 1.064] | – | 0.66 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |
| kd_timestral_a05_s0 | 1.003 | [0.936, 1.078] | – | 0.72 |
| kd_timestral_s1 | 1.003 | [0.950, 1.063] | – | 0.77 |
| kd_weighted_a05_s0 | 1.026 | [0.959, 1.108] | – | 0.65 |
| kd_ensemble_s1 | 1.027 | [0.966, 1.099] | – | 0.69 |
| kd_timestral_s0 | 1.035 | [0.971, 1.109] | – | 0.71 |
| kd_timestral_s2 | 1.036 | [0.975, 1.104] | – | 0.69 |
| kd_weighted_s1 | 1.042 | [0.973, 1.122] | – | 0.67 |
| kd_toto_s2 | 1.045 | [0.982, 1.117] | – | 0.64 |
| kd_weighted_s2 | 1.049 | [0.987, 1.121] | – | 0.60 |
| kd_ensemble_s2 | 1.054 | [0.991, 1.122] | – | 0.59 |
| kd_ensemble_a05_s0 | 1.055 | [0.988, 1.139] | – | 0.61 |
| kd_toto_a05_s1 | 1.061 | [0.991, 1.143] | – | 0.67 |
| kd_t0_a05_s1 | 1.065 | [0.988, 1.155] | – | 0.62 |
| kd_timestral_a05_s1 | 1.068 | [0.998, 1.149] | – | 0.72 |
| kd_t0_a05_s0 | 1.078 | [0.999, 1.173] | – | 0.59 |
| kd_ensemble_a05_s1 | 1.085 | [1.008, 1.173] | – | 0.63 |
| kd_t0_s0 | 1.086 | [1.015, 1.165] | – | 0.68 |
| kd_t0_s1 | 1.096 | [1.020, 1.183] | – | 0.68 |
| kd_weighted_a05_s1 | 1.101 | [1.027, 1.185] | – | 0.60 |
| kd_t0_a05_s2 | 1.102 | [1.027, 1.187] | – | 0.66 |
| kd_t0_s2 | 1.105 | [1.029, 1.194] | – | 0.61 |
| kd_toto_a05_s2 | 1.130 | [1.052, 1.219] | – | 0.58 |
| kd_timestral_a05_s2 | 1.136 | [1.061, 1.224] | – | 0.65 |
| gt_s2 | 1.143 | [1.078, 1.214] | – | 0.65 |
| gt_s1 | 1.148 | [1.075, 1.226] | – | 0.62 |
| kd_weighted_a05_s2 | 1.168 | [1.087, 1.257] | – | 0.59 |
| kd_ensemble_a05_s2 | 1.170 | [1.084, 1.266] | – | 0.56 |
| gt_s0 | 1.208 | [1.127, 1.298] | – | 0.71 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| teacher labels vs ground truth (Timestral, 3 seeds): `kd_timestral` vs `gt` | 0.878 | [0.851, 0.904] | a better |
| teacher: Toto-313m vs Timestral-209M: `kd_toto` vs `kd_timestral` | 0.980 | [0.974, 0.986] | a better |
| teacher: t0-gamma vs Timestral-209M: `kd_t0` vs `kd_timestral` | 1.070 | [1.057, 1.083] | b better |
| ensemble of 3 vs Timestral alone: `kd_ensemble` vs `kd_timestral` | 0.978 | [0.973, 0.982] | a better |
| per-window weighting vs plain ensemble: `kd_weighted` vs `kd_ensemble` | 1.017 | [1.007, 1.026] | b better |
| alpha 0.5 vs 1 (Timestral): `kd_timestral_a05` vs `kd_timestral` | 1.045 | [1.034, 1.057] | b better |
| alpha 0.5 vs 1 (Toto: no native tails): `kd_toto_a05` vs `kd_toto` | 1.059 | [1.045, 1.076] | b better |
| alpha 0.5 vs 1 (ensemble): `kd_ensemble_a05` vs `kd_ensemble` | 1.103 | [1.084, 1.124] | b better |
