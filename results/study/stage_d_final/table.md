# stage_d_final: relative WQL vs seasonal naive (95% CI); paired ratio vs `distilled_zero_shot` (<1 = better)

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_warm_fr_1y_scratch_s1 | 0.653 | [0.604, 0.702] | 0.821 [0.784, 0.862] | 0.77 |
| ft_warm_fr_1y_distilled_s1 | 0.661 | [0.613, 0.712] | 0.832 [0.792, 0.870] | 0.70 |
| ft_warm_fr_1y_scratch_s2 | 0.663 | [0.621, 0.707] | 0.835 [0.813, 0.859] | 0.73 |
| ft_warm_fr_1y_scratch_s0 | 0.664 | [0.628, 0.702] | 0.836 [0.807, 0.869] | 0.76 |
| ft_warm_fr_1y_distilled_s2 | 0.693 | [0.642, 0.750] | 0.872 [0.828, 0.922] | 0.61 |
| ft_warm_fr_1y_distilled_s0 | 0.704 | [0.651, 0.763] | 0.886 [0.842, 0.935] | 0.59 |
| ft_warm_fr_1w_distilled_s2 | 0.775 | [0.723, 0.830] | 0.975 [0.946, 1.006] | 0.67 |
| ft_warm_fr_1w_distilled_s0 | 0.778 | [0.732, 0.826] | 0.979 [0.963, 0.993] | 0.69 |
| distilled_zero_shot | 0.795 | [0.750, 0.840] | – | 0.76 |
| ft_warm_fr_1w_distilled_s1 | 0.810 | [0.759, 0.865] | 1.020 [0.989, 1.048] | 0.59 |
| ft_warm_fr_1m_distilled_s1 | 0.881 | [0.819, 0.947] | 1.109 [1.060, 1.158] | 0.42 |
| ft_warm_fr_1m_distilled_s2 | 0.913 | [0.842, 0.989] | 1.149 [1.090, 1.210] | 0.38 |
| ft_warm_fr_1m_distilled_s0 | 0.927 | [0.863, 0.996] | 1.167 [1.113, 1.223] | 0.39 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |
| ft_warm_fr_1m_scratch_s0 | 1.068 | [0.973, 1.166] | 1.344 [1.257, 1.430] | 0.31 |
| ft_warm_fr_1m_scratch_s1 | 1.081 | [0.978, 1.192] | 1.360 [1.271, 1.453] | 0.31 |
| ft_warm_fr_1m_scratch_s2 | 1.122 | [1.032, 1.212] | 1.413 [1.336, 1.491] | 0.31 |
| ft_warm_fr_1w_scratch_s0 | 1.446 | [1.332, 1.569] | 1.820 [1.727, 1.926] | 0.46 |
| ft_warm_fr_1w_scratch_s2 | 1.467 | [1.360, 1.583] | 1.846 [1.760, 1.943] | 0.55 |
| ft_warm_fr_1w_scratch_s1 | 1.496 | [1.394, 1.602] | 1.882 [1.805, 1.974] | 0.60 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| warm fr_power 1w: distilled init vs scratch: `ft_warm_fr_1w_distilled` vs `ft_warm_fr_1w_scratch` | 0.537 | [0.510, 0.564] | a better |
| warm fr_power 1w: fine-tuned vs zero-shot student: `ft_warm_fr_1w_distilled` vs `distilled_zero_shot` | 0.996 | [0.972, 1.018] | no detectable difference |
| warm fr_power 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_fr_1w_distilled` vs `zoo:toto-2.0-4m` | 1.117 | [1.077, 1.161] | b better |
| warm fr_power 1w: fine-tuned student vs timestral-5m zero-shot: `ft_warm_fr_1w_distilled` vs `zoo:timestral-5m` | 1.236 | [1.179, 1.291] | b better |
| warm fr_power 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_fr_1w_distilled` vs `zoo:toto-2.0-313m` | 1.279 | [1.227, 1.330] | b better |
| warm fr_power 1w: fine-tuned student vs timestral-209m zero-shot: `ft_warm_fr_1w_distilled` vs `zoo:timestral-209m` | 1.290 | [1.243, 1.332] | b better |
| warm fr_power 1m: distilled init vs scratch: `ft_warm_fr_1m_distilled` vs `ft_warm_fr_1m_scratch` | 0.831 | [0.813, 0.850] | a better |
| warm fr_power 1m: fine-tuned vs zero-shot student: `ft_warm_fr_1m_distilled` vs `distilled_zero_shot` | 1.144 | [1.091, 1.200] | b better |
| warm fr_power 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_fr_1m_distilled` vs `zoo:toto-2.0-4m` | 1.283 | [1.223, 1.356] | b better |
| warm fr_power 1m: fine-tuned student vs timestral-5m zero-shot: `ft_warm_fr_1m_distilled` vs `zoo:timestral-5m` | 1.419 | [1.337, 1.513] | b better |
| warm fr_power 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_fr_1m_distilled` vs `zoo:toto-2.0-313m` | 1.469 | [1.397, 1.554] | b better |
| warm fr_power 1m: fine-tuned student vs timestral-209m zero-shot: `ft_warm_fr_1m_distilled` vs `zoo:timestral-209m` | 1.481 | [1.413, 1.561] | b better |
| warm fr_power 1y: distilled init vs scratch: `ft_warm_fr_1y_distilled` vs `ft_warm_fr_1y_scratch` | 1.042 | [1.015, 1.072] | b better |
| warm fr_power 1y: fine-tuned vs zero-shot student: `ft_warm_fr_1y_distilled` vs `distilled_zero_shot` | 0.876 | [0.834, 0.920] | a better |
| warm fr_power 1y: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_fr_1y_distilled` vs `zoo:toto-2.0-4m` | 0.982 | [0.946, 1.020] | no detectable difference |
| warm fr_power 1y: fine-tuned student vs timestral-5m zero-shot: `ft_warm_fr_1y_distilled` vs `zoo:timestral-5m` | 1.086 | [1.059, 1.113] | b better |
| warm fr_power 1y: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_fr_1y_distilled` vs `zoo:toto-2.0-313m` | 1.124 | [1.100, 1.151] | b better |
| warm fr_power 1y: fine-tuned student vs timestral-209m zero-shot: `ft_warm_fr_1y_distilled` vs `zoo:timestral-209m` | 1.134 | [1.113, 1.155] | b better |

## fresh_fr_power@168 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| distilled_zero_shot | 0.823 | [0.800, 0.846] | – | 0.80 |
| ft_cold_fr_1w_distilled | 0.966 | [0.928, 1.006] | 1.174 [1.142, 1.206] | 0.41 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.38 |
| ft_cold_fr_1w_scratch | 1.414 | [1.314, 1.533] | 1.718 [1.613, 1.839] | 0.77 |

zero-shot models (same suite, same baseline): toto-2.0-313m 0.609, timestral-209m 0.615, timestral-5m 0.650, toto-2.0-4m 0.816

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| cold fr_power 1w: distilled init vs scratch: `ft_cold_fr_1w_distilled` vs `ft_cold_fr_1w_scratch` | 0.683 | [0.649, 0.715] | a better |
| cold fr_power 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_cold_fr_1w_distilled` vs `zoo:toto-2.0-4m` | 1.184 | [1.158, 1.208] | b better |
| cold fr_power 1w: fine-tuned student vs timestral-5m zero-shot: `ft_cold_fr_1w_distilled` vs `zoo:timestral-5m` | 1.487 | [1.433, 1.545] | b better |
| cold fr_power 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_cold_fr_1w_distilled` vs `zoo:toto-2.0-313m` | 1.586 | [1.511, 1.671] | b better |
| cold fr_power 1w: fine-tuned student vs timestral-209m zero-shot: `ft_cold_fr_1w_distilled` vs `zoo:timestral-209m` | 1.571 | [1.499, 1.649] | b better |

## fresh_fr_power@720 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| distilled_zero_shot | 0.795 | [0.750, 0.840] | – | 0.76 |
| ft_cold_fr_1m_scratch | 0.880 | [0.802, 0.956] | 1.107 [1.046, 1.174] | 0.45 |
| ft_cold_fr_1m_distilled | 0.897 | [0.832, 0.962] | 1.129 [1.074, 1.186] | 0.43 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, toto-2.0-313m 0.619, timestral-5m 0.641, toto-2.0-4m 0.709

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| cold fr_power 1m: distilled init vs scratch: `ft_cold_fr_1m_distilled` vs `ft_cold_fr_1m_scratch` | 1.020 | [0.989, 1.056] | no detectable difference |
| cold fr_power 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_cold_fr_1m_distilled` vs `zoo:toto-2.0-4m` | 1.266 | [1.211, 1.327] | b better |
| cold fr_power 1m: fine-tuned student vs timestral-5m zero-shot: `ft_cold_fr_1m_distilled` vs `zoo:timestral-5m` | 1.401 | [1.329, 1.480] | b better |
| cold fr_power 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_cold_fr_1m_distilled` vs `zoo:toto-2.0-313m` | 1.450 | [1.387, 1.519] | b better |
| cold fr_power 1m: fine-tuned student vs timestral-209m zero-shot: `ft_cold_fr_1m_distilled` vs `zoo:timestral-209m` | 1.462 | [1.396, 1.534] | b better |

## pedestrian_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_warm_ped_1y_distilled_s2 | 0.590 | [0.551, 0.631] | 0.778 [0.746, 0.810] | 0.77 |
| ft_warm_ped_1y_distilled_s0 | 0.591 | [0.553, 0.630] | 0.778 [0.748, 0.809] | 0.76 |
| ft_warm_ped_1y_distilled_s1 | 0.591 | [0.553, 0.630] | 0.779 [0.748, 0.809] | 0.77 |
| ft_warm_ped_1m_scratch_s2 | 0.603 | [0.561, 0.648] | 0.794 [0.760, 0.828] | 0.79 |
| ft_warm_ped_1y_scratch_s2 | 0.613 | [0.570, 0.658] | 0.808 [0.773, 0.842] | 0.81 |
| ft_warm_ped_1m_distilled_s2 | 0.615 | [0.576, 0.654] | 0.811 [0.779, 0.844] | 0.75 |
| ft_warm_ped_1m_distilled_s0 | 0.619 | [0.578, 0.659] | 0.816 [0.784, 0.849] | 0.74 |
| ft_warm_ped_1m_distilled_s1 | 0.621 | [0.581, 0.661] | 0.819 [0.786, 0.853] | 0.75 |
| ft_warm_ped_1m_scratch_s1 | 0.623 | [0.580, 0.668] | 0.820 [0.785, 0.858] | 0.79 |
| ft_warm_ped_1y_scratch_s0 | 0.623 | [0.578, 0.670] | 0.821 [0.784, 0.856] | 0.80 |
| ft_warm_ped_1y_scratch_s1 | 0.626 | [0.580, 0.674] | 0.825 [0.787, 0.862] | 0.81 |
| ft_warm_ped_1m_scratch_s0 | 0.629 | [0.588, 0.673] | 0.829 [0.796, 0.864] | 0.79 |
| ft_warm_ped_1w_distilled_s0 | 0.656 | [0.618, 0.696] | 0.865 [0.831, 0.898] | 0.67 |
| ft_warm_ped_1w_distilled_s2 | 0.659 | [0.620, 0.700] | 0.868 [0.835, 0.902] | 0.67 |
| ft_warm_ped_1w_distilled_s1 | 0.661 | [0.622, 0.702] | 0.872 [0.837, 0.906] | 0.67 |
| ft_warm_ped_1w_scratch_s2 | 0.673 | [0.633, 0.715] | 0.887 [0.855, 0.919] | 0.76 |
| ft_warm_ped_1w_scratch_s1 | 0.677 | [0.640, 0.716] | 0.892 [0.863, 0.920] | 0.76 |
| ft_warm_ped_1w_scratch_s0 | 0.685 | [0.646, 0.725] | 0.902 [0.871, 0.935] | 0.76 |
| distilled_zero_shot | 0.759 | [0.724, 0.794] | – | 0.75 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.346, chronos-2 0.419, chronos-2-small 0.459, timestral-209m 0.460, toto-2.0-313m 0.463, timestral-28m 0.491, toto-2.0-22m 0.527, timestral-5m 0.540, toto-2.0-4m 0.648, ets 1.565

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| warm pedestrian 1w: distilled init vs scratch: `ft_warm_ped_1w_distilled` vs `ft_warm_ped_1w_scratch` | 0.967 | [0.947, 0.986] | a better |
| warm pedestrian 1w: fine-tuned vs zero-shot student: `ft_warm_ped_1w_distilled` vs `distilled_zero_shot` | 0.869 | [0.835, 0.902] | a better |
| warm pedestrian 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_ped_1w_distilled` vs `zoo:toto-2.0-4m` | 1.018 | [0.987, 1.050] | no detectable difference |
| warm pedestrian 1w: fine-tuned student vs timestral-5m zero-shot: `ft_warm_ped_1w_distilled` vs `zoo:timestral-5m` | 1.222 | [1.189, 1.261] | b better |
| warm pedestrian 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_ped_1w_distilled` vs `zoo:toto-2.0-313m` | 1.425 | [1.364, 1.495] | b better |
| warm pedestrian 1w: fine-tuned student vs timestral-209m zero-shot: `ft_warm_ped_1w_distilled` vs `zoo:timestral-209m` | 1.434 | [1.367, 1.512] | b better |
| warm pedestrian 1m: distilled init vs scratch: `ft_warm_ped_1m_distilled` vs `ft_warm_ped_1m_scratch` | 0.995 | [0.970, 1.022] | no detectable difference |
| warm pedestrian 1m: fine-tuned vs zero-shot student: `ft_warm_ped_1m_distilled` vs `distilled_zero_shot` | 0.816 | [0.784, 0.849] | a better |
| warm pedestrian 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_ped_1m_distilled` vs `zoo:toto-2.0-4m` | 0.956 | [0.918, 0.996] | a better |
| warm pedestrian 1m: fine-tuned student vs timestral-5m zero-shot: `ft_warm_ped_1m_distilled` vs `zoo:timestral-5m` | 1.147 | [1.105, 1.192] | b better |
| warm pedestrian 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_ped_1m_distilled` vs `zoo:toto-2.0-313m` | 1.338 | [1.272, 1.412] | b better |
| warm pedestrian 1m: fine-tuned student vs timestral-209m zero-shot: `ft_warm_ped_1m_distilled` vs `zoo:timestral-209m` | 1.346 | [1.271, 1.431] | b better |
| warm pedestrian 1y: distilled init vs scratch: `ft_warm_ped_1y_distilled` vs `ft_warm_ped_1y_scratch` | 0.950 | [0.932, 0.968] | a better |
| warm pedestrian 1y: fine-tuned vs zero-shot student: `ft_warm_ped_1y_distilled` vs `distilled_zero_shot` | 0.779 | [0.749, 0.809] | a better |
| warm pedestrian 1y: fine-tuned student vs toto-2.0-4m zero-shot: `ft_warm_ped_1y_distilled` vs `zoo:toto-2.0-4m` | 0.913 | [0.882, 0.948] | a better |
| warm pedestrian 1y: fine-tuned student vs timestral-5m zero-shot: `ft_warm_ped_1y_distilled` vs `zoo:timestral-5m` | 1.096 | [1.065, 1.132] | b better |
| warm pedestrian 1y: fine-tuned student vs toto-2.0-313m zero-shot: `ft_warm_ped_1y_distilled` vs `zoo:toto-2.0-313m` | 1.279 | [1.221, 1.343] | b better |
| warm pedestrian 1y: fine-tuned student vs timestral-209m zero-shot: `ft_warm_ped_1y_distilled` vs `zoo:timestral-209m` | 1.286 | [1.227, 1.357] | b better |

## pedestrian_test@168 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_cold_ped_1w_distilled | 0.816 | [0.779, 0.857] | 0.932 [0.898, 0.968] | 0.67 |
| distilled_zero_shot | 0.876 | [0.840, 0.911] | – | 0.87 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.45 |
| ft_cold_ped_1w_scratch | 1.042 | [0.992, 1.094] | 1.190 [1.147, 1.233] | 0.70 |

zero-shot models (same suite, same baseline): toto-2.0-313m 0.553, timestral-209m 0.566, timestral-5m 0.707, toto-2.0-4m 0.852

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| cold pedestrian 1w: distilled init vs scratch: `ft_cold_ped_1w_distilled` vs `ft_cold_ped_1w_scratch` | 0.783 | [0.752, 0.814] | a better |
| cold pedestrian 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_cold_ped_1w_distilled` vs `zoo:toto-2.0-4m` | 0.958 | [0.918, 1.003] | no detectable difference |
| cold pedestrian 1w: fine-tuned student vs timestral-5m zero-shot: `ft_cold_ped_1w_distilled` vs `zoo:timestral-5m` | 1.154 | [1.113, 1.201] | b better |
| cold pedestrian 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_cold_ped_1w_distilled` vs `zoo:toto-2.0-313m` | 1.477 | [1.390, 1.581] | b better |
| cold pedestrian 1w: fine-tuned student vs timestral-209m zero-shot: `ft_cold_ped_1w_distilled` vs `zoo:timestral-209m` | 1.443 | [1.359, 1.543] | b better |

## pedestrian_test@720 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_cold_ped_1m_distilled | 0.636 | [0.597, 0.676] | 0.838 [0.806, 0.870] | 0.74 |
| ft_cold_ped_1m_scratch | 0.656 | [0.621, 0.690] | 0.864 [0.841, 0.885] | 0.81 |
| distilled_zero_shot | 0.759 | [0.724, 0.794] | – | 0.75 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): timestral-209m 0.460, toto-2.0-313m 0.463, timestral-5m 0.540, toto-2.0-4m 0.648

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| cold pedestrian 1m: distilled init vs scratch: `ft_cold_ped_1m_distilled` vs `ft_cold_ped_1m_scratch` | 0.970 | [0.942, 0.998] | a better |
| cold pedestrian 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_cold_ped_1m_distilled` vs `zoo:toto-2.0-4m` | 0.982 | [0.949, 1.015] | no detectable difference |
| cold pedestrian 1m: fine-tuned student vs timestral-5m zero-shot: `ft_cold_ped_1m_distilled` vs `zoo:timestral-5m` | 1.179 | [1.139, 1.220] | b better |
| cold pedestrian 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_cold_ped_1m_distilled` vs `zoo:toto-2.0-313m` | 1.375 | [1.309, 1.449] | b better |
| cold pedestrian 1m: fine-tuned student vs timestral-209m zero-shot: `ft_cold_ped_1m_distilled` vs `zoo:timestral-209m` | 1.383 | [1.311, 1.465] | b better |
