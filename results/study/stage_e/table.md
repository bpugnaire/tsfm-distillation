# stage_e: relative WQL vs seasonal naive (95% CI); paired ratio vs `zero_shot` (<1 = better)

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mix_fr_1y_s0 | 0.654 | [0.603, 0.705] | 0.822 [0.784, 0.862] | 0.69 |
| mix_fr_1y_s1 | 0.655 | [0.608, 0.706] | 0.825 [0.790, 0.860] | 0.70 |
| mix_fr_1y_s2 | 0.656 | [0.608, 0.706] | 0.825 [0.789, 0.863] | 0.69 |
| gt_fr_1y_s1 | 0.661 | [0.613, 0.712] | 0.832 [0.792, 0.870] | 0.70 |
| kd_fr_1y_s2 | 0.675 | [0.627, 0.725] | 0.850 [0.819, 0.882] | 0.69 |
| kd_fr_1y_s1 | 0.676 | [0.625, 0.726] | 0.850 [0.817, 0.885] | 0.67 |
| kd_fr_1y_s0 | 0.679 | [0.629, 0.728] | 0.854 [0.821, 0.888] | 0.69 |
| gt_fr_1y_s2 | 0.693 | [0.642, 0.750] | 0.872 [0.828, 0.922] | 0.61 |
| gt_fr_1y_s0 | 0.704 | [0.651, 0.763] | 0.886 [0.842, 0.935] | 0.59 |
| kd_fr_1m_s1 | 0.713 | [0.668, 0.759] | 0.897 [0.872, 0.923] | 0.66 |
| kd_fr_1m_s2 | 0.714 | [0.667, 0.762] | 0.899 [0.872, 0.927] | 0.65 |
| kd_fr_1m_s0 | 0.717 | [0.671, 0.764] | 0.902 [0.876, 0.929] | 0.66 |
| mix_fr_1w_s2 | 0.750 | [0.706, 0.796] | 0.944 [0.926, 0.962] | 0.74 |
| mix_fr_1w_s0 | 0.770 | [0.728, 0.815] | 0.969 [0.950, 0.989] | 0.71 |
| kd_fr_1w_s0 | 0.774 | [0.729, 0.822] | 0.975 [0.958, 0.991] | 0.74 |
| gt_fr_1w_s2 | 0.775 | [0.723, 0.830] | 0.975 [0.946, 1.006] | 0.67 |
| kd_fr_1w_s2 | 0.775 | [0.734, 0.818] | 0.975 [0.954, 0.998] | 0.76 |
| mix_fr_1w_s1 | 0.777 | [0.733, 0.825] | 0.978 [0.958, 0.999] | 0.72 |
| gt_fr_1w_s0 | 0.778 | [0.732, 0.826] | 0.979 [0.963, 0.993] | 0.69 |
| mix_fr_1m_s1 | 0.786 | [0.737, 0.837] | 0.990 [0.953, 1.026] | 0.54 |
| kd_fr_1w_s1 | 0.788 | [0.748, 0.831] | 0.992 [0.969, 1.017] | 0.74 |
| zero_shot | 0.795 | [0.750, 0.840] | – | 0.76 |
| mix_fr_1m_s0 | 0.796 | [0.741, 0.852] | 1.002 [0.961, 1.042] | 0.54 |
| mix_fr_1m_s2 | 0.796 | [0.742, 0.852] | 1.002 [0.960, 1.043] | 0.53 |
| gt_fr_1w_s1 | 0.810 | [0.759, 0.865] | 1.020 [0.989, 1.048] | 0.59 |
| gt_fr_1m_s1 | 0.881 | [0.819, 0.947] | 1.109 [1.060, 1.158] | 0.42 |
| gt_fr_1m_s2 | 0.913 | [0.842, 0.989] | 1.149 [1.090, 1.210] | 0.38 |
| gt_fr_1m_s0 | 0.927 | [0.863, 0.996] | 1.167 [1.113, 1.223] | 0.39 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_power 1w: teacher labels vs real future (fine-tuning target): `kd_fr_1w` vs `gt_fr_1w` | 0.987 | [0.964, 1.005] | no detectable difference |
| fr_power 1w: 50/50 mix vs real future: `mix_fr_1w` vs `gt_fr_1w` | 0.969 | [0.954, 0.982] | a better |
| fr_power 1w: teacher-label fine-tune vs no fine-tune: `kd_fr_1w` vs `zero_shot` | 0.983 | [0.963, 1.002] | no detectable difference |
| fr_power 1w: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_fr_1w` vs `zoo:toto-2.0-4m` | 1.102 | [1.066, 1.144] | b better |
| fr_power 1w: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_fr_1w` vs `zoo:toto-2.0-4m` | 1.082 | [1.046, 1.125] | b better |
| fr_power 1w: teacher-label fine-tune vs timestral-5m zero-shot: `kd_fr_1w` vs `zoo:timestral-5m` | 1.219 | [1.173, 1.261] | b better |
| fr_power 1w: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_fr_1w` vs `zoo:timestral-5m` | 1.197 | [1.149, 1.242] | b better |
| fr_power 1w: teacher-label fine-tune vs timestral-209m zero-shot: `kd_fr_1w` vs `zoo:timestral-209m` | 1.272 | [1.234, 1.309] | b better |
| fr_power 1w: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_fr_1w` vs `zoo:timestral-209m` | 1.250 | [1.212, 1.285] | b better |
| fr_power 1m: teacher labels vs real future (fine-tuning target): `kd_fr_1m` vs `gt_fr_1m` | 0.786 | [0.760, 0.810] | a better |
| fr_power 1m: 50/50 mix vs real future: `mix_fr_1m` vs `gt_fr_1m` | 0.873 | [0.855, 0.889] | a better |
| fr_power 1m: teacher-label fine-tune vs no fine-tune: `kd_fr_1m` vs `zero_shot` | 0.900 | [0.874, 0.927] | a better |
| fr_power 1m: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_fr_1m` vs `zoo:toto-2.0-4m` | 1.009 | [0.979, 1.041] | no detectable difference |
| fr_power 1m: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_fr_1m` vs `zoo:toto-2.0-4m` | 1.120 | [1.079, 1.171] | b better |
| fr_power 1m: teacher-label fine-tune vs timestral-5m zero-shot: `kd_fr_1m` vs `zoo:timestral-5m` | 1.116 | [1.075, 1.157] | b better |
| fr_power 1m: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_fr_1m` vs `zoo:timestral-5m` | 1.239 | [1.186, 1.299] | b better |
| fr_power 1m: teacher-label fine-tune vs timestral-209m zero-shot: `kd_fr_1m` vs `zoo:timestral-209m` | 1.164 | [1.136, 1.197] | b better |
| fr_power 1m: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_fr_1m` vs `zoo:timestral-209m` | 1.293 | [1.251, 1.343] | b better |
| fr_power 1y: teacher labels vs real future (fine-tuning target): `kd_fr_1y` vs `gt_fr_1y` | 0.973 | [0.942, 1.005] | no detectable difference |
| fr_power 1y: 50/50 mix vs real future: `mix_fr_1y` vs `gt_fr_1y` | 0.942 | [0.923, 0.961] | a better |
| fr_power 1y: teacher-label fine-tune vs no fine-tune: `kd_fr_1y` vs `zero_shot` | 0.852 | [0.820, 0.886] | a better |
| fr_power 1y: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_fr_1y` vs `zoo:toto-2.0-4m` | 0.956 | [0.929, 0.979] | a better |
| fr_power 1y: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_fr_1y` vs `zoo:toto-2.0-4m` | 0.925 | [0.895, 0.954] | a better |
| fr_power 1y: teacher-label fine-tune vs timestral-5m zero-shot: `kd_fr_1y` vs `zoo:timestral-5m` | 1.057 | [1.020, 1.098] | b better |
| fr_power 1y: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_fr_1y` vs `zoo:timestral-5m` | 1.024 | [0.996, 1.052] | no detectable difference |
| fr_power 1y: teacher-label fine-tune vs timestral-209m zero-shot: `kd_fr_1y` vs `zoo:timestral-209m` | 1.103 | [1.072, 1.135] | b better |
| fr_power 1y: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_fr_1y` vs `zoo:timestral-209m` | 1.068 | [1.042, 1.092] | b better |
| fr_power: teacher labels on 1 month vs real future on 1 year: `kd_fr_1m` vs `gt_fr_1y` | 1.027 | [0.999, 1.057] | no detectable difference |

## pedestrian_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mix_ped_1y_s0 | 0.581 | [0.543, 0.620] | 0.766 [0.736, 0.795] | 0.79 |
| mix_ped_1y_s2 | 0.581 | [0.542, 0.622] | 0.766 [0.735, 0.797] | 0.79 |
| mix_ped_1y_s1 | 0.583 | [0.544, 0.623] | 0.768 [0.737, 0.798] | 0.79 |
| kd_ped_1m_s2 | 0.587 | [0.549, 0.625] | 0.774 [0.744, 0.803] | 0.77 |
| kd_ped_1y_s0 | 0.587 | [0.548, 0.628] | 0.774 [0.743, 0.804] | 0.79 |
| kd_ped_1y_s2 | 0.588 | [0.549, 0.630] | 0.775 [0.743, 0.806] | 0.79 |
| kd_ped_1m_s0 | 0.589 | [0.551, 0.627] | 0.776 [0.747, 0.804] | 0.78 |
| kd_ped_1m_s1 | 0.589 | [0.552, 0.627] | 0.776 [0.747, 0.804] | 0.79 |
| kd_ped_1y_s1 | 0.589 | [0.549, 0.630] | 0.776 [0.745, 0.807] | 0.79 |
| gt_ped_1y_s2 | 0.590 | [0.551, 0.631] | 0.778 [0.746, 0.810] | 0.77 |
| gt_ped_1y_s0 | 0.591 | [0.553, 0.630] | 0.778 [0.748, 0.809] | 0.76 |
| gt_ped_1y_s1 | 0.591 | [0.553, 0.630] | 0.779 [0.748, 0.809] | 0.77 |
| mix_ped_1m_s2 | 0.592 | [0.555, 0.628] | 0.779 [0.750, 0.808] | 0.79 |
| mix_ped_1m_s1 | 0.592 | [0.554, 0.629] | 0.780 [0.750, 0.809] | 0.78 |
| mix_ped_1m_s0 | 0.593 | [0.556, 0.631] | 0.782 [0.753, 0.811] | 0.78 |
| kd_ped_1w_s2 | 0.601 | [0.565, 0.638] | 0.792 [0.764, 0.820] | 0.76 |
| kd_ped_1w_s1 | 0.602 | [0.565, 0.638] | 0.793 [0.765, 0.822] | 0.77 |
| kd_ped_1w_s0 | 0.603 | [0.567, 0.640] | 0.795 [0.766, 0.823] | 0.77 |
| mix_ped_1w_s2 | 0.611 | [0.575, 0.649] | 0.806 [0.775, 0.836] | 0.74 |
| mix_ped_1w_s1 | 0.611 | [0.575, 0.650] | 0.806 [0.775, 0.837] | 0.74 |
| mix_ped_1w_s0 | 0.613 | [0.576, 0.651] | 0.808 [0.777, 0.838] | 0.74 |
| gt_ped_1m_s2 | 0.615 | [0.576, 0.654] | 0.811 [0.779, 0.844] | 0.75 |
| gt_ped_1m_s0 | 0.619 | [0.578, 0.659] | 0.816 [0.784, 0.849] | 0.74 |
| gt_ped_1m_s1 | 0.621 | [0.581, 0.661] | 0.819 [0.786, 0.853] | 0.75 |
| gt_ped_1w_s0 | 0.656 | [0.618, 0.696] | 0.865 [0.831, 0.898] | 0.67 |
| gt_ped_1w_s2 | 0.659 | [0.620, 0.700] | 0.868 [0.835, 0.902] | 0.67 |
| gt_ped_1w_s1 | 0.661 | [0.622, 0.702] | 0.872 [0.837, 0.906] | 0.67 |
| zero_shot | 0.759 | [0.724, 0.794] | – | 0.75 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.346, chronos-2 0.419, chronos-2-small 0.459, timestral-209m 0.460, toto-2.0-313m 0.463, timestral-28m 0.491, toto-2.0-22m 0.527, timestral-5m 0.540, toto-2.0-4m 0.648, ets 1.565

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| pedestrian 1w: teacher labels vs real future (fine-tuning target): `kd_ped_1w` vs `gt_ped_1w` | 0.913 | [0.899, 0.927] | a better |
| pedestrian 1w: 50/50 mix vs real future: `mix_ped_1w` vs `gt_ped_1w` | 0.928 | [0.919, 0.936] | a better |
| pedestrian 1w: teacher-label fine-tune vs no fine-tune: `kd_ped_1w` vs `zero_shot` | 0.794 | [0.765, 0.822] | a better |
| pedestrian 1w: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_ped_1w` vs `zoo:toto-2.0-4m` | 0.930 | [0.904, 0.957] | a better |
| pedestrian 1w: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_ped_1w` vs `zoo:toto-2.0-4m` | 0.945 | [0.916, 0.975] | a better |
| pedestrian 1w: teacher-label fine-tune vs timestral-5m zero-shot: `kd_ped_1w` vs `zoo:timestral-5m` | 1.116 | [1.089, 1.148] | b better |
| pedestrian 1w: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_ped_1w` vs `zoo:timestral-5m` | 1.134 | [1.107, 1.168] | b better |
| pedestrian 1w: teacher-label fine-tune vs timestral-209m zero-shot: `kd_ped_1w` vs `zoo:timestral-209m` | 1.310 | [1.251, 1.379] | b better |
| pedestrian 1w: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_ped_1w` vs `zoo:timestral-209m` | 1.331 | [1.271, 1.400] | b better |
| pedestrian 1m: teacher labels vs real future (fine-tuning target): `kd_ped_1m` vs `gt_ped_1m` | 0.951 | [0.931, 0.969] | a better |
| pedestrian 1m: 50/50 mix vs real future: `mix_ped_1m` vs `gt_ped_1m` | 0.957 | [0.945, 0.968] | a better |
| pedestrian 1m: teacher-label fine-tune vs no fine-tune: `kd_ped_1m` vs `zero_shot` | 0.776 | [0.746, 0.804] | a better |
| pedestrian 1m: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_ped_1m` vs `zoo:toto-2.0-4m` | 0.909 | [0.878, 0.940] | a better |
| pedestrian 1m: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_ped_1m` vs `zoo:toto-2.0-4m` | 0.915 | [0.883, 0.947] | a better |
| pedestrian 1m: teacher-label fine-tune vs timestral-5m zero-shot: `kd_ped_1m` vs `zoo:timestral-5m` | 1.091 | [1.061, 1.125] | b better |
| pedestrian 1m: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_ped_1m` vs `zoo:timestral-5m` | 1.098 | [1.066, 1.136] | b better |
| pedestrian 1m: teacher-label fine-tune vs timestral-209m zero-shot: `kd_ped_1m` vs `zoo:timestral-209m` | 1.280 | [1.220, 1.349] | b better |
| pedestrian 1m: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_ped_1m` vs `zoo:timestral-209m` | 1.289 | [1.225, 1.364] | b better |
| pedestrian 1y: teacher labels vs real future (fine-tuning target): `kd_ped_1y` vs `gt_ped_1y` | 0.995 | [0.984, 1.004] | no detectable difference |
| pedestrian 1y: 50/50 mix vs real future: `mix_ped_1y` vs `gt_ped_1y` | 0.984 | [0.978, 0.989] | a better |
| pedestrian 1y: teacher-label fine-tune vs no fine-tune: `kd_ped_1y` vs `zero_shot` | 0.775 | [0.744, 0.806] | a better |
| pedestrian 1y: teacher-label fine-tune vs toto-2.0-4m zero-shot: `kd_ped_1y` vs `zoo:toto-2.0-4m` | 0.908 | [0.875, 0.943] | a better |
| pedestrian 1y: 50/50-mix fine-tune vs toto-2.0-4m zero-shot: `mix_ped_1y` vs `zoo:toto-2.0-4m` | 0.898 | [0.866, 0.932] | a better |
| pedestrian 1y: teacher-label fine-tune vs timestral-5m zero-shot: `kd_ped_1y` vs `zoo:timestral-5m` | 1.090 | [1.059, 1.125] | b better |
| pedestrian 1y: 50/50-mix fine-tune vs timestral-5m zero-shot: `mix_ped_1y` vs `zoo:timestral-5m` | 1.078 | [1.049, 1.112] | b better |
| pedestrian 1y: teacher-label fine-tune vs timestral-209m zero-shot: `kd_ped_1y` vs `zoo:timestral-209m` | 1.279 | [1.221, 1.349] | b better |
| pedestrian 1y: 50/50-mix fine-tune vs timestral-209m zero-shot: `mix_ped_1y` vs `zoo:timestral-209m` | 1.266 | [1.209, 1.332] | b better |
| pedestrian: teacher labels on 1 month vs real future on 1 year: `kd_ped_1m` vs `gt_ped_1y` | 0.995 | [0.984, 1.007] | no detectable difference |
