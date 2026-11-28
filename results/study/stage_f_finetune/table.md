# stage_f_finetune: relative WQL vs seasonal naive (95% CI); paired ratio vs `zero_shot` (<1 = better)

## fresh_fr_power (494 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| replay_fr_1y_s0 | 0.646 | [0.603, 0.689] | 0.954 [0.947, 0.960] | 0.71 |
| kd_fr_1y_s2 | 0.657 | [0.616, 0.699] | 0.971 [0.964, 0.981] | 0.67 |
| head_fr_1y_s2 | 0.662 | [0.620, 0.703] | 0.978 [0.969, 0.988] | 0.75 |
| kd_fr_1m_s2 | 0.662 | [0.623, 0.702] | 0.979 [0.971, 0.989] | 0.69 |
| kd_fr_1y_s1 | 0.664 | [0.622, 0.706] | 0.981 [0.970, 0.993] | 0.68 |
| head_fr_1y_s1 | 0.665 | [0.623, 0.705] | 0.982 [0.972, 0.993] | 0.75 |
| kd_fr_1y_s0 | 0.665 | [0.623, 0.708] | 0.983 [0.971, 0.998] | 0.67 |
| last_fr_1y_s2 | 0.666 | [0.623, 0.709] | 0.984 [0.975, 0.994] | 0.77 |
| l2sp_l_fr_1y_s0 | 0.667 | [0.622, 0.713] | 0.986 [0.972, 1.002] | 0.72 |
| full_lowlr_fr_1y_s1 | 0.668 | [0.623, 0.713] | 0.987 [0.971, 1.005] | 0.66 |
| l2sp_l_fr_1y_s1 | 0.668 | [0.621, 0.718] | 0.987 [0.969, 1.007] | 0.65 |
| kd_fr_1m_s0 | 0.669 | [0.629, 0.707] | 0.988 [0.978, 0.998] | 0.67 |
| kd_fr_1w_s0 | 0.669 | [0.627, 0.710] | 0.988 [0.980, 0.997] | 0.68 |
| last_fr_1y_s1 | 0.669 | [0.628, 0.709] | 0.988 [0.971, 1.006] | 0.74 |
| l2sp_s_fr_1y_s0 | 0.670 | [0.625, 0.716] | 0.990 [0.976, 1.006] | 0.72 |
| head_fr_1w_s1 | 0.670 | [0.626, 0.712] | 0.990 [0.981, 1.000] | 0.68 |
| full_fr_1y_s0 | 0.670 | [0.625, 0.716] | 0.991 [0.976, 1.007] | 0.72 |
| l2sp_l_fr_1y_s2 | 0.670 | [0.629, 0.712] | 0.991 [0.977, 1.006] | 0.66 |
| head_fr_1y_s0 | 0.671 | [0.629, 0.712] | 0.992 [0.983, 1.000] | 0.75 |
| kd_fr_1w_s2 | 0.672 | [0.631, 0.712] | 0.993 [0.988, 0.998] | 0.67 |
| last_fr_1y_s0 | 0.672 | [0.629, 0.714] | 0.993 [0.983, 1.003] | 0.75 |
| head_fr_1w_s2 | 0.672 | [0.629, 0.715] | 0.993 [0.984, 1.004] | 0.67 |
| full_lowlr_fr_1y_s2 | 0.674 | [0.630, 0.719] | 0.996 [0.982, 1.013] | 0.69 |
| head_fr_1w_s0 | 0.675 | [0.631, 0.718] | 0.997 [0.988, 1.010] | 0.66 |
| l2sp_s_fr_1y_s2 | 0.676 | [0.634, 0.717] | 0.999 [0.982, 1.017] | 0.65 |
| full_fr_1y_s2 | 0.676 | [0.634, 0.718] | 1.000 [0.983, 1.018] | 0.65 |
| zero_shot | 0.677 | [0.635, 0.719] | – | 0.69 |
| kd_fr_1m_s1 | 0.678 | [0.638, 0.716] | 1.002 [0.992, 1.013] | 0.67 |
| kd_fr_1w_s1 | 0.678 | [0.637, 0.719] | 1.002 [0.995, 1.010] | 0.68 |
| scratch_fr_1y_s0 | 0.680 | [0.641, 0.719] | 1.005 [0.979, 1.033] | 0.68 |
| head_fr_1m_s2 | 0.682 | [0.641, 0.722] | 1.008 [0.997, 1.020] | 0.67 |
| scratch_fr_1y_s1 | 0.683 | [0.642, 0.726] | 1.009 [0.992, 1.029] | 0.69 |
| last_fr_1w_s0 | 0.686 | [0.646, 0.724] | 1.013 [1.000, 1.028] | 0.58 |
| l2sp_s_fr_1y_s1 | 0.686 | [0.635, 0.739] | 1.014 [0.990, 1.040] | 0.62 |
| full_fr_1y_s1 | 0.687 | [0.635, 0.740] | 1.015 [0.990, 1.042] | 0.62 |
| head_fr_1m_s1 | 0.689 | [0.648, 0.729] | 1.018 [1.006, 1.030] | 0.67 |
| head_fr_1m_s0 | 0.689 | [0.647, 0.729] | 1.018 [1.006, 1.031] | 0.66 |
| last_fr_1w_s1 | 0.690 | [0.649, 0.729] | 1.019 [1.003, 1.036] | 0.58 |
| replay_fr_1y_s2 | 0.690 | [0.643, 0.738] | 1.020 [1.000, 1.040] | 0.65 |
| last_fr_1w_s2 | 0.691 | [0.651, 0.730] | 1.021 [1.006, 1.038] | 0.57 |
| full_lowlr_fr_1w_s1 | 0.698 | [0.653, 0.741] | 1.032 [1.009, 1.055] | 0.58 |
| full_lowlr_fr_1y_s0 | 0.698 | [0.646, 0.757] | 1.032 [0.999, 1.069] | 0.61 |
| l2sp_l_fr_1w_s2 | 0.698 | [0.654, 0.741] | 1.032 [1.013, 1.052] | 0.61 |
| scratch_fr_1y_s2 | 0.702 | [0.658, 0.744] | 1.037 [1.016, 1.061] | 0.68 |
| replay_fr_1m_s1 | 0.708 | [0.664, 0.749] | 1.047 [1.031, 1.063] | 0.59 |
| last_fr_1m_s2 | 0.709 | [0.668, 0.750] | 1.048 [1.033, 1.065] | 0.59 |
| replay_fr_1w_s1 | 0.712 | [0.668, 0.753] | 1.052 [1.025, 1.083] | 0.60 |
| full_lowlr_fr_1w_s0 | 0.713 | [0.666, 0.758] | 1.054 [1.029, 1.082] | 0.54 |
| l2sp_l_fr_1m_s2 | 0.715 | [0.671, 0.757] | 1.056 [1.031, 1.086] | 0.59 |
| full_lowlr_fr_1w_s2 | 0.719 | [0.671, 0.763] | 1.062 [1.036, 1.091] | 0.55 |
| l2sp_l_fr_1m_s0 | 0.719 | [0.672, 0.764] | 1.063 [1.040, 1.090] | 0.55 |
| replay_fr_1y_s1 | 0.725 | [0.663, 0.792] | 1.071 [1.029, 1.123] | 0.57 |
| replay_fr_1w_s2 | 0.726 | [0.680, 0.771] | 1.073 [1.044, 1.105] | 0.54 |
| l2sp_s_fr_1w_s2 | 0.728 | [0.681, 0.772] | 1.076 [1.048, 1.106] | 0.53 |
| l2sp_l_fr_1w_s0 | 0.728 | [0.680, 0.774] | 1.076 [1.051, 1.106] | 0.60 |
| full_fr_1w_s2 | 0.735 | [0.688, 0.780] | 1.086 [1.056, 1.118] | 0.52 |
| l2sp_s_fr_1m_s1 | 0.737 | [0.689, 0.784] | 1.089 [1.064, 1.119] | 0.55 |
| full_lowlr_fr_1m_s0 | 0.755 | [0.705, 0.803] | 1.115 [1.084, 1.153] | 0.50 |
| full_lowlr_fr_1m_s2 | 0.755 | [0.706, 0.802] | 1.116 [1.087, 1.153] | 0.47 |
| full_lowlr_fr_1m_s1 | 0.762 | [0.712, 0.811] | 1.126 [1.093, 1.167] | 0.48 |
| l2sp_l_fr_1m_s1 | 0.765 | [0.715, 0.814] | 1.130 [1.098, 1.169] | 0.51 |
| replay_fr_1w_s0 | 0.766 | [0.723, 0.811] | 1.133 [1.089, 1.182] | 0.50 |
| replay_fr_1m_s2 | 0.766 | [0.719, 0.813] | 1.133 [1.110, 1.159] | 0.53 |
| replay_fr_1m_s0 | 0.774 | [0.728, 0.819] | 1.144 [1.121, 1.170] | 0.47 |
| l2sp_s_fr_1m_s0 | 0.778 | [0.725, 0.830] | 1.150 [1.115, 1.190] | 0.46 |
| full_fr_1m_s2 | 0.779 | [0.723, 0.833] | 1.152 [1.114, 1.193] | 0.44 |
| full_fr_1m_s1 | 0.787 | [0.730, 0.843] | 1.162 [1.121, 1.211] | 0.46 |
| last_fr_1m_s0 | 0.787 | [0.730, 0.848] | 1.163 [1.121, 1.216] | 0.45 |
| l2sp_s_fr_1w_s0 | 0.792 | [0.740, 0.848] | 1.171 [1.125, 1.223] | 0.49 |
| l2sp_s_fr_1m_s2 | 0.794 | [0.740, 0.845] | 1.173 [1.133, 1.220] | 0.42 |
| l2sp_l_fr_1w_s1 | 0.795 | [0.745, 0.845] | 1.175 [1.136, 1.220] | 0.55 |
| full_fr_1w_s1 | 0.796 | [0.743, 0.851] | 1.176 [1.128, 1.230] | 0.45 |
| full_fr_1w_s0 | 0.801 | [0.747, 0.859] | 1.184 [1.136, 1.240] | 0.46 |
| l2sp_s_fr_1w_s1 | 0.810 | [0.757, 0.865] | 1.198 [1.151, 1.252] | 0.46 |
| full_fr_1m_s0 | 0.846 | [0.784, 0.905] | 1.250 [1.206, 1.299] | 0.35 |
| last_fr_1m_s1 | 0.858 | [0.790, 0.931] | 1.268 [1.210, 1.335] | 0.37 |
| scratch_fr_1m_s1 | 0.969 | [0.889, 1.051] | 1.432 [1.377, 1.493] | 0.33 |
| scratch_fr_1m_s2 | 0.989 | [0.912, 1.071] | 1.461 [1.393, 1.537] | 0.32 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.75 |
| scratch_fr_1m_s0 | 1.016 | [0.936, 1.096] | 1.502 [1.427, 1.576] | 0.30 |
| scratch_fr_1w_s0 | 1.208 | [1.113, 1.305] | 1.785 [1.697, 1.885] | 0.61 |
| scratch_fr_1w_s1 | 1.277 | [1.202, 1.351] | 1.888 [1.807, 1.980] | 0.63 |
| scratch_fr_1w_s2 | 1.463 | [1.347, 1.600] | 2.162 [2.017, 2.322] | 0.68 |

zero-shot models (same suite, same baseline): timestral-209m 0.618, timestral-28m 0.625, chronos-2 0.629, toto-2.0-313m 0.635, chronos-bolt-mini 0.643, chronos-2-small 0.651, timestral-5m 0.653, chronos-bolt-tiny 0.656, t0-gamma 0.663, toto-2.0-22m 0.668, toto-2.0-4m 0.727, ets 1.361

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_power 1w: full vs zero-shot student: `full_fr_1w` vs `zero_shot` | 1.156 | [1.113, 1.204] | b better |
| fr_power 1w: full_lowlr vs zero-shot student: `full_lowlr_fr_1w` vs `zero_shot` | 1.050 | [1.025, 1.077] | b better |
| fr_power 1w: full_lowlr vs full fine-tuning: `full_lowlr_fr_1w` vs `full_fr_1w` | 0.908 | [0.892, 0.923] | a better |
| fr_power 1w: head vs zero-shot student: `head_fr_1w` vs `zero_shot` | 0.993 | [0.984, 1.005] | no detectable difference |
| fr_power 1w: head vs full fine-tuning: `head_fr_1w` vs `full_fr_1w` | 0.860 | [0.831, 0.889] | a better |
| fr_power 1w: last vs zero-shot student: `last_fr_1w` vs `zero_shot` | 1.018 | [1.003, 1.034] | b better |
| fr_power 1w: last vs full fine-tuning: `last_fr_1w` vs `full_fr_1w` | 0.881 | [0.856, 0.904] | a better |
| fr_power 1w: l2sp_s vs zero-shot student: `l2sp_s_fr_1w` vs `zero_shot` | 1.156 | [1.116, 1.203] | b better |
| fr_power 1w: l2sp_s vs full fine-tuning: `l2sp_s_fr_1w` vs `full_fr_1w` | 1.000 | [0.997, 1.005] | no detectable difference |
| fr_power 1w: l2sp_l vs zero-shot student: `l2sp_l_fr_1w` vs `zero_shot` | 1.103 | [1.076, 1.135] | b better |
| fr_power 1w: l2sp_l vs full fine-tuning: `l2sp_l_fr_1w` vs `full_fr_1w` | 0.954 | [0.941, 0.968] | a better |
| fr_power 1w: replay vs zero-shot student: `replay_fr_1w` vs `zero_shot` | 1.091 | [1.059, 1.127] | b better |
| fr_power 1w: replay vs full fine-tuning: `replay_fr_1w` vs `full_fr_1w` | 0.944 | [0.932, 0.957] | a better |
| fr_power 1w: kd vs zero-shot student: `kd_fr_1w` vs `zero_shot` | 0.995 | [0.989, 1.002] | no detectable difference |
| fr_power 1w: kd vs full fine-tuning: `kd_fr_1w` vs `full_fr_1w` | 0.861 | [0.827, 0.893] | a better |
| fr_power 1w: scratch vs zero-shot student: `scratch_fr_1w` vs `zero_shot` | 1.959 | [1.863, 2.071] | b better |
| fr_power 1w: scratch vs full fine-tuning: `scratch_fr_1w` vs `full_fr_1w` | 1.695 | [1.590, 1.818] | b better |
| fr_power 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1w` vs `zoo:toto-2.0-4m` | 1.076 | [1.029, 1.125] | b better |
| fr_power 1w: full fine-tuning vs timestral-5m zero-shot: `full_fr_1w` vs `zoo:timestral-5m` | 1.197 | [1.150, 1.250] | b better |
| fr_power 1m: full vs zero-shot student: `full_fr_1m` vs `zero_shot` | 1.191 | [1.151, 1.236] | b better |
| fr_power 1m: full_lowlr vs zero-shot student: `full_lowlr_fr_1m` vs `zero_shot` | 1.120 | [1.088, 1.158] | b better |
| fr_power 1m: full_lowlr vs full fine-tuning: `full_lowlr_fr_1m` vs `full_fr_1m` | 0.940 | [0.932, 0.949] | a better |
| fr_power 1m: head vs zero-shot student: `head_fr_1m` vs `zero_shot` | 1.015 | [1.003, 1.027] | b better |
| fr_power 1m: head vs full fine-tuning: `head_fr_1m` vs `full_fr_1m` | 0.852 | [0.823, 0.880] | a better |
| fr_power 1m: last vs zero-shot student: `last_fr_1m` vs `zero_shot` | 1.170 | [1.133, 1.215] | b better |
| fr_power 1m: last vs full fine-tuning: `last_fr_1m` vs `full_fr_1m` | 0.983 | [0.957, 1.008] | no detectable difference |
| fr_power 1m: l2sp_s vs zero-shot student: `l2sp_s_fr_1m` vs `zero_shot` | 1.141 | [1.108, 1.180] | b better |
| fr_power 1m: l2sp_s vs full fine-tuning: `l2sp_s_fr_1m` vs `full_fr_1m` | 0.958 | [0.951, 0.965] | a better |
| fr_power 1m: l2sp_l vs zero-shot student: `l2sp_l_fr_1m` vs `zero_shot` | 1.085 | [1.059, 1.116] | b better |
| fr_power 1m: l2sp_l vs full fine-tuning: `l2sp_l_fr_1m` vs `full_fr_1m` | 0.911 | [0.899, 0.923] | a better |
| fr_power 1m: replay vs zero-shot student: `replay_fr_1m` vs `zero_shot` | 1.112 | [1.092, 1.135] | b better |
| fr_power 1m: replay vs full fine-tuning: `replay_fr_1m` vs `full_fr_1m` | 0.934 | [0.911, 0.955] | a better |
| fr_power 1m: kd vs zero-shot student: `kd_fr_1m` vs `zero_shot` | 0.991 | [0.983, 1.001] | no detectable difference |
| fr_power 1m: kd vs full fine-tuning: `kd_fr_1m` vs `full_fr_1m` | 0.832 | [0.805, 0.858] | a better |
| fr_power 1m: scratch vs zero-shot student: `scratch_fr_1m` vs `zero_shot` | 1.473 | [1.408, 1.543] | b better |
| fr_power 1m: scratch vs full fine-tuning: `scratch_fr_1m` vs `full_fr_1m` | 1.237 | [1.194, 1.288] | b better |
| fr_power 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1m` vs `zoo:toto-2.0-4m` | 1.109 | [1.069, 1.153] | b better |
| fr_power 1m: full fine-tuning vs timestral-5m zero-shot: `full_fr_1m` vs `zoo:timestral-5m` | 1.234 | [1.180, 1.293] | b better |
| fr_power 1y: full vs zero-shot student: `full_fr_1y` vs `zero_shot` | 1.010 | [0.994, 1.026] | no detectable difference |
| fr_power 1y: full_lowlr vs zero-shot student: `full_lowlr_fr_1y` vs `zero_shot` | 1.012 | [0.993, 1.032] | no detectable difference |
| fr_power 1y: full_lowlr vs full fine-tuning: `full_lowlr_fr_1y` vs `full_fr_1y` | 1.002 | [0.993, 1.011] | no detectable difference |
| fr_power 1y: head vs zero-shot student: `head_fr_1y` vs `zero_shot` | 0.984 | [0.975, 0.994] | a better |
| fr_power 1y: head vs full fine-tuning: `head_fr_1y` vs `full_fr_1y` | 0.975 | [0.960, 0.989] | a better |
| fr_power 1y: last vs zero-shot student: `last_fr_1y` vs `zero_shot` | 0.990 | [0.979, 1.002] | no detectable difference |
| fr_power 1y: last vs full fine-tuning: `last_fr_1y` vs `full_fr_1y` | 0.981 | [0.970, 0.993] | a better |
| fr_power 1y: l2sp_s vs zero-shot student: `l2sp_s_fr_1y` vs `zero_shot` | 1.008 | [0.993, 1.025] | no detectable difference |
| fr_power 1y: l2sp_s vs full fine-tuning: `l2sp_s_fr_1y` vs `full_fr_1y` | 0.999 | [0.998, 1.000] | a better |
| fr_power 1y: l2sp_l vs zero-shot student: `l2sp_l_fr_1y` vs `zero_shot` | 0.993 | [0.981, 1.007] | no detectable difference |
| fr_power 1y: l2sp_l vs full fine-tuning: `l2sp_l_fr_1y` vs `full_fr_1y` | 0.984 | [0.979, 0.989] | a better |
| fr_power 1y: replay vs zero-shot student: `replay_fr_1y` vs `zero_shot` | 1.031 | [1.009, 1.056] | b better |
| fr_power 1y: replay vs full fine-tuning: `replay_fr_1y` vs `full_fr_1y` | 1.021 | [1.008, 1.035] | b better |
| fr_power 1y: kd vs zero-shot student: `kd_fr_1y` vs `zero_shot` | 0.979 | [0.969, 0.991] | a better |
| fr_power 1y: kd vs full fine-tuning: `kd_fr_1y` vs `full_fr_1y` | 0.970 | [0.958, 0.982] | a better |
| fr_power 1y: scratch vs zero-shot student: `scratch_fr_1y` vs `zero_shot` | 1.031 | [1.013, 1.049] | b better |
| fr_power 1y: scratch vs full fine-tuning: `scratch_fr_1y` vs `full_fr_1y` | 1.021 | [1.000, 1.047] | no detectable difference |
| fr_power 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1y` vs `zoo:toto-2.0-4m` | 0.940 | [0.923, 0.958] | a better |
| fr_power 1y: full fine-tuning vs timestral-5m zero-shot: `full_fr_1y` vs `zoo:timestral-5m` | 1.046 | [1.019, 1.072] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| replay_ped_1y_s1 | 0.665 | [0.633, 0.701] | 0.983 [0.974, 0.992] | 0.81 |
| replay_ped_1y_s0 | 0.667 | [0.635, 0.702] | 0.985 [0.975, 0.996] | 0.81 |
| replay_ped_1m_s1 | 0.667 | [0.634, 0.704] | 0.986 [0.977, 0.998] | 0.80 |
| replay_ped_1m_s0 | 0.668 | [0.637, 0.704] | 0.988 [0.978, 0.998] | 0.81 |
| replay_ped_1y_s2 | 0.668 | [0.635, 0.706] | 0.988 [0.976, 0.998] | 0.80 |
| l2sp_l_ped_1y_s1 | 0.669 | [0.638, 0.703] | 0.989 [0.974, 1.004] | 0.75 |
| l2sp_l_ped_1y_s2 | 0.670 | [0.638, 0.705] | 0.990 [0.974, 1.007] | 0.75 |
| l2sp_l_ped_1y_s0 | 0.671 | [0.639, 0.705] | 0.991 [0.976, 1.007] | 0.75 |
| replay_ped_1m_s2 | 0.673 | [0.639, 0.711] | 0.995 [0.984, 1.006] | 0.82 |
| zero_shot | 0.676 | [0.644, 0.712] | – | 0.80 |
| replay_ped_1w_s2 | 0.678 | [0.645, 0.715] | 1.002 [0.992, 1.013] | 0.79 |
| replay_fr_1m_s0 | 0.680 | [0.651, 0.715] | 1.005 [0.995, 1.016] | 0.79 |
| replay_fr_1m_s2 | 0.681 | [0.648, 0.719] | 1.007 [0.995, 1.020] | 0.80 |
| full_lowlr_ped_1y_s2 | 0.683 | [0.651, 0.720] | 1.010 [0.982, 1.045] | 0.72 |
| l2sp_l_ped_1m_s1 | 0.684 | [0.652, 0.719] | 1.011 [0.990, 1.033] | 0.71 |
| l2sp_l_ped_1m_s2 | 0.684 | [0.651, 0.721] | 1.011 [0.989, 1.037] | 0.72 |
| replay_ped_1w_s1 | 0.684 | [0.652, 0.721] | 1.011 [1.000, 1.022] | 0.79 |
| replay_fr_1m_s1 | 0.684 | [0.651, 0.723] | 1.012 [0.995, 1.031] | 0.81 |
| head_fr_1y_s0 | 0.685 | [0.655, 0.720] | 1.013 [1.000, 1.028] | 0.83 |
| replay_ped_1w_s0 | 0.685 | [0.653, 0.724] | 1.013 [1.003, 1.026] | 0.79 |
| full_lowlr_ped_1y_s1 | 0.686 | [0.655, 0.721] | 1.013 [0.987, 1.046] | 0.72 |
| last_fr_1y_s0 | 0.686 | [0.655, 0.720] | 1.014 [0.999, 1.031] | 0.79 |
| replay_fr_1w_s2 | 0.687 | [0.654, 0.724] | 1.016 [1.005, 1.029] | 0.78 |
| last_fr_1m_s2 | 0.688 | [0.655, 0.724] | 1.017 [1.008, 1.025] | 0.74 |
| l2sp_l_ped_1m_s0 | 0.689 | [0.658, 0.724] | 1.019 [0.996, 1.043] | 0.72 |
| kd_fr_1w_s0 | 0.691 | [0.658, 0.729] | 1.021 [1.007, 1.036] | 0.79 |
| l2sp_s_ped_1y_s2 | 0.691 | [0.656, 0.731] | 1.021 [0.990, 1.060] | 0.73 |
| full_lowlr_ped_1y_s0 | 0.693 | [0.661, 0.729] | 1.024 [0.995, 1.059] | 0.71 |
| replay_fr_1w_s0 | 0.694 | [0.658, 0.733] | 1.026 [1.015, 1.038] | 0.80 |
| kd_fr_1m_s2 | 0.694 | [0.662, 0.732] | 1.026 [1.010, 1.045] | 0.79 |
| l2sp_s_ped_1y_s0 | 0.695 | [0.663, 0.732] | 1.028 [0.998, 1.061] | 0.73 |
| l2sp_s_ped_1y_s1 | 0.695 | [0.660, 0.735] | 1.028 [0.997, 1.065] | 0.73 |
| replay_fr_1w_s1 | 0.696 | [0.661, 0.737] | 1.028 [1.017, 1.042] | 0.79 |
| replay_fr_1y_s1 | 0.696 | [0.664, 0.734] | 1.029 [1.014, 1.043] | 0.81 |
| kd_ped_1m_s1 | 0.696 | [0.662, 0.735] | 1.029 [1.003, 1.063] | 0.78 |
| kd_fr_1w_s2 | 0.697 | [0.663, 0.736] | 1.030 [1.011, 1.052] | 0.78 |
| l2sp_l_fr_1y_s2 | 0.698 | [0.664, 0.737] | 1.032 [1.011, 1.057] | 0.78 |
| replay_fr_1y_s2 | 0.700 | [0.667, 0.740] | 1.035 [1.020, 1.053] | 0.81 |
| replay_fr_1y_s0 | 0.702 | [0.670, 0.738] | 1.037 [1.020, 1.054] | 0.80 |
| kd_ped_1m_s0 | 0.703 | [0.667, 0.743] | 1.039 [1.011, 1.075] | 0.78 |
| head_ped_1w_s0 | 0.705 | [0.667, 0.750] | 1.043 [1.014, 1.077] | 0.73 |
| l2sp_s_fr_1m_s1 | 0.706 | [0.670, 0.747] | 1.044 [1.019, 1.073] | 0.76 |
| head_fr_1y_s2 | 0.706 | [0.673, 0.743] | 1.044 [1.022, 1.072] | 0.83 |
| head_ped_1w_s1 | 0.706 | [0.668, 0.751] | 1.044 [1.015, 1.079] | 0.73 |
| head_ped_1m_s2 | 0.706 | [0.670, 0.750] | 1.044 [1.019, 1.073] | 0.74 |
| head_ped_1w_s2 | 0.707 | [0.668, 0.752] | 1.045 [1.016, 1.079] | 0.73 |
| head_ped_1m_s0 | 0.707 | [0.671, 0.750] | 1.046 [1.021, 1.074] | 0.74 |
| kd_fr_1y_s2 | 0.708 | [0.677, 0.742] | 1.046 [1.031, 1.062] | 0.83 |
| full_lowlr_ped_1m_s0 | 0.708 | [0.671, 0.751] | 1.047 [1.015, 1.086] | 0.66 |
| last_fr_1w_s1 | 0.708 | [0.672, 0.750] | 1.047 [1.022, 1.076] | 0.74 |
| head_ped_1m_s1 | 0.709 | [0.672, 0.752] | 1.047 [1.023, 1.076] | 0.74 |
| l2sp_s_fr_1y_s2 | 0.709 | [0.673, 0.750] | 1.048 [1.022, 1.080] | 0.77 |
| last_fr_1w_s2 | 0.709 | [0.673, 0.750] | 1.048 [1.023, 1.079] | 0.74 |
| last_fr_1w_s0 | 0.709 | [0.673, 0.751] | 1.048 [1.024, 1.078] | 0.74 |
| full_lowlr_fr_1m_s0 | 0.709 | [0.669, 0.757] | 1.049 [1.017, 1.084] | 0.72 |
| full_lowlr_fr_1m_s2 | 0.710 | [0.670, 0.754] | 1.049 [1.018, 1.083] | 0.71 |
| l2sp_l_fr_1m_s0 | 0.710 | [0.671, 0.755] | 1.049 [1.019, 1.085] | 0.73 |
| full_fr_1y_s2 | 0.711 | [0.675, 0.752] | 1.051 [1.024, 1.084] | 0.76 |
| l2sp_l_fr_1m_s2 | 0.711 | [0.671, 0.757] | 1.051 [1.018, 1.090] | 0.75 |
| full_lowlr_fr_1m_s1 | 0.712 | [0.670, 0.759] | 1.052 [1.019, 1.089] | 0.70 |
| full_lowlr_fr_1y_s2 | 0.712 | [0.677, 0.753] | 1.053 [1.026, 1.086] | 0.77 |
| kd_fr_1m_s0 | 0.712 | [0.677, 0.754] | 1.053 [1.030, 1.078] | 0.79 |
| kd_fr_1w_s1 | 0.713 | [0.677, 0.755] | 1.054 [1.029, 1.084] | 0.78 |
| kd_fr_1y_s0 | 0.714 | [0.684, 0.748] | 1.056 [1.037, 1.073] | 0.83 |
| l2sp_l_ped_1w_s0 | 0.715 | [0.679, 0.751] | 1.057 [1.026, 1.085] | 0.67 |
| kd_ped_1w_s2 | 0.715 | [0.678, 0.757] | 1.057 [1.027, 1.093] | 0.76 |
| head_fr_1y_s1 | 0.715 | [0.681, 0.755] | 1.057 [1.032, 1.092] | 0.83 |
| kd_fr_1y_s1 | 0.716 | [0.686, 0.749] | 1.059 [1.040, 1.077] | 0.83 |
| last_fr_1y_s2 | 0.717 | [0.684, 0.755] | 1.060 [1.035, 1.090] | 0.77 |
| head_fr_1m_s2 | 0.718 | [0.683, 0.758] | 1.062 [1.037, 1.089] | 0.75 |
| last_ped_1y_s1 | 0.719 | [0.674, 0.771] | 1.063 [1.023, 1.110] | 0.64 |
| l2sp_l_fr_1m_s1 | 0.719 | [0.676, 0.769] | 1.063 [1.026, 1.107] | 0.71 |
| full_lowlr_fr_1w_s1 | 0.720 | [0.683, 0.763] | 1.064 [1.037, 1.100] | 0.75 |
| head_fr_1m_s0 | 0.720 | [0.684, 0.761] | 1.065 [1.039, 1.094] | 0.75 |
| head_fr_1m_s1 | 0.720 | [0.685, 0.761] | 1.065 [1.039, 1.094] | 0.75 |
| last_ped_1y_s0 | 0.721 | [0.676, 0.772] | 1.065 [1.024, 1.114] | 0.64 |
| kd_ped_1w_s1 | 0.721 | [0.683, 0.765] | 1.066 [1.033, 1.107] | 0.75 |
| full_lowlr_ped_1m_s1 | 0.721 | [0.681, 0.769] | 1.066 [1.030, 1.111] | 0.64 |
| l2sp_l_fr_1y_s0 | 0.722 | [0.685, 0.765] | 1.067 [1.036, 1.107] | 0.77 |
| l2sp_l_fr_1y_s1 | 0.722 | [0.685, 0.764] | 1.068 [1.038, 1.105] | 0.73 |
| head_ped_1y_s0 | 0.723 | [0.679, 0.773] | 1.068 [1.032, 1.109] | 0.73 |
| l2sp_l_ped_1w_s1 | 0.723 | [0.687, 0.760] | 1.068 [1.036, 1.102] | 0.67 |
| l2sp_s_fr_1m_s0 | 0.723 | [0.682, 0.771] | 1.069 [1.036, 1.108] | 0.69 |
| last_ped_1m_s2 | 0.723 | [0.686, 0.766] | 1.069 [1.046, 1.094] | 0.61 |
| full_lowlr_fr_1w_s2 | 0.725 | [0.687, 0.768] | 1.071 [1.041, 1.110] | 0.75 |
| last_ped_1w_s0 | 0.725 | [0.680, 0.777] | 1.071 [1.034, 1.108] | 0.57 |
| head_ped_1y_s1 | 0.725 | [0.681, 0.775] | 1.072 [1.034, 1.115] | 0.73 |
| last_ped_1m_s0 | 0.726 | [0.689, 0.767] | 1.072 [1.049, 1.097] | 0.62 |
| last_ped_1m_s1 | 0.726 | [0.688, 0.768] | 1.073 [1.049, 1.097] | 0.62 |
| head_ped_1y_s2 | 0.726 | [0.682, 0.777] | 1.073 [1.034, 1.118] | 0.72 |
| l2sp_l_ped_1w_s2 | 0.726 | [0.690, 0.763] | 1.073 [1.039, 1.106] | 0.66 |
| kd_fr_1m_s1 | 0.727 | [0.691, 0.768] | 1.074 [1.048, 1.105] | 0.80 |
| head_fr_1w_s1 | 0.727 | [0.691, 0.769] | 1.075 [1.055, 1.098] | 0.77 |
| kd_ped_1y_s1 | 0.727 | [0.691, 0.769] | 1.075 [1.044, 1.114] | 0.78 |
| full_lowlr_fr_1w_s0 | 0.727 | [0.689, 0.771] | 1.075 [1.045, 1.116] | 0.74 |
| last_ped_1w_s1 | 0.727 | [0.682, 0.780] | 1.075 [1.040, 1.112] | 0.56 |
| full_lowlr_ped_1m_s2 | 0.728 | [0.686, 0.777] | 1.076 [1.040, 1.121] | 0.64 |
| last_ped_1y_s2 | 0.728 | [0.681, 0.782] | 1.076 [1.032, 1.128] | 0.62 |
| l2sp_s_fr_1m_s2 | 0.728 | [0.684, 0.779] | 1.076 [1.042, 1.114] | 0.68 |
| head_fr_1w_s2 | 0.728 | [0.693, 0.770] | 1.077 [1.054, 1.103] | 0.78 |
| last_ped_1w_s2 | 0.730 | [0.684, 0.783] | 1.079 [1.041, 1.119] | 0.57 |
| l2sp_l_fr_1w_s2 | 0.731 | [0.692, 0.775] | 1.081 [1.050, 1.121] | 0.76 |
| kd_ped_1y_s2 | 0.733 | [0.693, 0.778] | 1.083 [1.050, 1.125] | 0.79 |
| l2sp_s_fr_1y_s0 | 0.733 | [0.695, 0.778] | 1.083 [1.048, 1.129] | 0.75 |
| full_fr_1m_s2 | 0.734 | [0.690, 0.784] | 1.085 [1.053, 1.122] | 0.66 |
| kd_ped_1m_s2 | 0.734 | [0.692, 0.783] | 1.085 [1.048, 1.132] | 0.76 |
| full_fr_1m_s1 | 0.735 | [0.692, 0.784] | 1.086 [1.049, 1.130] | 0.69 |
| full_fr_1y_s0 | 0.735 | [0.696, 0.781] | 1.087 [1.050, 1.133] | 0.75 |
| kd_ped_1w_s0 | 0.735 | [0.694, 0.783] | 1.087 [1.050, 1.134] | 0.74 |
| head_fr_1w_s0 | 0.738 | [0.700, 0.783] | 1.091 [1.063, 1.124] | 0.77 |
| kd_ped_1y_s0 | 0.738 | [0.699, 0.784] | 1.091 [1.058, 1.133] | 0.78 |
| full_lowlr_fr_1y_s1 | 0.739 | [0.696, 0.788] | 1.092 [1.052, 1.142] | 0.72 |
| full_ped_1y_s2 | 0.739 | [0.696, 0.791] | 1.093 [1.050, 1.147] | 0.68 |
| l2sp_s_fr_1w_s2 | 0.741 | [0.700, 0.786] | 1.095 [1.060, 1.136] | 0.74 |
| full_fr_1w_s2 | 0.744 | [0.702, 0.790] | 1.099 [1.063, 1.142] | 0.74 |
| last_fr_1y_s1 | 0.747 | [0.710, 0.792] | 1.104 [1.071, 1.145] | 0.72 |
| l2sp_s_fr_1y_s1 | 0.747 | [0.704, 0.799] | 1.105 [1.063, 1.158] | 0.70 |
| full_ped_1y_s1 | 0.749 | [0.708, 0.797] | 1.107 [1.066, 1.158] | 0.67 |
| full_fr_1m_s0 | 0.751 | [0.704, 0.805] | 1.110 [1.069, 1.153] | 0.63 |
| last_fr_1m_s0 | 0.752 | [0.712, 0.797] | 1.112 [1.080, 1.140] | 0.61 |
| full_lowlr_ped_1w_s1 | 0.752 | [0.712, 0.793] | 1.112 [1.073, 1.150] | 0.59 |
| full_ped_1y_s0 | 0.755 | [0.712, 0.806] | 1.116 [1.075, 1.170] | 0.68 |
| full_lowlr_ped_1w_s0 | 0.755 | [0.715, 0.797] | 1.117 [1.078, 1.155] | 0.59 |
| full_fr_1y_s1 | 0.758 | [0.712, 0.812] | 1.120 [1.076, 1.178] | 0.69 |
| l2sp_l_fr_1w_s0 | 0.758 | [0.714, 0.809] | 1.120 [1.081, 1.169] | 0.75 |
| l2sp_s_ped_1m_s1 | 0.758 | [0.710, 0.813] | 1.121 [1.078, 1.174] | 0.61 |
| full_lowlr_ped_1w_s2 | 0.760 | [0.720, 0.801] | 1.124 [1.081, 1.164] | 0.58 |
| full_lowlr_fr_1y_s0 | 0.766 | [0.719, 0.825] | 1.133 [1.085, 1.194] | 0.71 |
| l2sp_l_fr_1w_s1 | 0.769 | [0.721, 0.824] | 1.136 [1.093, 1.190] | 0.73 |
| l2sp_s_ped_1m_s2 | 0.772 | [0.722, 0.832] | 1.142 [1.097, 1.197] | 0.60 |
| l2sp_s_fr_1w_s0 | 0.773 | [0.726, 0.828] | 1.142 [1.099, 1.198] | 0.74 |
| l2sp_s_fr_1w_s1 | 0.775 | [0.726, 0.832] | 1.145 [1.099, 1.202] | 0.73 |
| full_fr_1w_s0 | 0.777 | [0.729, 0.833] | 1.148 [1.103, 1.204] | 0.74 |
| l2sp_s_ped_1m_s0 | 0.777 | [0.726, 0.835] | 1.148 [1.104, 1.202] | 0.60 |
| full_fr_1w_s1 | 0.781 | [0.731, 0.840] | 1.155 [1.107, 1.214] | 0.72 |
| last_fr_1m_s1 | 0.793 | [0.750, 0.842] | 1.172 [1.136, 1.208] | 0.55 |
| l2sp_s_ped_1w_s0 | 0.794 | [0.750, 0.840] | 1.173 [1.124, 1.220] | 0.58 |
| l2sp_s_ped_1w_s1 | 0.794 | [0.750, 0.843] | 1.174 [1.126, 1.223] | 0.57 |
| l2sp_s_ped_1w_s2 | 0.803 | [0.758, 0.849] | 1.186 [1.134, 1.236] | 0.57 |
| full_ped_1w_s1 | 0.828 | [0.780, 0.878] | 1.223 [1.170, 1.276] | 0.50 |
| full_ped_1w_s0 | 0.839 | [0.791, 0.890] | 1.240 [1.186, 1.293] | 0.50 |
| full_ped_1m_s2 | 0.853 | [0.786, 0.933] | 1.261 [1.194, 1.341] | 0.50 |
| full_ped_1w_s2 | 0.854 | [0.801, 0.907] | 1.262 [1.200, 1.323] | 0.50 |
| full_ped_1m_s1 | 0.855 | [0.786, 0.936] | 1.264 [1.195, 1.346] | 0.51 |
| full_ped_1m_s0 | 0.857 | [0.792, 0.930] | 1.267 [1.205, 1.339] | 0.52 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_power 1w: full vs zero-shot student: `full_fr_1w` vs `zero_shot` | 1.137 | [1.094, 1.190] | b better |
| fr_power 1w: full_lowlr vs zero-shot student: `full_lowlr_fr_1w` vs `zero_shot` | 1.070 | [1.041, 1.109] | b better |
| fr_power 1w: full_lowlr vs full fine-tuning: `full_lowlr_fr_1w` vs `full_fr_1w` | 0.941 | [0.924, 0.956] | a better |
| fr_power 1w: head vs zero-shot student: `head_fr_1w` vs `zero_shot` | 1.081 | [1.058, 1.108] | b better |
| fr_power 1w: head vs full fine-tuning: `head_fr_1w` vs `full_fr_1w` | 0.951 | [0.925, 0.973] | a better |
| fr_power 1w: last vs zero-shot student: `last_fr_1w` vs `zero_shot` | 1.048 | [1.023, 1.077] | b better |
| fr_power 1w: last vs full fine-tuning: `last_fr_1w` vs `full_fr_1w` | 0.921 | [0.898, 0.941] | a better |
| fr_power 1w: l2sp_s vs zero-shot student: `l2sp_s_fr_1w` vs `zero_shot` | 1.131 | [1.089, 1.183] | b better |
| fr_power 1w: l2sp_s vs full fine-tuning: `l2sp_s_fr_1w` vs `full_fr_1w` | 0.994 | [0.992, 0.996] | a better |
| fr_power 1w: l2sp_l vs zero-shot student: `l2sp_l_fr_1w` vs `zero_shot` | 1.116 | [1.077, 1.164] | b better |
| fr_power 1w: l2sp_l vs full fine-tuning: `l2sp_l_fr_1w` vs `full_fr_1w` | 0.981 | [0.974, 0.988] | a better |
| fr_power 1w: replay vs zero-shot student: `replay_fr_1w` vs `zero_shot` | 1.026 | [1.016, 1.037] | b better |
| fr_power 1w: replay vs full fine-tuning: `replay_fr_1w` vs `full_fr_1w` | 0.902 | [0.866, 0.933] | a better |
| fr_power 1w: kd vs zero-shot student: `kd_fr_1w` vs `zero_shot` | 1.037 | [1.018, 1.060] | b better |
| fr_power 1w: kd vs full fine-tuning: `kd_fr_1w` vs `full_fr_1w` | 0.912 | [0.884, 0.935] | a better |
| fr_power 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1w` vs `zoo:toto-2.0-4m` | 1.068 | [1.027, 1.118] | b better |
| fr_power 1w: full fine-tuning vs timestral-5m zero-shot: `full_fr_1w` vs `zoo:timestral-5m` | 1.166 | [1.122, 1.223] | b better |
| fr_power 1m: full vs zero-shot student: `full_fr_1m` vs `zero_shot` | 1.097 | [1.060, 1.138] | b better |
| fr_power 1m: full_lowlr vs zero-shot student: `full_lowlr_fr_1m` vs `zero_shot` | 1.050 | [1.019, 1.086] | b better |
| fr_power 1m: full_lowlr vs full fine-tuning: `full_lowlr_fr_1m` vs `full_fr_1m` | 0.958 | [0.948, 0.966] | a better |
| fr_power 1m: head vs zero-shot student: `head_fr_1m` vs `zero_shot` | 1.064 | [1.038, 1.092] | b better |
| fr_power 1m: head vs full fine-tuning: `head_fr_1m` vs `full_fr_1m` | 0.970 | [0.940, 0.999] | a better |
| fr_power 1m: last vs zero-shot student: `last_fr_1m` vs `zero_shot` | 1.110 | [1.084, 1.135] | b better |
| fr_power 1m: last vs full fine-tuning: `last_fr_1m` vs `full_fr_1m` | 1.012 | [0.977, 1.049] | no detectable difference |
| fr_power 1m: l2sp_s vs zero-shot student: `l2sp_s_fr_1m` vs `zero_shot` | 1.066 | [1.036, 1.102] | b better |
| fr_power 1m: l2sp_s vs full fine-tuning: `l2sp_s_fr_1m` vs `full_fr_1m` | 0.972 | [0.964, 0.980] | a better |
| fr_power 1m: l2sp_l vs zero-shot student: `l2sp_l_fr_1m` vs `zero_shot` | 1.055 | [1.022, 1.095] | b better |
| fr_power 1m: l2sp_l vs full fine-tuning: `l2sp_l_fr_1m` vs `full_fr_1m` | 0.962 | [0.948, 0.976] | a better |
| fr_power 1m: replay vs zero-shot student: `replay_fr_1m` vs `zero_shot` | 1.012 | [1.002, 1.024] | b better |
| fr_power 1m: replay vs full fine-tuning: `replay_fr_1m` vs `full_fr_1m` | 0.923 | [0.895, 0.949] | a better |
| fr_power 1m: kd vs zero-shot student: `kd_fr_1m` vs `zero_shot` | 1.053 | [1.032, 1.078] | b better |
| fr_power 1m: kd vs full fine-tuning: `kd_fr_1m` vs `full_fr_1m` | 0.960 | [0.938, 0.981] | a better |
| fr_power 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1m` vs `zoo:toto-2.0-4m` | 1.029 | [0.996, 1.070] | no detectable difference |
| fr_power 1m: full fine-tuning vs timestral-5m zero-shot: `full_fr_1m` vs `zoo:timestral-5m` | 1.124 | [1.087, 1.166] | b better |
| fr_power 1y: full vs zero-shot student: `full_fr_1y` vs `zero_shot` | 1.091 | [1.055, 1.138] | b better |
| fr_power 1y: full_lowlr vs zero-shot student: `full_lowlr_fr_1y` vs `zero_shot` | 1.099 | [1.060, 1.148] | b better |
| fr_power 1y: full_lowlr vs full fine-tuning: `full_lowlr_fr_1y` vs `full_fr_1y` | 1.007 | [1.001, 1.013] | b better |
| fr_power 1y: head vs zero-shot student: `head_fr_1y` vs `zero_shot` | 1.041 | [1.021, 1.068] | b better |
| fr_power 1y: head vs full fine-tuning: `head_fr_1y` vs `full_fr_1y` | 0.954 | [0.935, 0.972] | a better |
| fr_power 1y: last vs zero-shot student: `last_fr_1y` vs `zero_shot` | 1.068 | [1.043, 1.099] | b better |
| fr_power 1y: last vs full fine-tuning: `last_fr_1y` vs `full_fr_1y` | 0.978 | [0.960, 0.996] | a better |
| fr_power 1y: l2sp_s vs zero-shot student: `l2sp_s_fr_1y` vs `zero_shot` | 1.083 | [1.049, 1.128] | b better |
| fr_power 1y: l2sp_s vs full fine-tuning: `l2sp_s_fr_1y` vs `full_fr_1y` | 0.993 | [0.990, 0.995] | a better |
| fr_power 1y: l2sp_l vs zero-shot student: `l2sp_l_fr_1y` vs `zero_shot` | 1.059 | [1.032, 1.093] | b better |
| fr_power 1y: l2sp_l vs full fine-tuning: `l2sp_l_fr_1y` vs `full_fr_1y` | 0.971 | [0.959, 0.981] | a better |
| fr_power 1y: replay vs zero-shot student: `replay_fr_1y` vs `zero_shot` | 1.041 | [1.029, 1.054] | b better |
| fr_power 1y: replay vs full fine-tuning: `replay_fr_1y` vs `full_fr_1y` | 0.954 | [0.922, 0.981] | a better |
| fr_power 1y: kd vs zero-shot student: `kd_fr_1y` vs `zero_shot` | 1.055 | [1.039, 1.072] | b better |
| fr_power 1y: kd vs full fine-tuning: `kd_fr_1y` vs `full_fr_1y` | 0.967 | [0.932, 0.999] | a better |
| fr_power 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1y` vs `zoo:toto-2.0-4m` | 1.024 | [0.988, 1.068] | no detectable difference |
| fr_power 1y: full fine-tuning vs timestral-5m zero-shot: `full_fr_1y` vs `zoo:timestral-5m` | 1.119 | [1.083, 1.165] | b better |
| pedestrian 1w: full vs zero-shot student: `full_ped_1w` vs `zero_shot` | 1.243 | [1.186, 1.297] | b better |
| pedestrian 1w: full_lowlr vs zero-shot student: `full_lowlr_ped_1w` vs `zero_shot` | 1.118 | [1.078, 1.156] | b better |
| pedestrian 1w: full_lowlr vs full fine-tuning: `full_lowlr_ped_1w` vs `full_ped_1w` | 0.899 | [0.881, 0.918] | a better |
| pedestrian 1w: head vs zero-shot student: `head_ped_1w` vs `zero_shot` | 1.044 | [1.015, 1.078] | b better |
| pedestrian 1w: head vs full fine-tuning: `head_ped_1w` vs `full_ped_1w` | 0.840 | [0.806, 0.874] | a better |
| pedestrian 1w: last vs zero-shot student: `last_ped_1w` vs `zero_shot` | 1.075 | [1.038, 1.114] | b better |
| pedestrian 1w: last vs full fine-tuning: `last_ped_1w` vs `full_ped_1w` | 0.865 | [0.818, 0.912] | a better |
| pedestrian 1w: l2sp_s vs zero-shot student: `l2sp_s_ped_1w` vs `zero_shot` | 1.179 | [1.128, 1.227] | b better |
| pedestrian 1w: l2sp_s vs full fine-tuning: `l2sp_s_ped_1w` vs `full_ped_1w` | 0.948 | [0.937, 0.961] | a better |
| pedestrian 1w: l2sp_l vs zero-shot student: `l2sp_l_ped_1w` vs `zero_shot` | 1.067 | [1.035, 1.098] | b better |
| pedestrian 1w: l2sp_l vs full fine-tuning: `l2sp_l_ped_1w` vs `full_ped_1w` | 0.858 | [0.837, 0.882] | a better |
| pedestrian 1w: replay vs zero-shot student: `replay_ped_1w` vs `zero_shot` | 1.011 | [1.002, 1.021] | b better |
| pedestrian 1w: replay vs full fine-tuning: `replay_ped_1w` vs `full_ped_1w` | 0.813 | [0.781, 0.852] | a better |
| pedestrian 1w: kd vs zero-shot student: `kd_ped_1w` vs `zero_shot` | 1.071 | [1.038, 1.112] | b better |
| pedestrian 1w: kd vs full fine-tuning: `kd_ped_1w` vs `full_ped_1w` | 0.861 | [0.830, 0.894] | a better |
| pedestrian 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1w` vs `zoo:toto-2.0-4m` | 1.167 | [1.110, 1.223] | b better |
| pedestrian 1w: full fine-tuning vs timestral-5m zero-shot: `full_ped_1w` vs `zoo:timestral-5m` | 1.274 | [1.217, 1.334] | b better |
| pedestrian 1m: full vs zero-shot student: `full_ped_1m` vs `zero_shot` | 1.267 | [1.204, 1.343] | b better |
| pedestrian 1m: full_lowlr vs zero-shot student: `full_lowlr_ped_1m` vs `zero_shot` | 1.064 | [1.029, 1.107] | b better |
| pedestrian 1m: full_lowlr vs full fine-tuning: `full_lowlr_ped_1m` vs `full_ped_1m` | 0.840 | [0.808, 0.869] | a better |
| pedestrian 1m: head vs zero-shot student: `head_ped_1m` vs `zero_shot` | 1.046 | [1.021, 1.075] | b better |
| pedestrian 1m: head vs full fine-tuning: `head_ped_1m` vs `full_ped_1m` | 0.826 | [0.790, 0.858] | a better |
| pedestrian 1m: last vs zero-shot student: `last_ped_1m` vs `zero_shot` | 1.072 | [1.049, 1.096] | b better |
| pedestrian 1m: last vs full fine-tuning: `last_ped_1m` vs `full_ped_1m` | 0.846 | [0.808, 0.883] | a better |
| pedestrian 1m: l2sp_s vs zero-shot student: `l2sp_s_ped_1m` vs `zero_shot` | 1.138 | [1.095, 1.192] | b better |
| pedestrian 1m: l2sp_s vs full fine-tuning: `l2sp_s_ped_1m` vs `full_ped_1m` | 0.898 | [0.874, 0.919] | a better |
| pedestrian 1m: l2sp_l vs zero-shot student: `l2sp_l_ped_1m` vs `zero_shot` | 1.014 | [0.993, 1.038] | no detectable difference |
| pedestrian 1m: l2sp_l vs full fine-tuning: `l2sp_l_ped_1m` vs `full_ped_1m` | 0.801 | [0.759, 0.838] | a better |
| pedestrian 1m: replay vs zero-shot student: `replay_ped_1m` vs `zero_shot` | 0.991 | [0.982, 1.001] | no detectable difference |
| pedestrian 1m: replay vs full fine-tuning: `replay_ped_1m` vs `full_ped_1m` | 0.782 | [0.738, 0.822] | a better |
| pedestrian 1m: kd vs zero-shot student: `kd_ped_1m` vs `zero_shot` | 1.054 | [1.023, 1.092] | b better |
| pedestrian 1m: kd vs full fine-tuning: `kd_ped_1m` vs `full_ped_1m` | 0.832 | [0.797, 0.865] | a better |
| pedestrian 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1m` vs `zoo:toto-2.0-4m` | 1.189 | [1.121, 1.269] | b better |
| pedestrian 1m: full fine-tuning vs timestral-5m zero-shot: `full_ped_1m` vs `zoo:timestral-5m` | 1.299 | [1.228, 1.384] | b better |
| pedestrian 1y: full vs zero-shot student: `full_ped_1y` vs `zero_shot` | 1.107 | [1.066, 1.160] | b better |
| pedestrian 1y: full_lowlr vs zero-shot student: `full_lowlr_ped_1y` vs `zero_shot` | 1.017 | [0.989, 1.051] | no detectable difference |
| pedestrian 1y: full_lowlr vs full fine-tuning: `full_lowlr_ped_1y` vs `full_ped_1y` | 0.918 | [0.897, 0.938] | a better |
| pedestrian 1y: head vs zero-shot student: `head_ped_1y` vs `zero_shot` | 1.071 | [1.033, 1.114] | b better |
| pedestrian 1y: head vs full fine-tuning: `head_ped_1y` vs `full_ped_1y` | 0.967 | [0.940, 0.997] | a better |
| pedestrian 1y: last vs zero-shot student: `last_ped_1y` vs `zero_shot` | 1.068 | [1.026, 1.117] | b better |
| pedestrian 1y: last vs full fine-tuning: `last_ped_1y` vs `full_ped_1y` | 0.965 | [0.943, 0.989] | a better |
| pedestrian 1y: l2sp_s vs zero-shot student: `l2sp_s_ped_1y` vs `zero_shot` | 1.026 | [0.996, 1.063] | no detectable difference |
| pedestrian 1y: l2sp_s vs full fine-tuning: `l2sp_s_ped_1y` vs `full_ped_1y` | 0.926 | [0.907, 0.944] | a better |
| pedestrian 1y: l2sp_l vs zero-shot student: `l2sp_l_ped_1y` vs `zero_shot` | 0.990 | [0.975, 1.006] | no detectable difference |
| pedestrian 1y: l2sp_l vs full fine-tuning: `l2sp_l_ped_1y` vs `full_ped_1y` | 0.894 | [0.860, 0.923] | a better |
| pedestrian 1y: replay vs zero-shot student: `replay_ped_1y` vs `zero_shot` | 0.987 | [0.978, 0.995] | a better |
| pedestrian 1y: replay vs full fine-tuning: `replay_ped_1y` vs `full_ped_1y` | 0.891 | [0.852, 0.926] | a better |
| pedestrian 1y: kd vs zero-shot student: `kd_ped_1y` vs `zero_shot` | 1.084 | [1.051, 1.124] | b better |
| pedestrian 1y: kd vs full fine-tuning: `kd_ped_1y` vs `full_ped_1y` | 0.979 | [0.957, 0.997] | a better |
| pedestrian 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1y` vs `zoo:toto-2.0-4m` | 1.039 | [0.998, 1.089] | no detectable difference |
| pedestrian 1y: full fine-tuning vs timestral-5m zero-shot: `full_ped_1y` vs `zoo:timestral-5m` | 1.135 | [1.093, 1.188] | b better |

## pedestrian_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| full_ped_1y_s1 | 0.466 | [0.421, 0.510] | 0.838 [0.806, 0.870] | 0.77 |
| full_ped_1y_s2 | 0.469 | [0.427, 0.513] | 0.844 [0.813, 0.877] | 0.76 |
| full_ped_1y_s0 | 0.470 | [0.426, 0.514] | 0.845 [0.814, 0.880] | 0.77 |
| replay_ped_1y_s2 | 0.473 | [0.431, 0.516] | 0.851 [0.825, 0.877] | 0.77 |
| replay_ped_1y_s0 | 0.474 | [0.431, 0.517] | 0.852 [0.825, 0.880] | 0.78 |
| full_lowlr_ped_1y_s1 | 0.474 | [0.431, 0.518] | 0.853 [0.826, 0.881] | 0.78 |
| full_lowlr_ped_1y_s2 | 0.476 | [0.433, 0.519] | 0.856 [0.827, 0.884] | 0.78 |
| replay_ped_1y_s1 | 0.476 | [0.432, 0.519] | 0.856 [0.829, 0.882] | 0.78 |
| full_lowlr_ped_1y_s0 | 0.476 | [0.433, 0.520] | 0.857 [0.829, 0.886] | 0.77 |
| l2sp_s_ped_1y_s1 | 0.483 | [0.440, 0.525] | 0.868 [0.843, 0.894] | 0.79 |
| l2sp_s_ped_1y_s0 | 0.484 | [0.441, 0.526] | 0.871 [0.846, 0.897] | 0.77 |
| l2sp_s_ped_1y_s2 | 0.486 | [0.444, 0.529] | 0.875 [0.848, 0.903] | 0.77 |
| replay_ped_1m_s2 | 0.489 | [0.446, 0.532] | 0.881 [0.853, 0.908] | 0.76 |
| replay_ped_1m_s1 | 0.492 | [0.448, 0.534] | 0.884 [0.856, 0.913] | 0.75 |
| replay_ped_1m_s0 | 0.493 | [0.448, 0.537] | 0.887 [0.858, 0.916] | 0.75 |
| full_lowlr_ped_1m_s0 | 0.497 | [0.453, 0.542] | 0.895 [0.865, 0.926] | 0.76 |
| full_lowlr_ped_1m_s2 | 0.498 | [0.453, 0.544] | 0.897 [0.866, 0.930] | 0.76 |
| l2sp_s_ped_1m_s2 | 0.500 | [0.454, 0.545] | 0.900 [0.871, 0.930] | 0.75 |
| full_ped_1m_s2 | 0.501 | [0.454, 0.548] | 0.901 [0.866, 0.939] | 0.71 |
| full_lowlr_ped_1m_s1 | 0.501 | [0.455, 0.547] | 0.902 [0.869, 0.935] | 0.74 |
| full_ped_1m_s0 | 0.502 | [0.456, 0.549] | 0.903 [0.868, 0.943] | 0.71 |
| last_ped_1y_s0 | 0.504 | [0.461, 0.546] | 0.906 [0.884, 0.929] | 0.78 |
| last_ped_1y_s2 | 0.504 | [0.460, 0.546] | 0.906 [0.884, 0.929] | 0.78 |
| l2sp_s_ped_1m_s0 | 0.504 | [0.458, 0.548] | 0.907 [0.877, 0.938] | 0.74 |
| last_ped_1y_s1 | 0.505 | [0.462, 0.547] | 0.909 [0.887, 0.930] | 0.78 |
| l2sp_s_ped_1m_s1 | 0.506 | [0.460, 0.550] | 0.910 [0.880, 0.941] | 0.76 |
| full_ped_1m_s1 | 0.506 | [0.459, 0.554] | 0.910 [0.874, 0.950] | 0.70 |
| l2sp_l_ped_1y_s0 | 0.509 | [0.467, 0.551] | 0.917 [0.899, 0.935] | 0.79 |
| l2sp_l_ped_1y_s1 | 0.511 | [0.469, 0.552] | 0.919 [0.901, 0.938] | 0.78 |
| l2sp_l_ped_1y_s2 | 0.512 | [0.470, 0.553] | 0.921 [0.901, 0.941] | 0.77 |
| l2sp_l_ped_1m_s2 | 0.515 | [0.475, 0.554] | 0.927 [0.909, 0.944] | 0.79 |
| last_ped_1m_s2 | 0.518 | [0.474, 0.562] | 0.933 [0.908, 0.958] | 0.77 |
| last_ped_1m_s1 | 0.518 | [0.474, 0.562] | 0.933 [0.907, 0.958] | 0.76 |
| l2sp_l_ped_1m_s1 | 0.520 | [0.478, 0.562] | 0.935 [0.914, 0.956] | 0.77 |
| last_ped_1m_s0 | 0.520 | [0.476, 0.563] | 0.936 [0.910, 0.962] | 0.76 |
| kd_ped_1y_s0 | 0.525 | [0.484, 0.564] | 0.944 [0.929, 0.958] | 0.83 |
| kd_ped_1y_s2 | 0.525 | [0.483, 0.564] | 0.945 [0.929, 0.959] | 0.83 |
| kd_ped_1m_s2 | 0.525 | [0.485, 0.565] | 0.945 [0.930, 0.958] | 0.83 |
| kd_ped_1y_s1 | 0.526 | [0.485, 0.565] | 0.946 [0.931, 0.960] | 0.83 |
| kd_ped_1w_s0 | 0.527 | [0.486, 0.567] | 0.949 [0.934, 0.962] | 0.83 |
| replay_ped_1w_s1 | 0.527 | [0.480, 0.573] | 0.949 [0.916, 0.979] | 0.67 |
| kd_ped_1m_s0 | 0.527 | [0.487, 0.568] | 0.949 [0.933, 0.964] | 0.82 |
| replay_ped_1w_s2 | 0.528 | [0.480, 0.575] | 0.950 [0.915, 0.985] | 0.67 |
| replay_ped_1w_s0 | 0.528 | [0.481, 0.575] | 0.951 [0.916, 0.984] | 0.67 |
| kd_ped_1w_s1 | 0.528 | [0.487, 0.568] | 0.951 [0.935, 0.966] | 0.82 |
| kd_ped_1m_s1 | 0.528 | [0.487, 0.570] | 0.951 [0.936, 0.964] | 0.83 |
| l2sp_l_ped_1w_s0 | 0.529 | [0.487, 0.570] | 0.952 [0.933, 0.971] | 0.73 |
| kd_ped_1w_s2 | 0.530 | [0.490, 0.570] | 0.954 [0.940, 0.969] | 0.83 |
| l2sp_l_ped_1w_s2 | 0.530 | [0.488, 0.571] | 0.954 [0.934, 0.976] | 0.73 |
| l2sp_l_ped_1m_s0 | 0.532 | [0.489, 0.578] | 0.958 [0.934, 0.985] | 0.76 |
| full_lowlr_ped_1w_s2 | 0.533 | [0.485, 0.582] | 0.959 [0.925, 0.994] | 0.67 |
| full_lowlr_ped_1w_s1 | 0.533 | [0.485, 0.581] | 0.959 [0.925, 0.994] | 0.67 |
| l2sp_l_ped_1w_s1 | 0.534 | [0.492, 0.575] | 0.961 [0.942, 0.981] | 0.74 |
| full_lowlr_ped_1w_s0 | 0.534 | [0.486, 0.581] | 0.961 [0.927, 0.996] | 0.68 |
| last_ped_1w_s1 | 0.535 | [0.490, 0.579] | 0.962 [0.937, 0.986] | 0.69 |
| last_ped_1w_s0 | 0.535 | [0.490, 0.579] | 0.962 [0.937, 0.987] | 0.69 |
| last_ped_1w_s2 | 0.536 | [0.491, 0.580] | 0.964 [0.939, 0.989] | 0.69 |
| head_ped_1m_s1 | 0.536 | [0.495, 0.576] | 0.965 [0.955, 0.974] | 0.78 |
| head_ped_1m_s0 | 0.536 | [0.496, 0.576] | 0.965 [0.955, 0.974] | 0.78 |
| head_ped_1m_s2 | 0.536 | [0.496, 0.576] | 0.965 [0.956, 0.974] | 0.78 |
| head_ped_1y_s0 | 0.537 | [0.495, 0.578] | 0.966 [0.954, 0.978] | 0.78 |
| head_ped_1y_s2 | 0.537 | [0.495, 0.578] | 0.966 [0.955, 0.978] | 0.77 |
| head_ped_1y_s1 | 0.537 | [0.495, 0.578] | 0.967 [0.955, 0.978] | 0.78 |
| full_ped_1w_s1 | 0.539 | [0.489, 0.588] | 0.970 [0.931, 1.007] | 0.64 |
| head_ped_1w_s1 | 0.541 | [0.501, 0.581] | 0.973 [0.964, 0.983] | 0.76 |
| head_ped_1w_s0 | 0.542 | [0.501, 0.582] | 0.975 [0.965, 0.985] | 0.75 |
| head_ped_1w_s2 | 0.542 | [0.502, 0.582] | 0.975 [0.965, 0.985] | 0.76 |
| l2sp_s_ped_1w_s2 | 0.542 | [0.494, 0.589] | 0.975 [0.941, 1.010] | 0.67 |
| l2sp_s_ped_1w_s1 | 0.542 | [0.495, 0.590] | 0.976 [0.942, 1.009] | 0.67 |
| full_ped_1w_s0 | 0.543 | [0.491, 0.594] | 0.977 [0.935, 1.016] | 0.64 |
| full_ped_1w_s2 | 0.543 | [0.492, 0.593] | 0.977 [0.936, 1.016] | 0.64 |
| scratch_ped_1y_s0 | 0.544 | [0.498, 0.591] | 0.978 [0.947, 1.011] | 0.82 |
| l2sp_s_ped_1w_s0 | 0.546 | [0.497, 0.594] | 0.983 [0.947, 1.019] | 0.67 |
| scratch_ped_1y_s2 | 0.551 | [0.506, 0.599] | 0.992 [0.959, 1.028] | 0.81 |
| scratch_ped_1y_s1 | 0.553 | [0.508, 0.601] | 0.995 [0.963, 1.033] | 0.81 |
| scratch_ped_1m_s2 | 0.556 | [0.511, 0.601] | 1.000 [0.970, 1.031] | 0.80 |
| zero_shot | 0.556 | [0.516, 0.595] | – | 0.84 |
| scratch_ped_1m_s1 | 0.559 | [0.514, 0.605] | 1.005 [0.971, 1.043] | 0.77 |
| scratch_ped_1m_s0 | 0.566 | [0.519, 0.614] | 1.018 [0.979, 1.063] | 0.77 |
| scratch_ped_1w_s0 | 0.592 | [0.543, 0.642] | 1.065 [1.029, 1.103] | 0.73 |
| scratch_ped_1w_s1 | 0.606 | [0.558, 0.654] | 1.091 [1.057, 1.124] | 0.74 |
| scratch_ped_1w_s2 | 0.612 | [0.561, 0.662] | 1.101 [1.064, 1.141] | 0.74 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.346, chronos-2 0.419, chronos-2-small 0.459, timestral-209m 0.460, toto-2.0-313m 0.463, timestral-28m 0.491, toto-2.0-22m 0.527, timestral-5m 0.540, toto-2.0-4m 0.648, ets 1.565

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| pedestrian 1w: full vs zero-shot student: `full_ped_1w` vs `zero_shot` | 0.975 | [0.935, 1.014] | no detectable difference |
| pedestrian 1w: full_lowlr vs zero-shot student: `full_lowlr_ped_1w` vs `zero_shot` | 0.960 | [0.926, 0.995] | a better |
| pedestrian 1w: full_lowlr vs full fine-tuning: `full_lowlr_ped_1w` vs `full_ped_1w` | 0.985 | [0.974, 0.996] | a better |
| pedestrian 1w: head vs zero-shot student: `head_ped_1w` vs `zero_shot` | 0.975 | [0.965, 0.984] | a better |
| pedestrian 1w: head vs full fine-tuning: `head_ped_1w` vs `full_ped_1w` | 1.000 | [0.966, 1.035] | no detectable difference |
| pedestrian 1w: last vs zero-shot student: `last_ped_1w` vs `zero_shot` | 0.963 | [0.937, 0.987] | a better |
| pedestrian 1w: last vs full fine-tuning: `last_ped_1w` vs `full_ped_1w` | 0.988 | [0.965, 1.011] | no detectable difference |
| pedestrian 1w: l2sp_s vs zero-shot student: `l2sp_s_ped_1w` vs `zero_shot` | 0.979 | [0.944, 1.013] | no detectable difference |
| pedestrian 1w: l2sp_s vs full fine-tuning: `l2sp_s_ped_1w` vs `full_ped_1w` | 1.004 | [0.993, 1.014] | no detectable difference |
| pedestrian 1w: l2sp_l vs zero-shot student: `l2sp_l_ped_1w` vs `zero_shot` | 0.956 | [0.937, 0.976] | a better |
| pedestrian 1w: l2sp_l vs full fine-tuning: `l2sp_l_ped_1w` vs `full_ped_1w` | 0.981 | [0.957, 1.006] | no detectable difference |
| pedestrian 1w: replay vs zero-shot student: `replay_ped_1w` vs `zero_shot` | 0.951 | [0.917, 0.984] | a better |
| pedestrian 1w: replay vs full fine-tuning: `replay_ped_1w` vs `full_ped_1w` | 0.975 | [0.964, 0.986] | a better |
| pedestrian 1w: kd vs zero-shot student: `kd_ped_1w` vs `zero_shot` | 0.951 | [0.937, 0.966] | a better |
| pedestrian 1w: kd vs full fine-tuning: `kd_ped_1w` vs `full_ped_1w` | 0.976 | [0.941, 1.012] | no detectable difference |
| pedestrian 1w: scratch vs zero-shot student: `scratch_ped_1w` vs `zero_shot` | 1.093 | [1.058, 1.128] | b better |
| pedestrian 1w: scratch vs full fine-tuning: `scratch_ped_1w` vs `full_ped_1w` | 1.121 | [1.086, 1.158] | b better |
| pedestrian 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1w` vs `zoo:toto-2.0-4m` | 0.837 | [0.790, 0.881] | a better |
| pedestrian 1w: full fine-tuning vs timestral-5m zero-shot: `full_ped_1w` vs `zoo:timestral-5m` | 1.004 | [0.967, 1.042] | no detectable difference |
| pedestrian 1m: full vs zero-shot student: `full_ped_1m` vs `zero_shot` | 0.907 | [0.872, 0.946] | a better |
| pedestrian 1m: full_lowlr vs zero-shot student: `full_lowlr_ped_1m` vs `zero_shot` | 0.899 | [0.868, 0.931] | a better |
| pedestrian 1m: full_lowlr vs full fine-tuning: `full_lowlr_ped_1m` vs `full_ped_1m` | 0.991 | [0.976, 1.006] | no detectable difference |
| pedestrian 1m: head vs zero-shot student: `head_ped_1m` vs `zero_shot` | 0.965 | [0.955, 0.974] | a better |
| pedestrian 1m: head vs full fine-tuning: `head_ped_1m` vs `full_ped_1m` | 1.064 | [1.025, 1.105] | b better |
| pedestrian 1m: last vs zero-shot student: `last_ped_1m` vs `zero_shot` | 0.934 | [0.908, 0.960] | a better |
| pedestrian 1m: last vs full fine-tuning: `last_ped_1m` vs `full_ped_1m` | 1.030 | [1.000, 1.059] | b better |
| pedestrian 1m: l2sp_s vs zero-shot student: `l2sp_s_ped_1m` vs `zero_shot` | 0.906 | [0.877, 0.936] | a better |
| pedestrian 1m: l2sp_s vs full fine-tuning: `l2sp_s_ped_1m` vs `full_ped_1m` | 0.999 | [0.986, 1.013] | no detectable difference |
| pedestrian 1m: l2sp_l vs zero-shot student: `l2sp_l_ped_1m` vs `zero_shot` | 0.942 | [0.921, 0.963] | a better |
| pedestrian 1m: l2sp_l vs full fine-tuning: `l2sp_l_ped_1m` vs `full_ped_1m` | 1.038 | [1.006, 1.069] | b better |
| pedestrian 1m: replay vs zero-shot student: `replay_ped_1m` vs `zero_shot` | 0.885 | [0.857, 0.913] | a better |
| pedestrian 1m: replay vs full fine-tuning: `replay_ped_1m` vs `full_ped_1m` | 0.976 | [0.960, 0.992] | a better |
| pedestrian 1m: kd vs zero-shot student: `kd_ped_1m` vs `zero_shot` | 0.949 | [0.934, 0.962] | a better |
| pedestrian 1m: kd vs full fine-tuning: `kd_ped_1m` vs `full_ped_1m` | 1.046 | [1.009, 1.083] | b better |
| pedestrian 1m: scratch vs zero-shot student: `scratch_ped_1m` vs `zero_shot` | 1.014 | [0.979, 1.051] | no detectable difference |
| pedestrian 1m: scratch vs full fine-tuning: `scratch_ped_1m` vs `full_ped_1m` | 1.118 | [1.081, 1.157] | b better |
| pedestrian 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1m` vs `zoo:toto-2.0-4m` | 0.778 | [0.736, 0.821] | a better |
| pedestrian 1m: full fine-tuning vs timestral-5m zero-shot: `full_ped_1m` vs `zoo:timestral-5m` | 0.934 | [0.899, 0.971] | a better |
| pedestrian 1y: full vs zero-shot student: `full_ped_1y` vs `zero_shot` | 0.844 | [0.812, 0.877] | a better |
| pedestrian 1y: full_lowlr vs zero-shot student: `full_lowlr_ped_1y` vs `zero_shot` | 0.856 | [0.828, 0.884] | a better |
| pedestrian 1y: full_lowlr vs full fine-tuning: `full_lowlr_ped_1y` vs `full_ped_1y` | 1.014 | [1.006, 1.023] | b better |
| pedestrian 1y: head vs zero-shot student: `head_ped_1y` vs `zero_shot` | 0.966 | [0.955, 0.978] | a better |
| pedestrian 1y: head vs full fine-tuning: `head_ped_1y` vs `full_ped_1y` | 1.146 | [1.109, 1.181] | b better |
| pedestrian 1y: last vs zero-shot student: `last_ped_1y` vs `zero_shot` | 0.907 | [0.885, 0.930] | a better |
| pedestrian 1y: last vs full fine-tuning: `last_ped_1y` vs `full_ped_1y` | 1.075 | [1.053, 1.098] | b better |
| pedestrian 1y: l2sp_s vs zero-shot student: `l2sp_s_ped_1y` vs `zero_shot` | 0.872 | [0.846, 0.898] | a better |
| pedestrian 1y: l2sp_s vs full fine-tuning: `l2sp_s_ped_1y` vs `full_ped_1y` | 1.034 | [1.023, 1.045] | b better |
| pedestrian 1y: l2sp_l vs zero-shot student: `l2sp_l_ped_1y` vs `zero_shot` | 0.919 | [0.900, 0.938] | a better |
| pedestrian 1y: l2sp_l vs full fine-tuning: `l2sp_l_ped_1y` vs `full_ped_1y` | 1.089 | [1.063, 1.115] | b better |
| pedestrian 1y: replay vs zero-shot student: `replay_ped_1y` vs `zero_shot` | 0.854 | [0.827, 0.880] | a better |
| pedestrian 1y: replay vs full fine-tuning: `replay_ped_1y` vs `full_ped_1y` | 1.012 | [1.000, 1.023] | b better |
| pedestrian 1y: kd vs zero-shot student: `kd_ped_1y` vs `zero_shot` | 0.945 | [0.930, 0.959] | a better |
| pedestrian 1y: kd vs full fine-tuning: `kd_ped_1y` vs `full_ped_1y` | 1.120 | [1.088, 1.153] | b better |
| pedestrian 1y: scratch vs zero-shot student: `scratch_ped_1y` vs `zero_shot` | 0.990 | [0.959, 1.026] | no detectable difference |
| pedestrian 1y: scratch vs full fine-tuning: `scratch_ped_1y` vs `full_ped_1y` | 1.174 | [1.142, 1.207] | b better |
| pedestrian 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1y` vs `zoo:toto-2.0-4m` | 0.724 | [0.684, 0.764] | a better |
| pedestrian 1y: full fine-tuning vs timestral-5m zero-shot: `full_ped_1y` vs `zoo:timestral-5m` | 0.869 | [0.840, 0.899] | a better |

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| zero_shot | 0.481 | [0.436, 0.533] | – | 0.83 |
| replay_ped_1w_s2 | 0.482 | [0.437, 0.535] | 1.002 [0.989, 1.015] | 0.85 |
| replay_ped_1w_s0 | 0.484 | [0.439, 0.536] | 1.006 [0.995, 1.018] | 0.85 |
| replay_ped_1y_s0 | 0.486 | [0.442, 0.538] | 1.010 [0.994, 1.027] | 0.84 |
| replay_ped_1y_s2 | 0.487 | [0.441, 0.540] | 1.011 [0.996, 1.026] | 0.82 |
| replay_ped_1m_s2 | 0.489 | [0.443, 0.543] | 1.016 [1.001, 1.032] | 0.83 |
| replay_ped_1m_s0 | 0.490 | [0.445, 0.544] | 1.018 [1.001, 1.036] | 0.83 |
| replay_ped_1w_s1 | 0.495 | [0.449, 0.547] | 1.028 [1.006, 1.050] | 0.84 |
| replay_ped_1m_s1 | 0.497 | [0.451, 0.549] | 1.032 [1.012, 1.054] | 0.82 |
| replay_fr_1w_s0 | 0.497 | [0.450, 0.553] | 1.034 [1.019, 1.049] | 0.84 |
| replay_ped_1y_s1 | 0.498 | [0.452, 0.550] | 1.034 [1.011, 1.059] | 0.82 |
| replay_fr_1m_s0 | 0.505 | [0.458, 0.560] | 1.049 [1.030, 1.070] | 0.83 |
| replay_fr_1m_s2 | 0.510 | [0.464, 0.564] | 1.059 [1.034, 1.087] | 0.82 |
| replay_fr_1w_s2 | 0.518 | [0.469, 0.572] | 1.076 [1.047, 1.106] | 0.80 |
| kd_fr_1w_s0 | 0.522 | [0.476, 0.577] | 1.085 [1.064, 1.108] | 0.83 |
| replay_fr_1w_s1 | 0.533 | [0.479, 0.597] | 1.107 [1.078, 1.138] | 0.81 |
| kd_fr_1m_s2 | 0.537 | [0.491, 0.591] | 1.115 [1.083, 1.151] | 0.80 |
| replay_fr_1m_s1 | 0.548 | [0.495, 0.612] | 1.138 [1.083, 1.201] | 0.81 |
| last_fr_1m_s2 | 0.554 | [0.505, 0.613] | 1.152 [1.116, 1.189] | 0.75 |
| replay_fr_1y_s1 | 0.557 | [0.504, 0.620] | 1.158 [1.125, 1.192] | 0.83 |
| replay_fr_1y_s2 | 0.562 | [0.508, 0.627] | 1.168 [1.133, 1.205] | 0.81 |
| kd_fr_1w_s2 | 0.579 | [0.532, 0.636] | 1.203 [1.158, 1.248] | 0.78 |
| replay_fr_1y_s0 | 0.591 | [0.545, 0.647] | 1.229 [1.160, 1.301] | 0.74 |
| full_lowlr_fr_1m_s0 | 0.596 | [0.544, 0.657] | 1.238 [1.185, 1.294] | 0.77 |
| last_fr_1y_s0 | 0.605 | [0.551, 0.668] | 1.257 [1.210, 1.306] | 0.76 |
| kd_fr_1m_s0 | 0.608 | [0.555, 0.673] | 1.264 [1.211, 1.317] | 0.79 |
| head_fr_1y_s0 | 0.618 | [0.566, 0.682] | 1.284 [1.234, 1.336] | 0.80 |
| full_lowlr_fr_1m_s1 | 0.622 | [0.569, 0.684] | 1.293 [1.232, 1.355] | 0.76 |
| full_lowlr_fr_1m_s2 | 0.626 | [0.569, 0.691] | 1.301 [1.239, 1.367] | 0.74 |
| l2sp_s_fr_1m_s1 | 0.628 | [0.569, 0.697] | 1.306 [1.235, 1.381] | 0.76 |
| l2sp_l_fr_1m_s0 | 0.676 | [0.613, 0.748] | 1.405 [1.311, 1.511] | 0.74 |
| full_fr_1m_s2 | 0.678 | [0.614, 0.755] | 1.408 [1.331, 1.489] | 0.73 |
| kd_fr_1m_s1 | 0.679 | [0.617, 0.754] | 1.411 [1.330, 1.494] | 0.76 |
| l2sp_s_fr_1m_s0 | 0.680 | [0.618, 0.750] | 1.412 [1.325, 1.503] | 0.72 |
| l2sp_l_ped_1y_s1 | 0.681 | [0.611, 0.768] | 1.416 [1.316, 1.526] | 0.62 |
| kd_fr_1w_s1 | 0.683 | [0.625, 0.752] | 1.420 [1.333, 1.516] | 0.69 |
| l2sp_l_ped_1m_s2 | 0.686 | [0.616, 0.773] | 1.426 [1.323, 1.541] | 0.60 |
| l2sp_l_ped_1m_s1 | 0.689 | [0.618, 0.775] | 1.432 [1.330, 1.545] | 0.60 |
| l2sp_l_ped_1y_s0 | 0.693 | [0.622, 0.782] | 1.441 [1.333, 1.562] | 0.62 |
| l2sp_l_ped_1w_s0 | 0.694 | [0.627, 0.776] | 1.441 [1.325, 1.566] | 0.61 |
| l2sp_s_fr_1m_s2 | 0.696 | [0.632, 0.775] | 1.446 [1.364, 1.529] | 0.73 |
| kd_fr_1y_s2 | 0.697 | [0.636, 0.771] | 1.447 [1.369, 1.530] | 0.84 |
| l2sp_l_ped_1y_s2 | 0.699 | [0.627, 0.789] | 1.453 [1.340, 1.579] | 0.60 |
| kd_fr_1y_s0 | 0.701 | [0.642, 0.772] | 1.456 [1.361, 1.559] | 0.80 |
| l2sp_l_fr_1m_s2 | 0.707 | [0.640, 0.781] | 1.468 [1.365, 1.581] | 0.75 |
| l2sp_l_fr_1y_s2 | 0.711 | [0.649, 0.783] | 1.478 [1.366, 1.605] | 0.66 |
| full_fr_1m_s1 | 0.730 | [0.659, 0.813] | 1.518 [1.403, 1.638] | 0.66 |
| full_fr_1m_s0 | 0.733 | [0.665, 0.816] | 1.523 [1.431, 1.618] | 0.68 |
| l2sp_l_fr_1m_s1 | 0.734 | [0.664, 0.814] | 1.525 [1.405, 1.657] | 0.69 |
| kd_fr_1y_s1 | 0.739 | [0.676, 0.817] | 1.535 [1.432, 1.647] | 0.78 |
| l2sp_l_ped_1w_s1 | 0.747 | [0.672, 0.839] | 1.552 [1.416, 1.703] | 0.58 |
| last_fr_1w_s1 | 0.750 | [0.663, 0.863] | 1.557 [1.446, 1.683] | 0.60 |
| l2sp_l_ped_1w_s2 | 0.758 | [0.681, 0.853] | 1.575 [1.432, 1.735] | 0.56 |
| l2sp_l_ped_1m_s0 | 0.760 | [0.683, 0.852] | 1.579 [1.448, 1.722] | 0.55 |
| full_lowlr_fr_1w_s1 | 0.763 | [0.682, 0.863] | 1.586 [1.467, 1.718] | 0.69 |
| last_fr_1w_s0 | 0.764 | [0.675, 0.879] | 1.588 [1.471, 1.719] | 0.59 |
| last_fr_1m_s0 | 0.768 | [0.686, 0.867] | 1.595 [1.492, 1.704] | 0.58 |
| last_fr_1w_s2 | 0.771 | [0.681, 0.888] | 1.602 [1.484, 1.735] | 0.59 |
| l2sp_s_fr_1y_s2 | 0.777 | [0.706, 0.862] | 1.615 [1.474, 1.782] | 0.62 |
| head_fr_1y_s2 | 0.788 | [0.717, 0.873] | 1.637 [1.518, 1.771] | 0.68 |
| full_fr_1y_s2 | 0.789 | [0.715, 0.876] | 1.640 [1.493, 1.811] | 0.61 |
| last_fr_1y_s2 | 0.789 | [0.713, 0.880] | 1.640 [1.509, 1.786] | 0.63 |
| head_ped_1m_s0 | 0.806 | [0.719, 0.906] | 1.674 [1.530, 1.835] | 0.63 |
| head_ped_1m_s2 | 0.810 | [0.723, 0.910] | 1.684 [1.538, 1.848] | 0.63 |
| head_ped_1m_s1 | 0.815 | [0.728, 0.915] | 1.694 [1.548, 1.859] | 0.62 |
| full_lowlr_fr_1w_s2 | 0.816 | [0.730, 0.922] | 1.696 [1.560, 1.850] | 0.66 |
| full_lowlr_fr_1w_s0 | 0.822 | [0.732, 0.932] | 1.707 [1.564, 1.869] | 0.67 |
| full_lowlr_fr_1y_s2 | 0.832 | [0.752, 0.922] | 1.729 [1.567, 1.917] | 0.60 |
| l2sp_l_fr_1w_s2 | 0.833 | [0.745, 0.944] | 1.732 [1.587, 1.892] | 0.72 |
| l2sp_l_fr_1y_s1 | 0.840 | [0.757, 0.937] | 1.746 [1.574, 1.952] | 0.60 |
| head_fr_1m_s2 | 0.845 | [0.759, 0.945] | 1.755 [1.580, 1.956] | 0.54 |
| last_ped_1m_s2 | 0.865 | [0.761, 0.998] | 1.798 [1.649, 1.957] | 0.50 |
| head_fr_1m_s0 | 0.870 | [0.778, 0.975] | 1.808 [1.618, 2.032] | 0.54 |
| last_fr_1m_s1 | 0.874 | [0.779, 0.992] | 1.816 [1.684, 1.959] | 0.52 |
| head_fr_1m_s1 | 0.876 | [0.783, 0.984] | 1.821 [1.631, 2.044] | 0.53 |
| l2sp_s_fr_1w_s2 | 0.877 | [0.779, 1.000] | 1.823 [1.663, 2.005] | 0.68 |
| full_fr_1w_s2 | 0.887 | [0.787, 1.012] | 1.842 [1.678, 2.029] | 0.67 |
| head_ped_1w_s0 | 0.890 | [0.804, 0.994] | 1.849 [1.682, 2.036] | 0.60 |
| head_ped_1w_s1 | 0.893 | [0.806, 0.998] | 1.855 [1.684, 2.043] | 0.60 |
| head_ped_1w_s2 | 0.897 | [0.809, 1.004] | 1.864 [1.693, 2.051] | 0.59 |
| last_ped_1m_s1 | 0.897 | [0.790, 1.032] | 1.865 [1.704, 2.036] | 0.49 |
| kd_ped_1m_s1 | 0.898 | [0.797, 1.025] | 1.867 [1.669, 2.091] | 0.61 |
| l2sp_l_fr_1y_s0 | 0.899 | [0.810, 1.002] | 1.868 [1.682, 2.087] | 0.61 |
| last_ped_1m_s0 | 0.901 | [0.791, 1.041] | 1.872 [1.715, 2.048] | 0.49 |
| head_fr_1y_s1 | 0.904 | [0.816, 1.012] | 1.878 [1.700, 2.081] | 0.62 |
| head_ped_1y_s0 | 0.906 | [0.805, 1.031] | 1.883 [1.691, 2.098] | 0.58 |
| full_lowlr_ped_1y_s1 | 0.912 | [0.805, 1.040] | 1.894 [1.692, 2.123] | 0.54 |
| last_ped_1w_s1 | 0.914 | [0.799, 1.058] | 1.898 [1.740, 2.084] | 0.47 |
| kd_ped_1m_s0 | 0.919 | [0.812, 1.049] | 1.910 [1.695, 2.160] | 0.63 |
| kd_ped_1w_s2 | 0.921 | [0.819, 1.051] | 1.915 [1.710, 2.153] | 0.62 |
| full_lowlr_fr_1y_s1 | 0.927 | [0.828, 1.041] | 1.925 [1.704, 2.189] | 0.56 |
| last_ped_1w_s2 | 0.928 | [0.811, 1.074] | 1.927 [1.763, 2.115] | 0.47 |
| last_ped_1w_s0 | 0.928 | [0.811, 1.073] | 1.927 [1.763, 2.123] | 0.47 |
| full_lowlr_ped_1y_s2 | 0.928 | [0.823, 1.059] | 1.929 [1.720, 2.165] | 0.53 |
| head_fr_1w_s1 | 0.932 | [0.821, 1.071] | 1.937 [1.767, 2.133] | 0.53 |
| l2sp_s_fr_1y_s1 | 0.933 | [0.832, 1.048] | 1.940 [1.715, 2.208] | 0.56 |
| head_ped_1y_s1 | 0.938 | [0.832, 1.069] | 1.949 [1.744, 2.181] | 0.56 |
| head_ped_1y_s2 | 0.944 | [0.836, 1.078] | 1.961 [1.751, 2.198] | 0.56 |
| full_lowlr_ped_1w_s0 | 0.948 | [0.843, 1.076] | 1.969 [1.751, 2.220] | 0.48 |
| l2sp_l_fr_1w_s0 | 0.956 | [0.849, 1.087] | 1.987 [1.777, 2.219] | 0.71 |
| full_lowlr_ped_1w_s2 | 0.959 | [0.850, 1.092] | 1.992 [1.768, 2.248] | 0.47 |
| l2sp_s_fr_1y_s0 | 0.963 | [0.861, 1.082] | 2.000 [1.779, 2.258] | 0.57 |
| l2sp_s_ped_1y_s2 | 0.965 | [0.846, 1.116] | 2.006 [1.770, 2.281] | 0.54 |
| head_fr_1w_s2 | 0.966 | [0.852, 1.108] | 2.007 [1.825, 2.215] | 0.53 |
| l2sp_s_ped_1y_s0 | 0.967 | [0.847, 1.121] | 2.010 [1.772, 2.287] | 0.54 |
| full_fr_1y_s1 | 0.968 | [0.859, 1.091] | 2.012 [1.771, 2.301] | 0.55 |
| l2sp_s_ped_1y_s1 | 0.971 | [0.849, 1.126] | 2.017 [1.778, 2.291] | 0.54 |
| full_fr_1y_s0 | 0.972 | [0.868, 1.096] | 2.021 [1.796, 2.287] | 0.56 |
| full_lowlr_ped_1y_s0 | 0.974 | [0.857, 1.122] | 2.025 [1.790, 2.293] | 0.52 |
| kd_ped_1w_s1 | 0.975 | [0.860, 1.119] | 2.025 [1.792, 2.298] | 0.59 |
| full_lowlr_ped_1w_s1 | 0.976 | [0.864, 1.112] | 2.028 [1.794, 2.294] | 0.47 |
| full_lowlr_ped_1m_s0 | 0.980 | [0.857, 1.133] | 2.036 [1.802, 2.303] | 0.49 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| last_fr_1y_s1 | 1.002 | [0.882, 1.141] | 2.083 [1.850, 2.359] | 0.54 |
| head_fr_1w_s0 | 1.028 | [0.906, 1.182] | 2.136 [1.928, 2.376] | 0.51 |
| kd_ped_1y_s1 | 1.034 | [0.908, 1.190] | 2.149 [1.890, 2.455] | 0.60 |
| l2sp_s_fr_1w_s0 | 1.048 | [0.927, 1.186] | 2.178 [1.929, 2.449] | 0.67 |
| l2sp_s_fr_1w_s1 | 1.050 | [0.928, 1.191] | 2.183 [1.930, 2.454] | 0.64 |
| kd_ped_1y_s2 | 1.052 | [0.921, 1.215] | 2.185 [1.915, 2.497] | 0.60 |
| kd_ped_1w_s0 | 1.059 | [0.931, 1.219] | 2.199 [1.930, 2.510] | 0.57 |
| full_fr_1w_s0 | 1.059 | [0.937, 1.200] | 2.200 [1.949, 2.470] | 0.67 |
| l2sp_l_fr_1w_s1 | 1.063 | [0.936, 1.203] | 2.209 [1.942, 2.502] | 0.62 |
| full_lowlr_ped_1m_s1 | 1.063 | [0.923, 1.241] | 2.209 [1.933, 2.526] | 0.47 |
| full_lowlr_ped_1m_s2 | 1.071 | [0.928, 1.249] | 2.225 [1.945, 2.549] | 0.46 |
| full_lowlr_fr_1y_s0 | 1.073 | [0.941, 1.223] | 2.230 [1.930, 2.585] | 0.53 |
| kd_ped_1y_s0 | 1.077 | [0.940, 1.246] | 2.238 [1.955, 2.570] | 0.58 |
| last_ped_1y_s1 | 1.081 | [0.943, 1.251] | 2.245 [1.988, 2.531] | 0.37 |
| full_fr_1w_s1 | 1.085 | [0.956, 1.239] | 2.255 [1.989, 2.549] | 0.59 |
| kd_ped_1m_s2 | 1.088 | [0.948, 1.265] | 2.261 [1.976, 2.584] | 0.61 |
| last_ped_1y_s0 | 1.096 | [0.953, 1.272] | 2.277 [2.008, 2.580] | 0.38 |
| full_ped_1y_s2 | 1.109 | [0.976, 1.271] | 2.304 [2.021, 2.640] | 0.54 |
| last_ped_1y_s2 | 1.141 | [0.992, 1.327] | 2.371 [2.088, 2.689] | 0.36 |
| l2sp_s_ped_1m_s1 | 1.178 | [1.019, 1.382] | 2.449 [2.109, 2.829] | 0.43 |
| full_ped_1y_s1 | 1.186 | [1.032, 1.377] | 2.464 [2.127, 2.847] | 0.49 |
| l2sp_s_ped_1w_s2 | 1.191 | [1.026, 1.389] | 2.475 [2.122, 2.894] | 0.42 |
| l2sp_s_ped_1w_s0 | 1.194 | [1.035, 1.386] | 2.481 [2.137, 2.889] | 0.42 |
| l2sp_s_ped_1w_s1 | 1.195 | [1.033, 1.390] | 2.483 [2.131, 2.894] | 0.41 |
| l2sp_s_ped_1m_s2 | 1.200 | [1.031, 1.414] | 2.494 [2.142, 2.899] | 0.42 |
| full_ped_1y_s0 | 1.214 | [1.054, 1.411] | 2.522 [2.180, 2.927] | 0.51 |
| l2sp_s_ped_1m_s0 | 1.215 | [1.044, 1.429] | 2.525 [2.168, 2.932] | 0.41 |
| full_ped_1w_s1 | 1.250 | [1.072, 1.462] | 2.597 [2.229, 3.035] | 0.41 |
| full_ped_1w_s2 | 1.302 | [1.113, 1.527] | 2.705 [2.313, 3.175] | 0.41 |
| full_ped_1w_s0 | 1.325 | [1.135, 1.548] | 2.752 [2.357, 3.220] | 0.40 |
| full_ped_1m_s1 | 1.562 | [1.307, 1.870] | 3.245 [2.715, 3.885] | 0.40 |
| full_ped_1m_s0 | 1.566 | [1.306, 1.882] | 3.255 [2.709, 3.910] | 0.40 |
| full_ped_1m_s2 | 1.589 | [1.324, 1.918] | 3.301 [2.752, 3.956] | 0.39 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_power 1w: full vs zero-shot student: `full_fr_1w` vs `zero_shot` | 2.119 | [1.890, 2.368] | b better |
| fr_power 1w: full_lowlr vs zero-shot student: `full_lowlr_fr_1w` vs `zero_shot` | 1.665 | [1.533, 1.816] | b better |
| fr_power 1w: full_lowlr vs full fine-tuning: `full_lowlr_fr_1w` vs `full_fr_1w` | 0.786 | [0.755, 0.818] | a better |
| fr_power 1w: head vs zero-shot student: `head_fr_1w` vs `zero_shot` | 2.030 | [1.844, 2.245] | b better |
| fr_power 1w: head vs full fine-tuning: `head_fr_1w` vs `full_fr_1w` | 0.958 | [0.893, 1.026] | no detectable difference |
| fr_power 1w: last vs zero-shot student: `last_fr_1w` vs `zero_shot` | 1.583 | [1.469, 1.713] | b better |
| fr_power 1w: last vs full fine-tuning: `last_fr_1w` vs `full_fr_1w` | 0.747 | [0.697, 0.804] | a better |
| fr_power 1w: l2sp_s vs zero-shot student: `l2sp_s_fr_1w` vs `zero_shot` | 2.083 | [1.862, 2.324] | b better |
| fr_power 1w: l2sp_s vs full fine-tuning: `l2sp_s_fr_1w` vs `full_fr_1w` | 0.983 | [0.978, 0.988] | a better |
| fr_power 1w: l2sp_l vs zero-shot student: `l2sp_l_fr_1w` vs `zero_shot` | 2.000 | [1.789, 2.226] | b better |
| fr_power 1w: l2sp_l vs full fine-tuning: `l2sp_l_fr_1w` vs `full_fr_1w` | 0.944 | [0.930, 0.957] | a better |
| fr_power 1w: replay vs zero-shot student: `replay_fr_1w` vs `zero_shot` | 1.081 | [1.061, 1.102] | b better |
| fr_power 1w: replay vs full fine-tuning: `replay_fr_1w` vs `full_fr_1w` | 0.510 | [0.461, 0.566] | a better |
| fr_power 1w: kd vs zero-shot student: `kd_fr_1w` vs `zero_shot` | 1.251 | [1.200, 1.306] | b better |
| fr_power 1w: kd vs full fine-tuning: `kd_fr_1w` vs `full_fr_1w` | 0.590 | [0.546, 0.638] | a better |
| fr_power 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1w` vs `zoo:toto-2.0-4m` | 1.964 | [1.777, 2.165] | b better |
| fr_power 1w: full fine-tuning vs timestral-5m zero-shot: `full_fr_1w` vs `zoo:timestral-5m` | 2.074 | [1.838, 2.320] | b better |
| fr_power 1m: full vs zero-shot student: `full_fr_1m` vs `zero_shot` | 1.497 | [1.402, 1.594] | b better |
| fr_power 1m: full_lowlr vs zero-shot student: `full_lowlr_fr_1m` vs `zero_shot` | 1.279 | [1.221, 1.340] | b better |
| fr_power 1m: full_lowlr vs full fine-tuning: `full_lowlr_fr_1m` vs `full_fr_1m` | 0.855 | [0.834, 0.876] | a better |
| fr_power 1m: head vs zero-shot student: `head_fr_1m` vs `zero_shot` | 1.796 | [1.612, 2.009] | b better |
| fr_power 1m: head vs full fine-tuning: `head_fr_1m` vs `full_fr_1m` | 1.200 | [1.122, 1.289] | b better |
| fr_power 1m: last vs zero-shot student: `last_fr_1m` vs `zero_shot` | 1.547 | [1.460, 1.640] | b better |
| fr_power 1m: last vs full fine-tuning: `last_fr_1m` vs `full_fr_1m` | 1.034 | [0.997, 1.071] | no detectable difference |
| fr_power 1m: l2sp_s vs zero-shot student: `l2sp_s_fr_1m` vs `zero_shot` | 1.396 | [1.318, 1.478] | b better |
| fr_power 1m: l2sp_s vs full fine-tuning: `l2sp_s_fr_1m` vs `full_fr_1m` | 0.933 | [0.921, 0.945] | a better |
| fr_power 1m: l2sp_l vs zero-shot student: `l2sp_l_fr_1m` vs `zero_shot` | 1.469 | [1.363, 1.587] | b better |
| fr_power 1m: l2sp_l vs full fine-tuning: `l2sp_l_fr_1m` vs `full_fr_1m` | 0.982 | [0.954, 1.009] | no detectable difference |
| fr_power 1m: replay vs zero-shot student: `replay_fr_1m` vs `zero_shot` | 1.101 | [1.069, 1.139] | b better |
| fr_power 1m: replay vs full fine-tuning: `replay_fr_1m` vs `full_fr_1m` | 0.736 | [0.700, 0.776] | a better |
| fr_power 1m: kd vs zero-shot student: `kd_fr_1m` vs `zero_shot` | 1.274 | [1.219, 1.330] | b better |
| fr_power 1m: kd vs full fine-tuning: `kd_fr_1m` vs `full_fr_1m` | 0.851 | [0.824, 0.880] | a better |
| fr_power 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1m` vs `zoo:toto-2.0-4m` | 1.387 | [1.306, 1.471] | b better |
| fr_power 1m: full fine-tuning vs timestral-5m zero-shot: `full_fr_1m` vs `zoo:timestral-5m` | 1.464 | [1.362, 1.564] | b better |
| fr_power 1y: full vs zero-shot student: `full_fr_1y` vs `zero_shot` | 1.909 | [1.701, 2.153] | b better |
| fr_power 1y: full_lowlr vs zero-shot student: `full_lowlr_fr_1y` vs `zero_shot` | 1.980 | [1.746, 2.253] | b better |
| fr_power 1y: full_lowlr vs full fine-tuning: `full_lowlr_fr_1y` vs `full_fr_1y` | 1.037 | [1.023, 1.052] | b better |
| fr_power 1y: head vs zero-shot student: `head_fr_1y` vs `zero_shot` | 1.636 | [1.515, 1.769] | b better |
| fr_power 1y: head vs full fine-tuning: `head_fr_1y` vs `full_fr_1y` | 0.857 | [0.815, 0.899] | a better |
| fr_power 1y: last vs zero-shot student: `last_fr_1y` vs `zero_shot` | 1.722 | [1.577, 1.887] | b better |
| fr_power 1y: last vs full fine-tuning: `last_fr_1y` vs `full_fr_1y` | 0.902 | [0.864, 0.942] | a better |
| fr_power 1y: l2sp_s vs zero-shot student: `l2sp_s_fr_1y` vs `zero_shot` | 1.869 | [1.670, 2.100] | b better |
| fr_power 1y: l2sp_s vs full fine-tuning: `l2sp_s_fr_1y` vs `full_fr_1y` | 0.979 | [0.975, 0.983] | a better |
| fr_power 1y: l2sp_l vs zero-shot student: `l2sp_l_fr_1y` vs `zero_shot` | 1.711 | [1.553, 1.900] | b better |
| fr_power 1y: l2sp_l vs full fine-tuning: `l2sp_l_fr_1y` vs `full_fr_1y` | 0.897 | [0.878, 0.915] | a better |
| fr_power 1y: replay vs zero-shot student: `replay_fr_1y` vs `zero_shot` | 1.204 | [1.161, 1.250] | b better |
| fr_power 1y: replay vs full fine-tuning: `replay_fr_1y` vs `full_fr_1y` | 0.631 | [0.578, 0.688] | a better |
| fr_power 1y: kd vs zero-shot student: `kd_fr_1y` vs `zero_shot` | 1.489 | [1.398, 1.588] | b better |
| fr_power 1y: kd vs full fine-tuning: `kd_fr_1y` vs `full_fr_1y` | 0.780 | [0.726, 0.839] | a better |
| fr_power 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_fr_1y` vs `zoo:toto-2.0-4m` | 1.769 | [1.614, 1.947] | b better |
| fr_power 1y: full fine-tuning vs timestral-5m zero-shot: `full_fr_1y` vs `zoo:timestral-5m` | 1.868 | [1.663, 2.102] | b better |
| pedestrian 1w: full vs zero-shot student: `full_ped_1w` vs `zero_shot` | 2.686 | [2.303, 3.144] | b better |
| pedestrian 1w: full_lowlr vs zero-shot student: `full_lowlr_ped_1w` vs `zero_shot` | 1.997 | [1.772, 2.255] | b better |
| pedestrian 1w: full_lowlr vs full fine-tuning: `full_lowlr_ped_1w` vs `full_ped_1w` | 0.743 | [0.713, 0.776] | a better |
| pedestrian 1w: head vs zero-shot student: `head_ped_1w` vs `zero_shot` | 1.856 | [1.687, 2.044] | b better |
| pedestrian 1w: head vs full fine-tuning: `head_ped_1w` vs `full_ped_1w` | 0.691 | [0.644, 0.742] | a better |
| pedestrian 1w: last vs zero-shot student: `last_ped_1w` vs `zero_shot` | 1.919 | [1.757, 2.108] | b better |
| pedestrian 1w: last vs full fine-tuning: `last_ped_1w` vs `full_ped_1w` | 0.714 | [0.640, 0.796] | a better |
| pedestrian 1w: l2sp_s vs zero-shot student: `l2sp_s_ped_1w` vs `zero_shot` | 2.481 | [2.131, 2.891] | b better |
| pedestrian 1w: l2sp_s vs full fine-tuning: `l2sp_s_ped_1w` vs `full_ped_1w` | 0.924 | [0.906, 0.942] | a better |
| pedestrian 1w: l2sp_l vs zero-shot student: `l2sp_l_ped_1w` vs `zero_shot` | 1.526 | [1.395, 1.672] | b better |
| pedestrian 1w: l2sp_l vs full fine-tuning: `l2sp_l_ped_1w` vs `full_ped_1w` | 0.568 | [0.518, 0.621] | a better |
| pedestrian 1w: replay vs zero-shot student: `replay_ped_1w` vs `zero_shot` | 1.014 | [1.000, 1.028] | b better |
| pedestrian 1w: replay vs full fine-tuning: `replay_ped_1w` vs `full_ped_1w` | 0.377 | [0.326, 0.436] | a better |
| pedestrian 1w: kd vs zero-shot student: `kd_ped_1w` vs `zero_shot` | 2.052 | [1.816, 2.325] | b better |
| pedestrian 1w: kd vs full fine-tuning: `kd_ped_1w` vs `full_ped_1w` | 0.764 | [0.733, 0.795] | a better |
| pedestrian 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1w` vs `zoo:toto-2.0-4m` | 2.489 | [2.167, 2.851] | b better |
| pedestrian 1w: full fine-tuning vs timestral-5m zero-shot: `full_ped_1w` vs `zoo:timestral-5m` | 2.629 | [2.249, 3.060] | b better |
| pedestrian 1m: full vs zero-shot student: `full_ped_1m` vs `zero_shot` | 3.272 | [2.728, 3.920] | b better |
| pedestrian 1m: full_lowlr vs zero-shot student: `full_lowlr_ped_1m` vs `zero_shot` | 2.159 | [1.895, 2.459] | b better |
| pedestrian 1m: full_lowlr vs full fine-tuning: `full_lowlr_ped_1m` vs `full_ped_1m` | 0.660 | [0.621, 0.701] | a better |
| pedestrian 1m: head vs zero-shot student: `head_ped_1m` vs `zero_shot` | 1.684 | [1.539, 1.848] | b better |
| pedestrian 1m: head vs full fine-tuning: `head_ped_1m` vs `full_ped_1m` | 0.515 | [0.455, 0.583] | a better |
| pedestrian 1m: last vs zero-shot student: `last_ped_1m` vs `zero_shot` | 1.846 | [1.692, 2.015] | b better |
| pedestrian 1m: last vs full fine-tuning: `last_ped_1m` vs `full_ped_1m` | 0.564 | [0.502, 0.637] | a better |
| pedestrian 1m: l2sp_s vs zero-shot student: `l2sp_s_ped_1m` vs `zero_shot` | 2.491 | [2.139, 2.892] | b better |
| pedestrian 1m: l2sp_s vs full fine-tuning: `l2sp_s_ped_1m` vs `full_ped_1m` | 0.761 | [0.721, 0.802] | a better |
| pedestrian 1m: l2sp_l vs zero-shot student: `l2sp_l_ped_1m` vs `zero_shot` | 1.484 | [1.373, 1.608] | b better |
| pedestrian 1m: l2sp_l vs full fine-tuning: `l2sp_l_ped_1m` vs `full_ped_1m` | 0.454 | [0.403, 0.511] | a better |
| pedestrian 1m: replay vs zero-shot student: `replay_ped_1m` vs `zero_shot` | 1.025 | [1.009, 1.041] | b better |
| pedestrian 1m: replay vs full fine-tuning: `replay_ped_1m` vs `full_ped_1m` | 0.313 | [0.264, 0.372] | a better |
| pedestrian 1m: kd vs zero-shot student: `kd_ped_1m` vs `zero_shot` | 2.022 | [1.790, 2.290] | b better |
| pedestrian 1m: kd vs full fine-tuning: `kd_ped_1m` vs `full_ped_1m` | 0.618 | [0.569, 0.667] | a better |
| pedestrian 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1m` vs `zoo:toto-2.0-4m` | 3.032 | [2.560, 3.553] | b better |
| pedestrian 1m: full fine-tuning vs timestral-5m zero-shot: `full_ped_1m` vs `zoo:timestral-5m` | 3.202 | [2.666, 3.833] | b better |
| pedestrian 1y: full vs zero-shot student: `full_ped_1y` vs `zero_shot` | 2.439 | [2.117, 2.812] | b better |
| pedestrian 1y: full_lowlr vs zero-shot student: `full_lowlr_ped_1y` vs `zero_shot` | 1.952 | [1.738, 2.200] | b better |
| pedestrian 1y: full_lowlr vs full fine-tuning: `full_lowlr_ped_1y` vs `full_ped_1y` | 0.800 | [0.772, 0.829] | a better |
| pedestrian 1y: head vs zero-shot student: `head_ped_1y` vs `zero_shot` | 1.932 | [1.729, 2.160] | b better |
| pedestrian 1y: head vs full fine-tuning: `head_ped_1y` vs `full_ped_1y` | 0.792 | [0.751, 0.836] | a better |
| pedestrian 1y: last vs zero-shot student: `last_ped_1y` vs `zero_shot` | 2.300 | [2.029, 2.601] | b better |
| pedestrian 1y: last vs full fine-tuning: `last_ped_1y` vs `full_ped_1y` | 0.943 | [0.903, 0.985] | a better |
| pedestrian 1y: l2sp_s vs zero-shot student: `l2sp_s_ped_1y` vs `zero_shot` | 2.011 | [1.774, 2.287] | b better |
| pedestrian 1y: l2sp_s vs full fine-tuning: `l2sp_s_ped_1y` vs `full_ped_1y` | 0.825 | [0.789, 0.862] | a better |
| pedestrian 1y: l2sp_l vs zero-shot student: `l2sp_l_ped_1y` vs `zero_shot` | 1.438 | [1.331, 1.556] | b better |
| pedestrian 1y: l2sp_l vs full fine-tuning: `l2sp_l_ped_1y` vs `full_ped_1y` | 0.590 | [0.541, 0.641] | a better |
| pedestrian 1y: replay vs zero-shot student: `replay_ped_1y` vs `zero_shot` | 1.021 | [1.005, 1.038] | b better |
| pedestrian 1y: replay vs full fine-tuning: `replay_ped_1y` vs `full_ped_1y` | 0.419 | [0.367, 0.478] | a better |
| pedestrian 1y: kd vs zero-shot student: `kd_ped_1y` vs `zero_shot` | 2.193 | [1.921, 2.507] | b better |
| pedestrian 1y: kd vs full fine-tuning: `kd_ped_1y` vs `full_ped_1y` | 0.899 | [0.876, 0.924] | a better |
| pedestrian 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_ped_1y` vs `zoo:toto-2.0-4m` | 2.259 | [1.999, 2.547] | b better |
| pedestrian 1y: full fine-tuning vs timestral-5m zero-shot: `full_ped_1y` vs `zoo:timestral-5m` | 2.386 | [2.064, 2.743] | b better |
