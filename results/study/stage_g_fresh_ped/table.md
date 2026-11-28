# stage_g_fresh_ped: relative WQL vs seasonal naive (95% CI); paired ratio vs `zero_shot` (<1 = better)

## fresh_mel_ped (2207 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| full_mel_1y_s1 | 0.479 | [0.450, 0.505] | 0.854 [0.839, 0.870] | 0.78 |
| replay_mel_1y_s2 | 0.480 | [0.451, 0.507] | 0.857 [0.843, 0.871] | 0.79 |
| full_lowlr_mel_1y_s1 | 0.480 | [0.452, 0.507] | 0.857 [0.844, 0.871] | 0.79 |
| replay_mel_1y_s1 | 0.480 | [0.451, 0.507] | 0.857 [0.844, 0.871] | 0.79 |
| full_lowlr_mel_1y_s2 | 0.481 | [0.452, 0.507] | 0.858 [0.845, 0.871] | 0.79 |
| full_mel_1y_s2 | 0.481 | [0.452, 0.508] | 0.859 [0.843, 0.875] | 0.78 |
| full_lowlr_mel_1y_s0 | 0.482 | [0.453, 0.508] | 0.860 [0.847, 0.872] | 0.81 |
| full_mel_1y_s0 | 0.482 | [0.452, 0.509] | 0.860 [0.844, 0.876] | 0.80 |
| l2sp_s_mel_1y_s1 | 0.484 | [0.456, 0.511] | 0.864 [0.852, 0.877] | 0.80 |
| replay_mel_1y_s0 | 0.484 | [0.455, 0.511] | 0.864 [0.851, 0.878] | 0.79 |
| l2sp_s_mel_1y_s2 | 0.485 | [0.456, 0.512] | 0.865 [0.852, 0.879] | 0.79 |
| l2sp_s_mel_1y_s0 | 0.491 | [0.463, 0.517] | 0.876 [0.864, 0.888] | 0.82 |
| last_mel_1y_s2 | 0.498 | [0.469, 0.525] | 0.889 [0.877, 0.900] | 0.82 |
| last_mel_1y_s1 | 0.498 | [0.470, 0.525] | 0.890 [0.879, 0.901] | 0.82 |
| last_mel_1y_s0 | 0.499 | [0.470, 0.526] | 0.891 [0.880, 0.902] | 0.82 |
| l2sp_l_mel_1y_s1 | 0.504 | [0.476, 0.530] | 0.899 [0.890, 0.910] | 0.81 |
| l2sp_l_mel_1y_s2 | 0.504 | [0.476, 0.530] | 0.900 [0.890, 0.910] | 0.81 |
| l2sp_l_mel_1y_s0 | 0.505 | [0.478, 0.531] | 0.902 [0.893, 0.911] | 0.82 |
| scratch_mel_1y_s1 | 0.522 | [0.497, 0.546] | 0.932 [0.922, 0.942] | 0.81 |
| scratch_mel_1y_s2 | 0.522 | [0.497, 0.547] | 0.933 [0.923, 0.943] | 0.84 |
| scratch_mel_1y_s0 | 0.523 | [0.499, 0.547] | 0.934 [0.924, 0.946] | 0.83 |
| kd_mel_1y_s0 | 0.529 | [0.499, 0.556] | 0.944 [0.935, 0.953] | 0.86 |
| kd_mel_1y_s1 | 0.533 | [0.503, 0.561] | 0.952 [0.944, 0.961] | 0.86 |
| kd_mel_1y_s2 | 0.534 | [0.504, 0.562] | 0.954 [0.945, 0.962] | 0.86 |
| kd_mel_1m_s2 | 0.534 | [0.504, 0.562] | 0.954 [0.945, 0.962] | 0.85 |
| kd_mel_1m_s1 | 0.537 | [0.507, 0.564] | 0.958 [0.950, 0.966] | 0.86 |
| kd_mel_1m_s0 | 0.538 | [0.509, 0.566] | 0.961 [0.953, 0.969] | 0.86 |
| head_mel_1y_s1 | 0.538 | [0.510, 0.565] | 0.961 [0.957, 0.965] | 0.81 |
| head_mel_1y_s2 | 0.538 | [0.510, 0.565] | 0.961 [0.957, 0.965] | 0.81 |
| head_mel_1y_s0 | 0.539 | [0.510, 0.565] | 0.962 [0.958, 0.966] | 0.81 |
| l2sp_s_mel_1m_s1 | 0.560 | [0.531, 0.587] | 1.000 [0.988, 1.013] | 0.81 |
| zero_shot | 0.560 | [0.531, 0.586] | – | 0.85 |
| full_mel_1m_s1 | 0.562 | [0.533, 0.589] | 1.004 [0.992, 1.018] | 0.80 |
| head_mel_1m_s2 | 0.563 | [0.537, 0.587] | 1.005 [0.998, 1.014] | 0.82 |
| head_mel_1m_s1 | 0.564 | [0.538, 0.588] | 1.006 [1.000, 1.014] | 0.81 |
| head_mel_1m_s0 | 0.565 | [0.539, 0.589] | 1.009 [1.002, 1.017] | 0.82 |
| full_lowlr_mel_1m_s1 | 0.568 | [0.539, 0.595] | 1.015 [1.002, 1.029] | 0.79 |
| kd_mel_1w_s1 | 0.574 | [0.546, 0.600] | 1.026 [1.018, 1.033] | 0.84 |
| l2sp_l_mel_1m_s0 | 0.580 | [0.550, 0.607] | 1.035 [1.023, 1.047] | 0.81 |
| kd_mel_1w_s0 | 0.582 | [0.554, 0.607] | 1.039 [1.031, 1.047] | 0.86 |
| l2sp_l_mel_1m_s1 | 0.582 | [0.552, 0.612] | 1.039 [1.025, 1.055] | 0.78 |
| l2sp_l_mel_1m_s2 | 0.582 | [0.552, 0.610] | 1.039 [1.029, 1.051] | 0.81 |
| kd_mel_1w_s2 | 0.582 | [0.554, 0.608] | 1.039 [1.032, 1.047] | 0.86 |
| replay_mel_1m_s0 | 0.582 | [0.553, 0.610] | 1.040 [1.027, 1.053] | 0.75 |
| l2sp_s_mel_1m_s2 | 0.584 | [0.555, 0.613] | 1.044 [1.031, 1.059] | 0.80 |
| full_lowlr_mel_1m_s0 | 0.585 | [0.555, 0.613] | 1.044 [1.030, 1.058] | 0.78 |
| full_lowlr_mel_1m_s2 | 0.586 | [0.557, 0.615] | 1.047 [1.035, 1.059] | 0.77 |
| last_mel_1m_s1 | 0.592 | [0.561, 0.622] | 1.056 [1.042, 1.070] | 0.76 |
| last_mel_1m_s2 | 0.593 | [0.562, 0.623] | 1.059 [1.045, 1.073] | 0.76 |
| last_mel_1m_s0 | 0.594 | [0.562, 0.625] | 1.060 [1.046, 1.075] | 0.76 |
| replay_mel_1m_s2 | 0.595 | [0.563, 0.625] | 1.062 [1.048, 1.077] | 0.74 |
| full_mel_1m_s2 | 0.597 | [0.566, 0.626] | 1.066 [1.051, 1.081] | 0.78 |
| replay_mel_1w_s0 | 0.599 | [0.572, 0.624] | 1.070 [1.056, 1.084] | 0.66 |
| replay_mel_1m_s1 | 0.601 | [0.567, 0.633] | 1.074 [1.056, 1.091] | 0.71 |
| replay_mel_1w_s2 | 0.603 | [0.575, 0.628] | 1.076 [1.063, 1.090] | 0.63 |
| replay_mel_1w_s1 | 0.603 | [0.576, 0.628] | 1.076 [1.060, 1.095] | 0.64 |
| full_mel_1m_s0 | 0.603 | [0.573, 0.632] | 1.077 [1.064, 1.091] | 0.79 |
| l2sp_s_mel_1m_s0 | 0.608 | [0.575, 0.639] | 1.085 [1.068, 1.104] | 0.76 |
| l2sp_l_mel_1w_s2 | 0.613 | [0.586, 0.640] | 1.095 [1.076, 1.117] | 0.59 |
| scratch_mel_1m_s1 | 0.617 | [0.596, 0.639] | 1.102 [1.078, 1.130] | 0.79 |
| full_lowlr_mel_1w_s1 | 0.622 | [0.597, 0.647] | 1.111 [1.090, 1.134] | 0.58 |
| l2sp_l_mel_1w_s0 | 0.634 | [0.605, 0.662] | 1.132 [1.111, 1.156] | 0.60 |
| head_mel_1w_s0 | 0.636 | [0.613, 0.658] | 1.135 [1.114, 1.161] | 0.66 |
| head_mel_1w_s1 | 0.636 | [0.614, 0.659] | 1.136 [1.114, 1.162] | 0.66 |
| scratch_mel_1m_s0 | 0.636 | [0.614, 0.659] | 1.136 [1.112, 1.163] | 0.81 |
| head_mel_1w_s2 | 0.638 | [0.615, 0.660] | 1.138 [1.117, 1.164] | 0.65 |
| l2sp_s_mel_1w_s2 | 0.638 | [0.610, 0.665] | 1.139 [1.118, 1.162] | 0.60 |
| l2sp_l_mel_1w_s1 | 0.639 | [0.611, 0.667] | 1.141 [1.120, 1.165] | 0.57 |
| l2sp_s_mel_1w_s0 | 0.641 | [0.614, 0.668] | 1.145 [1.122, 1.169] | 0.56 |
| full_lowlr_mel_1w_s0 | 0.643 | [0.615, 0.669] | 1.147 [1.126, 1.171] | 0.54 |
| full_lowlr_mel_1w_s2 | 0.645 | [0.619, 0.672] | 1.152 [1.129, 1.177] | 0.53 |
| l2sp_s_mel_1w_s1 | 0.649 | [0.622, 0.676] | 1.159 [1.139, 1.181] | 0.53 |
| scratch_mel_1m_s2 | 0.657 | [0.635, 0.678] | 1.172 [1.144, 1.205] | 0.83 |
| full_mel_1w_s0 | 0.660 | [0.631, 0.689] | 1.179 [1.154, 1.206] | 0.52 |
| full_mel_1w_s2 | 0.660 | [0.629, 0.691] | 1.179 [1.153, 1.209] | 0.54 |
| full_mel_1w_s1 | 0.668 | [0.638, 0.697] | 1.192 [1.163, 1.226] | 0.52 |
| last_mel_1w_s1 | 0.694 | [0.666, 0.721] | 1.239 [1.208, 1.274] | 0.55 |
| last_mel_1w_s0 | 0.695 | [0.667, 0.722] | 1.241 [1.208, 1.276] | 0.55 |
| last_mel_1w_s2 | 0.700 | [0.672, 0.728] | 1.251 [1.217, 1.286] | 0.54 |
| scratch_mel_1w_s1 | 0.791 | [0.765, 0.820] | 1.413 [1.371, 1.461] | 0.65 |
| scratch_mel_1w_s2 | 0.800 | [0.770, 0.832] | 1.428 [1.370, 1.499] | 0.56 |
| scratch_mel_1w_s0 | 0.812 | [0.781, 0.845] | 1.450 [1.397, 1.508] | 0.55 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.80 |

zero-shot models (same suite, same baseline): chronos-2 0.445, t0-gamma 0.460, timestral-209m 0.467, chronos-2-small 0.470, chronos-bolt-mini 0.476, toto-2.0-313m 0.484, chronos-bolt-tiny 0.490, timestral-28m 0.508, timestral-5m 0.547, toto-2.0-22m 0.559, toto-2.0-4m 0.654, ets 2.567

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| mel_ped 1w: full vs zero-shot student: `full_mel_1w` vs `zero_shot` | 1.185 | [1.158, 1.215] | b better |
| mel_ped 1w: full_lowlr vs zero-shot student: `full_lowlr_mel_1w` vs `zero_shot` | 1.137 | [1.116, 1.161] | b better |
| mel_ped 1w: full_lowlr vs full fine-tuning: `full_lowlr_mel_1w` vs `full_mel_1w` | 0.960 | [0.952, 0.968] | a better |
| mel_ped 1w: head vs zero-shot student: `head_mel_1w` vs `zero_shot` | 1.136 | [1.115, 1.163] | b better |
| mel_ped 1w: head vs full fine-tuning: `head_mel_1w` vs `full_mel_1w` | 0.959 | [0.937, 0.981] | a better |
| mel_ped 1w: last vs zero-shot student: `last_mel_1w` vs `zero_shot` | 1.244 | [1.211, 1.279] | b better |
| mel_ped 1w: last vs full fine-tuning: `last_mel_1w` vs `full_mel_1w` | 1.050 | [1.032, 1.068] | b better |
| mel_ped 1w: l2sp_s vs zero-shot student: `l2sp_s_mel_1w` vs `zero_shot` | 1.148 | [1.127, 1.171] | b better |
| mel_ped 1w: l2sp_s vs full fine-tuning: `l2sp_s_mel_1w` vs `full_mel_1w` | 0.969 | [0.962, 0.975] | a better |
| mel_ped 1w: l2sp_l vs zero-shot student: `l2sp_l_mel_1w` vs `zero_shot` | 1.124 | [1.103, 1.146] | b better |
| mel_ped 1w: l2sp_l vs full fine-tuning: `l2sp_l_mel_1w` vs `full_mel_1w` | 0.948 | [0.939, 0.957] | a better |
| mel_ped 1w: replay vs zero-shot student: `replay_mel_1w` vs `zero_shot` | 1.076 | [1.061, 1.091] | b better |
| mel_ped 1w: replay vs full fine-tuning: `replay_mel_1w` vs `full_mel_1w` | 0.908 | [0.895, 0.919] | a better |
| mel_ped 1w: kd vs zero-shot student: `kd_mel_1w` vs `zero_shot` | 1.035 | [1.027, 1.043] | b better |
| mel_ped 1w: kd vs full fine-tuning: `kd_mel_1w` vs `full_mel_1w` | 0.873 | [0.852, 0.893] | a better |
| mel_ped 1w: scratch vs zero-shot student: `scratch_mel_1w` vs `zero_shot` | 1.439 | [1.387, 1.501] | b better |
| mel_ped 1w: scratch vs full fine-tuning: `scratch_mel_1w` vs `full_mel_1w` | 1.214 | [1.187, 1.249] | b better |
| mel_ped 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-4m` | 1.015 | [0.989, 1.045] | no detectable difference |
| mel_ped 1w: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-4m` | 0.921 | [0.905, 0.939] | a better |
| mel_ped 1w: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-22m` | 1.186 | [1.155, 1.219] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-22m` | 1.077 | [1.057, 1.097] | b better |
| mel_ped 1w: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-313m` | 1.372 | [1.335, 1.410] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-313m` | 1.245 | [1.219, 1.275] | b better |
| mel_ped 1w: full fine-tuning vs timestral-5m zero-shot: `full_mel_1w` vs `zoo:timestral-5m` | 1.213 | [1.184, 1.244] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1w` vs `zoo:timestral-5m` | 1.101 | [1.083, 1.119] | b better |
| mel_ped 1w: full fine-tuning vs timestral-28m zero-shot: `full_mel_1w` vs `zoo:timestral-28m` | 1.305 | [1.269, 1.343] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1w` vs `zoo:timestral-28m` | 1.185 | [1.158, 1.213] | b better |
| mel_ped 1w: full fine-tuning vs timestral-209m zero-shot: `full_mel_1w` vs `zoo:timestral-209m` | 1.422 | [1.380, 1.467] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1w` vs `zoo:timestral-209m` | 1.291 | [1.257, 1.327] | b better |
| mel_ped 1w: full fine-tuning vs chronos-2 zero-shot: `full_mel_1w` vs `zoo:chronos-2` | 1.490 | [1.450, 1.533] | b better |
| mel_ped 1w: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1w` vs `zoo:chronos-2` | 1.353 | [1.324, 1.383] | b better |
| mel_ped 1w: full fine-tuning vs t0-gamma zero-shot: `full_mel_1w` vs `zoo:t0-gamma` | 1.441 | [1.399, 1.488] | b better |
| mel_ped 1w: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1w` vs `zoo:t0-gamma` | 1.308 | [1.280, 1.340] | b better |
| mel_ped 1m: full vs zero-shot student: `full_mel_1m` vs `zero_shot` | 1.052 | [1.040, 1.066] | b better |
| mel_ped 1m: full_lowlr vs zero-shot student: `full_lowlr_mel_1m` vs `zero_shot` | 1.037 | [1.025, 1.050] | b better |
| mel_ped 1m: full_lowlr vs full fine-tuning: `full_lowlr_mel_1m` vs `full_mel_1m` | 0.986 | [0.983, 0.988] | a better |
| mel_ped 1m: head vs zero-shot student: `head_mel_1m` vs `zero_shot` | 1.007 | [1.000, 1.015] | b better |
| mel_ped 1m: head vs full fine-tuning: `head_mel_1m` vs `full_mel_1m` | 0.957 | [0.946, 0.967] | a better |
| mel_ped 1m: last vs zero-shot student: `last_mel_1m` vs `zero_shot` | 1.059 | [1.045, 1.073] | b better |
| mel_ped 1m: last vs full fine-tuning: `last_mel_1m` vs `full_mel_1m` | 1.006 | [0.996, 1.016] | no detectable difference |
| mel_ped 1m: l2sp_s vs zero-shot student: `l2sp_s_mel_1m` vs `zero_shot` | 1.046 | [1.033, 1.060] | b better |
| mel_ped 1m: l2sp_s vs full fine-tuning: `l2sp_s_mel_1m` vs `full_mel_1m` | 0.994 | [0.991, 0.997] | a better |
| mel_ped 1m: l2sp_l vs zero-shot student: `l2sp_l_mel_1m` vs `zero_shot` | 1.040 | [1.029, 1.053] | b better |
| mel_ped 1m: l2sp_l vs full fine-tuning: `l2sp_l_mel_1m` vs `full_mel_1m` | 0.988 | [0.984, 0.993] | a better |
| mel_ped 1m: replay vs zero-shot student: `replay_mel_1m` vs `zero_shot` | 1.061 | [1.047, 1.076] | b better |
| mel_ped 1m: replay vs full fine-tuning: `replay_mel_1m` vs `full_mel_1m` | 1.008 | [1.002, 1.015] | b better |
| mel_ped 1m: kd vs zero-shot student: `kd_mel_1m` vs `zero_shot` | 0.958 | [0.949, 0.966] | a better |
| mel_ped 1m: kd vs full fine-tuning: `kd_mel_1m` vs `full_mel_1m` | 0.910 | [0.900, 0.920] | a better |
| mel_ped 1m: scratch vs zero-shot student: `scratch_mel_1m` vs `zero_shot` | 1.142 | [1.117, 1.169] | b better |
| mel_ped 1m: scratch vs full fine-tuning: `scratch_mel_1m` vs `full_mel_1m` | 1.085 | [1.063, 1.110] | b better |
| mel_ped 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-4m` | 0.902 | [0.884, 0.919] | a better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-4m` | 0.909 | [0.891, 0.927] | a better |
| mel_ped 1m: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-22m` | 1.053 | [1.040, 1.068] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-22m` | 1.062 | [1.048, 1.077] | b better |
| mel_ped 1m: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-313m` | 1.218 | [1.198, 1.239] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-313m` | 1.229 | [1.209, 1.249] | b better |
| mel_ped 1m: full fine-tuning vs timestral-5m zero-shot: `full_mel_1m` vs `zoo:timestral-5m` | 1.077 | [1.065, 1.090] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1m` vs `zoo:timestral-5m` | 1.086 | [1.073, 1.100] | b better |
| mel_ped 1m: full fine-tuning vs timestral-28m zero-shot: `full_mel_1m` vs `zoo:timestral-28m` | 1.160 | [1.142, 1.178] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1m` vs `zoo:timestral-28m` | 1.169 | [1.150, 1.189] | b better |
| mel_ped 1m: full fine-tuning vs timestral-209m zero-shot: `full_mel_1m` vs `zoo:timestral-209m` | 1.263 | [1.237, 1.289] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1m` vs `zoo:timestral-209m` | 1.273 | [1.248, 1.299] | b better |
| mel_ped 1m: full fine-tuning vs chronos-2 zero-shot: `full_mel_1m` vs `zoo:chronos-2` | 1.324 | [1.301, 1.347] | b better |
| mel_ped 1m: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1m` vs `zoo:chronos-2` | 1.335 | [1.311, 1.358] | b better |
| mel_ped 1m: full fine-tuning vs t0-gamma zero-shot: `full_mel_1m` vs `zoo:t0-gamma` | 1.280 | [1.258, 1.303] | b better |
| mel_ped 1m: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1m` vs `zoo:t0-gamma` | 1.291 | [1.268, 1.313] | b better |
| mel_ped 1y: full vs zero-shot student: `full_mel_1y` vs `zero_shot` | 0.859 | [0.843, 0.874] | a better |
| mel_ped 1y: full_lowlr vs zero-shot student: `full_lowlr_mel_1y` vs `zero_shot` | 0.859 | [0.845, 0.872] | a better |
| mel_ped 1y: full_lowlr vs full fine-tuning: `full_lowlr_mel_1y` vs `full_mel_1y` | 1.000 | [0.995, 1.005] | no detectable difference |
| mel_ped 1y: head vs zero-shot student: `head_mel_1y` vs `zero_shot` | 0.962 | [0.958, 0.966] | a better |
| mel_ped 1y: head vs full fine-tuning: `head_mel_1y` vs `full_mel_1y` | 1.120 | [1.103, 1.137] | b better |
| mel_ped 1y: last vs zero-shot student: `last_mel_1y` vs `zero_shot` | 0.890 | [0.879, 0.901] | a better |
| mel_ped 1y: last vs full fine-tuning: `last_mel_1y` vs `full_mel_1y` | 1.036 | [1.028, 1.046] | b better |
| mel_ped 1y: l2sp_s vs zero-shot student: `l2sp_s_mel_1y` vs `zero_shot` | 0.869 | [0.857, 0.882] | a better |
| mel_ped 1y: l2sp_s vs full fine-tuning: `l2sp_s_mel_1y` vs `full_mel_1y` | 1.012 | [1.006, 1.019] | b better |
| mel_ped 1y: l2sp_l vs zero-shot student: `l2sp_l_mel_1y` vs `zero_shot` | 0.900 | [0.891, 0.910] | a better |
| mel_ped 1y: l2sp_l vs full fine-tuning: `l2sp_l_mel_1y` vs `full_mel_1y` | 1.049 | [1.038, 1.060] | b better |
| mel_ped 1y: replay vs zero-shot student: `replay_mel_1y` vs `zero_shot` | 0.860 | [0.846, 0.874] | a better |
| mel_ped 1y: replay vs full fine-tuning: `replay_mel_1y` vs `full_mel_1y` | 1.001 | [0.995, 1.007] | no detectable difference |
| mel_ped 1y: kd vs zero-shot student: `kd_mel_1y` vs `zero_shot` | 0.950 | [0.941, 0.959] | a better |
| mel_ped 1y: kd vs full fine-tuning: `kd_mel_1y` vs `full_mel_1y` | 1.107 | [1.094, 1.121] | b better |
| mel_ped 1y: scratch vs zero-shot student: `scratch_mel_1y` vs `zero_shot` | 0.934 | [0.924, 0.944] | a better |
| mel_ped 1y: scratch vs full fine-tuning: `scratch_mel_1y` vs `full_mel_1y` | 1.088 | [1.073, 1.105] | b better |
| mel_ped 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-4m` | 0.736 | [0.714, 0.757] | a better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-4m` | 0.737 | [0.716, 0.757] | a better |
| mel_ped 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-22m` | 0.860 | [0.843, 0.876] | a better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-22m` | 0.861 | [0.845, 0.876] | a better |
| mel_ped 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-313m` | 0.994 | [0.987, 1.002] | no detectable difference |
| mel_ped 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-313m` | 0.996 | [0.988, 1.002] | no detectable difference |
| mel_ped 1y: full fine-tuning vs timestral-5m zero-shot: `full_mel_1y` vs `zoo:timestral-5m` | 0.879 | [0.866, 0.891] | a better |
| mel_ped 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1y` vs `zoo:timestral-5m` | 0.880 | [0.869, 0.890] | a better |
| mel_ped 1y: full fine-tuning vs timestral-28m zero-shot: `full_mel_1y` vs `zoo:timestral-28m` | 0.946 | [0.935, 0.957] | a better |
| mel_ped 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1y` vs `zoo:timestral-28m` | 0.947 | [0.937, 0.957] | a better |
| mel_ped 1y: full fine-tuning vs timestral-209m zero-shot: `full_mel_1y` vs `zoo:timestral-209m` | 1.031 | [1.021, 1.040] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1y` vs `zoo:timestral-209m` | 1.032 | [1.022, 1.041] | b better |
| mel_ped 1y: full fine-tuning vs chronos-2 zero-shot: `full_mel_1y` vs `zoo:chronos-2` | 1.080 | [1.069, 1.091] | b better |
| mel_ped 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1y` vs `zoo:chronos-2` | 1.082 | [1.073, 1.090] | b better |
| mel_ped 1y: full fine-tuning vs t0-gamma zero-shot: `full_mel_1y` vs `zoo:t0-gamma` | 1.045 | [1.031, 1.058] | b better |
| mel_ped 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1y` vs `zoo:t0-gamma` | 1.046 | [1.034, 1.058] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| replay_mel_1y_s2 | 0.673 | [0.641, 0.711] | 0.996 [0.985, 1.005] | 0.80 |
| l2sp_l_mel_1y_s1 | 0.675 | [0.642, 0.712] | 0.997 [0.985, 1.010] | 0.75 |
| l2sp_l_mel_1y_s2 | 0.675 | [0.642, 0.713] | 0.998 [0.986, 1.011] | 0.75 |
| l2sp_l_mel_1y_s0 | 0.675 | [0.642, 0.712] | 0.998 [0.985, 1.012] | 0.75 |
| replay_mel_1y_s1 | 0.676 | [0.645, 0.713] | 1.000 [0.991, 1.009] | 0.79 |
| zero_shot | 0.676 | [0.644, 0.712] | – | 0.80 |
| replay_mel_1y_s0 | 0.677 | [0.645, 0.714] | 1.000 [0.992, 1.009] | 0.80 |
| replay_mel_1m_s2 | 0.682 | [0.649, 0.721] | 1.008 [0.996, 1.023] | 0.81 |
| replay_mel_1m_s1 | 0.686 | [0.652, 0.725] | 1.014 [1.001, 1.030] | 0.79 |
| replay_mel_1m_s0 | 0.687 | [0.655, 0.724] | 1.015 [0.999, 1.036] | 0.81 |
| full_lowlr_mel_1m_s1 | 0.687 | [0.650, 0.729] | 1.015 [0.988, 1.048] | 0.75 |
| l2sp_s_mel_1y_s0 | 0.688 | [0.657, 0.724] | 1.017 [0.991, 1.050] | 0.74 |
| l2sp_s_mel_1y_s2 | 0.689 | [0.655, 0.728] | 1.018 [0.989, 1.054] | 0.73 |
| replay_mel_1w_s0 | 0.689 | [0.658, 0.725] | 1.019 [1.005, 1.034] | 0.78 |
| l2sp_s_mel_1m_s1 | 0.689 | [0.651, 0.731] | 1.019 [0.994, 1.049] | 0.76 |
| full_lowlr_mel_1y_s0 | 0.690 | [0.656, 0.728] | 1.020 [0.990, 1.058] | 0.72 |
| full_mel_1m_s1 | 0.690 | [0.652, 0.731] | 1.020 [0.994, 1.052] | 0.76 |
| full_lowlr_mel_1m_s0 | 0.691 | [0.649, 0.738] | 1.021 [0.982, 1.065] | 0.73 |
| replay_mel_1w_s2 | 0.692 | [0.659, 0.729] | 1.022 [1.007, 1.040] | 0.77 |
| full_lowlr_mel_1y_s2 | 0.692 | [0.658, 0.730] | 1.023 [0.994, 1.059] | 0.71 |
| full_lowlr_mel_1y_s1 | 0.693 | [0.659, 0.732] | 1.025 [0.995, 1.063] | 0.71 |
| l2sp_s_mel_1y_s1 | 0.694 | [0.659, 0.735] | 1.026 [0.995, 1.065] | 0.73 |
| replay_mel_1w_s1 | 0.697 | [0.661, 0.736] | 1.030 [1.014, 1.047] | 0.80 |
| l2sp_l_mel_1m_s0 | 0.698 | [0.658, 0.745] | 1.031 [0.996, 1.074] | 0.74 |
| full_lowlr_mel_1m_s2 | 0.698 | [0.650, 0.753] | 1.032 [0.981, 1.087] | 0.70 |
| l2sp_l_mel_1m_s2 | 0.698 | [0.657, 0.746] | 1.033 [0.993, 1.079] | 0.75 |
| l2sp_l_mel_1m_s1 | 0.701 | [0.659, 0.750] | 1.036 [0.997, 1.082] | 0.73 |
| full_mel_1m_s0 | 0.701 | [0.657, 0.752] | 1.037 [0.991, 1.088] | 0.72 |
| l2sp_s_mel_1m_s0 | 0.709 | [0.662, 0.763] | 1.048 [0.997, 1.106] | 0.70 |
| l2sp_s_mel_1m_s2 | 0.711 | [0.662, 0.766] | 1.051 [1.000, 1.108] | 0.71 |
| full_mel_1m_s2 | 0.717 | [0.667, 0.773] | 1.060 [1.006, 1.119] | 0.69 |
| l2sp_l_mel_1w_s0 | 0.718 | [0.679, 0.765] | 1.061 [1.036, 1.089] | 0.71 |
| last_mel_1y_s0 | 0.718 | [0.677, 0.768] | 1.062 [1.027, 1.105] | 0.68 |
| last_mel_1y_s1 | 0.719 | [0.675, 0.769] | 1.062 [1.026, 1.109] | 0.67 |
| kd_mel_1m_s0 | 0.719 | [0.683, 0.757] | 1.062 [1.035, 1.093] | 0.76 |
| last_mel_1y_s2 | 0.719 | [0.678, 0.769] | 1.063 [1.030, 1.106] | 0.67 |
| head_mel_1y_s2 | 0.720 | [0.677, 0.771] | 1.065 [1.030, 1.105] | 0.74 |
| kd_mel_1w_s1 | 0.721 | [0.685, 0.762] | 1.065 [1.037, 1.102] | 0.78 |
| l2sp_l_mel_1w_s2 | 0.722 | [0.683, 0.768] | 1.067 [1.041, 1.096] | 0.69 |
| l2sp_l_mel_1w_s1 | 0.722 | [0.683, 0.769] | 1.067 [1.041, 1.097] | 0.69 |
| head_mel_1y_s0 | 0.722 | [0.678, 0.773] | 1.067 [1.030, 1.110] | 0.73 |
| head_mel_1y_s1 | 0.724 | [0.680, 0.776] | 1.071 [1.034, 1.113] | 0.73 |
| kd_mel_1m_s2 | 0.731 | [0.693, 0.774] | 1.080 [1.049, 1.118] | 0.76 |
| kd_mel_1w_s2 | 0.734 | [0.697, 0.775] | 1.084 [1.053, 1.124] | 0.77 |
| kd_mel_1m_s1 | 0.734 | [0.696, 0.777] | 1.085 [1.054, 1.122] | 0.76 |
| kd_mel_1w_s0 | 0.741 | [0.703, 0.785] | 1.096 [1.063, 1.138] | 0.78 |
| kd_mel_1y_s2 | 0.742 | [0.703, 0.784] | 1.096 [1.063, 1.135] | 0.77 |
| kd_mel_1y_s0 | 0.742 | [0.704, 0.784] | 1.097 [1.064, 1.134] | 0.77 |
| kd_mel_1y_s1 | 0.742 | [0.704, 0.786] | 1.097 [1.064, 1.137] | 0.77 |
| full_lowlr_mel_1w_s1 | 0.742 | [0.706, 0.784] | 1.098 [1.070, 1.128] | 0.69 |
| full_mel_1y_s0 | 0.746 | [0.702, 0.798] | 1.102 [1.060, 1.158] | 0.66 |
| l2sp_s_mel_1w_s0 | 0.747 | [0.703, 0.797] | 1.104 [1.068, 1.148] | 0.67 |
| full_lowlr_mel_1w_s0 | 0.751 | [0.713, 0.794] | 1.110 [1.082, 1.141] | 0.67 |
| l2sp_s_mel_1w_s2 | 0.753 | [0.708, 0.805] | 1.113 [1.075, 1.158] | 0.67 |
| full_mel_1y_s2 | 0.753 | [0.710, 0.804] | 1.114 [1.072, 1.165] | 0.64 |
| last_mel_1m_s0 | 0.755 | [0.706, 0.812] | 1.116 [1.074, 1.165] | 0.60 |
| last_mel_1m_s1 | 0.756 | [0.706, 0.814] | 1.118 [1.074, 1.168] | 0.60 |
| full_mel_1y_s1 | 0.758 | [0.711, 0.813] | 1.120 [1.075, 1.175] | 0.65 |
| last_mel_1m_s2 | 0.759 | [0.708, 0.818] | 1.122 [1.077, 1.175] | 0.58 |
| l2sp_s_mel_1w_s1 | 0.761 | [0.714, 0.815] | 1.125 [1.085, 1.174] | 0.66 |
| full_lowlr_mel_1w_s2 | 0.765 | [0.724, 0.813] | 1.131 [1.098, 1.171] | 0.64 |
| full_mel_1w_s0 | 0.796 | [0.748, 0.855] | 1.177 [1.133, 1.229] | 0.63 |
| full_mel_1w_s2 | 0.797 | [0.750, 0.851] | 1.179 [1.134, 1.227] | 0.63 |
| head_mel_1m_s1 | 0.807 | [0.750, 0.876] | 1.193 [1.135, 1.261] | 0.69 |
| head_mel_1m_s0 | 0.817 | [0.757, 0.890] | 1.208 [1.145, 1.283] | 0.69 |
| head_mel_1m_s2 | 0.826 | [0.763, 0.901] | 1.221 [1.153, 1.301] | 0.70 |
| full_mel_1w_s1 | 0.846 | [0.787, 0.918] | 1.251 [1.195, 1.317] | 0.59 |
| last_mel_1w_s0 | 0.923 | [0.852, 1.010] | 1.365 [1.296, 1.446] | 0.52 |
| last_mel_1w_s2 | 0.924 | [0.853, 1.011] | 1.366 [1.296, 1.448] | 0.50 |
| last_mel_1w_s1 | 0.924 | [0.853, 1.012] | 1.366 [1.296, 1.449] | 0.52 |
| head_mel_1w_s0 | 0.963 | [0.890, 1.051] | 1.423 [1.347, 1.512] | 0.62 |
| head_mel_1w_s2 | 0.963 | [0.891, 1.051] | 1.423 [1.349, 1.510] | 0.61 |
| head_mel_1w_s1 | 0.963 | [0.891, 1.052] | 1.424 [1.349, 1.511] | 0.61 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| mel_ped 1w: full vs zero-shot student: `full_mel_1w` vs `zero_shot` | 1.207 | [1.162, 1.260] | b better |
| mel_ped 1w: full_lowlr vs zero-shot student: `full_lowlr_mel_1w` vs `zero_shot` | 1.115 | [1.087, 1.149] | b better |
| mel_ped 1w: full_lowlr vs full fine-tuning: `full_lowlr_mel_1w` vs `full_mel_1w` | 0.924 | [0.901, 0.944] | a better |
| mel_ped 1w: head vs zero-shot student: `head_mel_1w` vs `zero_shot` | 1.423 | [1.349, 1.511] | b better |
| mel_ped 1w: head vs full fine-tuning: `head_mel_1w` vs `full_mel_1w` | 1.179 | [1.133, 1.228] | b better |
| mel_ped 1w: last vs zero-shot student: `last_mel_1w` vs `zero_shot` | 1.366 | [1.296, 1.448] | b better |
| mel_ped 1w: last vs full fine-tuning: `last_mel_1w` vs `full_mel_1w` | 1.131 | [1.099, 1.167] | b better |
| mel_ped 1w: l2sp_s vs zero-shot student: `l2sp_s_mel_1w` vs `zero_shot` | 1.115 | [1.077, 1.161] | b better |
| mel_ped 1w: l2sp_s vs full fine-tuning: `l2sp_s_mel_1w` vs `full_mel_1w` | 0.924 | [0.904, 0.942] | a better |
| mel_ped 1w: l2sp_l vs zero-shot student: `l2sp_l_mel_1w` vs `zero_shot` | 1.066 | [1.040, 1.095] | b better |
| mel_ped 1w: l2sp_l vs full fine-tuning: `l2sp_l_mel_1w` vs `full_mel_1w` | 0.883 | [0.857, 0.906] | a better |
| mel_ped 1w: replay vs zero-shot student: `replay_mel_1w` vs `zero_shot` | 1.027 | [1.014, 1.042] | b better |
| mel_ped 1w: replay vs full fine-tuning: `replay_mel_1w` vs `full_mel_1w` | 0.851 | [0.818, 0.882] | a better |
| mel_ped 1w: kd vs zero-shot student: `kd_mel_1w` vs `zero_shot` | 1.083 | [1.052, 1.122] | b better |
| mel_ped 1w: kd vs full fine-tuning: `kd_mel_1w` vs `full_mel_1w` | 0.897 | [0.868, 0.923] | a better |
| mel_ped 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-4m` | 1.133 | [1.083, 1.193] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-4m` | 0.964 | [0.942, 0.988] | a better |
| mel_ped 1w: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-22m` | 1.199 | [1.144, 1.262] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-22m` | 1.020 | [0.998, 1.047] | no detectable difference |
| mel_ped 1w: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-313m` | 1.225 | [1.168, 1.290] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-313m` | 1.042 | [1.017, 1.071] | b better |
| mel_ped 1w: full fine-tuning vs timestral-5m zero-shot: `full_mel_1w` vs `zoo:timestral-5m` | 1.238 | [1.186, 1.302] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1w` vs `zoo:timestral-5m` | 1.053 | [1.038, 1.072] | b better |
| mel_ped 1w: full fine-tuning vs timestral-28m zero-shot: `full_mel_1w` vs `zoo:timestral-28m` | 1.281 | [1.227, 1.343] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1w` vs `zoo:timestral-28m` | 1.090 | [1.073, 1.109] | b better |
| mel_ped 1w: full fine-tuning vs timestral-209m zero-shot: `full_mel_1w` vs `zoo:timestral-209m` | 1.301 | [1.246, 1.364] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1w` vs `zoo:timestral-209m` | 1.107 | [1.088, 1.129] | b better |
| mel_ped 1w: full fine-tuning vs chronos-2 zero-shot: `full_mel_1w` vs `zoo:chronos-2` | 1.329 | [1.272, 1.394] | b better |
| mel_ped 1w: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1w` vs `zoo:chronos-2` | 1.131 | [1.107, 1.159] | b better |
| mel_ped 1w: full fine-tuning vs t0-gamma zero-shot: `full_mel_1w` vs `zoo:t0-gamma` | 1.599 | [1.444, 1.794] | b better |
| mel_ped 1w: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1w` vs `zoo:t0-gamma` | 1.361 | [1.248, 1.492] | b better |
| mel_ped 1m: full vs zero-shot student: `full_mel_1m` vs `zero_shot` | 1.044 | [1.003, 1.091] | b better |
| mel_ped 1m: full_lowlr vs zero-shot student: `full_lowlr_mel_1m` vs `zero_shot` | 1.026 | [0.988, 1.070] | no detectable difference |
| mel_ped 1m: full_lowlr vs full fine-tuning: `full_lowlr_mel_1m` vs `full_mel_1m` | 0.983 | [0.978, 0.987] | a better |
| mel_ped 1m: head vs zero-shot student: `head_mel_1m` vs `zero_shot` | 1.208 | [1.144, 1.283] | b better |
| mel_ped 1m: head vs full fine-tuning: `head_mel_1m` vs `full_mel_1m` | 1.157 | [1.123, 1.195] | b better |
| mel_ped 1m: last vs zero-shot student: `last_mel_1m` vs `zero_shot` | 1.119 | [1.076, 1.170] | b better |
| mel_ped 1m: last vs full fine-tuning: `last_mel_1m` vs `full_mel_1m` | 1.072 | [1.044, 1.099] | b better |
| mel_ped 1m: l2sp_s vs zero-shot student: `l2sp_s_mel_1m` vs `zero_shot` | 1.045 | [1.003, 1.092] | b better |
| mel_ped 1m: l2sp_s vs full fine-tuning: `l2sp_s_mel_1m` vs `full_mel_1m` | 1.001 | [0.997, 1.004] | no detectable difference |
| mel_ped 1m: l2sp_l vs zero-shot student: `l2sp_l_mel_1m` vs `zero_shot` | 1.034 | [0.996, 1.080] | no detectable difference |
| mel_ped 1m: l2sp_l vs full fine-tuning: `l2sp_l_mel_1m` vs `full_mel_1m` | 0.990 | [0.978, 1.001] | no detectable difference |
| mel_ped 1m: replay vs zero-shot student: `replay_mel_1m` vs `zero_shot` | 1.019 | [1.007, 1.033] | b better |
| mel_ped 1m: replay vs full fine-tuning: `replay_mel_1m` vs `full_mel_1m` | 0.976 | [0.944, 1.006] | no detectable difference |
| mel_ped 1m: kd vs zero-shot student: `kd_mel_1m` vs `zero_shot` | 1.077 | [1.048, 1.112] | b better |
| mel_ped 1m: kd vs full fine-tuning: `kd_mel_1m` vs `full_mel_1m` | 1.032 | [1.003, 1.063] | b better |
| mel_ped 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-4m` | 0.980 | [0.945, 1.022] | no detectable difference |
| mel_ped 1m: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-4m` | 0.956 | [0.935, 0.980] | a better |
| mel_ped 1m: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-22m` | 1.037 | [0.990, 1.087] | no detectable difference |
| mel_ped 1m: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-22m` | 1.012 | [0.990, 1.038] | no detectable difference |
| mel_ped 1m: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-313m` | 1.059 | [1.015, 1.109] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-313m` | 1.034 | [1.010, 1.061] | b better |
| mel_ped 1m: full fine-tuning vs timestral-5m zero-shot: `full_mel_1m` vs `zoo:timestral-5m` | 1.070 | [1.032, 1.115] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1m` vs `zoo:timestral-5m` | 1.045 | [1.029, 1.064] | b better |
| mel_ped 1m: full fine-tuning vs timestral-28m zero-shot: `full_mel_1m` vs `zoo:timestral-28m` | 1.107 | [1.067, 1.154] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1m` vs `zoo:timestral-28m` | 1.081 | [1.064, 1.100] | b better |
| mel_ped 1m: full fine-tuning vs timestral-209m zero-shot: `full_mel_1m` vs `zoo:timestral-209m` | 1.125 | [1.085, 1.172] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1m` vs `zoo:timestral-209m` | 1.098 | [1.079, 1.121] | b better |
| mel_ped 1m: full fine-tuning vs chronos-2 zero-shot: `full_mel_1m` vs `zoo:chronos-2` | 1.150 | [1.110, 1.197] | b better |
| mel_ped 1m: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1m` vs `zoo:chronos-2` | 1.122 | [1.099, 1.149] | b better |
| mel_ped 1m: full fine-tuning vs t0-gamma zero-shot: `full_mel_1m` vs `zoo:t0-gamma` | 1.383 | [1.251, 1.540] | b better |
| mel_ped 1m: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1m` vs `zoo:t0-gamma` | 1.350 | [1.238, 1.481] | b better |
| mel_ped 1y: full vs zero-shot student: `full_mel_1y` vs `zero_shot` | 1.114 | [1.071, 1.167] | b better |
| mel_ped 1y: full_lowlr vs zero-shot student: `full_lowlr_mel_1y` vs `zero_shot` | 1.023 | [0.993, 1.060] | no detectable difference |
| mel_ped 1y: full_lowlr vs full fine-tuning: `full_lowlr_mel_1y` vs `full_mel_1y` | 0.918 | [0.898, 0.936] | a better |
| mel_ped 1y: head vs zero-shot student: `head_mel_1y` vs `zero_shot` | 1.068 | [1.031, 1.109] | b better |
| mel_ped 1y: head vs full fine-tuning: `head_mel_1y` vs `full_mel_1y` | 0.958 | [0.930, 0.988] | a better |
| mel_ped 1y: last vs zero-shot student: `last_mel_1y` vs `zero_shot` | 1.063 | [1.028, 1.107] | b better |
| mel_ped 1y: last vs full fine-tuning: `last_mel_1y` vs `full_mel_1y` | 0.954 | [0.934, 0.974] | a better |
| mel_ped 1y: l2sp_s vs zero-shot student: `l2sp_s_mel_1y` vs `zero_shot` | 1.022 | [0.993, 1.057] | no detectable difference |
| mel_ped 1y: l2sp_s vs full fine-tuning: `l2sp_s_mel_1y` vs `full_mel_1y` | 0.917 | [0.896, 0.934] | a better |
| mel_ped 1y: l2sp_l vs zero-shot student: `l2sp_l_mel_1y` vs `zero_shot` | 0.998 | [0.985, 1.011] | no detectable difference |
| mel_ped 1y: l2sp_l vs full fine-tuning: `l2sp_l_mel_1y` vs `full_mel_1y` | 0.896 | [0.859, 0.927] | a better |
| mel_ped 1y: replay vs zero-shot student: `replay_mel_1y` vs `zero_shot` | 1.000 | [0.991, 1.008] | no detectable difference |
| mel_ped 1y: replay vs full fine-tuning: `replay_mel_1y` vs `full_mel_1y` | 0.897 | [0.858, 0.932] | a better |
| mel_ped 1y: kd vs zero-shot student: `kd_mel_1y` vs `zero_shot` | 1.097 | [1.064, 1.136] | b better |
| mel_ped 1y: kd vs full fine-tuning: `kd_mel_1y` vs `full_mel_1y` | 0.985 | [0.960, 1.008] | no detectable difference |
| mel_ped 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-4m` | 1.046 | [1.003, 1.097] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-4m` | 0.938 | [0.915, 0.962] | a better |
| mel_ped 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-22m` | 1.106 | [1.058, 1.165] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-22m` | 0.993 | [0.973, 1.015] | no detectable difference |
| mel_ped 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-313m` | 1.130 | [1.084, 1.187] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-313m` | 1.014 | [0.992, 1.038] | no detectable difference |
| mel_ped 1y: full fine-tuning vs timestral-5m zero-shot: `full_mel_1y` vs `zoo:timestral-5m` | 1.142 | [1.098, 1.197] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1y` vs `zoo:timestral-5m` | 1.025 | [1.008, 1.043] | b better |
| mel_ped 1y: full fine-tuning vs timestral-28m zero-shot: `full_mel_1y` vs `zoo:timestral-28m` | 1.182 | [1.137, 1.237] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1y` vs `zoo:timestral-28m` | 1.060 | [1.042, 1.077] | b better |
| mel_ped 1y: full fine-tuning vs timestral-209m zero-shot: `full_mel_1y` vs `zoo:timestral-209m` | 1.201 | [1.155, 1.256] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1y` vs `zoo:timestral-209m` | 1.077 | [1.058, 1.096] | b better |
| mel_ped 1y: full fine-tuning vs chronos-2 zero-shot: `full_mel_1y` vs `zoo:chronos-2` | 1.227 | [1.181, 1.282] | b better |
| mel_ped 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1y` vs `zoo:chronos-2` | 1.101 | [1.076, 1.125] | b better |
| mel_ped 1y: full fine-tuning vs t0-gamma zero-shot: `full_mel_1y` vs `zoo:t0-gamma` | 1.476 | [1.341, 1.633] | b better |
| mel_ped 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1y` vs `zoo:t0-gamma` | 1.324 | [1.217, 1.448] | b better |

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| zero_shot | 0.481 | [0.436, 0.533] | – | 0.83 |
| replay_mel_1y_s1 | 0.482 | [0.437, 0.534] | 1.002 [0.992, 1.012] | 0.84 |
| replay_mel_1w_s0 | 0.486 | [0.440, 0.540] | 1.010 [1.002, 1.017] | 0.84 |
| replay_mel_1y_s2 | 0.491 | [0.446, 0.545] | 1.021 [1.005, 1.037] | 0.83 |
| replay_mel_1y_s0 | 0.495 | [0.450, 0.549] | 1.029 [1.016, 1.043] | 0.82 |
| replay_mel_1m_s2 | 0.515 | [0.469, 0.568] | 1.071 [1.050, 1.093] | 0.84 |
| replay_mel_1m_s1 | 0.517 | [0.469, 0.573] | 1.075 [1.047, 1.105] | 0.82 |
| replay_mel_1w_s2 | 0.521 | [0.474, 0.576] | 1.083 [1.053, 1.112] | 0.82 |
| replay_mel_1w_s1 | 0.529 | [0.480, 0.586] | 1.099 [1.071, 1.127] | 0.85 |
| replay_mel_1m_s0 | 0.532 | [0.484, 0.591] | 1.105 [1.076, 1.137] | 0.82 |
| l2sp_l_mel_1y_s2 | 0.566 | [0.505, 0.640] | 1.176 [1.122, 1.240] | 0.75 |
| l2sp_l_mel_1y_s1 | 0.579 | [0.518, 0.655] | 1.204 [1.147, 1.272] | 0.75 |
| l2sp_l_mel_1y_s0 | 0.612 | [0.550, 0.689] | 1.272 [1.202, 1.349] | 0.71 |
| l2sp_s_mel_1m_s1 | 0.720 | [0.646, 0.804] | 1.495 [1.384, 1.616] | 0.65 |
| full_lowlr_mel_1m_s1 | 0.726 | [0.658, 0.806] | 1.509 [1.403, 1.625] | 0.62 |
| l2sp_l_mel_1w_s0 | 0.728 | [0.650, 0.827] | 1.513 [1.417, 1.617] | 0.66 |
| full_mel_1m_s1 | 0.732 | [0.657, 0.818] | 1.521 [1.406, 1.648] | 0.64 |
| full_lowlr_mel_1m_s0 | 0.800 | [0.723, 0.892] | 1.662 [1.537, 1.802] | 0.59 |
| l2sp_l_mel_1w_s1 | 0.808 | [0.719, 0.919] | 1.679 [1.549, 1.817] | 0.60 |
| full_lowlr_mel_1m_s2 | 0.813 | [0.731, 0.913] | 1.689 [1.546, 1.847] | 0.59 |
| l2sp_l_mel_1w_s2 | 0.815 | [0.725, 0.926] | 1.693 [1.554, 1.840] | 0.59 |
| l2sp_l_mel_1m_s2 | 0.826 | [0.742, 0.935] | 1.717 [1.566, 1.873] | 0.66 |
| l2sp_l_mel_1m_s1 | 0.870 | [0.782, 0.980] | 1.807 [1.655, 1.969] | 0.63 |
| head_mel_1y_s2 | 0.871 | [0.774, 0.991] | 1.809 [1.645, 1.993] | 0.62 |
| full_mel_1m_s0 | 0.874 | [0.786, 0.983] | 1.815 [1.656, 1.994] | 0.57 |
| kd_mel_1m_s0 | 0.882 | [0.784, 1.001] | 1.832 [1.645, 2.049] | 0.63 |
| l2sp_l_mel_1m_s0 | 0.897 | [0.807, 1.013] | 1.864 [1.702, 2.040] | 0.64 |
| head_mel_1y_s0 | 0.897 | [0.797, 1.023] | 1.864 [1.686, 2.065] | 0.60 |
| full_lowlr_mel_1y_s2 | 0.898 | [0.793, 1.029] | 1.866 [1.662, 2.103] | 0.56 |
| full_lowlr_mel_1w_s1 | 0.901 | [0.791, 1.032] | 1.873 [1.692, 2.073] | 0.56 |
| head_mel_1y_s1 | 0.902 | [0.802, 1.028] | 1.874 [1.694, 2.077] | 0.60 |
| l2sp_s_mel_1m_s2 | 0.908 | [0.810, 1.028] | 1.886 [1.704, 2.091] | 0.58 |
| l2sp_s_mel_1y_s0 | 0.910 | [0.795, 1.055] | 1.891 [1.668, 2.154] | 0.59 |
| full_lowlr_mel_1y_s1 | 0.915 | [0.806, 1.054] | 1.902 [1.684, 2.154] | 0.55 |
| kd_mel_1w_s1 | 0.916 | [0.812, 1.042] | 1.903 [1.702, 2.125] | 0.60 |
| full_lowlr_mel_1y_s0 | 0.923 | [0.813, 1.063] | 1.919 [1.697, 2.178] | 0.55 |
| l2sp_s_mel_1m_s0 | 0.932 | [0.833, 1.055] | 1.936 [1.759, 2.128] | 0.55 |
| l2sp_s_mel_1y_s2 | 0.946 | [0.821, 1.101] | 1.965 [1.726, 2.244] | 0.55 |
| full_lowlr_mel_1w_s0 | 0.948 | [0.830, 1.091] | 1.970 [1.771, 2.182] | 0.54 |
| l2sp_s_mel_1y_s1 | 0.957 | [0.833, 1.114] | 1.988 [1.743, 2.268] | 0.55 |
| kd_mel_1m_s2 | 0.959 | [0.853, 1.090] | 1.994 [1.779, 2.245] | 0.64 |
| full_mel_1m_s2 | 0.964 | [0.855, 1.098] | 2.003 [1.791, 2.238] | 0.54 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| kd_mel_1m_s1 | 1.006 | [0.888, 1.154] | 2.090 [1.850, 2.366] | 0.61 |
| kd_mel_1w_s2 | 1.007 | [0.886, 1.155] | 2.092 [1.847, 2.375] | 0.57 |
| last_mel_1y_s0 | 1.045 | [0.913, 1.208] | 2.172 [1.926, 2.453] | 0.48 |
| last_mel_1y_s2 | 1.048 | [0.915, 1.214] | 2.178 [1.934, 2.457] | 0.47 |
| kd_mel_1y_s0 | 1.050 | [0.918, 1.212] | 2.181 [1.915, 2.496] | 0.61 |
| last_mel_1y_s1 | 1.067 | [0.932, 1.234] | 2.217 [1.966, 2.504] | 0.46 |
| l2sp_s_mel_1w_s0 | 1.073 | [0.937, 1.235] | 2.230 [1.965, 2.536] | 0.49 |
| kd_mel_1w_s0 | 1.079 | [0.945, 1.239] | 2.242 [1.970, 2.543] | 0.56 |
| kd_mel_1y_s2 | 1.099 | [0.958, 1.272] | 2.283 [1.993, 2.617] | 0.59 |
| kd_mel_1y_s1 | 1.110 | [0.965, 1.287] | 2.306 [2.011, 2.649] | 0.60 |
| l2sp_s_mel_1w_s2 | 1.116 | [0.975, 1.290] | 2.320 [2.031, 2.655] | 0.48 |
| full_lowlr_mel_1w_s2 | 1.152 | [1.002, 1.335] | 2.395 [2.106, 2.727] | 0.48 |
| l2sp_s_mel_1w_s1 | 1.156 | [1.007, 1.343] | 2.403 [2.102, 2.742] | 0.44 |
| last_mel_1m_s2 | 1.209 | [1.043, 1.422] | 2.511 [2.214, 2.835] | 0.43 |
| last_mel_1m_s0 | 1.213 | [1.047, 1.428] | 2.520 [2.222, 2.844] | 0.43 |
| last_mel_1m_s1 | 1.218 | [1.049, 1.437] | 2.531 [2.225, 2.857] | 0.45 |
| full_mel_1y_s0 | 1.256 | [1.073, 1.479] | 2.610 [2.237, 3.061] | 0.48 |
| full_mel_1w_s0 | 1.282 | [1.107, 1.502] | 2.664 [2.311, 3.075] | 0.46 |
| full_mel_1y_s2 | 1.287 | [1.095, 1.521] | 2.674 [2.278, 3.145] | 0.47 |
| full_mel_1y_s1 | 1.293 | [1.097, 1.534] | 2.686 [2.288, 3.172] | 0.47 |
| full_mel_1w_s2 | 1.299 | [1.114, 1.527] | 2.698 [2.321, 3.130] | 0.46 |
| head_mel_1m_s1 | 1.307 | [1.132, 1.520] | 2.716 [2.343, 3.137] | 0.51 |
| head_mel_1m_s0 | 1.378 | [1.190, 1.609] | 2.864 [2.471, 3.316] | 0.51 |
| head_mel_1m_s2 | 1.430 | [1.229, 1.674] | 2.970 [2.551, 3.452] | 0.51 |
| full_mel_1w_s1 | 1.520 | [1.296, 1.797] | 3.158 [2.701, 3.719] | 0.40 |
| last_mel_1w_s0 | 1.593 | [1.355, 1.883] | 3.309 [2.822, 3.875] | 0.34 |
| last_mel_1w_s2 | 1.609 | [1.367, 1.903] | 3.343 [2.856, 3.911] | 0.32 |
| last_mel_1w_s1 | 1.627 | [1.381, 1.925] | 3.380 [2.885, 3.962] | 0.33 |
| head_mel_1w_s2 | 2.119 | [1.773, 2.540] | 4.404 [3.664, 5.296] | 0.33 |
| head_mel_1w_s0 | 2.143 | [1.792, 2.568] | 4.452 [3.698, 5.357] | 0.33 |
| head_mel_1w_s1 | 2.152 | [1.801, 2.576] | 4.472 [3.721, 5.380] | 0.32 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| mel_ped 1w: full vs zero-shot student: `full_mel_1w` vs `zero_shot` | 2.856 | [2.461, 3.324] | b better |
| mel_ped 1w: full_lowlr vs zero-shot student: `full_lowlr_mel_1w` vs `zero_shot` | 2.109 | [1.885, 2.362] | b better |
| mel_ped 1w: full_lowlr vs full fine-tuning: `full_lowlr_mel_1w` vs `full_mel_1w` | 0.738 | [0.700, 0.776] | a better |
| mel_ped 1w: head vs zero-shot student: `head_mel_1w` vs `zero_shot` | 4.443 | [3.695, 5.343] | b better |
| mel_ped 1w: head vs full fine-tuning: `head_mel_1w` vs `full_mel_1w` | 1.556 | [1.447, 1.668] | b better |
| mel_ped 1w: last vs zero-shot student: `last_mel_1w` vs `zero_shot` | 3.345 | [2.855, 3.922] | b better |
| mel_ped 1w: last vs full fine-tuning: `last_mel_1w` vs `full_mel_1w` | 1.171 | [1.129, 1.214] | b better |
| mel_ped 1w: l2sp_s vs zero-shot student: `l2sp_s_mel_1w` vs `zero_shot` | 2.321 | [2.036, 2.647] | b better |
| mel_ped 1w: l2sp_s vs full fine-tuning: `l2sp_s_mel_1w` vs `full_mel_1w` | 0.813 | [0.791, 0.835] | a better |
| mel_ped 1w: l2sp_l vs zero-shot student: `l2sp_l_mel_1w` vs `zero_shot` | 1.635 | [1.513, 1.764] | b better |
| mel_ped 1w: l2sp_l vs full fine-tuning: `l2sp_l_mel_1w` vs `full_mel_1w` | 0.572 | [0.525, 0.622] | a better |
| mel_ped 1w: replay vs zero-shot student: `replay_mel_1w` vs `zero_shot` | 1.069 | [1.051, 1.089] | b better |
| mel_ped 1w: replay vs full fine-tuning: `replay_mel_1w` vs `full_mel_1w` | 0.374 | [0.326, 0.429] | a better |
| mel_ped 1w: kd vs zero-shot student: `kd_mel_1w` vs `zero_shot` | 2.085 | [1.848, 2.357] | b better |
| mel_ped 1w: kd vs full fine-tuning: `kd_mel_1w` vs `full_mel_1w` | 0.730 | [0.690, 0.767] | a better |
| mel_ped 1w: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-4m` | 2.647 | [2.319, 3.011] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-4m` | 0.991 | [0.955, 1.027] | no detectable difference |
| mel_ped 1w: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-22m` | 2.893 | [2.511, 3.329] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-22m` | 1.083 | [1.031, 1.138] | b better |
| mel_ped 1w: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1w` vs `zoo:toto-2.0-313m` | 3.129 | [2.690, 3.636] | b better |
| mel_ped 1w: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1w` vs `zoo:toto-2.0-313m` | 1.171 | [1.122, 1.227] | b better |
| mel_ped 1w: full fine-tuning vs timestral-5m zero-shot: `full_mel_1w` vs `zoo:timestral-5m` | 2.795 | [2.402, 3.240] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1w` vs `zoo:timestral-5m` | 1.046 | [1.019, 1.075] | b better |
| mel_ped 1w: full fine-tuning vs timestral-28m zero-shot: `full_mel_1w` vs `zoo:timestral-28m` | 3.077 | [2.632, 3.578] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1w` vs `zoo:timestral-28m` | 1.152 | [1.114, 1.194] | b better |
| mel_ped 1w: full fine-tuning vs timestral-209m zero-shot: `full_mel_1w` vs `zoo:timestral-209m` | 3.417 | [2.911, 3.986] | b better |
| mel_ped 1w: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1w` vs `zoo:timestral-209m` | 1.279 | [1.231, 1.335] | b better |
| mel_ped 1w: full fine-tuning vs chronos-2 zero-shot: `full_mel_1w` vs `zoo:chronos-2` | 2.728 | [2.317, 3.208] | b better |
| mel_ped 1w: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1w` vs `zoo:chronos-2` | 1.021 | [0.959, 1.087] | no detectable difference |
| mel_ped 1w: full fine-tuning vs t0-gamma zero-shot: `full_mel_1w` vs `zoo:t0-gamma` | 2.833 | [2.422, 3.300] | b better |
| mel_ped 1w: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1w` vs `zoo:t0-gamma` | 1.060 | [1.006, 1.117] | b better |
| mel_ped 1m: full vs zero-shot student: `full_mel_1m` vs `zero_shot` | 1.818 | [1.654, 1.999] | b better |
| mel_ped 1m: full_lowlr vs zero-shot student: `full_lowlr_mel_1m` vs `zero_shot` | 1.645 | [1.517, 1.786] | b better |
| mel_ped 1m: full_lowlr vs full fine-tuning: `full_lowlr_mel_1m` vs `full_mel_1m` | 0.905 | [0.887, 0.921] | a better |
| mel_ped 1m: head vs zero-shot student: `head_mel_1m` vs `zero_shot` | 2.851 | [2.456, 3.302] | b better |
| mel_ped 1m: head vs full fine-tuning: `head_mel_1m` vs `full_mel_1m` | 1.568 | [1.469, 1.666] | b better |
| mel_ped 1m: last vs zero-shot student: `last_mel_1m` vs `zero_shot` | 2.523 | [2.220, 2.848] | b better |
| mel_ped 1m: last vs full fine-tuning: `last_mel_1m` vs `full_mel_1m` | 1.388 | [1.306, 1.483] | b better |
| mel_ped 1m: l2sp_s vs zero-shot student: `l2sp_s_mel_1m` vs `zero_shot` | 1.806 | [1.648, 1.980] | b better |
| mel_ped 1m: l2sp_s vs full fine-tuning: `l2sp_s_mel_1m` vs `full_mel_1m` | 0.993 | [0.985, 1.001] | no detectable difference |
| mel_ped 1m: l2sp_l vs zero-shot student: `l2sp_l_mel_1m` vs `zero_shot` | 1.800 | [1.645, 1.966] | b better |
| mel_ped 1m: l2sp_l vs full fine-tuning: `l2sp_l_mel_1m` vs `full_mel_1m` | 0.990 | [0.960, 1.022] | no detectable difference |
| mel_ped 1m: replay vs zero-shot student: `replay_mel_1m` vs `zero_shot` | 1.089 | [1.067, 1.113] | b better |
| mel_ped 1m: replay vs full fine-tuning: `replay_mel_1m` vs `full_mel_1m` | 0.599 | [0.555, 0.648] | a better |
| mel_ped 1m: kd vs zero-shot student: `kd_mel_1m` vs `zero_shot` | 1.976 | [1.764, 2.225] | b better |
| mel_ped 1m: kd vs full fine-tuning: `kd_mel_1m` vs `full_mel_1m` | 1.087 | [1.052, 1.127] | b better |
| mel_ped 1m: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-4m` | 1.685 | [1.558, 1.820] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-4m` | 1.009 | [0.979, 1.044] | no detectable difference |
| mel_ped 1m: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-22m` | 1.841 | [1.686, 2.011] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-22m` | 1.103 | [1.056, 1.154] | b better |
| mel_ped 1m: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1m` vs `zoo:toto-2.0-313m` | 1.992 | [1.806, 2.187] | b better |
| mel_ped 1m: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1m` vs `zoo:toto-2.0-313m` | 1.193 | [1.144, 1.249] | b better |
| mel_ped 1m: full fine-tuning vs timestral-5m zero-shot: `full_mel_1m` vs `zoo:timestral-5m` | 1.779 | [1.615, 1.951] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1m` vs `zoo:timestral-5m` | 1.066 | [1.036, 1.095] | b better |
| mel_ped 1m: full fine-tuning vs timestral-28m zero-shot: `full_mel_1m` vs `zoo:timestral-28m` | 1.959 | [1.779, 2.149] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1m` vs `zoo:timestral-28m` | 1.174 | [1.136, 1.214] | b better |
| mel_ped 1m: full fine-tuning vs timestral-209m zero-shot: `full_mel_1m` vs `zoo:timestral-209m` | 2.175 | [1.971, 2.399] | b better |
| mel_ped 1m: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1m` vs `zoo:timestral-209m` | 1.303 | [1.251, 1.360] | b better |
| mel_ped 1m: full fine-tuning vs chronos-2 zero-shot: `full_mel_1m` vs `zoo:chronos-2` | 1.736 | [1.553, 1.932] | b better |
| mel_ped 1m: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1m` vs `zoo:chronos-2` | 1.040 | [0.979, 1.106] | no detectable difference |
| mel_ped 1m: full fine-tuning vs t0-gamma zero-shot: `full_mel_1m` vs `zoo:t0-gamma` | 1.803 | [1.628, 1.990] | b better |
| mel_ped 1m: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1m` vs `zoo:t0-gamma` | 1.080 | [1.029, 1.133] | b better |
| mel_ped 1y: full vs zero-shot student: `full_mel_1y` vs `zero_shot` | 2.661 | [2.271, 3.132] | b better |
| mel_ped 1y: full_lowlr vs zero-shot student: `full_lowlr_mel_1y` vs `zero_shot` | 1.897 | [1.682, 2.144] | b better |
| mel_ped 1y: full_lowlr vs full fine-tuning: `full_lowlr_mel_1y` vs `full_mel_1y` | 0.713 | [0.675, 0.749] | a better |
| mel_ped 1y: head vs zero-shot student: `head_mel_1y` vs `zero_shot` | 1.850 | [1.675, 2.045] | b better |
| mel_ped 1y: head vs full fine-tuning: `head_mel_1y` vs `full_mel_1y` | 0.695 | [0.638, 0.755] | a better |
| mel_ped 1y: last vs zero-shot student: `last_mel_1y` vs `zero_shot` | 2.189 | [1.943, 2.474] | b better |
| mel_ped 1y: last vs full fine-tuning: `last_mel_1y` vs `full_mel_1y` | 0.823 | [0.778, 0.871] | a better |
| mel_ped 1y: l2sp_s vs zero-shot student: `l2sp_s_mel_1y` vs `zero_shot` | 1.951 | [1.717, 2.225] | b better |
| mel_ped 1y: l2sp_s vs full fine-tuning: `l2sp_s_mel_1y` vs `full_mel_1y` | 0.733 | [0.694, 0.772] | a better |
| mel_ped 1y: l2sp_l vs zero-shot student: `l2sp_l_mel_1y` vs `zero_shot` | 1.220 | [1.158, 1.289] | b better |
| mel_ped 1y: l2sp_l vs full fine-tuning: `l2sp_l_mel_1y` vs `full_mel_1y` | 0.458 | [0.400, 0.522] | a better |
| mel_ped 1y: replay vs zero-shot student: `replay_mel_1y` vs `zero_shot` | 1.018 | [1.007, 1.031] | b better |
| mel_ped 1y: replay vs full fine-tuning: `replay_mel_1y` vs `full_mel_1y` | 0.383 | [0.327, 0.446] | a better |
| mel_ped 1y: kd vs zero-shot student: `kd_mel_1y` vs `zero_shot` | 2.258 | [1.974, 2.589] | b better |
| mel_ped 1y: kd vs full fine-tuning: `kd_mel_1y` vs `full_mel_1y` | 0.849 | [0.817, 0.879] | a better |
| mel_ped 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-4m` | 2.465 | [2.141, 2.837] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-4m` | 0.944 | [0.908, 0.981] | a better |
| mel_ped 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-22m` | 2.694 | [2.320, 3.129] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-22m` | 1.031 | [0.981, 1.082] | no detectable difference |
| mel_ped 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_mel_1y` vs `zoo:toto-2.0-313m` | 2.915 | [2.482, 3.413] | b better |
| mel_ped 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_mel_1y` vs `zoo:toto-2.0-313m` | 1.116 | [1.070, 1.167] | b better |
| mel_ped 1y: full fine-tuning vs timestral-5m zero-shot: `full_mel_1y` vs `zoo:timestral-5m` | 2.603 | [2.223, 3.057] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_mel_1y` vs `zoo:timestral-5m` | 0.997 | [0.975, 1.019] | no detectable difference |
| mel_ped 1y: full fine-tuning vs timestral-28m zero-shot: `full_mel_1y` vs `zoo:timestral-28m` | 2.866 | [2.442, 3.378] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_mel_1y` vs `zoo:timestral-28m` | 1.097 | [1.061, 1.136] | b better |
| mel_ped 1y: full fine-tuning vs timestral-209m zero-shot: `full_mel_1y` vs `zoo:timestral-209m` | 3.183 | [2.698, 3.754] | b better |
| mel_ped 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_mel_1y` vs `zoo:timestral-209m` | 1.218 | [1.173, 1.269] | b better |
| mel_ped 1y: full fine-tuning vs chronos-2 zero-shot: `full_mel_1y` vs `zoo:chronos-2` | 2.541 | [2.138, 2.998] | b better |
| mel_ped 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_mel_1y` vs `zoo:chronos-2` | 0.973 | [0.915, 1.034] | no detectable difference |
| mel_ped 1y: full fine-tuning vs t0-gamma zero-shot: `full_mel_1y` vs `zoo:t0-gamma` | 2.638 | [2.245, 3.106] | b better |
| mel_ped 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_mel_1y` vs `zoo:t0-gamma` | 1.010 | [0.961, 1.061] | no detectable difference |
