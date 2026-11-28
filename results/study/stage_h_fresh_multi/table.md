# stage_h_fresh_multi: relative WQL vs seasonal naive (95% CI); paired ratio vs `zero_shot` (<1 = better)

## fresh_fr_solar (468 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| replay_solar_1y_s0 | 0.704 | [0.677, 0.735] | 0.888 [0.864, 0.910] | 0.89 |
| replay_solar_1y_s2 | 0.710 | [0.677, 0.747] | 0.895 [0.857, 0.932] | 0.89 |
| full_solar_1y_s2 | 0.719 | [0.692, 0.750] | 0.907 [0.878, 0.934] | 0.91 |
| full_solar_1y_s0 | 0.724 | [0.691, 0.761] | 0.912 [0.879, 0.943] | 0.91 |
| full_solar_1y_s1 | 0.745 | [0.707, 0.786] | 0.939 [0.893, 0.983] | 0.91 |
| replay_solar_1y_s1 | 0.750 | [0.714, 0.789] | 0.945 [0.901, 0.988] | 0.91 |
| kd_solar_1y_s2 | 0.780 | [0.741, 0.819] | 0.983 [0.939, 1.028] | 0.87 |
| kd_solar_1y_s1 | 0.782 | [0.743, 0.823] | 0.987 [0.941, 1.032] | 0.85 |
| zero_shot | 0.793 | [0.767, 0.821] | – | 0.76 |
| kd_solar_1y_s0 | 0.797 | [0.759, 0.837] | 1.005 [0.960, 1.051] | 0.89 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.82 |

zero-shot models (same suite, same baseline): t0-gamma 0.702, toto-2.0-313m 0.722, timestral-209m 0.728, timestral-28m 0.735, chronos-bolt-mini 0.752, chronos-bolt-tiny 0.759, chronos-2 0.760, timestral-5m 0.771, toto-2.0-22m 0.773, toto-2.0-4m 0.781, chronos-2-small 0.795, ets 2.668

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_solar 1y: replay vs zero-shot student: `replay_solar_1y` vs `zero_shot` | 0.912 | [0.876, 0.945] | a better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-4m` | 0.926 | [0.893, 0.959] | a better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-22m` | 0.936 | [0.912, 0.961] | a better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-313m` | 1.002 | [0.984, 1.019] | no detectable difference |
| fr_solar 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_solar_1y` vs `zoo:timestral-5m` | 0.938 | [0.926, 0.949] | a better |
| fr_solar 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_solar_1y` vs `zoo:timestral-28m` | 0.984 | [0.968, 0.999] | a better |
| fr_solar 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_solar_1y` vs `zoo:timestral-209m` | 0.994 | [0.976, 1.009] | no detectable difference |
| fr_solar 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_solar_1y` vs `zoo:chronos-2` | 0.952 | [0.932, 0.974] | a better |
| fr_solar 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_solar_1y` vs `zoo:t0-gamma` | 1.031 | [1.015, 1.044] | b better |
| fr_solar 1y: full vs zero-shot student: `full_solar_1y` vs `zero_shot` | 0.921 | [0.885, 0.955] | a better |
| fr_solar 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-4m` | 0.935 | [0.903, 0.968] | a better |
| fr_solar 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-22m` | 0.945 | [0.920, 0.971] | a better |
| fr_solar 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-313m` | 1.011 | [0.995, 1.029] | no detectable difference |
| fr_solar 1y: full fine-tuning vs timestral-5m zero-shot: `full_solar_1y` vs `zoo:timestral-5m` | 0.947 | [0.934, 0.960] | a better |
| fr_solar 1y: full fine-tuning vs timestral-28m zero-shot: `full_solar_1y` vs `zoo:timestral-28m` | 0.994 | [0.977, 1.010] | no detectable difference |
| fr_solar 1y: full fine-tuning vs timestral-209m zero-shot: `full_solar_1y` vs `zoo:timestral-209m` | 1.003 | [0.986, 1.019] | no detectable difference |
| fr_solar 1y: full fine-tuning vs chronos-2 zero-shot: `full_solar_1y` vs `zoo:chronos-2` | 0.961 | [0.941, 0.983] | a better |
| fr_solar 1y: full fine-tuning vs t0-gamma zero-shot: `full_solar_1y` vs `zoo:t0-gamma` | 1.041 | [1.025, 1.054] | b better |
| fr_solar 1y: kd vs zero-shot student: `kd_solar_1y` vs `zero_shot` | 0.992 | [0.947, 1.037] | no detectable difference |
| fr_solar 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-4m` | 1.007 | [0.968, 1.046] | no detectable difference |
| fr_solar 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-22m` | 1.017 | [0.985, 1.047] | no detectable difference |
| fr_solar 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-313m` | 1.089 | [1.069, 1.108] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_solar_1y` vs `zoo:timestral-5m` | 1.020 | [1.013, 1.028] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_solar_1y` vs `zoo:timestral-28m` | 1.070 | [1.060, 1.082] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_solar_1y` vs `zoo:timestral-209m` | 1.080 | [1.068, 1.093] | b better |
| fr_solar 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_solar_1y` vs `zoo:chronos-2` | 1.035 | [1.005, 1.070] | b better |
| fr_solar 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_solar_1y` vs `zoo:t0-gamma` | 1.121 | [1.098, 1.146] | b better |

## fresh_fr_wind (429 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| full_wind_1y_s1 | 0.722 | [0.681, 0.767] | 0.980 [0.958, 1.002] | 0.70 |
| kd_wind_1y_s1 | 0.722 | [0.692, 0.755] | 0.981 [0.975, 0.988] | 0.78 |
| replay_wind_1y_s1 | 0.727 | [0.689, 0.769] | 0.988 [0.965, 1.009] | 0.70 |
| kd_wind_1y_s0 | 0.727 | [0.697, 0.760] | 0.988 [0.983, 0.993] | 0.78 |
| kd_wind_1y_s2 | 0.730 | [0.697, 0.763] | 0.991 [0.987, 0.995] | 0.78 |
| replay_wind_1y_s2 | 0.735 | [0.696, 0.775] | 0.998 [0.983, 1.013] | 0.79 |
| zero_shot | 0.736 | [0.702, 0.773] | – | 0.78 |
| full_wind_1y_s0 | 0.745 | [0.707, 0.786] | 1.011 [0.986, 1.031] | 0.69 |
| full_wind_1y_s2 | 0.752 | [0.707, 0.803] | 1.021 [0.992, 1.051] | 0.60 |
| replay_wind_1y_s0 | 0.754 | [0.713, 0.798] | 1.024 [1.007, 1.041] | 0.76 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.80 |

zero-shot models (same suite, same baseline): timestral-209m 0.723, timestral-28m 0.723, timestral-5m 0.724, t0-gamma 0.726, chronos-2 0.730, toto-2.0-4m 0.732, chronos-2-small 0.733, toto-2.0-22m 0.734, toto-2.0-313m 0.740, chronos-bolt-tiny 0.748, chronos-bolt-mini 0.756, ets 0.950

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_wind 1y: replay vs zero-shot student: `replay_wind_1y` vs `zero_shot` | 1.007 | [0.991, 1.023] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-4m` | 1.013 | [0.990, 1.037] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-22m` | 1.011 | [0.992, 1.032] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-313m` | 1.003 | [0.984, 1.023] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_wind_1y` vs `zoo:timestral-5m` | 1.024 | [1.009, 1.040] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_wind_1y` vs `zoo:timestral-28m` | 1.025 | [1.010, 1.041] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_wind_1y` vs `zoo:timestral-209m` | 1.026 | [1.011, 1.041] | b better |
| fr_wind 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_wind_1y` vs `zoo:chronos-2` | 1.016 | [1.000, 1.032] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_wind_1y` vs `zoo:t0-gamma` | 1.021 | [1.013, 1.029] | b better |
| fr_wind 1y: full vs zero-shot student: `full_wind_1y` vs `zero_shot` | 1.013 | [0.990, 1.036] | no detectable difference |
| fr_wind 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-4m` | 1.020 | [0.992, 1.047] | no detectable difference |
| fr_wind 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-22m` | 1.017 | [0.993, 1.044] | no detectable difference |
| fr_wind 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-313m` | 1.009 | [0.986, 1.034] | no detectable difference |
| fr_wind 1y: full fine-tuning vs timestral-5m zero-shot: `full_wind_1y` vs `zoo:timestral-5m` | 1.030 | [1.011, 1.052] | b better |
| fr_wind 1y: full fine-tuning vs timestral-28m zero-shot: `full_wind_1y` vs `zoo:timestral-28m` | 1.032 | [1.010, 1.054] | b better |
| fr_wind 1y: full fine-tuning vs timestral-209m zero-shot: `full_wind_1y` vs `zoo:timestral-209m` | 1.032 | [1.011, 1.054] | b better |
| fr_wind 1y: full fine-tuning vs chronos-2 zero-shot: `full_wind_1y` vs `zoo:chronos-2` | 1.022 | [1.006, 1.041] | b better |
| fr_wind 1y: full fine-tuning vs t0-gamma zero-shot: `full_wind_1y` vs `zoo:t0-gamma` | 1.027 | [1.016, 1.039] | b better |
| fr_wind 1y: kd vs zero-shot student: `kd_wind_1y` vs `zero_shot` | 0.987 | [0.982, 0.992] | a better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-4m` | 0.993 | [0.974, 1.008] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-22m` | 0.990 | [0.977, 1.002] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-313m` | 0.982 | [0.970, 0.993] | a better |
| fr_wind 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_wind_1y` vs `zoo:timestral-5m` | 1.003 | [0.995, 1.012] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_wind_1y` vs `zoo:timestral-28m` | 1.005 | [0.996, 1.014] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_wind_1y` vs `zoo:timestral-209m` | 1.005 | [0.999, 1.012] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_wind_1y` vs `zoo:chronos-2` | 0.995 | [0.969, 1.016] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_wind_1y` vs `zoo:t0-gamma` | 1.000 | [0.980, 1.019] | no detectable difference |

## fresh_paris_bike (4151 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| full_bike_1y_s0 | 0.499 | [0.488, 0.511] | 0.888 [0.878, 0.899] | 0.81 |
| replay_bike_1y_s2 | 0.502 | [0.491, 0.513] | 0.892 [0.882, 0.903] | 0.78 |
| full_bike_1y_s2 | 0.509 | [0.497, 0.521] | 0.906 [0.894, 0.918] | 0.75 |
| full_bike_1y_s1 | 0.519 | [0.508, 0.529] | 0.923 [0.915, 0.931] | 0.81 |
| replay_bike_1y_s0 | 0.519 | [0.509, 0.529] | 0.924 [0.916, 0.931] | 0.76 |
| replay_bike_1y_s1 | 0.522 | [0.511, 0.533] | 0.928 [0.918, 0.937] | 0.76 |
| kd_bike_1y_s2 | 0.542 | [0.532, 0.552] | 0.964 [0.957, 0.969] | 0.86 |
| kd_bike_1y_s1 | 0.544 | [0.534, 0.555] | 0.968 [0.960, 0.973] | 0.85 |
| kd_bike_1y_s0 | 0.549 | [0.539, 0.559] | 0.976 [0.969, 0.981] | 0.85 |
| zero_shot | 0.562 | [0.554, 0.571] | – | 0.87 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.83 |

zero-shot models (same suite, same baseline): t0-gamma 0.444, chronos-2 0.458, timestral-209m 0.482, chronos-2-small 0.485, toto-2.0-313m 0.501, timestral-28m 0.522, chronos-bolt-mini 0.530, toto-2.0-22m 0.548, timestral-5m 0.559, chronos-bolt-tiny 0.560, toto-2.0-4m 0.720, ets 0.997

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| paris_bike 1y: replay vs zero-shot student: `replay_bike_1y` vs `zero_shot` | 0.917 | [0.908, 0.926] | a better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-4m` | 0.717 | [0.704, 0.730] | a better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-22m` | 0.942 | [0.935, 0.949] | a better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-313m` | 1.031 | [1.023, 1.039] | b better |
| paris_bike 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_bike_1y` vs `zoo:timestral-5m` | 0.923 | [0.915, 0.932] | a better |
| paris_bike 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_bike_1y` vs `zoo:timestral-28m` | 0.987 | [0.980, 0.995] | a better |
| paris_bike 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_bike_1y` vs `zoo:timestral-209m` | 1.070 | [1.063, 1.077] | b better |
| paris_bike 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_bike_1y` vs `zoo:chronos-2` | 1.126 | [1.117, 1.134] | b better |
| paris_bike 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_bike_1y` vs `zoo:t0-gamma` | 1.163 | [1.153, 1.172] | b better |
| paris_bike 1y: full vs zero-shot student: `full_bike_1y` vs `zero_shot` | 0.908 | [0.898, 0.918] | a better |
| paris_bike 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-4m` | 0.709 | [0.695, 0.724] | a better |
| paris_bike 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-22m` | 0.932 | [0.924, 0.940] | a better |
| paris_bike 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-313m` | 1.020 | [1.011, 1.029] | b better |
| paris_bike 1y: full fine-tuning vs timestral-5m zero-shot: `full_bike_1y` vs `zoo:timestral-5m` | 0.913 | [0.904, 0.923] | a better |
| paris_bike 1y: full fine-tuning vs timestral-28m zero-shot: `full_bike_1y` vs `zoo:timestral-28m` | 0.977 | [0.969, 0.986] | a better |
| paris_bike 1y: full fine-tuning vs timestral-209m zero-shot: `full_bike_1y` vs `zoo:timestral-209m` | 1.059 | [1.052, 1.067] | b better |
| paris_bike 1y: full fine-tuning vs chronos-2 zero-shot: `full_bike_1y` vs `zoo:chronos-2` | 1.114 | [1.106, 1.123] | b better |
| paris_bike 1y: full fine-tuning vs t0-gamma zero-shot: `full_bike_1y` vs `zoo:t0-gamma` | 1.150 | [1.141, 1.160] | b better |
| paris_bike 1y: kd vs zero-shot student: `kd_bike_1y` vs `zero_shot` | 0.969 | [0.962, 0.974] | a better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-4m` | 0.758 | [0.747, 0.768] | a better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-22m` | 0.995 | [0.989, 1.002] | no detectable difference |
| paris_bike 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-313m` | 1.089 | [1.081, 1.097] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_bike_1y` vs `zoo:timestral-5m` | 0.975 | [0.973, 0.978] | a better |
| paris_bike 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_bike_1y` vs `zoo:timestral-28m` | 1.043 | [1.039, 1.048] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_bike_1y` vs `zoo:timestral-209m` | 1.131 | [1.121, 1.141] | b better |
| paris_bike 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_bike_1y` vs `zoo:chronos-2` | 1.189 | [1.179, 1.201] | b better |
| paris_bike 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_bike_1y` vs `zoo:t0-gamma` | 1.228 | [1.215, 1.243] | b better |

## multidomain_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| zero_shot | 0.676 | [0.644, 0.712] | – | 0.80 |
| replay_bike_1y_s0 | 0.678 | [0.648, 0.712] | 1.002 [0.989, 1.015] | 0.79 |
| replay_bike_1y_s2 | 0.679 | [0.648, 0.714] | 1.004 [0.992, 1.016] | 0.80 |
| replay_bike_1y_s1 | 0.681 | [0.649, 0.718] | 1.006 [0.995, 1.017] | 0.80 |
| replay_solar_1y_s1 | 0.682 | [0.651, 0.718] | 1.008 [0.995, 1.022] | 0.82 |
| replay_solar_1y_s2 | 0.685 | [0.653, 0.722] | 1.013 [1.001, 1.027] | 0.81 |
| replay_solar_1y_s0 | 0.687 | [0.654, 0.726] | 1.016 [1.003, 1.029] | 0.81 |
| full_solar_1y_s1 | 0.697 | [0.668, 0.726] | 1.030 [1.007, 1.052] | 0.72 |
| full_solar_1y_s2 | 0.701 | [0.673, 0.732] | 1.037 [1.016, 1.058] | 0.72 |
| full_solar_1y_s0 | 0.703 | [0.674, 0.732] | 1.039 [1.015, 1.062] | 0.71 |
| replay_wind_1y_s2 | 0.714 | [0.668, 0.757] | 1.056 [1.011, 1.113] | 0.80 |
| replay_wind_1y_s0 | 0.719 | [0.676, 0.759] | 1.063 [1.021, 1.118] | 0.82 |
| kd_wind_1y_s2 | 0.720 | [0.687, 0.757] | 1.064 [1.038, 1.096] | 0.80 |
| replay_wind_1y_s1 | 0.721 | [0.677, 0.763] | 1.066 [1.021, 1.127] | 0.79 |
| kd_bike_1y_s0 | 0.723 | [0.690, 0.759] | 1.069 [1.038, 1.104] | 0.77 |
| kd_wind_1y_s1 | 0.729 | [0.694, 0.768] | 1.077 [1.048, 1.115] | 0.80 |
| kd_wind_1y_s0 | 0.732 | [0.696, 0.773] | 1.082 [1.050, 1.124] | 0.81 |
| kd_bike_1y_s1 | 0.732 | [0.696, 0.772] | 1.083 [1.050, 1.119] | 0.76 |
| full_bike_1y_s1 | 0.733 | [0.691, 0.771] | 1.083 [1.028, 1.146] | 0.74 |
| kd_bike_1y_s2 | 0.736 | [0.699, 0.778] | 1.089 [1.056, 1.126] | 0.76 |
| full_bike_1y_s2 | 0.753 | [0.708, 0.794] | 1.113 [1.052, 1.183] | 0.67 |
| full_bike_1y_s0 | 0.773 | [0.718, 0.828] | 1.143 [1.071, 1.231] | 0.67 |
| kd_solar_1y_s0 | 0.793 | [0.740, 0.854] | 1.173 [1.114, 1.242] | 0.66 |
| kd_solar_1y_s1 | 0.796 | [0.740, 0.858] | 1.176 [1.115, 1.249] | 0.65 |
| kd_solar_1y_s2 | 0.804 | [0.747, 0.869] | 1.189 [1.125, 1.264] | 0.65 |
| full_wind_1y_s1 | 0.807 | [0.750, 0.862] | 1.192 [1.120, 1.281] | 0.72 |
| full_wind_1y_s0 | 0.823 | [0.757, 0.895] | 1.216 [1.127, 1.337] | 0.71 |
| full_wind_1y_s2 | 0.932 | [0.857, 1.009] | 1.378 [1.273, 1.507] | 0.64 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |

zero-shot models (same suite, same baseline): t0-gamma 0.511, chronos-2 0.614, timestral-209m 0.628, chronos-bolt-mini 0.637, timestral-28m 0.638, chronos-2-small 0.642, chronos-bolt-tiny 0.651, timestral-5m 0.660, toto-2.0-313m 0.667, toto-2.0-22m 0.681, toto-2.0-4m 0.721, ttm-r2.1-l1 0.884, ttm-r2 0.973, ets 1.220

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_solar 1y: replay vs zero-shot student: `replay_solar_1y` vs `zero_shot` | 1.016 | [1.006, 1.028] | b better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-4m` | 0.954 | [0.932, 0.977] | a better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-22m` | 1.009 | [0.988, 1.034] | no detectable difference |
| fr_solar 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-313m` | 1.031 | [1.008, 1.058] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_solar_1y` vs `zoo:timestral-5m` | 1.042 | [1.027, 1.059] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_solar_1y` vs `zoo:timestral-28m` | 1.078 | [1.061, 1.097] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_solar_1y` vs `zoo:timestral-209m` | 1.095 | [1.076, 1.117] | b better |
| fr_solar 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_solar_1y` vs `zoo:chronos-2` | 1.119 | [1.095, 1.144] | b better |
| fr_solar 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_solar_1y` vs `zoo:t0-gamma` | 1.346 | [1.237, 1.479] | b better |
| fr_solar 1y: full vs zero-shot student: `full_solar_1y` vs `zero_shot` | 1.036 | [1.013, 1.057] | b better |
| fr_solar 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-4m` | 0.973 | [0.945, 1.000] | a better |
| fr_solar 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-22m` | 1.029 | [1.005, 1.052] | b better |
| fr_solar 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-313m` | 1.051 | [1.022, 1.080] | b better |
| fr_solar 1y: full fine-tuning vs timestral-5m zero-shot: `full_solar_1y` vs `zoo:timestral-5m` | 1.062 | [1.037, 1.087] | b better |
| fr_solar 1y: full fine-tuning vs timestral-28m zero-shot: `full_solar_1y` vs `zoo:timestral-28m` | 1.099 | [1.073, 1.125] | b better |
| fr_solar 1y: full fine-tuning vs timestral-209m zero-shot: `full_solar_1y` vs `zoo:timestral-209m` | 1.117 | [1.087, 1.145] | b better |
| fr_solar 1y: full fine-tuning vs chronos-2 zero-shot: `full_solar_1y` vs `zoo:chronos-2` | 1.141 | [1.106, 1.176] | b better |
| fr_solar 1y: full fine-tuning vs t0-gamma zero-shot: `full_solar_1y` vs `zoo:t0-gamma` | 1.373 | [1.261, 1.501] | b better |
| fr_solar 1y: kd vs zero-shot student: `kd_solar_1y` vs `zero_shot` | 1.180 | [1.118, 1.252] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-4m` | 1.107 | [1.054, 1.170] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-22m` | 1.171 | [1.103, 1.250] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-313m` | 1.197 | [1.130, 1.277] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_solar_1y` vs `zoo:timestral-5m` | 1.209 | [1.148, 1.283] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_solar_1y` vs `zoo:timestral-28m` | 1.251 | [1.187, 1.327] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_solar_1y` vs `zoo:timestral-209m` | 1.271 | [1.207, 1.348] | b better |
| fr_solar 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_solar_1y` vs `zoo:chronos-2` | 1.299 | [1.236, 1.377] | b better |
| fr_solar 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_solar_1y` vs `zoo:t0-gamma` | 1.562 | [1.406, 1.749] | b better |
| fr_wind 1y: replay vs zero-shot student: `replay_wind_1y` vs `zero_shot` | 1.066 | [1.024, 1.123] | b better |
| fr_wind 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-4m` | 1.001 | [0.947, 1.065] | no detectable difference |
| fr_wind 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-22m` | 1.059 | [1.012, 1.113] | b better |
| fr_wind 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-313m` | 1.082 | [1.029, 1.144] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_wind_1y` vs `zoo:timestral-5m` | 1.093 | [1.045, 1.156] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_wind_1y` vs `zoo:timestral-28m` | 1.131 | [1.081, 1.196] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_wind_1y` vs `zoo:timestral-209m` | 1.149 | [1.095, 1.216] | b better |
| fr_wind 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_wind_1y` vs `zoo:chronos-2` | 1.174 | [1.115, 1.246] | b better |
| fr_wind 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_wind_1y` vs `zoo:t0-gamma` | 1.413 | [1.284, 1.546] | b better |
| fr_wind 1y: full vs zero-shot student: `full_wind_1y` vs `zero_shot` | 1.276 | [1.187, 1.391] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-4m` | 1.198 | [1.104, 1.317] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-22m` | 1.267 | [1.178, 1.376] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-313m` | 1.295 | [1.199, 1.414] | b better |
| fr_wind 1y: full fine-tuning vs timestral-5m zero-shot: `full_wind_1y` vs `zoo:timestral-5m` | 1.309 | [1.214, 1.433] | b better |
| fr_wind 1y: full fine-tuning vs timestral-28m zero-shot: `full_wind_1y` vs `zoo:timestral-28m` | 1.354 | [1.255, 1.480] | b better |
| fr_wind 1y: full fine-tuning vs timestral-209m zero-shot: `full_wind_1y` vs `zoo:timestral-209m` | 1.375 | [1.274, 1.504] | b better |
| fr_wind 1y: full fine-tuning vs chronos-2 zero-shot: `full_wind_1y` vs `zoo:chronos-2` | 1.405 | [1.299, 1.542] | b better |
| fr_wind 1y: full fine-tuning vs t0-gamma zero-shot: `full_wind_1y` vs `zoo:t0-gamma` | 1.691 | [1.515, 1.865] | b better |
| fr_wind 1y: kd vs zero-shot student: `kd_wind_1y` vs `zero_shot` | 1.076 | [1.047, 1.112] | b better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-4m` | 1.010 | [0.982, 1.041] | no detectable difference |
| fr_wind 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-22m` | 1.068 | [1.034, 1.107] | b better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-313m` | 1.091 | [1.056, 1.133] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_wind_1y` vs `zoo:timestral-5m` | 1.103 | [1.074, 1.140] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_wind_1y` vs `zoo:timestral-28m` | 1.141 | [1.112, 1.179] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_wind_1y` vs `zoo:timestral-209m` | 1.159 | [1.128, 1.198] | b better |
| fr_wind 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_wind_1y` vs `zoo:chronos-2` | 1.184 | [1.151, 1.225] | b better |
| fr_wind 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_wind_1y` vs `zoo:t0-gamma` | 1.425 | [1.304, 1.570] | b better |
| paris_bike 1y: replay vs zero-shot student: `replay_bike_1y` vs `zero_shot` | 1.007 | [0.997, 1.017] | no detectable difference |
| paris_bike 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-4m` | 0.945 | [0.919, 0.970] | a better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-22m` | 0.999 | [0.979, 1.023] | no detectable difference |
| paris_bike 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-313m` | 1.021 | [0.998, 1.046] | no detectable difference |
| paris_bike 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_bike_1y` vs `zoo:timestral-5m` | 1.032 | [1.014, 1.050] | b better |
| paris_bike 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_bike_1y` vs `zoo:timestral-28m` | 1.068 | [1.049, 1.086] | b better |
| paris_bike 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_bike_1y` vs `zoo:timestral-209m` | 1.085 | [1.064, 1.105] | b better |
| paris_bike 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_bike_1y` vs `zoo:chronos-2` | 1.108 | [1.082, 1.134] | b better |
| paris_bike 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_bike_1y` vs `zoo:t0-gamma` | 1.333 | [1.222, 1.458] | b better |
| paris_bike 1y: full vs zero-shot student: `full_bike_1y` vs `zero_shot` | 1.120 | [1.057, 1.193] | b better |
| paris_bike 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-4m` | 1.051 | [0.988, 1.128] | no detectable difference |
| paris_bike 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-22m` | 1.112 | [1.052, 1.181] | b better |
| paris_bike 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-313m` | 1.136 | [1.071, 1.212] | b better |
| paris_bike 1y: full fine-tuning vs timestral-5m zero-shot: `full_bike_1y` vs `zoo:timestral-5m` | 1.148 | [1.084, 1.226] | b better |
| paris_bike 1y: full fine-tuning vs timestral-28m zero-shot: `full_bike_1y` vs `zoo:timestral-28m` | 1.188 | [1.120, 1.271] | b better |
| paris_bike 1y: full fine-tuning vs timestral-209m zero-shot: `full_bike_1y` vs `zoo:timestral-209m` | 1.206 | [1.137, 1.294] | b better |
| paris_bike 1y: full fine-tuning vs chronos-2 zero-shot: `full_bike_1y` vs `zoo:chronos-2` | 1.233 | [1.160, 1.324] | b better |
| paris_bike 1y: full fine-tuning vs t0-gamma zero-shot: `full_bike_1y` vs `zoo:t0-gamma` | 1.483 | [1.344, 1.627] | b better |
| paris_bike 1y: kd vs zero-shot student: `kd_bike_1y` vs `zero_shot` | 1.082 | [1.050, 1.118] | b better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-4m` | 1.016 | [0.979, 1.052] | no detectable difference |
| paris_bike 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-22m` | 1.075 | [1.041, 1.111] | b better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-313m` | 1.098 | [1.062, 1.135] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_bike_1y` vs `zoo:timestral-5m` | 1.110 | [1.076, 1.146] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_bike_1y` vs `zoo:timestral-28m` | 1.148 | [1.114, 1.184] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_bike_1y` vs `zoo:timestral-209m` | 1.166 | [1.131, 1.202] | b better |
| paris_bike 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_bike_1y` vs `zoo:chronos-2` | 1.192 | [1.154, 1.231] | b better |
| paris_bike 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_bike_1y` vs `zoo:t0-gamma` | 1.434 | [1.313, 1.575] | b better |

## shapes (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| zero_shot | 0.481 | [0.436, 0.533] | – | 0.83 |
| replay_bike_1y_s2 | 0.493 | [0.446, 0.548] | 1.023 [1.011, 1.036] | 0.84 |
| replay_bike_1y_s1 | 0.493 | [0.448, 0.546] | 1.025 [1.009, 1.042] | 0.84 |
| replay_solar_1y_s0 | 0.503 | [0.457, 0.555] | 1.045 [1.024, 1.068] | 0.85 |
| replay_solar_1y_s1 | 0.509 | [0.458, 0.570] | 1.058 [1.035, 1.084] | 0.82 |
| replay_bike_1y_s0 | 0.512 | [0.465, 0.566] | 1.064 [1.036, 1.092] | 0.83 |
| replay_solar_1y_s2 | 0.524 | [0.477, 0.579] | 1.089 [1.072, 1.106] | 0.86 |
| replay_wind_1y_s0 | 0.537 | [0.491, 0.589] | 1.115 [1.086, 1.143] | 0.89 |
| replay_wind_1y_s1 | 0.566 | [0.511, 0.628] | 1.176 [1.140, 1.212] | 0.86 |
| replay_wind_1y_s2 | 0.568 | [0.517, 0.626] | 1.181 [1.128, 1.239] | 0.86 |
| full_solar_1y_s1 | 0.639 | [0.586, 0.702] | 1.327 [1.259, 1.406] | 0.62 |
| full_solar_1y_s0 | 0.704 | [0.646, 0.771] | 1.463 [1.368, 1.572] | 0.55 |
| full_solar_1y_s2 | 0.712 | [0.651, 0.781] | 1.479 [1.368, 1.607] | 0.57 |
| kd_wind_1y_s2 | 0.763 | [0.688, 0.849] | 1.584 [1.458, 1.723] | 0.68 |
| full_wind_1y_s0 | 0.782 | [0.711, 0.862] | 1.624 [1.496, 1.773] | 0.61 |
| kd_wind_1y_s0 | 0.824 | [0.737, 0.930] | 1.712 [1.567, 1.870] | 0.72 |
| kd_wind_1y_s1 | 0.838 | [0.750, 0.943] | 1.741 [1.592, 1.905] | 0.70 |
| full_wind_1y_s1 | 0.902 | [0.807, 1.011] | 1.874 [1.680, 2.105] | 0.55 |
| full_wind_1y_s2 | 0.959 | [0.857, 1.079] | 1.992 [1.779, 2.235] | 0.58 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.76 |
| full_bike_1y_s0 | 1.012 | [0.907, 1.136] | 2.103 [1.871, 2.366] | 0.57 |
| full_bike_1y_s1 | 1.013 | [0.911, 1.133] | 2.105 [1.872, 2.386] | 0.57 |
| kd_bike_1y_s0 | 1.048 | [0.928, 1.191] | 2.177 [1.935, 2.441] | 0.64 |
| kd_bike_1y_s1 | 1.132 | [0.995, 1.298] | 2.352 [2.069, 2.655] | 0.58 |
| full_bike_1y_s2 | 1.138 | [1.005, 1.298] | 2.365 [2.076, 2.687] | 0.51 |
| kd_bike_1y_s2 | 1.178 | [1.033, 1.356] | 2.449 [2.147, 2.771] | 0.56 |
| kd_solar_1y_s0 | 1.412 | [1.214, 1.662] | 2.933 [2.540, 3.391] | 0.41 |
| kd_solar_1y_s1 | 1.451 | [1.244, 1.707] | 3.014 [2.603, 3.494] | 0.41 |
| kd_solar_1y_s2 | 1.480 | [1.271, 1.741] | 3.076 [2.655, 3.567] | 0.41 |

zero-shot models (same suite, same baseline): timestral-209m 0.402, toto-2.0-313m 0.439, timestral-28m 0.447, toto-2.0-22m 0.475, t0-gamma 0.485, timestral-5m 0.492, chronos-2-small 0.494, chronos-2 0.504, toto-2.0-4m 0.519, chronos-bolt-mini 0.651, chronos-bolt-tiny 0.686, ets 1.246, ttm-r2 1.380, ttm-r2.1-l1 1.790

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_solar 1y: replay vs zero-shot student: `replay_solar_1y` vs `zero_shot` | 1.069 | [1.052, 1.087] | b better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-4m` | 0.991 | [0.959, 1.026] | no detectable difference |
| fr_solar 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-22m` | 1.083 | [1.035, 1.136] | b better |
| fr_solar 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_solar_1y` vs `zoo:toto-2.0-313m` | 1.172 | [1.125, 1.225] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_solar_1y` vs `zoo:timestral-5m` | 1.047 | [1.020, 1.075] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_solar_1y` vs `zoo:timestral-28m` | 1.152 | [1.115, 1.194] | b better |
| fr_solar 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_solar_1y` vs `zoo:timestral-209m` | 1.279 | [1.227, 1.339] | b better |
| fr_solar 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_solar_1y` vs `zoo:chronos-2` | 1.021 | [0.959, 1.088] | no detectable difference |
| fr_solar 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_solar_1y` vs `zoo:t0-gamma` | 1.061 | [1.006, 1.118] | b better |
| fr_solar 1y: full vs zero-shot student: `full_solar_1y` vs `zero_shot` | 1.431 | [1.339, 1.535] | b better |
| fr_solar 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-4m` | 1.326 | [1.263, 1.392] | b better |
| fr_solar 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-22m` | 1.449 | [1.370, 1.528] | b better |
| fr_solar 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_solar_1y` vs `zoo:toto-2.0-313m` | 1.567 | [1.469, 1.670] | b better |
| fr_solar 1y: full fine-tuning vs timestral-5m zero-shot: `full_solar_1y` vs `zoo:timestral-5m` | 1.400 | [1.312, 1.497] | b better |
| fr_solar 1y: full fine-tuning vs timestral-28m zero-shot: `full_solar_1y` vs `zoo:timestral-28m` | 1.541 | [1.444, 1.647] | b better |
| fr_solar 1y: full fine-tuning vs timestral-209m zero-shot: `full_solar_1y` vs `zoo:timestral-209m` | 1.712 | [1.592, 1.842] | b better |
| fr_solar 1y: full fine-tuning vs chronos-2 zero-shot: `full_solar_1y` vs `zoo:chronos-2` | 1.366 | [1.260, 1.477] | b better |
| fr_solar 1y: full fine-tuning vs t0-gamma zero-shot: `full_solar_1y` vs `zoo:t0-gamma` | 1.419 | [1.310, 1.535] | b better |
| fr_solar 1y: kd vs zero-shot student: `kd_solar_1y` vs `zero_shot` | 3.009 | [2.601, 3.487] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-4m` | 2.788 | [2.454, 3.160] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-22m` | 3.047 | [2.648, 3.492] | b better |
| fr_solar 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_solar_1y` vs `zoo:toto-2.0-313m` | 3.296 | [2.842, 3.821] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_solar_1y` vs `zoo:timestral-5m` | 2.944 | [2.540, 3.396] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_solar_1y` vs `zoo:timestral-28m` | 3.241 | [2.794, 3.759] | b better |
| fr_solar 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_solar_1y` vs `zoo:timestral-209m` | 3.600 | [3.095, 4.196] | b better |
| fr_solar 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_solar_1y` vs `zoo:chronos-2` | 2.874 | [2.448, 3.367] | b better |
| fr_solar 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_solar_1y` vs `zoo:t0-gamma` | 2.984 | [2.554, 3.476] | b better |
| fr_wind 1y: replay vs zero-shot student: `replay_wind_1y` vs `zero_shot` | 1.170 | [1.133, 1.208] | b better |
| fr_wind 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-4m` | 1.084 | [1.042, 1.129] | b better |
| fr_wind 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-22m` | 1.185 | [1.129, 1.245] | b better |
| fr_wind 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_wind_1y` vs `zoo:toto-2.0-313m` | 1.282 | [1.227, 1.344] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_wind_1y` vs `zoo:timestral-5m` | 1.145 | [1.104, 1.187] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_wind_1y` vs `zoo:timestral-28m` | 1.260 | [1.212, 1.311] | b better |
| fr_wind 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_wind_1y` vs `zoo:timestral-209m` | 1.399 | [1.346, 1.460] | b better |
| fr_wind 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_wind_1y` vs `zoo:chronos-2` | 1.117 | [1.050, 1.186] | b better |
| fr_wind 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_wind_1y` vs `zoo:t0-gamma` | 1.160 | [1.098, 1.221] | b better |
| fr_wind 1y: full vs zero-shot student: `full_wind_1y` vs `zero_shot` | 1.854 | [1.678, 2.063] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-4m` | 1.717 | [1.586, 1.862] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-22m` | 1.877 | [1.714, 2.057] | b better |
| fr_wind 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_wind_1y` vs `zoo:toto-2.0-313m` | 2.031 | [1.840, 2.233] | b better |
| fr_wind 1y: full fine-tuning vs timestral-5m zero-shot: `full_wind_1y` vs `zoo:timestral-5m` | 1.814 | [1.637, 2.013] | b better |
| fr_wind 1y: full fine-tuning vs timestral-28m zero-shot: `full_wind_1y` vs `zoo:timestral-28m` | 1.997 | [1.806, 2.209] | b better |
| fr_wind 1y: full fine-tuning vs timestral-209m zero-shot: `full_wind_1y` vs `zoo:timestral-209m` | 2.217 | [1.997, 2.470] | b better |
| fr_wind 1y: full fine-tuning vs chronos-2 zero-shot: `full_wind_1y` vs `zoo:chronos-2` | 1.770 | [1.580, 1.972] | b better |
| fr_wind 1y: full fine-tuning vs t0-gamma zero-shot: `full_wind_1y` vs `zoo:t0-gamma` | 1.838 | [1.637, 2.051] | b better |
| fr_wind 1y: kd vs zero-shot student: `kd_wind_1y` vs `zero_shot` | 1.685 | [1.545, 1.838] | b better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-4m` | 1.562 | [1.456, 1.674] | b better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-22m` | 1.707 | [1.569, 1.855] | b better |
| fr_wind 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_wind_1y` vs `zoo:toto-2.0-313m` | 1.846 | [1.689, 2.020] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_wind_1y` vs `zoo:timestral-5m` | 1.649 | [1.507, 1.798] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_wind_1y` vs `zoo:timestral-28m` | 1.816 | [1.657, 1.981] | b better |
| fr_wind 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_wind_1y` vs `zoo:timestral-209m` | 2.016 | [1.840, 2.215] | b better |
| fr_wind 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_wind_1y` vs `zoo:chronos-2` | 1.610 | [1.443, 1.783] | b better |
| fr_wind 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_wind_1y` vs `zoo:t0-gamma` | 1.671 | [1.509, 1.844] | b better |
| paris_bike 1y: replay vs zero-shot student: `replay_bike_1y` vs `zero_shot` | 1.041 | [1.024, 1.058] | b better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-4m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-4m` | 0.964 | [0.934, 0.996] | a better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-22m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-22m` | 1.054 | [1.009, 1.102] | b better |
| paris_bike 1y: replay fine-tuning vs toto-2.0-313m zero-shot: `replay_bike_1y` vs `zoo:toto-2.0-313m` | 1.140 | [1.098, 1.188] | b better |
| paris_bike 1y: replay fine-tuning vs timestral-5m zero-shot: `replay_bike_1y` vs `zoo:timestral-5m` | 1.019 | [0.993, 1.044] | no detectable difference |
| paris_bike 1y: replay fine-tuning vs timestral-28m zero-shot: `replay_bike_1y` vs `zoo:timestral-28m` | 1.121 | [1.087, 1.160] | b better |
| paris_bike 1y: replay fine-tuning vs timestral-209m zero-shot: `replay_bike_1y` vs `zoo:timestral-209m` | 1.245 | [1.198, 1.299] | b better |
| paris_bike 1y: replay fine-tuning vs chronos-2 zero-shot: `replay_bike_1y` vs `zoo:chronos-2` | 0.994 | [0.937, 1.055] | no detectable difference |
| paris_bike 1y: replay fine-tuning vs t0-gamma zero-shot: `replay_bike_1y` vs `zoo:t0-gamma` | 1.032 | [0.981, 1.085] | no detectable difference |
| paris_bike 1y: full vs zero-shot student: `full_bike_1y` vs `zero_shot` | 2.222 | [1.967, 2.515] | b better |
| paris_bike 1y: full fine-tuning vs toto-2.0-4m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-4m` | 2.059 | [1.857, 2.278] | b better |
| paris_bike 1y: full fine-tuning vs toto-2.0-22m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-22m` | 2.250 | [2.020, 2.503] | b better |
| paris_bike 1y: full fine-tuning vs toto-2.0-313m zero-shot: `full_bike_1y` vs `zoo:toto-2.0-313m` | 2.434 | [2.161, 2.735] | b better |
| paris_bike 1y: full fine-tuning vs timestral-5m zero-shot: `full_bike_1y` vs `zoo:timestral-5m` | 2.174 | [1.928, 2.443] | b better |
| paris_bike 1y: full fine-tuning vs timestral-28m zero-shot: `full_bike_1y` vs `zoo:timestral-28m` | 2.394 | [2.121, 2.688] | b better |
| paris_bike 1y: full fine-tuning vs timestral-209m zero-shot: `full_bike_1y` vs `zoo:timestral-209m` | 2.658 | [2.343, 3.011] | b better |
| paris_bike 1y: full fine-tuning vs chronos-2 zero-shot: `full_bike_1y` vs `zoo:chronos-2` | 2.122 | [1.869, 2.396] | b better |
| paris_bike 1y: full fine-tuning vs t0-gamma zero-shot: `full_bike_1y` vs `zoo:t0-gamma` | 2.204 | [1.943, 2.493] | b better |
| paris_bike 1y: kd vs zero-shot student: `kd_bike_1y` vs `zero_shot` | 2.339 | [2.060, 2.632] | b better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-4m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-4m` | 2.167 | [1.929, 2.427] | b better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-22m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-22m` | 2.369 | [2.092, 2.677] | b better |
| paris_bike 1y: kd fine-tuning vs toto-2.0-313m zero-shot: `kd_bike_1y` vs `zoo:toto-2.0-313m` | 2.562 | [2.249, 2.910] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-5m zero-shot: `kd_bike_1y` vs `zoo:timestral-5m` | 2.289 | [2.018, 2.581] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-28m zero-shot: `kd_bike_1y` vs `zoo:timestral-28m` | 2.520 | [2.220, 2.848] | b better |
| paris_bike 1y: kd fine-tuning vs timestral-209m zero-shot: `kd_bike_1y` vs `zoo:timestral-209m` | 2.798 | [2.437, 3.194] | b better |
| paris_bike 1y: kd fine-tuning vs chronos-2 zero-shot: `kd_bike_1y` vs `zoo:chronos-2` | 2.234 | [1.964, 2.538] | b better |
| paris_bike 1y: kd fine-tuning vs t0-gamma zero-shot: `kd_bike_1y` vs `zoo:t0-gamma` | 2.320 | [2.042, 2.641] | b better |
