# stage_d_pilot: relative WQL vs seasonal naive (95% CI); paired ratio vs `distilled_zero_shot` (<1 = better)

## fresh_fr_power (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_fr_1y_scratch | 0.713 | [0.671, 0.754] | 0.689 [0.652, 0.724] | 0.76 |
| ft_fr_1y_distilled | 0.748 | [0.695, 0.803] | 0.723 [0.683, 0.763] | 0.72 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.74 |
| distilled_zero_shot | 1.035 | [0.971, 1.109] | – | 0.71 |
| ft_fr_1m_distilled | 1.109 | [1.016, 1.213] | 1.072 [1.037, 1.107] | 0.34 |
| ft_fr_1m_scratch | 1.135 | [1.017, 1.267] | 1.097 [1.040, 1.157] | 0.32 |
| ft_fr_1w_distilled | 1.309 | [1.238, 1.383] | 1.265 [1.226, 1.307] | 0.34 |
| ft_fr_1w_scratch | 2.192 | [1.998, 2.438] | 2.119 [1.970, 2.287] | 0.65 |

zero-shot models (same suite, same baseline): timestral-209m 0.614, timestral-28m 0.615, toto-2.0-313m 0.619, chronos-2 0.624, toto-2.0-22m 0.638, timestral-5m 0.641, chronos-2-small 0.642, t0-gamma 0.668, toto-2.0-4m 0.709, ets 1.391

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| fr_power 1w: distilled init vs scratch: `ft_fr_1w_distilled` vs `ft_fr_1w_scratch` | 0.597 | [0.541, 0.654] | a better |
| fr_power 1w: fine-tuned vs not fine-tuned: `ft_fr_1w_distilled` vs `distilled_zero_shot` | 1.265 | [1.226, 1.307] | b better |
| fr_power 1w: fine-tuned student vs timestral-209m zero-shot: `ft_fr_1w_distilled` vs `zoo:timestral-209m` | 2.132 | [2.022, 2.273] | b better |
| fr_power 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_fr_1w_distilled` vs `zoo:toto-2.0-313m` | 2.115 | [2.000, 2.247] | b better |
| fr_power 1w: fine-tuned student vs timestral-5m zero-shot: `ft_fr_1w_distilled` vs `zoo:timestral-5m` | 2.043 | [1.921, 2.189] | b better |
| fr_power 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_fr_1w_distilled` vs `zoo:toto-2.0-4m` | 1.847 | [1.746, 1.973] | b better |
| fr_power 1m: distilled init vs scratch: `ft_fr_1m_distilled` vs `ft_fr_1m_scratch` | 0.977 | [0.946, 1.010] | no detectable difference |
| fr_power 1m: fine-tuned vs not fine-tuned: `ft_fr_1m_distilled` vs `distilled_zero_shot` | 1.072 | [1.037, 1.107] | b better |
| fr_power 1m: fine-tuned student vs timestral-209m zero-shot: `ft_fr_1m_distilled` vs `zoo:timestral-209m` | 1.807 | [1.702, 1.937] | b better |
| fr_power 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_fr_1m_distilled` vs `zoo:toto-2.0-313m` | 1.792 | [1.685, 1.928] | b better |
| fr_power 1m: fine-tuned student vs timestral-5m zero-shot: `ft_fr_1m_distilled` vs `zoo:timestral-5m` | 1.731 | [1.613, 1.877] | b better |
| fr_power 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_fr_1m_distilled` vs `zoo:toto-2.0-4m` | 1.565 | [1.475, 1.669] | b better |
| fr_power 1y: distilled init vs scratch: `ft_fr_1y_distilled` vs `ft_fr_1y_scratch` | 1.049 | [1.028, 1.073] | b better |
| fr_power 1y: fine-tuned vs not fine-tuned: `ft_fr_1y_distilled` vs `distilled_zero_shot` | 0.723 | [0.683, 0.763] | a better |
| fr_power 1y: fine-tuned student vs timestral-209m zero-shot: `ft_fr_1y_distilled` vs `zoo:timestral-209m` | 1.219 | [1.184, 1.251] | b better |
| fr_power 1y: fine-tuned student vs toto-2.0-313m zero-shot: `ft_fr_1y_distilled` vs `zoo:toto-2.0-313m` | 1.209 | [1.182, 1.235] | b better |
| fr_power 1y: fine-tuned student vs timestral-5m zero-shot: `ft_fr_1y_distilled` vs `zoo:timestral-5m` | 1.168 | [1.134, 1.204] | b better |
| fr_power 1y: fine-tuned student vs toto-2.0-4m zero-shot: `ft_fr_1y_distilled` vs `zoo:toto-2.0-4m` | 1.056 | [1.035, 1.078] | b better |

## pedestrian_test (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| ft_ped_1y_distilled | 0.740 | [0.701, 0.781] | 0.780 [0.755, 0.806] | 0.82 |
| ft_ped_1y_scratch | 0.755 | [0.712, 0.803] | 0.796 [0.766, 0.825] | 0.79 |
| ft_ped_1m_scratch | 0.766 | [0.730, 0.803] | 0.807 [0.784, 0.831] | 0.82 |
| ft_ped_1m_distilled | 0.773 | [0.739, 0.807] | 0.815 [0.795, 0.835] | 0.81 |
| distilled_zero_shot | 0.948 | [0.905, 0.994] | – | 0.81 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.81 |
| ft_ped_1w_distilled | 1.215 | [1.139, 1.297] | 1.281 [1.233, 1.332] | 0.66 |
| ft_ped_1w_scratch | 1.929 | [1.785, 2.086] | 2.034 [1.927, 2.147] | 0.36 |

zero-shot models (same suite, same baseline): t0-gamma 0.346, chronos-2 0.419, chronos-2-small 0.459, timestral-209m 0.460, toto-2.0-313m 0.463, timestral-28m 0.491, toto-2.0-22m 0.527, timestral-5m 0.540, toto-2.0-4m 0.648, ets 1.565

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| pedestrian 1w: distilled init vs scratch: `ft_ped_1w_distilled` vs `ft_ped_1w_scratch` | 0.630 | [0.599, 0.662] | a better |
| pedestrian 1w: fine-tuned vs not fine-tuned: `ft_ped_1w_distilled` vs `distilled_zero_shot` | 1.281 | [1.233, 1.332] | b better |
| pedestrian 1w: fine-tuned student vs timestral-209m zero-shot: `ft_ped_1w_distilled` vs `zoo:timestral-209m` | 2.642 | [2.374, 2.941] | b better |
| pedestrian 1w: fine-tuned student vs toto-2.0-313m zero-shot: `ft_ped_1w_distilled` vs `zoo:toto-2.0-313m` | 2.626 | [2.369, 2.911] | b better |
| pedestrian 1w: fine-tuned student vs timestral-5m zero-shot: `ft_ped_1w_distilled` vs `zoo:timestral-5m` | 2.251 | [2.068, 2.456] | b better |
| pedestrian 1w: fine-tuned student vs toto-2.0-4m zero-shot: `ft_ped_1w_distilled` vs `zoo:toto-2.0-4m` | 1.875 | [1.738, 2.021] | b better |
| pedestrian 1m: distilled init vs scratch: `ft_ped_1m_distilled` vs `ft_ped_1m_scratch` | 1.009 | [0.997, 1.022] | no detectable difference |
| pedestrian 1m: fine-tuned vs not fine-tuned: `ft_ped_1m_distilled` vs `distilled_zero_shot` | 0.815 | [0.795, 0.835] | a better |
| pedestrian 1m: fine-tuned student vs timestral-209m zero-shot: `ft_ped_1m_distilled` vs `zoo:timestral-209m` | 1.681 | [1.549, 1.842] | b better |
| pedestrian 1m: fine-tuned student vs toto-2.0-313m zero-shot: `ft_ped_1m_distilled` vs `zoo:toto-2.0-313m` | 1.671 | [1.546, 1.817] | b better |
| pedestrian 1m: fine-tuned student vs timestral-5m zero-shot: `ft_ped_1m_distilled` vs `zoo:timestral-5m` | 1.432 | [1.353, 1.522] | b better |
| pedestrian 1m: fine-tuned student vs toto-2.0-4m zero-shot: `ft_ped_1m_distilled` vs `zoo:toto-2.0-4m` | 1.193 | [1.138, 1.252] | b better |
| pedestrian 1y: distilled init vs scratch: `ft_ped_1y_distilled` vs `ft_ped_1y_scratch` | 0.980 | [0.965, 0.996] | a better |
| pedestrian 1y: fine-tuned vs not fine-tuned: `ft_ped_1y_distilled` vs `distilled_zero_shot` | 0.780 | [0.755, 0.806] | a better |
| pedestrian 1y: fine-tuned student vs timestral-209m zero-shot: `ft_ped_1y_distilled` vs `zoo:timestral-209m` | 1.609 | [1.498, 1.742] | b better |
| pedestrian 1y: fine-tuned student vs toto-2.0-313m zero-shot: `ft_ped_1y_distilled` vs `zoo:toto-2.0-313m` | 1.600 | [1.496, 1.726] | b better |
| pedestrian 1y: fine-tuned student vs timestral-5m zero-shot: `ft_ped_1y_distilled` vs `zoo:timestral-5m` | 1.371 | [1.307, 1.445] | b better |
| pedestrian 1y: fine-tuned student vs toto-2.0-4m zero-shot: `ft_ped_1y_distilled` vs `zoo:toto-2.0-4m` | 1.142 | [1.094, 1.196] | b better |
