# stage_c11_ctx: relative WQL vs seasonal naive (95% CI); paired ratio vs `c8_mt_512` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_2048_p64 | 0.464 | [0.418, 0.516] | 0.974 [0.926, 1.022] | 0.86 |
| c8_mt_512 | 0.476 | [0.433, 0.522] | – | 0.82 |
| mt_2048_p16 | 0.539 | [0.492, 0.592] | 1.132 [1.056, 1.217] | 0.75 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 2048 steps at patch 16 vs 512 steps: `mt_2048_p16` vs `c8_mt_512` | 1.132 | [1.056, 1.217] | b better |
| 2048 steps at patch 64 vs 512 steps: `mt_2048_p64` vs `c8_mt_512` | 0.974 | [0.926, 1.022] | no detectable difference |
| patch 16 vs 64 at 2048 steps: `mt_2048_p16` vs `mt_2048_p64` | 1.162 | [1.109, 1.222] | b better |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_2048_p64 | 0.667 | [0.637, 0.701] | 0.954 [0.926, 0.985] | 0.81 |
| mt_2048_p16 | 0.672 | [0.637, 0.712] | 0.961 [0.927, 0.998] | 0.81 |
| c8_mt_512 | 0.700 | [0.661, 0.737] | – | 0.79 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 2048 steps at patch 16 vs 512 steps: `mt_2048_p16` vs `c8_mt_512` | 0.961 | [0.927, 0.998] | a better |
| 2048 steps at patch 64 vs 512 steps: `mt_2048_p64` vs `c8_mt_512` | 0.954 | [0.926, 0.985] | a better |
| patch 16 vs 64 at 2048 steps: `mt_2048_p16` vs `mt_2048_p64` | 1.007 | [0.989, 1.028] | no detectable difference |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_2048_p16 | 0.697 | [0.650, 0.743] | 0.942 [0.916, 0.969] | 0.72 |
| c8_mt_512 | 0.739 | [0.692, 0.786] | – | 0.70 |
| mt_2048_p64 | 0.748 | [0.706, 0.788] | 1.011 [0.991, 1.032] | 0.65 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| 2048 steps at patch 16 vs 512 steps: `mt_2048_p16` vs `c8_mt_512` | 0.942 | [0.916, 0.969] | a better |
| 2048 steps at patch 64 vs 512 steps: `mt_2048_p64` vs `c8_mt_512` | 1.011 | [0.991, 1.032] | no detectable difference |
| patch 16 vs 64 at 2048 steps: `mt_2048_p16` vs `mt_2048_p64` | 0.932 | [0.907, 0.959] | a better |
