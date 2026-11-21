# stage_c12_varctx: relative WQL vs seasonal naive (95% CI); paired ratio vs `c11_full_hist` (<1 = better)

## shapes_ctx2048 (200 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c8_512 | 0.476 | [0.433, 0.522] | 0.883 [0.821, 0.947] | 0.82 |
| c11_full_hist | 0.539 | [0.492, 0.592] | – | 0.75 |
| mt_2048_p16_var | 0.553 | [0.504, 0.608] | 1.025 [0.986, 1.065] | 0.78 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.77 |

zero-shot models (same suite, same baseline): timestral-28m 0.399, timestral-5m 0.431, toto-2.0-4m 0.481

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| variable-length histories (consistent labels) vs full histories only: `mt_2048_p16_var` vs `c11_full_hist` | 1.025 | [0.986, 1.065] | no detectable difference |
| 2048 steps with variable histories vs 512 steps: `mt_2048_p16_var` vs `c8_512` | 1.160 | [1.086, 1.241] | b better |

## multidomain_test_ctx2048 (600 windows)  — public datasets: teachers may have seen them in pretraining

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| mt_2048_p16_var | 0.665 | [0.628, 0.707] | 0.988 [0.969, 1.010] | 0.80 |
| c11_full_hist | 0.672 | [0.637, 0.712] | – | 0.81 |
| c8_512 | 0.700 | [0.661, 0.737] | 1.041 [1.002, 1.079] | 0.79 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.67 |

zero-shot models (same suite, same baseline): timestral-28m 0.592, timestral-5m 0.606, toto-2.0-4m 0.645

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| variable-length histories (consistent labels) vs full histories only: `mt_2048_p16_var` vs `c11_full_hist` | 0.988 | [0.969, 1.010] | no detectable difference |
| 2048 steps with variable histories vs 512 steps: `mt_2048_p16_var` vs `c8_512` | 0.950 | [0.911, 0.986] | a better |

## fresh_fr_power_ctx2048 (312 windows)

| arm | rel. WQL | 95% CI | vs reference | 80% cov. |
|---|---|---|---|---|
| c11_full_hist | 0.697 | [0.650, 0.743] | – | 0.72 |
| mt_2048_p16_var | 0.715 | [0.669, 0.761] | 1.027 [1.014, 1.039] | 0.68 |
| c8_512 | 0.739 | [0.692, 0.786] | 1.061 [1.032, 1.092] | 0.70 |
| seasonal_naive | 1.000 | [1.000, 1.000] | – | 0.72 |

zero-shot models (same suite, same baseline): timestral-28m 0.626, timestral-5m 0.645, toto-2.0-4m 0.705

| contrast (paired, same windows) | ratio a/b | 95% CI | verdict |
|---|---|---|---|
| variable-length histories (consistent labels) vs full histories only: `mt_2048_p16_var` vs `c11_full_hist` | 1.027 | [1.014, 1.039] | b better |
| 2048 steps with variable histories vs 512 steps: `mt_2048_p16_var` vs `c8_512` | 0.968 | [0.942, 0.995] | a better |
