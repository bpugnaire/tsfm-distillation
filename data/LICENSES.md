# Data and model licences

| Asset | Source | Licence | Use | Checked |
|---|---|---|---|---|
| RTE éCO2mix national/regional (real-time + consolidated) | https://odre.opendatasoft.com (datasets `eco2mix-national-tr`, `eco2mix-regional-tr`, `eco2mix-national-cons-def`, `eco2mix-regional-cons-def`) | Licence Ouverte v2.0 (Etalab) | train / val / fresh test | 2026-11-12 |
| Synthetic shapes + KernelSynth-style series | generated (this repo) | MIT (repo) | train / gallery | 2026-11-12 |
| Toto 2.0 (4m, 22m, 313m) | https://huggingface.co/Datadog | Apache-2.0 | teacher / comparator | 2026-11-12 |
| t0-gamma | https://huggingface.co/theforecastingcompany/t0-gamma | Apache-2.0 (not gated) | teacher | 2026-11-12 |
| Chronos-2, Chronos-2-small | https://huggingface.co/amazon/chronos-2, autogluon/chronos-2-small | Apache-2.0 | comparator only | 2026-11-12 |
| Timestral 5M/28M/123M/209M | https://huggingface.co/mistralai | Modified MIT (revenue clause > $20M/month); code Apache-2.0 | comparator (owner confirmed eligibility 2026-11-13, personal project) | 2026-11-13 |
| GIFT-Eval test data + harness | https://huggingface.co/datasets/Salesforce/GiftEval, https://github.com/SalesforceAIResearch/gift-eval (commit 0b57dd6) | Apache-2.0 (dataset card + LICENSE.txt) | evaluation only (public benchmark, flagged) | 2026-11-16 |
| Chronos-Bolt tiny / mini | https://huggingface.co/amazon/chronos-bolt-tiny, amazon/chronos-bolt-mini | Apache-2.0 | comparator only | 2026-11-16 |
| IBM Granite TTM-r2 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2 | Apache-2.0 | comparator only | 2026-11-16 |
| raws-labs/tinycast (146k params; not this project) | https://huggingface.co/raws-labs/tinycast (rev 9635a27), https://github.com/raws-labs/tinycast (commit 628c0be) | Apache-2.0 | comparator only (GIFT-Eval) | 2026-11-21 |

## Note on distillation from Timestral (2026-11-13)
The Timestral licence does not mention distillation or training on its outputs. Its revenue restriction "applies to the Model and any
derivatives, modifications, or combined works based on it"; whether a distilled student is a "derivative" is undefined. Until clarified,
a student distilled (even partly) from Timestral is released with Timestral's attribution notice and licence conditions, or the released
model is distilled from Apache-2.0 teachers only (pending owner decision, docs/status.yaml).

## City of Melbourne pedestrian counts (fresh pedestrian test, `fresh_mel_ped`, fine-tuning domain `mel_ped`)
- Source: City of Melbourne Open Data, "Pedestrian Counting System (counts per hour)",
  https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour
- Licence: CC BY (Creative Commons Attribution). Downloaded 2026-11-27 (rolling two years, 2024-11-27 to 2026-11-26).
- Use: evaluation (targets after splits.fresh_start) and fine-tuning on the history before it; never in distillation.
