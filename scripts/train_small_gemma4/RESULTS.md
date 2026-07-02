# Gemma4-12B DSpark Results

This file records successful DSpark + Gemma4-12B reproduction/eval runs on branch `dev/train_small_data`.

## Shared Setup

- Target model: `google/gemma-4-12B-it`
- DSpark config: `config/dspark/dspark_gemma4_12b_small.py`
- Eval tasks: `gsm8k:32,mt-bench:16,alpaca:32`
- Proposal length: approximately `5.00+1`

---

## Run A: 1k smoke baseline

- Draft checkpoint: `/home/aiscuser/checkpoints/deepspec_small/dspark_block5_gemma4_12b_1k/step_latest`
- Training data: 1000 regenerated Open-PerfectBlend samples
- Target cache: `/home/aiscuser/.cache/deepspec/gemma4_12b_target_cache_1k`
- Valid cache samples: 964 / 1000
- Train budget: 200 steps

### Acceptance Metrics

| dataset | target_model | draft_model | #propose | accept_len | verify_rate | accept_rate@0 | accept_rate@1 | accept_rate@2 | accept_rate@3 | accept_rate@4 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | gemma-4-12B-it | step_latest | 5.00+1 | 1.36 | 0.2268 | 0.2864 | 0.0679 | 0.0066 | 0.0001 | 0.0000 |
| mt-bench | gemma-4-12B-it | step_latest | 5.00+1 | 1.17 | 0.1954 | 0.1503 | 0.0196 | 0.0022 | 0.0000 | 0.0000 |
| alpaca | gemma-4-12B-it | step_latest | 5.00+1 | 1.16 | 0.1941 | 0.1478 | 0.0153 | 0.0016 | 0.0001 | 0.0000 |

### Confidence Head Reliability Metrics

| dataset | samples | proposals | ece_mean | auc_mean | brier_mean | pred_mean | target_mean | ece@0 | ece@1 | ece@2 | ece@3 | ece@4 | auc@0 | auc@1 | auc@2 | auc@3 | auc@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 32 | 7925 | 0.0077 | 0.7470 | 0.0393 | 0.0714 | 0.0722 | 0.0233 | 0.0128 | 0.0016 | 0.0005 | 0.0001 | 0.8175 | 0.8515 | 0.8745 | 0.4446 | NaN |
| mt-bench | 16 | 4956 | 0.0053 | 0.8512 | 0.0243 | 0.0371 | 0.0344 | 0.0210 | 0.0043 | 0.0014 | 0.0001 | 0.0000 | 0.7844 | 0.8994 | 0.8699 | NaN | NaN |
| alpaca | 32 | 9522 | 0.0051 | 0.7767 | 0.0235 | 0.0370 | 0.0330 | 0.0223 | 0.0026 | 0.0007 | 0.0000 | 0.0000 | 0.7945 | 0.9007 | 0.9196 | 0.4920 | NaN |

### Notes

- This was a smoke-test baseline, not an expected paper-level result.
- The pipeline successfully ran end to end: SGLang serving, 1000-sample regeneration, target-cache preparation, DSpark training, and small eval.
- Acceptance is low, especially after position 0, indicating severe suffix decay / undertraining with the 1k-sample 200-step run.

---

## Run B: 50k / step_3000

- Draft checkpoint: `/home/aiscuser/checkpoints/deepspec_small/dspark_block5_gemma4_12b_50k/step_3000`
- Training data: 50,000 regenerated Open-PerfectBlend samples
- Target cache: `/home/aiscuser/.cache/deepspec/gemma4_12b_target_cache_50k`
- Valid cache samples: 48,583 / 50,000
- Train budget: 3,000 steps

### Acceptance Metrics

| dataset | target_model | draft_model | #propose | accept_len | verify_rate | accept_rate@0 | accept_rate@1 | accept_rate@2 | accept_rate@3 | accept_rate@4 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | gemma-4-12B-it | step_3000 | 4.98+1 | 3.49 | 0.5840 | 0.7794 | 0.6252 | 0.4760 | 0.3594 | 0.2676 |
| mt-bench | gemma-4-12B-it | step_3000 | 5.00+1 | 1.97 | 0.3292 | 0.4929 | 0.2400 | 0.1243 | 0.0713 | 0.0473 |
| alpaca | gemma-4-12B-it | step_3000 | 4.99+1 | 1.85 | 0.3087 | 0.4602 | 0.1970 | 0.0964 | 0.0592 | 0.0402 |

### Confidence Head Reliability Metrics

| dataset | samples | proposals | ece_mean | auc_mean | brier_mean | pred_mean | target_mean | ece@0 | ece@1 | ece@2 | ece@3 | ece@4 | auc@0 | auc@1 | auc@2 | auc@3 | auc@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 32 | 3114 | 0.0387 | 0.8891 | 0.1247 | 0.5357 | 0.5020 | 0.0282 | 0.0400 | 0.0471 | 0.0427 | 0.0354 | 0.9096 | 0.8853 | 0.8814 | 0.8854 | 0.8836 |
| mt-bench | 16 | 2964 | 0.0193 | 0.9123 | 0.0769 | 0.2031 | 0.1952 | 0.0356 | 0.0220 | 0.0144 | 0.0126 | 0.0116 | 0.8660 | 0.8823 | 0.9103 | 0.9496 | 0.9536 |
| alpaca | 32 | 6026 | 0.0162 | 0.9176 | 0.0687 | 0.1822 | 0.1707 | 0.0241 | 0.0257 | 0.0157 | 0.0106 | 0.0049 | 0.8692 | 0.8853 | 0.9202 | 0.9465 | 0.9668 |

### 1k → 50k Acceptance Improvement

| dataset | accept_len 1k | accept_len 50k | Δ accept_len | accept_rate@0 1k | accept_rate@0 50k | accept_rate@1 1k | accept_rate@1 50k |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.36 | 3.49 | +2.13 | 0.2864 | 0.7794 | 0.0679 | 0.6252 |
| mt-bench | 1.17 | 1.97 | +0.80 | 0.1503 | 0.4929 | 0.0196 | 0.2400 |
| alpaca | 1.16 | 1.85 | +0.69 | 0.1478 | 0.4602 | 0.0153 | 0.1970 |

### Notes

- 50k/step_3000 substantially improves acceptance over the 1k smoke baseline.
- GSM8K improves the most: `accept_len` increases from 1.36 to 3.49 and suffix acceptance remains non-trivial through position 4.
- MT-Bench and Alpaca also improve, but their later-position acceptance remains much lower than GSM8K, suggesting strong domain/task dependence.
- Confidence head AUC is strong across datasets (~0.89-0.92), but confidence is mildly over-confident in aggregate (`pred_mean > target_mean`).
