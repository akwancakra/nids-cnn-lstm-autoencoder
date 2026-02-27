# RESEARCH_REPORT_CSE_F1_SPRINT1

- generated_at: 2026-02-20T10:42:18.480813
- source_summary: `D:/Codingan/Python/nids-cnn-lstm-autoencoder/results/metrics/summary_sprint1.csv`

## Sprint Goal

- Primary KPI: minimize domain gap using composite gap score.
- Composite score = 0.50*|f1_gap| + 0.30*|auc_gap| + 0.20*|accuracy_gap|.
- Guardrails: cse_fpr <= 0.2000 and cse_f1 >= 0.2400.

## Baseline

- id=rs00_release_baseline | cse_f1=0.2780 | cse_fpr=0.2474 | cic_f1=0.7333

## Run Status

- total_runs_in_plan: 13
- completed_with_metrics: 13
- pending: 0

## Ranking (by Composite Gap Score, lower is better)

| rank | id | phase | mode | threshold_method | score | cse_f1 | cse_fpr | f1_gap | auc_gap | guardrail |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | rs02_fewshot_target_percentile_frac001 | A_threshold | few_shot | target_percentile | 0.3511 | 0.0814 | 0.0102 | 0.3978 | 0.3782 | FAIL |
| 2 | rs04_zero_shot_target_percentile_threshold_only | A_threshold | zero_shot | target_percentile | 0.3513 | 0.0813 | 0.0102 | 0.3982 | 0.3782 | FAIL |
| 3 | rs03_fewshot_target_percentile_frac002 | A_threshold | few_shot | target_percentile | 0.3517 | 0.0814 | 0.0102 | 0.3988 | 0.3782 | FAIL |
| 4 | rs11_full_scaleguard_q995_clip20 | B_preprocess | zero_shot | percentile | 0.4019 | 0.2587 | 0.1788 | 0.4833 | 0.3165 | PASS |
| 5 | rs00_release_baseline | baseline | zero_shot | percentile | 0.4088 | 0.2780 | 0.2474 | 0.4553 | 0.3782 | FAIL |
| 6 | rs01_zero_shot_percentile_replay | A_threshold | zero_shot | percentile | 0.4088 | 0.2780 | 0.2476 | 0.4553 | 0.3782 | FAIL |
| 7 | rs12_full_scaleguard_q999_clip10 | B_preprocess | zero_shot | percentile | 0.4133 | 0.2600 | 0.1739 | 0.4728 | 0.3791 | PASS |
| 8 | rs05_zero_shot_target_gaussian_k20 | A_threshold | zero_shot | target_gaussian | 0.4145 | 0.2048 | 0.0447 | 0.4933 | 0.3782 | FAIL |
| 9 | rs10_full_feature_filter_corr090 | B_preprocess | zero_shot | percentile | 0.4155 | 0.2513 | 0.2511 | 0.4736 | 0.3636 | FAIL |
| 10 | rs07_fewshot_target_percentile_frac001_ft2_lr5e4 | C_fewshot | few_shot | target_percentile | 0.4298 | 0.1109 | 0.0102 | 0.5831 | 0.2709 | FAIL |

## Ranking (by CSE F1 under FPR cap)

| rank | id | phase | mode | threshold_method | cse_f1 | cse_fpr | f1_gap | auc_gap |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | rs12_full_scaleguard_q999_clip10 | B_preprocess | zero_shot | percentile | 0.2600 | 0.1739 | 0.4728 | 0.3791 |
| 2 | rs11_full_scaleguard_q995_clip20 | B_preprocess | zero_shot | percentile | 0.2587 | 0.1788 | 0.4833 | 0.3165 |
| 3 | rs05_zero_shot_target_gaussian_k20 | A_threshold | zero_shot | target_gaussian | 0.2048 | 0.0447 | 0.4933 | 0.3782 |
| 4 | rs06_fewshot_target_gaussian_k25 | C_fewshot | few_shot | target_gaussian | 0.1665 | 0.0274 | 0.5237 | 0.3782 |
| 5 | rs07_fewshot_target_percentile_frac001_ft2_lr5e4 | C_fewshot | few_shot | target_percentile | 0.1109 | 0.0102 | 0.5831 | 0.2709 |
| 6 | rs09_fewshot_target_percentile_frac002_ft2_lr2e4 | C_fewshot | few_shot | target_percentile | 0.0929 | 0.0102 | 0.6007 | 0.2902 |
| 7 | rs08_fewshot_target_percentile_frac001_ft2_lr2e4 | C_fewshot | few_shot | target_percentile | 0.0830 | 0.0099 | 0.6034 | 0.2781 |
| 8 | rs02_fewshot_target_percentile_frac001 | A_threshold | few_shot | target_percentile | 0.0814 | 0.0102 | 0.3978 | 0.3782 |
| 9 | rs03_fewshot_target_percentile_frac002 | A_threshold | few_shot | target_percentile | 0.0814 | 0.0102 | 0.3988 | 0.3782 |
| 10 | rs04_zero_shot_target_percentile_threshold_only | A_threshold | zero_shot | target_percentile | 0.0813 | 0.0102 | 0.3982 | 0.3782 |

## Recommendation

- Current best guardrail-safe candidate: `rs11_full_scaleguard_q995_clip20` (score=0.4019, cse_f1=0.2587, cse_fpr=0.1788, f1_gap=0.4833, auc_gap=0.3165).

## Next Steps

- Complete pending runs in the plan or sweep set.
- Re-run top-2 gap-balanced candidates once for variance check.
- If no guardrail-safe gains, run one preprocess retrain candidate only.
