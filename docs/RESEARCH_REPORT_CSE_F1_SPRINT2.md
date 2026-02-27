# RESEARCH_REPORT_CSE_F1_SPRINT1

- generated_at: 2026-02-20T10:42:25.328234
- source_summary: `D:/Codingan/Python/nids-cnn-lstm-autoencoder/results/metrics/summary_sprint2.csv`

## Sprint Goal

- Primary KPI: minimize domain gap using composite gap score.
- Composite score = 0.50*|f1_gap| + 0.30*|auc_gap| + 0.20*|accuracy_gap|.
- Guardrails: cse_fpr <= 0.2000 and cse_f1 >= 0.2400.

## Baseline

- Baseline `sp20_rs11_ref_p99p5` not found or metrics missing.

## Run Status

- total_runs_in_plan: 9
- completed_with_metrics: 0
- pending: 9

Pending IDs:
- sp20_rs11_ref_p99p5
- sp21_rs11_ft_f001_e1_lr1e4_p995
- sp22_rs11_ft_f001_e2_lr2e4_p995
- sp23_rs11_ft_f002_e2_lr1e4_p995
- sp24_rs11_ft_f002_e3_lr2e4_p99
- sp25_rs11_ft_f003_e2_lr2e4_tgk18
- sp26_rs11_ft_f003_e3_lr1e4_tgk20
- sp30_retrain_w20_huber_q995
- sp31_retrain_w30_huber_q995

## Ranking (by Composite Gap Score, lower is better)

| rank | id | phase | mode | threshold_method | score | cse_f1 | cse_fpr | f1_gap | auc_gap | guardrail |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

## Ranking (by CSE F1 under FPR cap)

| rank | id | phase | mode | threshold_method | cse_f1 | cse_fpr | f1_gap | auc_gap |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |

## Recommendation

- No configuration satisfies cse_fpr cap; widen search or adjust cap.

## Next Steps

- Complete pending runs in the plan or sweep set.
- Re-run top-2 gap-balanced candidates once for variance check.
- If no guardrail-safe gains, run one preprocess retrain candidate only.
