# RESEARCH REPORT CSE F1 - SPRINT 4

- generated_at: 2026-02-27T21:26:27.134453
- source_summary: `D:/Codingan/python/nids-cnn-lstm-autoencoder/results/sprint4/summary.csv`
- gate_decision: `D:/Codingan/python/nids-cnn-lstm-autoencoder/results/sprint4/gate_decision.json`

## Ringkasan / Summary

- Objective: maximize CSE recall dengan guardrail CIC FPR rendah.
- Objective (EN): maximize CSE recall under low CIC-FPR guardrail.
- Gate pass: `False` (reason: `stage3_no_candidate`).
- Adaptive recall target: `0.4` (max(0.4000, baseline(0.3442)+0.0500)).

## Stage Validity

| stage | required_valid_runs | valid_runs | status |
| --- | ---: | ---: | --- |
| stage1 | 5 | 0 | inconclusive |
| stage2 | 6 | 0 | inconclusive |
| stage3 | 5 | 0 | inconclusive |
| stage4 | 2 | 0 | skipped_by_design |

## Best Candidate Per Stage

| stage | run_id | mode | cse_recall | cse_precision | cse_f1 | cic_fpr |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| stage0 | - | - | - | - | - | - |
| stage1 | - | - | - | - | - | - |
| stage2 | - | - | - | - | - | - |
| stage3 | - | - | - | - | - | - |
| stage4 | - | - | - | - | - | - |

## Operational Notes

- ID Primary: Terminologi teknis tetap English untuk konsistensi.
- EN Mirror: Technical keywords are intentionally kept in English.

## Run Status

- total_runs: 24
- success: 1
- failed_experiment: 0
- failed_infra_exhausted: 0
- pending_or_skipped: 23
