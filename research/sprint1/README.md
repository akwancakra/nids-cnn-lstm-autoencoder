# Sprint1 Research Runs (CSE F1)

This folder defines the experiment matrix for `CSE_F1_SPRINT1`.

## Files

- `research/sprint1/experiments.yaml`: run definitions (A/B/C batches).
- `research/sprint1/generated_configs/`: auto-generated per-run configs (created by runner).

## Runner

Use `scripts/research_sprint1.py`.

### Dry-run command preview

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --dry-run
```

### Run selected IDs only

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --run-ids rs02_fewshot_target_percentile_frac001,rs03_fewshot_target_percentile_frac002
```

### Summarize only (no run execution)

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --summarize-only
```

### Eval-only threshold sweep (no retrain)

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --sweep-threshold --skip-existing
```

Optional overrides:

- `--candidate-tags rs10_full_feature_filter_corr090,rs11_full_scaleguard_q995_clip20,rs12_full_scaleguard_q999_clip10`
- `--sweep-percentiles 97,98,99,99.5`
- `--sweep-methods percentile,target_percentile,target_gaussian`
- `--sweep-k-sigmas 1.8,2.0,2.2`
- `--guardrail-cse-fpr 0.20 --guardrail-cse-f1 0.24`
- `--weight-f1-gap 0.5 --weight-auc-gap 0.3 --weight-accuracy-gap 0.2`

## Outputs

- `results/metrics/summary_sprint1.csv`
- `docs/RESEARCH_REPORT_CSE_F1_SPRINT1.md`

Summary columns now include:

- `composite_gap_score`
- `passes_guardrail`
- `rank_gap_balanced`
