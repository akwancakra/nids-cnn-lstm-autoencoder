# Sprint2 Research Runs (CSE F1 Under FPR Cap)

Sprint2 fokus menaikkan `cse_f1` dengan guardrail `cse_fpr <= 0.20`.

## Plan file

- `research/sprint2/experiments.yaml`

## Run commands

### Dry run

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint2/experiments.yaml --base-config config.yaml --dry-run
```

### Execute active runs

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint2/experiments.yaml --base-config config.yaml --skip-existing --summary-csv results/metrics/summary_sprint2.csv --report-md docs/RESEARCH_REPORT_CSE_F1_SPRINT2.md
```

### Summarize only

```powershell
c:\Users\nitro\.conda\envs\nids-tf210gpu\python.exe scripts/research_sprint1.py --plan research/sprint2/experiments.yaml --base-config config.yaml --summarize-only --summary-csv results/metrics/summary_sprint2.csv --report-md docs/RESEARCH_REPORT_CSE_F1_SPRINT2.md
```

## Notes

- Active runs default: eval/few-shot only.
- Full retrain candidates (`sp30`, `sp31`) diset `active: false`; nyalakan setelah shortlist eval selesai.
