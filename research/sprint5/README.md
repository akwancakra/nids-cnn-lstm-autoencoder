# Sprint 5 — CSE FPR Reduction & Domain Shift

Sprint 5 fokus pada pengurangan CSE FPR melalui target-aware threshold dan CIC guardrail relaxation.

## Quick Start

```bash
# Dry-run
python scripts/sprint5/research_runner.py --dry-run --stage-names stage0

# Execute stages
python scripts/sprint5/research_runner.py --stage-names stage0 --skip-existing
python scripts/sprint5/research_runner.py --stage-names stage1 --skip-existing
python scripts/sprint5/research_runner.py --stage-names stage3,stage4 --skip-existing

# Summarize
python scripts/sprint5/research_runner.py --summarize-only
```

## Registry

- **run_registry.yaml** — 12 run: Stage 0 (2), Stage 1 (7), Stage 3 (2), Stage 4 (2).

## Gate

- CSE recall min: 0.75
- CIC FPR max: 0.10
- CSE precision min: 0.30
