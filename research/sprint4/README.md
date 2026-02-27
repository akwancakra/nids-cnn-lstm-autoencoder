# Sprint 4 Research Runner

Sprint 4 uses `scripts/sprint4/research_runner.py` with anomaly-first ranking and CIC-FPR guardrail.

## Key Files

- `research/sprint4/run_registry.yaml`
- `config/sprint4/base.yaml`
- `scripts/sprint4/research_runner.py`
- `results/sprint4/summary.csv`
- `results/sprint4/gate_decision.json`
- `docs/sprint4/RESEARCH_REPORT_CSE_F1_SPRINT4.md`

## Colab + Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example root:

`/content/drive/MyDrive/nids-cnn-lstm-autoencoder`

## Runner Examples

Stage 0:

```bash
python scripts/sprint4/research_runner.py --stage-names stage0 --skip-existing
```

Stage 1:

```bash
python scripts/sprint4/research_runner.py --stage-names stage1 --skip-existing
```

Stage 2:

```bash
python scripts/sprint4/research_runner.py --stage-names stage2 --skip-existing
```

Stage 3 + Stage 4 (conditional):

```bash
python scripts/sprint4/research_runner.py --stage-names stage3,stage4 --skip-existing
python scripts/sprint4/research_runner.py --summarize-only
```
