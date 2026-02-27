# Sprint 3 Research Runner

Sprint 3 uses `scripts/research_sprint3.py` with staged runs and strict zero-shot evaluation.

## Key Files

- `research/sprint3/run_registry.yaml`
- `scripts/research_sprint3.py`
- `results/research/sprint3/summary.csv`
- `docs/RESEARCH_REPORT_CSE_F1_SPRINT3.md`

## Colab + Drive Setup

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example project root on Drive:

`/content/drive/MyDrive/nids-cnn-lstm-autoencoder`

Recommended symlink targets:

- `data/raw`
- `data/research`
- `models/research`
- `results/research`

## Runner Commands

Batch A (stage0 + stage1):

```bash
python scripts/research_sprint3.py --stage-names stage0,stage1 --skip-existing
```

Batch B (stage2):

```bash
python scripts/research_sprint3.py --stage-names stage2 --skip-existing
```

Batch C (stage3 + stage4 + summarize):

```bash
python scripts/research_sprint3.py --stage-names stage3,stage4 --skip-existing
python scripts/research_sprint3.py --summarize-only
```

## Notes

- Stage2 depends on completed metrics from Stage1.
- Stage3 depends on completed metrics from Stage2.
- Stage4 templates the best Stage3 config and re-runs with seed confirmation.

