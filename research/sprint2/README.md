# Sprint2 Research (Full Isolation)

Sprint2 sekarang memakai **full isolation per `run_id`** supaya eksperimen tidak saling mengganggu.

## Struktur

- `research/sprint2/run_registry.yaml`: sumber kebenaran semua run Sprint2.
- `research/sprint2/generated_configs/`: config final per run (auto-generated).
- `config/base.yaml`: baseline config global.
- `config/profiles/*.yaml`: profile config per skenario (hybrid/baseline, zero/few-shot).

## Runner

- `scripts/research_sprint2.py`

Runner akan:
1. merge `base + profile + run overrides`,
2. force path terisolasi per `run_id`,
3. untuk eval-only yang punya `reuse_artifacts_from`, input data/model diambil read-only dari run sumber,
4. tetap menulis output evaluasi ke folder run baru.

## Namespace Output

Untuk setiap `run_id=<RID>`:

- `data/research/<RID>/processed`
- `models/research/<RID>`
- `results/research/<RID>`
- `reports/research/<RID>` (opsional, jika ditambah alur report per-run)

Agregasi sprint:

- `results/research/sprint2/summary.csv`
- `docs/RESEARCH_REPORT_CSE_F1_SPRINT2.md`

## Command

### 1) Dry-run (cek command + generated config)

```powershell
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml --dry-run
```

### 2) Jalankan semua run aktif

```powershell
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml
```

### 3) Jalankan run tertentu saja

```powershell
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml --run-ids s2_hybrid_zero_seed42_base,s2_hybrid_few_seed42_tp99_eval
```

### 4) Summarize only

```powershell
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml --summarize-only
```

### 5) Skip run yang metrics-nya sudah ada

```powershell
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml --skip-existing
```

## Seed Policy

Primary:
1. run aktif default seed `42`,
2. rerun kandidat terpilih dengan seed `1234` (template run sudah ada, tinggal aktifkan `active: true`).
