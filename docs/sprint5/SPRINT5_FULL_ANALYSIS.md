# Sprint 5 — Full Run Analysis

- **Sprint:** Sprint 5 Plan v3 — CSE F1 / FPR Reduction
- **Branch:** `feat/sprint5-cse-fpr-reduction`
- **Objective:** Maximize CSE recall dibawah guardrail CIC FPR rendah (zero-shot cross-domain NIDS)
- **Environment:** Google Colab, Tesla T4 (13.6 GB), TF 2.19.0, Python 3.12 (`/usr/bin/python3`)
- **Generated at:** 2026-03-01

---

## Daftar Isi

1. [Konfigurasi Sprint 5](#konfigurasi-sprint-5)
2. [Bug Fixes yang Diterapkan](#bug-fixes-yang-diterapkan)
3. [Run 1 — Pre-Fix (Baseline)](#run-1--pre-fix-baseline)
4. [Run 2 — Post-Fix (2026-03-01)](#run-2--post-fix-2026-03-01)
5. [Gate Decision](#gate-decision)
6. [Analisis Mendalam — Trade-off CSE Recall vs CIC FPR](#analisis-mendalam--trade-off-cse-recall-vs-cic-fpr)
7. [Rekomendasi & Sprint 6 Plan](#rekomendasi--sprint-6-plan)

---

## Konfigurasi Sprint 5

### Struktur Stage

| Stage       | Run IDs                                              | Tahapan                                | Kondisi Eksekusi                      |
| ----------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| **Stage 0** | `s5_00_lock_baseline`, `s5_m07_rerun`                | `preprocess,train,eval` / `train,eval` | always                                |
| **Stage 1** | `s5_t01` s/d `s5_t07` (7 runs)                       | `eval` only                            | always                                |
| **Stage 2** | —                                                    | —                                      | skipped (tidak ada run di sprint ini) |
| **Stage 3** | `s5_h01_guardrail_fpr012`, `s5_h02_guardrail_fpr015` | `eval` only                            | always                                |
| **Stage 4** | `s5_c01_seed42`, `s5_c02_seed1234`                   | `preprocess,train,eval`                | `gate_pass = true`                    |

### Gate Rules

| Constraint     | Target                                      |
| -------------- | ------------------------------------------- |
| CSE recall     | ≥ `adaptive_recall_target` (floor = 0.75)   |
| CSE precision  | ≥ 0.30                                      |
| CIC FPR        | ≤ 0.10                                      |
| Collapse guard | recall ≥ 0.95 AND precision < 0.20 → reject |

`adaptive_recall_target = max(0.75, best_stage1_recall_under_guardrail + 0.0)`

### Registry Contract

- `base_config_hash`: dihitung dengan `json.dumps(cfg, sort_keys=True, separators=(',',':'))` → SHA-256
- Hash diverifikasi saat dry-run (Cell 9 notebook)
- Score mode eval: `recon_mse` (independen dari training loss)

---

## Bug Fixes yang Diterapkan

Sebelum Run 2, lima bug kritis diperbaiki:

### FIX-1 — Chain Resolution `reuse_data_from_best_stage` (P0)

**File:** `scripts/sprint5/research_runner.py`

**Masalah:** Fungsi `apply_data_reuse_inputs()` memilih `s5_m07_rerun` sebagai best-stage0 run (recall 0.698 > lock_baseline 0.626), lalu mengeset `shard_dir = data/sprint5/s5_m07_rerun/processed/shards`. Namun `s5_m07_rerun` sendiri memakai `reuse_data_from: s5_00_lock_baseline` (tidak punya preprocessed data di isolated dir-nya).

**Dampak:** Seluruh 7 run Stage 1 crash dengan `FileNotFoundError: data/sprint5/s5_m07_rerun/processed/cic_val.npz`.

**Fix:** Tambah fungsi `resolve_data_origin(runs, run_id, max_depth=10)` yang mengikuti rantai `reuse_data_from` hingga menemukan run dengan `preprocess` di `stages`. Diwire di dua lokasi: explicit reuse dan dynamic best-stage reuse.

```python
def resolve_data_origin(runs, run_id, max_depth=10):
    """Follow reuse_data_from chain until finding a run with 'preprocess' in stages."""
    seen = set()
    current = run_id
    for _ in range(max_depth):
        if current in seen:
            break  # circular chain guard
        seen.add(current)
        run_cfg = runs.get(current, {})
        if 'preprocess' in run_cfg.get('stages', []):
            return current
        next_id = run_cfg.get('reuse_data_from')
        if not next_id:
            break
        current = str(next_id)
    return run_id  # fallback
```

---

### FIX-2 — `sample_size: 0 → 200000` (P1)

**File:** `config/sprint5/base.yaml` (L91)

**Masalah:** `evaluation.sample_size: 0` membuat `ReservoirSampler` tidak terinisialisasi, sehingga `roc_auc` tidak bisa dihitung (menghasilkan `None`).

**Fix:** Ubah ke `sample_size: 200000`.

**Side effect:** Menginvalidasi `base_config_hash` → perlu update `run_registry.yaml`.

---

### FIX-3 — Label Notebook `Sprint4 → Sprint5` (P2)

**File:** `notebooks/sprint5_colab_runner.ipynb`

**Masalah:** Fungsi `run_stage()` memakai prefix `'Sprint4 {stage_name}'`. Path `report_path` menunjuk ke `RESEARCH_REPORT_CSE_F1_SPRINT4.md`. Cell workaround lama tidak dihapus.

**Fix:** Update semua label ke Sprint5, fix report path ke `SPRINT5.md`, hapus cell workaround.

---

### FIX-4 — PR-AUC Ditambahkan (Enhancement)

**File:** `scripts/eval_metrics.py`

**Fix:** Tambah `average_precision_score` di tiga lokasi:

- Shard path (setelah `ReservoirSampler`)
- Non-shard path (CIC + CSE)
- Generalization gap (`pr_auc_gap`)

---

### FIX-5 — Reset `run_status.json` (Operational)

**File:** `results/sprint5/runtime/run_status.json`

**Fix:** Reset status Stage 1–4 dari `failed_experiment` ke `pending` agar runner tidak skip runs yg perlu dijalankan ulang. Stage 0 (`success`) dibiarkan untuk di-skip dengan `--skip-existing`.

---

### Hash Fix — `base_config_hash` Update (P0.5)

**File:** `research/sprint5/run_registry.yaml`

**Masalah:** Setelah FIX-2 (`sample_size: 200000`), dry-run Cell 9 di Colab menghasilkan `ValueError: Registry base_config_hash mismatch`.

**Fix:** Update hash dari `b095223ef6403efcbe386f689cc36bd...` ke `d34ad4d8057193b8782b44a494664b0a...`.

---

## Run 1 — Pre-Fix (Baseline)

### Status Run Keseluruhan

| Status                 | Jumlah | Detail                                       |
| ---------------------- | -----: | -------------------------------------------- |
| `success`              |      2 | Stage 0 saja (`lock_baseline` + `m07_rerun`) |
| `failed_experiment`    |      7 | Seluruh Stage 1 — `FileNotFoundError`        |
| `skipped_inconclusive` |      2 | Stage 3 — bergantung stage1 yg gagal         |
| `skipped_gate`         |      2 | Stage 4 — gate_pass=False                    |

### Metrics Stage 0 (yang berhasil)

| Run                                | CSE Recall | CSE F1    | CSE FPR | CIC FPR | F1 Gap    | Wall Time |
| ---------------------------------- | ---------- | --------- | ------- | ------- | --------- | --------- |
| `s5_00_lock_baseline`              | 0.626      | 0.729     | 0.224   | ~0.10   | 0.103     | ~33m 50s  |
| `s5_m07_rerun` (mse_mae_mix α=0.7) | **0.698**  | **0.781** | 0.224   | ~0.11   | **0.046** | ~31m 36s  |

**Temuan Run 1:**

- `mse_mae_mix` loss menghasilkan CSE F1 lebih tinggi (+5.2pp) dan F1 gap jauh lebih kecil (0.046 vs 0.103)
- Gate decision: `gate_pass: false`, reason: `stage3_no_candidate` (stage1 semua gagal, stage3 bergantung stage1)
- `sprint_history_summary.count_under_guardrail = 0` — tidak ada run yang lolos CIC FPR guardrail

---

## Run 2 — Post-Fix (2026-03-01)

### Status Run Keseluruhan

| Status              | Jumlah | Detail                                  |
| ------------------- | -----: | --------------------------------------- |
| `success`           | **11** | Stage 0 (2) + Stage 1 (7) + Stage 3 (2) |
| `skipped_gate`      |      2 | Stage 4 — by design, gate_pass=False    |
| `failed_experiment` |  **0** | ✓ Tidak ada kegagalan                   |

**FIX-1 terkonfirmasi:** Stage 1 sukses 7/7. Chain `s5_m07_rerun → s5_00_lock_baseline` teresolve dengan benar.

---

### Stage 0 — Baseline (skipped via `--skip-existing`, pakai artifact lama)

| Run                   | CSE Recall | CSE Prec   | CSE F1     | CSE FPR | F1 Gap     | Wall Time |
| --------------------- | ---------- | ---------- | ---------- | ------- | ---------- | --------- |
| `s5_00_lock_baseline` | 0.6259     | 0.8741     | 0.7295     | 0.2237  | 0.1029     | 33m 51s   |
| `s5_m07_rerun`        | **0.6984** | **0.8858** | **0.7810** | 0.2235  | **0.0462** | 31m 36s   |

---

### Stage 1 — Target Percentile Variants (eval-only, model: `s5_00_lock_baseline`)

| Run                             | Threshold |     CIC F1 | CIC FPR | CSE Recall |   CSE Prec |     CSE F1 |    CSE FPR | F1 Gap |
| ------------------------------- | --------: | ---------: | ------: | ---------: | ---------: | ---------: | ---------: | -----: |
| `s5_t01_target_p95`             |     8.890 |     0.7788 |  0.0108 |     0.0907 |     0.8183 |     0.1633 |     0.0500 | 0.6155 |
| `s5_t02_target_p97`             |    10.727 |     0.7723 |  0.0051 |     0.0444 |     0.7860 |     0.0841 |     0.0300 | 0.6882 |
| `s5_t03_target_p98`             |    12.901 |     0.7519 |  0.0013 |     0.0153 |     0.6554 |     0.0299 |     0.0200 | 0.7219 |
| `s5_t04_target_p99`             |    14.621 |     0.6831 |  0.0007 |     0.0066 |     0.6221 |     0.0131 |     0.0100 | 0.6699 |
| `s5_t05_target_p97_sub5pct`     |    10.991 |     0.7710 |  0.0043 |     0.0399 |     0.7776 |     0.0758 |     0.0283 | 0.6952 |
| `s5_t06_target_p97_sub10pct`    |    10.861 |     0.7716 |  0.0048 |     0.0417 |     0.7802 |     0.0792 |     0.0292 | 0.6925 |
| `s5_t07_source_calib_guardrail` |     4.076 | **0.8324** |  0.1030 | **0.6259** | **0.8741** | **0.7295** | **0.2237** | 0.1029 |

**Observasi Stage 1:**

- Semakin tinggi percentile → threshold makin tinggi → CSE recall makin collapse (0.09 → 0.007)
- CIC FPR sangat rendah di p95–p99 (0.0007–0.0108) tapi CSE recall tidak viable
- Sub-sampling CSE benign (5%, 10%) di p97 tidak signifikan mengubah hasil
- `source_calib_guardrail` memberikan recall tertinggi (0.626) tapi CIC FPR = 0.103 (sedikit melanggar guardrail 0.10)
- **Tidak ada run Stage 1 yang memenuhi dua constraint sekaligus** (recall ≥ 0.75 DAN CIC FPR ≤ 0.10)

---

### Stage 3 — Guardrail Relax (eval-only, model dari `/content/sprint5_lock/`)

| Run                       | Threshold |     CIC F1 | CIC FPR | CSE Recall | CSE Prec |     CSE F1 | CSE FPR |     F1 Gap |
| ------------------------- | --------: | ---------: | ------: | ---------: | -------: | ---------: | ------: | ---------: |
| `s5_h01_guardrail_fpr012` |     3.822 |     0.8422 |  0.1207 |     0.6501 |   0.8701 |     0.7442 |  0.2409 |     0.0980 |
| `s5_h02_guardrail_fpr015` |     3.496 | **0.8474** |  0.1466 | **0.6925** |   0.8651 | **0.7692** |  0.2681 | **0.0782** |

**Observasi Stage 3:**

- Relaxed guardrail (FPR 0.12/0.15 vs 0.10) memang membuka recall lebih tinggi (0.65–0.69)
- F1 gap sangat kecil (0.078–0.098) — generalisasi terbaik di antara semua konfigurasi
- Namun CSE FPR tetap tinggi (0.24–0.27) karena threshold ikut turun (ambang deteksi lebih longgar)
- **`s5_h02`** adalah kandidat terbaik overall: CSE F1=0.769, recall=0.693, gap=0.078

---

### Stage 4 — Seed Confirmation (skipped — gate_pass=False)

| Run               | Status         | Keterangan                          |
| ----------------- | -------------- | ----------------------------------- |
| `s5_c01_seed42`   | `skipped_gate` | Gate tidak lolos, di-skip by design |
| `s5_c02_seed1234` | `skipped_gate` | Gate tidak lolos, di-skip by design |

---

## Gate Decision

```json
{
  "generated_at": "2026-03-01T10:31:20.773978",
  "gate_pass": false,
  "reason": "failed_constraints",
  "adaptive_recall_target": 0.75,
  "adaptive_target_reasoning": "max(0.7500, baseline(0.0907)+0.0000)",
  "best_stage3_run_id": "s5_h02_guardrail_fpr015",
  "best_stage3_metrics": {
    "cse_recall": 0.6925,
    "cse_precision": 0.8651,
    "cse_f1": 0.7692,
    "cic_fpr": 0.1466,
    "threshold": 3.496
  },
  "pivot_recommendation": "USAD",
  "stage_validity": {
    "stage1": { "required": 7, "valid_count": 7, "passed": true },
    "stage3": { "required": 2, "valid_count": 2, "passed": true },
    "stage4": {
      "required": 2,
      "valid_count": 0,
      "passed": true,
      "status": "skipped_by_design"
    }
  },
  "sprint_history_summary": {
    "recall_p50": 0.0408,
    "recall_p90": 0.0676,
    "best_recall_under_guardrail": 0.0907,
    "count_all": 11,
    "count_under_guardrail": 6
  }
}
```

### Breakdown Kegagalan Gate

| Constraint                   | Target | Best Achieved | Run    | Status         |
| ---------------------------- | ------ | ------------- | ------ | -------------- |
| CSE recall ≥ adaptive target | ≥ 0.75 | **0.6925**    | s5_h02 | ❌ gap -0.057  |
| CSE precision ≥ 0.30         | ≥ 0.30 | 0.8651        | s5_h02 | ✓              |
| CIC FPR ≤ 0.10               | ≤ 0.10 | **0.1466**    | s5_h02 | ❌ +0.047 over |

**Dari 11 valid runs:**

- 6 runs berhasil melewati CIC FPR ≤ 0.10 (stage1 runs)
- Best recall di antara yang lolos guardrail: **0.0907** (`s5_t01_target_p95`) — jauh dari target 0.75
- Distribusi recall stage1: p50=0.041, p90=0.068 — sangat skewed rendah

---

## Analisis Mendalam — Trade-off CSE Recall vs CIC FPR

### Pola yang Diamati

```
threshold lebih rendah  → recall ↑, FPR ↑ (lebih banyak anomali terdeteksi)
threshold lebih tinggi  → recall ↓, FPR ↓ (lebih selektif)
```

Masalahnya: **CIC dan CSE punya distribusi reconstruction error yang berbeda signifikan**.

- Di threshold rendah (guardrail ~3.5–4.0): CIC FPR sudah 0.10–0.15 tapi CSE recall baru 0.63–0.69
- Di threshold tinggi (p95–p99: 8.9–14.6): CIC FPR turun ke 0.001–0.011 tapi CSE recall collapse ke 0.007–0.091
- **Sweet spot yang memenuhi kedua constraint tidak ada** pada model ini

### Root Cause Fundamental

Model dilatih pada CIC-IDS2017 (source). Reconstruction error baseline benign di CIC sudah tinggi (distribution shift internal). Ketika diaplikasikan ke CSE-CIC-IDS2018 (target):

1. Benign CSE punya pola berbeda → reconstruction error berbeda dari CIC benign
2. Anomali CSE punya karakteristik berbeda → skor anomali tumpang tindih dengan benign di threshold tertentu
3. Akibatnya, threshold yang "aman" untuk CIC (FPR rendah) terlalu ketat untuk CSE (recall rendah)

### Metrik yang Masih Bermasalah

**`roc_auc = None` di semua run** — meskipun `sample_size: 200000` diset di `base.yaml`, nilai `auc_gap = 0.0` sebagai sentinel mengkonfirmasi AUC global tidak terhitung. Kemungkinan penyebab:

1. Generated configs di `research/sprint5/generated_configs/*.yaml` masih override `sample_size: 0`
2. Path pembacaan config di shard mode berbeda dari yang di base.yaml

Untuk didiagnosis di sprint berikutnya:

```python
# Di eval.py / eval_metrics.py, cek:
sample_size = cfg['evaluation'].get('sample_size', 0)
print(f"[DEBUG] sample_size = {sample_size}")
# Verifikasi apakah ini 0 atau 200000 di runtime
```

---

## Rekomendasi & Sprint 6 Plan

### Opsi Tindak Lanjut Gate Failure

| Opsi                                         | Deskripsi                                                                 | Effort  | Risk                            | Rekomendasi                                 |
| -------------------------------------------- | ------------------------------------------------------------------------- | ------- | ------------------------------- | ------------------------------------------- |
| **USAD Pivot**                               | Gate recommend USAD — arsitektur berbeda (unsupervised anomaly detection) | High    | Medium                          | ⭐ Sesuai gate recommendation               |
| **Relax CIC FPR → 0.15**                     | `s5_h02` hampir masuk (FPR=0.147), naikkan batas                          | Low     | High (CIC degradasi signifikan) | 🟡 Jika constraint operasional memungkinkan |
| **Retrain dengan focal loss / weighted BCE** | Tingkatkan CSE recall lewat loss weighting di training                    | High    | Low                             | 🟡 Butuh retraining massal                  |
| **Domain Adaptation**                        | Fine-tune pada subset CSE benign tanpa label anomali                      | Medium  | Low                             | 🟡 Perlu strategi split yang hati-hati      |
| **Terima `s5_h02` sebagai final**            | CSE F1=0.769, recall=0.693 — mendekati target                             | Minimal | Acceptable                      | 🔵 Jika target akademis, bukan produksi     |

### Best Candidate per Stage (Run 2)

```
Best CSE F1 overall   : s5_h02_guardrail_fpr015  → CSE F1=0.769, recall=0.693, CIC FPR=0.147
Best CSE recall       : s5_m07_rerun             → CSE recall=0.698, CSE F1=0.781, CIC FPR=0.224
Best transfer gap     : s5_h02_guardrail_fpr015  → F1 gap=0.078
Best CIC guard        : s5_t04_target_p99        → CIC FPR=0.0007, CSE recall=0.007 (collapsed)
Best CIC+CSE balance  : s5_t01_target_p95        → CIC FPR=0.0108, CSE recall=0.091 (tidak viable)
```

### Checklist Sprint 6

- [ ] **Diagnose `roc_auc=None`**: cek apakah generated configs override `sample_size`
- [ ] **Putuskan arah:** USAD pivot vs relax CIC FPR vs accept s5_h02
- [ ] Jika USAD: setup pipeline baru dengan data CSE-CIC-IDS2018 sebagai primary training set
- [ ] Jika relax constraint: update gate rules di `run_registry.yaml` → `cic_fpr_guardrail: 0.15`
- [ ] Fix PR-AUC tidak muncul di `summary.csv` (saat ini hanya ada di metrics JSON per-run)
- [ ] Dokumentasikan pilihan constraint di `docs/sprint6/SPRINT6_PLAN.md`

---

## Appendix — Wall Time Ringkasan

| Stage   | Run                             | Wall Time       |
| ------- | ------------------------------- | --------------- |
| Stage 0 | `s5_00_lock_baseline`           | 33m 51s (2030s) |
| Stage 0 | `s5_m07_rerun`                  | 31m 36s (1896s) |
| Stage 1 | `s5_t01_target_p95`             | 2m 29s (149s)   |
| Stage 1 | `s5_t02`–`s5_t06`               | ~50s each       |
| Stage 1 | `s5_t07_source_calib_guardrail` | 19m 15s (1155s) |
| Stage 3 | `s5_h01_guardrail_fpr012`       | 19m 47s (1187s) |
| Stage 3 | `s5_h02_guardrail_fpr015`       | 19m 41s (1180s) |
| Stage 4 | `s5_c01`, `s5_c02`              | 0s (skipped)    |

**Total wall time Run 2:** ~2h 45m  
**Stage 1 (t07) dan Stage 3 lambat** karena `source_calib_guardrail` method harus scanning seluruh CSE shards (54 shards) untuk menentukan threshold, bukan hanya CIC calib split.
