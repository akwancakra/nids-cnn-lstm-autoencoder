# Changelog — Sprint 4 → Sprint 5

## Perubahan Utama

### 1. Fix s4_m07 Crash

- **File:** `scripts/eval_metrics.py`
- **Perubahan:** `load_model(args.model)` → `load_model(args.model, compile=False)`
- **Alasan:** Model dengan custom loss (`mse_mae_mix`) gagal deserialisasi saat load tanpa `custom_objects`. Evaluasi hanya perlu inferensi, tidak perlu loss/optimizer.
- **Dampak:** Run s4_m07 / s5_m07_rerun dapat dieval tanpa crash.

### 2. Dukungan target_percentile di Shard Path

- **File:** `scripts/eval_metrics.py`
- **Perubahan:** Untuk `threshold_method in ("target_percentile", "target_gaussian")`:
  - Collect CSE scores + labels dari shard CSE test.
  - Filter `target_benign_errors = cse_scores[cse_labels == 0]`.
  - Opsional: `evaluation.target_benign_sample_frac` untuk semi-blind subsample.
  - Kirim ke `compute_threshold_value(...)`.
- **Dampak:** Threshold bisa dikalibrasi dari CSE benign (domain target) untuk mitigasi domain shift.

### 3. Sprint History Summary Fix

- **File:** `scripts/sprint5/research_runner.py`
- **Perubahan:** `collect_history_pairs` sekarang **include** `sprint4` dan `sprint5` (tidak skip).
- **Perubahan:** Tambah `_assert_history_populated(root, history)` — fail early jika ada `*_cse_metrics.json` di results/sprint4 atau sprint5 tetapi history kosong (indikasi bug include logic).
- **Dampak:** `sprint_history_summary` tidak lagi kosong saat hasil run ada.

### 4. Gate Sprint 5

- **CSE recall min:** 0.75 (fixed floor), naik dari 0.40 adaptive.
- **Pivot policy:** Sama seperti Sprint 4 (USAD on fail).
- **Stage validity:** stage1: 7, stage2: 0, stage3: 2, stage4: 2.

### 5. Struktur File Sprint 5

| Path | Deskripsi |
| ---- | --------- |
| `config/sprint5/base.yaml` | Base config, paths ke sprint5. |
| `config/sprint5/profiles/hybrid_zero_shot_anomaly.yaml` | Profile evaluasi. |
| `research/sprint5/run_registry.yaml` | 12 run, stage 0/1/3/4. |
| `scripts/sprint5/` | preprocess.py, train.py, eval.py, research_runner.py. |
| `notebooks/sprint5_colab_runner.ipynb` | Colab runner untuk Sprint 5. |
| `docs/sprint5/SPRINT5_PLAN.md` | Ringkasan plan. |
| `docs/sprint5/CHANGELOG_FROM_S4.md` | Dokumen ini. |

## Path Changes

| Sprint 4 | Sprint 5 |
| --------- | -------- |
| `data/sprint4/` | `data/sprint5/` |
| `models/sprint4/` | `models/sprint5/` |
| `results/sprint4/` | `results/sprint5/` |
